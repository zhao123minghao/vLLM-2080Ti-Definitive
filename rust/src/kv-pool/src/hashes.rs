// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Port of `BlockHashListWithBlockSize` and `resolve_block_hashes` from
//! `vllm/v1/core/kv_cache_utils.py`.
//!
//! The vLLM scheduler can hold the same prefix at two granularities: hashes are computed at
//! `hash_block_size`, while a KV cache group may use a larger `block_size`. Because every
//! hash chains over its whole prefix, the *last* `hash_block_size` hash inside a target block
//! already fingerprints that block, so the conversion is a lazy index view rather than a
//! re-hash. This module reproduces that view, including the parts that are easy to get wrong
//! in a rewrite:
//!
//! * the index mapping `(idx + 1) * scale_factor - 1`, with Python's negative indices and
//!   `IndexError` for anything out of range;
//! * Python slice semantics (`indices()` normalisation, negative steps, `ValueError` for a
//!   zero step);
//! * `resolve_block_hashes`, which keeps the raw hashes for fine-grained partial hits, passes
//!   an existing view through, and asserts that the sizes divide each other.
//!
//! Hashes are opaque bytes here: the parity harness carries the values Python computed.

use crate::PyError;

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Lazy view of `block_hashes` at a coarser `target_block_size`.
#[derive(Debug, Clone)]
pub struct BlockHashListView {
    hashes: Vec<Vec<u8>>,
    scale_factor: usize,
}

impl BlockHashListView {
    /// Mirror of `BlockHashListWithBlockSize(...)`, including its assert.
    ///
    /// A zero `hash_block_size` raises `ZeroDivisionError` in Python; the variant exists so
    /// the log line matches rather than silently reporting an assertion.
    pub fn new(
        hashes: Vec<Vec<u8>>,
        hash_block_size: usize,
        target_block_size: usize,
    ) -> Result<Self, PyError> {
        if hash_block_size == 0 {
            return Err(PyError::Zero);
        }
        if !target_block_size.is_multiple_of(hash_block_size) {
            return Err(PyError::Assert);
        }
        Ok(Self {
            hashes,
            scale_factor: target_block_size / hash_block_size,
        })
    }

    /// `__len__`: the number of target-size blocks covered.
    pub fn len(&self) -> usize {
        self.hashes.len() / self.scale_factor
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scale factor between `hash_block_size` and `target_block_size`.
    pub fn scale_factor(&self) -> usize {
        self.scale_factor
    }

    /// `_get_value_at`: the hash at target-block index `idx` (Python allows negatives).
    fn value_at(&self, idx: i64) -> Result<&[u8], PyError> {
        let raw = (idx + 1) * self.scale_factor as i64 - 1;
        let resolved = if raw < 0 {
            let shifted = raw + self.hashes.len() as i64;
            if shifted < 0 {
                return Err(PyError::Index);
            }
            shifted
        } else {
            raw
        };
        self.hashes.get(resolved as usize).map(Vec::as_slice).ok_or(PyError::Index)
    }

    /// `__iter__` / the list produced by a slice, materialised.
    fn values(&self) -> Vec<Vec<u8>> {
        (0..self.len() as i64)
            .map(|index| self.value_at(index).expect("in-range index").to_vec())
            .collect()
    }

    /// `__getitem__` for one integer index, with Python's negative-index handling.
    pub fn get(&self, index: i64) -> Result<Vec<u8>, PyError> {
        self.value_at(index).map(<[u8]>::to_vec)
    }

    /// `__getitem__` for a slice, mirroring `slice.indices()` plus `range()`.
    ///
    /// `None` stands for a missing part of the slice, exactly like `slice(None, None, None)`.
    pub fn slice(
        &self,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Vec<Vec<u8>>, PyError> {
        let (start, stop, step) = slice_indices(self.len(), start, stop, step)?;
        let mut out = Vec::new();
        let mut index = start;
        if step > 0 {
            while index < stop {
                out.push(self.value_at(index)?);
                index += step;
            }
        } else {
            while index > stop {
                out.push(self.value_at(index)?);
                index += step;
            }
        }
        Ok(out.into_iter().map(|hash| hash.to_vec()).collect())
    }
}

/// CPython's `PySlice_AdjustIndices`, so the port slices exactly like Python.
fn slice_indices(
    length: usize,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<(i64, i64, i64), PyError> {
    let step = step.unwrap_or(1);
    if step == 0 {
        // Python: ValueError("slice step cannot be zero")
        return Err(PyError::Value);
    }
    let len = length as i64;

    let adjust = |value: i64, is_start: bool| -> i64 {
        let mut value = value;
        if value < 0 {
            value += len;
            if value < 0 {
                return if step < 0 { -1 } else { 0 };
            }
        } else if value >= len {
            return if step < 0 { len - 1 } else { len };
        }
        if is_start && value < 0 && step < 0 {
            return -1;
        }
        value
    };

    let start = match start {
        Some(value) => adjust(value, true),
        None => {
            if step < 0 {
                len - 1
            } else {
                0
            }
        }
    };
    let stop = match stop {
        Some(value) => adjust(value, false),
        None => {
            if step < 0 {
                -1
            } else {
                len
            }
        }
    };
    Ok((start, stop, step))
}

/// Either the raw hashes or a view over them — what `block_hashes` can be at a call site.
#[derive(Debug, Clone)]
pub enum HashList {
    /// `list[BlockHash]`, i.e. hashes at `hash_block_size` granularity.
    Raw(Vec<Vec<u8>>),
    /// `BlockHashListWithBlockSize`.
    View(BlockHashListView),
}

impl HashList {
    /// Number of entries, whatever the representation.
    pub fn len(&self) -> usize {
        match self {
            HashList::Raw(hashes) => hashes.len(),
            HashList::View(view) => view.len(),
        }
    }

    /// Whether the list holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every entry, materialised in order.
    pub fn values(&self) -> Vec<Vec<u8>> {
        match self {
            HashList::Raw(hashes) => hashes.clone(),
            HashList::View(view) => view.values(),
        }
    }

    /// One entry by index.
    pub fn get(&self, index: i64) -> Result<Vec<u8>, PyError> {
        match self {
            HashList::Raw(hashes) => {
                let resolved = if index < 0 {
                    let shifted = index + hashes.len() as i64;
                    if shifted < 0 {
                        return Err(PyError::Index);
                    }
                    shifted
                } else {
                    index
                };
                hashes.get(resolved as usize).cloned().ok_or(PyError::Index)
            }
            HashList::View(view) => view.get(index),
        }
    }

    /// One slice.
    pub fn slice(
        &self,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Vec<Vec<u8>>, PyError> {
        match self {
            HashList::Raw(hashes) => {
                let (start, stop, step) = slice_indices(hashes.len(), start, stop, step)?;
                let mut out = Vec::new();
                let mut index = start;
                if step > 0 {
                    while index < stop {
                        out.push(hashes[index as usize].clone());
                        index += step;
                    }
                } else {
                    while index > stop {
                        out.push(hashes[index as usize].clone());
                        index += step;
                    }
                }
                Ok(out)
            }
            HashList::View(view) => view.slice(start, stop, step),
        }
    }

    /// Scale factor when this is a view (Python reads `block_hashes.scale_factor`).
    pub fn scale_factor(&self) -> Option<usize> {
        match self {
            HashList::Raw(_) => None,
            HashList::View(view) => Some(view.scale_factor()),
        }
    }
}

/// `resolve_block_hashes`: the view at `block_size` for the given input.
pub fn resolve_block_hashes(
    block_hashes: &HashList,
    hash_block_size: usize,
    block_size: usize,
    supports_fine_grained_hash_lookup: bool,
    alignment_tokens: Option<usize>,
) -> Result<HashList, PyError> {
    if block_size == hash_block_size {
        return Ok(block_hashes.clone());
    }
    if let HashList::View(view) = block_hashes {
        // Python asserts the existing view is already at this block size.
        if hash_block_size == 0 || view.scale_factor() != block_size / hash_block_size {
            return Err(PyError::Assert);
        }
        return Ok(block_hashes.clone());
    }
    // Fine-grained partial hits keep the raw hashes.
    if supports_fine_grained_hash_lookup
        && let Some(alignment) = alignment_tokens
        && alignment < block_size
        && block_size.is_multiple_of(alignment)
    {
        return Ok(block_hashes.clone());
    }
    if hash_block_size == 0 {
        return Err(PyError::Zero);
    }
    if !block_size.is_multiple_of(hash_block_size) {
        return Err(PyError::Assert);
    }
    let HashList::Raw(hashes) = block_hashes else {
        unreachable!("views returned above");
    };
    Ok(HashList::View(BlockHashListView::new(
        hashes.clone(),
        hash_block_size,
        block_size,
    )?))
}

/// Format a hash list the way every trace log line prints block hashes.
pub fn format_hashes(hashes: &[Vec<u8>]) -> String {
    hashes.iter().map(|hash| hex_encode(hash)).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(names: &[&str]) -> Vec<Vec<u8>> {
        names.iter().map(|name| name.as_bytes().to_vec()).collect()
    }

    #[test]
    fn the_view_maps_each_target_block_to_its_last_hash() {
        // scale 2: target block 0 -> hash 1, target block 1 -> hash 3.
        let view = BlockHashListView::new(hashes(&["a", "b", "c", "d"]), 16, 32).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0).unwrap(), b"b".to_vec());
        assert_eq!(view.get(1).unwrap(), b"d".to_vec());
        assert_eq!(view.get(2), Err(PyError::Index));
        // Python negative indices fall out of the same arithmetic.
        assert_eq!(view.get(-1).unwrap(), b"d".to_vec());
        assert_eq!(view.get(-2).unwrap(), b"b".to_vec());
        assert_eq!(view.get(-3), Err(PyError::Index));
    }

    #[test]
    fn scale_one_is_the_identity() {
        let view = BlockHashListView::new(hashes(&["a", "b"]), 16, 16).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.values(), hashes(&["a", "b"]));
    }

    #[test]
    fn a_non_multiple_target_block_size_is_rejected() {
        assert_eq!(
            BlockHashListView::new(hashes(&["a"]), 16, 24).map(|_| ()),
            Err(PyError::Assert)
        );
        assert_eq!(
            BlockHashListView::new(hashes(&["a"]), 0, 16).map(|_| ()),
            Err(PyError::Zero)
        );
    }

    #[test]
    fn slices_follow_python_rules() {
        let view = BlockHashListView::new(hashes(&["a", "b", "c", "d", "e", "f"]), 1, 2).unwrap();
        // View of [a,b,c,d,e,f] at scale 2 is [b,d,f].
        assert_eq!(view.len(), 3);
        assert_eq!(
            view.slice(None, None, None).unwrap(),
            hashes(&["b", "d", "f"])
        );
        assert_eq!(
            view.slice(Some(1), None, None).unwrap(),
            hashes(&["d", "f"])
        );
        assert_eq!(
            view.slice(None, Some(-1), None).unwrap(),
            hashes(&["b", "d"])
        );
        assert_eq!(
            view.slice(None, None, Some(-1)).unwrap(),
            hashes(&["f", "d", "b"])
        );
        assert_eq!(
            view.slice(Some(0), Some(0), None).unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            view.slice(None, Some(99), None).unwrap(),
            hashes(&["b", "d", "f"])
        );
        assert_eq!(view.slice(None, None, Some(0)), Err(PyError::Value));
    }

    #[test]
    fn resolve_keeps_raw_hashes_for_fine_grained_hits() {
        let raw = HashList::Raw(hashes(&["a", "b", "c", "d"]));
        // Same size: the input is returned untouched.
        let same = resolve_block_hashes(&raw, 16, 16, false, None).unwrap();
        assert_eq!(same.values(), hashes(&["a", "b", "c", "d"]));

        // Coarser block size: a view.
        let view = resolve_block_hashes(&raw, 16, 32, false, None).unwrap();
        assert_eq!(view.scale_factor(), Some(2));
        assert_eq!(view.values(), hashes(&["b", "d"]));

        // Fine-grained lookup with a smaller alignment keeps the raw hashes.
        let fine = resolve_block_hashes(&raw, 16, 32, true, Some(16)).unwrap();
        assert_eq!(fine.scale_factor(), None);
        assert_eq!(fine.values(), hashes(&["a", "b", "c", "d"]));

        // An existing view passes through, and a mismatched one is rejected.
        let again = resolve_block_hashes(&view, 16, 32, false, None).unwrap();
        assert_eq!(again.values(), hashes(&["b", "d"]));
        assert_eq!(
            resolve_block_hashes(&view, 16, 64, false, None).map(|_| ()),
            Err(PyError::Assert)
        );

        // A block size that does not divide asks for an assertion (48 would scale by 3).
        assert_eq!(
            resolve_block_hashes(&raw, 16, 40, false, None).map(|_| ()),
            Err(PyError::Assert)
        );
    }
}
