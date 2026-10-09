// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Port of `BlockPool` from `vllm/v1/core/block_pool.py`.
//!
//! This is the layer on top of [`FreeQueue`](crate::FreeQueue) and
//! [`BlockHashToBlockMap`](crate::BlockHashToBlockMap): reference counts, prefix-cache
//! bookkeeping per block, the eviction order of freed blocks, and the reset path.
//!
//! Out of scope on purpose (see `docs/v1-core-rust-migration.md`): the KV-cache event
//! queue, the metrics collector, hashing itself (the trace carries the hashes that Python
//! computed), and `cache_full_blocks` / `cache_partial_block`, which need a `Request`.
//! Cached entries are therefore installed through the same private helper Python uses
//! ([`BlockPool::insert_block_hash`] mirrors `_insert_block_hash`).
//!
//! Everything a caller can observe about the state — the free list, reference counts, the
//! per-block hash metadata, the cached-hash map, the pending reuse watchers — is what
//! [`BlockPool::dump`] prints, which is what the parity harness compares.

use crate::{BlockHashToBlockMap, FreeQueue, PyError};

/// Per-block state that Python keeps on `KVCacheBlock`.
#[derive(Debug, Clone, Default)]
struct BlockState {
    ref_cnt: i64,
    is_null: bool,
    hash: Option<Vec<u8>>,
    num_tokens: Option<i64>,
    /// `cached_block_hashes_by_block[block_id]`: the extra hashes pointing at a block that
    /// already had a primary hash. Python stores a set; this keeps them sorted because a
    /// set has no order to compare against.
    extra_hashes: Vec<Vec<u8>>,
}

/// Port of `BlockPool`.
pub struct BlockPool {
    num_gpu_blocks: usize,
    enable_caching: bool,
    blocks: Vec<BlockState>,
    queue: FreeQueue,
    map: BlockHashToBlockMap,
    /// `_reuse_watchers`: block ids whose contents are still being read. The callbacks are
    /// no-ops in the harness, but *when they fire* is state, so the ids are kept.
    reuse_watchers: Vec<u64>,
}

impl BlockPool {
    /// Mirror of `BlockPool(num_gpu_blocks=..., enable_caching=...)`.
    ///
    /// `hash_block_size` is not modelled: it is only used by the hashing paths, which are
    /// out of scope here.
    pub fn new(num_gpu_blocks: usize, enable_caching: bool) -> Result<Self, PyError> {
        if num_gpu_blocks == 0 {
            // Python asserts `num_gpu_blocks > 0`.
            return Err(PyError::Assert);
        }
        let mut queue = FreeQueue::new(num_gpu_blocks);
        let mut blocks = vec![BlockState::default(); num_gpu_blocks];
        // The pool reserves one block as the null block; its ref count is never maintained.
        let null_block = queue.popleft()?;
        blocks[null_block].is_null = true;
        Ok(Self {
            num_gpu_blocks,
            enable_caching,
            blocks,
            queue,
            map: BlockHashToBlockMap::new(),
            reuse_watchers: Vec::new(),
        })
    }

    /// `BlockPool.get_num_free_blocks`.
    pub fn num_free_blocks(&self) -> i64 {
        self.queue.num_free_blocks()
    }

    /// `BlockPool.get_usage`.
    pub fn usage(&self) -> f64 {
        let total_gpu_blocks = self.num_gpu_blocks as i64 - 1;
        if total_gpu_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.num_free_blocks() as f64 / total_gpu_blocks as f64)
    }

    /// Block ids in the free list, in eviction order.
    pub fn free_order(&self) -> Vec<u64> {
        self.queue.all_free_blocks().iter().map(|id| *id as u64).collect()
    }

    /// `BlockPool.get_new_blocks`.
    pub fn get_new_blocks(&mut self, num_blocks: usize) -> Result<Vec<u64>, PyError> {
        if self.num_free_blocks() < num_blocks as i64 {
            // Python raises ValueError("Cannot get N free blocks from the pool").
            return Err(PyError::Value);
        }
        let ret: Vec<u64> = self.queue.popleft_n(num_blocks)?.iter().map(|id| *id as u64).collect();
        self.notify_reuse(&ret);
        for id in &ret {
            let index = *id as usize;
            if self.enable_caching {
                self.maybe_evict_cached_block(*id);
            }
            if self.blocks[index].ref_cnt != 0 {
                // Python asserts the popped block was unreferenced.
                return Err(PyError::Assert);
            }
            self.blocks[index].ref_cnt += 1;
        }
        Ok(ret)
    }

    /// `BlockPool._notify_reuse`: every waiting watcher fires and is dropped.
    fn notify_reuse(&mut self, blocks: &[u64]) {
        for id in blocks {
            self.reuse_watchers.retain(|watcher| watcher != id);
        }
    }

    /// `BlockPool.unpin_blocks`.
    pub fn unpin_blocks(&mut self, blocks: &[u64]) -> Result<(), PyError> {
        let mut released: Vec<u64> = Vec::new();
        for id in blocks {
            let index = *id as usize;
            let block = &mut self.blocks[index];
            if block.ref_cnt <= 0 || block.is_null {
                // Python asserts `ref_cnt > 0 and not is_null`.
                return Err(PyError::Assert);
            }
            block.ref_cnt -= 1;
            if !self.reuse_watchers.contains(id) {
                self.reuse_watchers.push(*id);
            }
            if block.ref_cnt == 0 {
                released.push(*id);
            }
        }
        self.queue.append_n(&released.iter().map(|id| *id as usize).collect::<Vec<_>>())
    }

    /// `BlockPool.free_blocks`.
    pub fn free_blocks(&mut self, ordered_blocks: &[u64]) -> Result<(), PyError> {
        // Cached blocks are reused last (LRU), uncached blocks first (LIFO locality).
        let mut to_evict_last: Vec<usize> = Vec::new();
        let mut to_evict_first: Vec<usize> = Vec::new();
        for id in ordered_blocks {
            let index = *id as usize;
            let block = &mut self.blocks[index];
            block.ref_cnt -= 1;
            if block.ref_cnt == 0 && !block.is_null {
                if block.hash.is_none() || !self.enable_caching {
                    to_evict_first.push(index);
                } else {
                    to_evict_last.push(index);
                }
            }
        }
        self.queue.prepend_n(&to_evict_first)?;
        self.queue.append_n(&to_evict_last)
    }

    /// `BlockPool.touch`.
    pub fn touch(&mut self, blocks: &[u64]) -> Result<(), PyError> {
        for id in blocks {
            let index = *id as usize;
            if self.blocks[index].ref_cnt == 0 && !self.blocks[index].is_null {
                self.queue.remove(index)?;
            }
            self.blocks[index].ref_cnt += 1;
        }
        Ok(())
    }

    /// `BlockPool.is_block_writable`.
    pub fn is_block_writable(&self, id: u64) -> bool {
        let block = &self.blocks[id as usize];
        !block.is_null && block.ref_cnt == 1 && block.hash.is_none()
    }

    /// `BlockPool.get_cached_block`: the first cached block for a key, if any.
    pub fn get_cached_block(&self, key: &[u8]) -> Option<u64> {
        self.map.get_one_block(key)
    }

    /// `BlockPool._insert_block_hash`. `num_tokens` only applies to the primary insertion,
    /// exactly like Python.
    pub fn insert_block_hash(
        &mut self,
        key: &[u8],
        id: u64,
        num_tokens: Option<i64>,
    ) -> Result<(), PyError> {
        let index = id as usize;
        if self.blocks[index].hash.as_deref() == Some(key) {
            return Ok(());
        }
        if self.map.contain(key, id) {
            return Ok(());
        }
        if self.blocks[index].hash.is_none() {
            // Python's set_block_hash asserts both hash fields are empty.
            if self.blocks[index].num_tokens.is_some() {
                return Err(PyError::Assert);
            }
            let block = &mut self.blocks[index];
            block.hash = Some(key.to_vec());
            block.num_tokens = num_tokens;
        } else {
            let extras = &mut self.blocks[index].extra_hashes;
            if !extras.iter().any(|hash| hash.as_slice() == key) {
                extras.push(key.to_vec());
                extras.sort();
            }
        }
        self.map.insert(key, id);
        Ok(())
    }

    /// `BlockPool._maybe_evict_cached_block`: drop a block's cache entries.
    pub fn maybe_evict_cached_block(&mut self, id: u64) -> bool {
        !self.remove_cached_block_hashes(id).is_empty()
    }

    /// `BlockPool._remove_cached_block_hashes`.
    fn remove_cached_block_hashes(&mut self, id: u64) -> Vec<Vec<u8>> {
        let index = id as usize;
        let mut hashes: Vec<Vec<u8>> = Vec::new();
        if let Some(hash) = &self.blocks[index].hash {
            hashes.push(hash.clone());
        }
        hashes.append(&mut self.blocks[index].extra_hashes);
        if hashes.is_empty() {
            return Vec::new();
        }
        let mut removed = Vec::new();
        for hash in &hashes {
            if self.map.pop(hash, id).is_some() {
                removed.push(hash.clone());
            }
        }
        // `block.reset_hash()`.
        self.blocks[index].hash = None;
        self.blocks[index].num_tokens = None;
        removed
    }

    /// `BlockPool.evict_blocks`.
    pub fn evict_blocks(&mut self, block_ids: &[u64]) -> Result<(), PyError> {
        for id in block_ids {
            if *id as usize >= self.blocks.len() {
                // Python asserts the id was allocated by the scheduler.
                return Err(PyError::Assert);
            }
            self.maybe_evict_cached_block(*id);
        }
        Ok(())
    }

    /// `BlockPool.reset_prefix_cache`: only possible when every block but the null block is
    /// free. Returns whether the reset happened.
    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used_blocks = self.num_gpu_blocks as i64 - self.num_free_blocks();
        if num_used_blocks != 1 {
            return false;
        }
        self.map = BlockHashToBlockMap::new();
        let watchers: Vec<u64> = self.free_order();
        self.notify_reuse(&watchers);
        for block in &mut self.blocks {
            // `block.reset_hash()` for every block, including the null block.
            block.hash = None;
            block.num_tokens = None;
            block.extra_hashes.clear();
        }
        true
    }

    /// The full observable state, one canonical string.
    ///
    /// Format (space separated, dash for absent values):
    /// `free=<n> map=<n> watchers=<ids|-> id:ref:hash:tokens:extras ...`
    pub fn dump(&self) -> String {
        // Watcher ids come out of a Python dict, so the dump sorts them — and the Python side
        // sorts as well. Nothing about a watcher depends on when it was registered.
        let mut watchers = self.reuse_watchers.clone();
        watchers.sort_unstable();
        let mut tokens = vec![
            format!("free={}", self.num_free_blocks()),
            format!("map={}", self.map.len()),
            format!(
                "watchers={}",
                join_or_dash(watchers.iter().map(u64::to_string))
            ),
        ];
        for (index, block) in self.blocks.iter().enumerate() {
            let extras = block
                .extra_hashes
                .iter()
                .map(|hash| hex_encode(hash))
                .collect::<Vec<_>>()
                .join(",");
            tokens.push(format!(
                "{}:{}:{}:{}:{}",
                index,
                block.ref_cnt,
                block.hash.as_deref().map(hex_encode).unwrap_or_else(|| "-".to_string()),
                block
                    .num_tokens
                    .map(|tokens| tokens.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                if extras.is_empty() {
                    "-".to_string()
                } else {
                    extras
                },
            ));
        }
        tokens.join(" ")
    }
}

fn join_or_dash<I: Iterator<Item = String>>(values: I) -> String {
    let joined = values.collect::<Vec<_>>().join(",");
    if joined.is_empty() {
        "-".to_string()
    } else {
        joined
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_reserves_a_null_block() {
        let mut pool = BlockPool::new(4, true).unwrap();
        assert_eq!(pool.num_free_blocks(), 3);
        assert_eq!(pool.free_order(), vec![1, 2, 3]);
        assert!(pool.blocks[0].is_null);
        assert_eq!(pool.get_new_blocks(3), Ok(vec![1, 2, 3]));
        // The null block is never handed out.
        assert_eq!(pool.get_new_blocks(1), Err(PyError::Value));
    }

    #[test]
    fn freed_blocks_are_reused_lifo_and_cached_ones_lru() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(3).unwrap();
        // 1 has a hash, 2 does not: 2 goes to the front (reused first), 1 to the back.
        pool.insert_block_hash(b"aa", 1, Some(32)).unwrap();
        pool.free_blocks(&blocks).unwrap();
        assert_eq!(pool.free_order(), vec![2, 3, 1]);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn eviction_drops_the_cache_entry_but_keeps_the_ref_count() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        let id = blocks[0];
        pool.insert_block_hash(b"aa", id, Some(32)).unwrap();
        assert_eq!(pool.get_cached_block(b"aa"), Some(id));
        pool.free_blocks(&blocks).unwrap();
        // Cached blocks are reused last, so a free block keeps serving the cache.
        assert_eq!(pool.get_cached_block(b"aa"), Some(id));
        // Handing it out again is what evicts the entry; it is the last free block to go.
        let reused = pool.get_new_blocks(3).unwrap();
        assert_eq!(reused.last(), Some(&id));
        assert_eq!(pool.get_cached_block(b"aa"), None);
        assert_eq!(pool.blocks[id as usize].ref_cnt, 1);
        assert_eq!(pool.blocks[id as usize].hash, None);
    }

    #[test]
    fn touch_unpins_a_cached_block_from_the_free_list() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.free_blocks(&blocks).unwrap();
        assert_eq!(pool.num_free_blocks(), 3);
        pool.touch(&blocks).unwrap();
        assert_eq!(pool.num_free_blocks(), 2);
        assert_eq!(pool.blocks[blocks[0] as usize].ref_cnt, 1);
    }

    #[test]
    fn unpin_registers_a_reuse_watcher() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.unpin_blocks(&blocks).unwrap();
        assert_eq!(pool.reuse_watchers, vec![blocks[0]]);
        assert_eq!(pool.num_free_blocks(), 3);
        // Handing the block out again fires and clears the watcher.
        pool.get_new_blocks(3).unwrap();
        assert!(pool.reuse_watchers.is_empty());
    }

    #[test]
    fn reset_requires_every_block_but_the_null_block_to_be_free() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.insert_block_hash(b"aa", blocks[0], Some(32)).unwrap();
        assert!(!pool.reset_prefix_cache());
        pool.free_blocks(&blocks).unwrap();
        assert!(pool.reset_prefix_cache());
        assert_eq!(pool.get_cached_block(b"aa"), None);
        assert_eq!(pool.blocks[blocks[0] as usize].hash, None);
    }

    #[test]
    fn a_second_hash_on_the_same_block_lands_in_the_extra_set() {
        let mut pool = BlockPool::new(4, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        let id = blocks[0];
        pool.insert_block_hash(b"aa", id, Some(32)).unwrap();
        pool.insert_block_hash(b"bb", id, None).unwrap();
        assert_eq!(pool.blocks[id as usize].extra_hashes, vec![b"bb".to_vec()]);
        assert_eq!(pool.get_cached_block(b"bb"), Some(id));
        assert!(pool.maybe_evict_cached_block(id));
        assert_eq!(pool.get_cached_block(b"aa"), None);
        assert_eq!(pool.get_cached_block(b"bb"), None);
        // A second eviction finds nothing left to drop.
        assert!(!pool.maybe_evict_cached_block(id));
    }

    #[test]
    fn dump_is_stable_and_contains_the_state() {
        let mut pool = BlockPool::new(3, true).unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.insert_block_hash(b"aa", blocks[0], Some(32)).unwrap();
        let dump = pool.dump();
        assert_eq!(
            dump,
            "free=1 map=1 watchers=- 0:0:-:-:- 1:1:6161:32:- 2:0:-:-:-"
        );
    }
}
