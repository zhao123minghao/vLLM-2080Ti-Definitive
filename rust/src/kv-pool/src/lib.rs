// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Structural port of the KV-cache block bookkeeping in `vllm/v1/core`.
//!
//! This is the second step of the `v1/core` migration spike after the prefix-cache
//! hashing, and it follows the same rule as the first one: the Python code is the
//! source of truth and every difference is a bug in this crate.
//!
//! Covered here:
//!
//! * [`FreeQueue`] — `FreeKVCacheBlockQueue` from
//!   `vllm/v1/core/kv_cache_utils.py`: a doubly linked list threaded through the
//!   blocks' `prev_free_block` / `next_free_block` fields, with fake head and tail
//!   sentinels. Python mutates the blocks; this crate stores the same links in an
//!   index-addressed table, which is the natural shape for a Rust port.
//! * [`BlockHashToBlockMap`] — the map of the same name in `vllm/v1/core/block_pool.py`,
//!   where one hash key holds either a single block or an insertion-ordered group of
//!   blocks.
//! * [`BlockPool`] — the pool built on both of them: reference counts, the free-list order
//!   after `free_blocks`, cache eviction, the reuse watchers and the reset path.
//!
//! Validation lives in `tools/check_block_pool_parity.sh`: `tools/gen_block_pool_trace.py`
//! drives the real Python classes with a seeded random operation sequence and records
//! both the trace and the resulting log; [`replay`] runs the same trace through this
//! crate and the two logs are diffed line by line.
//!
//! Operations that Python rejects are reported as the exception it would have raised
//! ([`PyError`]), because "raises the same error" is part of the contract.

use std::fmt::Write as _;

pub mod hashes;
pub mod pool;

pub use hashes::{BlockHashListView, HashList, format_hashes, resolve_block_hashes};
pub use pool::BlockPool;
use pool::RequestView;

/// The Python exception a rejected operation corresponds to.
///
/// The variants are named after the Python exception classes on purpose: the log
/// produced by [`replay`] must match `type(exc).__name__` on the Python side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyError {
    /// `ValueError`, for example popping from an empty free queue or a slice step of zero.
    Value,
    /// `AssertionError`, for example popping more blocks than the queue holds.
    Assert,
    /// `RuntimeError`, for example removing a block that is not in the list.
    Runtime,
    /// `IndexError`, for example reading a block hash past the end of a list.
    Index,
    /// `ZeroDivisionError`, for example a block size of zero.
    Zero,
}

impl PyError {
    /// The Python exception class name, as written to the trace log.
    pub fn python_name(self) -> &'static str {
        match self {
            PyError::Value => "ValueError",
            PyError::Assert => "AssertionError",
            PyError::Runtime => "RuntimeError",
            PyError::Index => "IndexError",
            PyError::Zero => "ZeroDivisionError",
        }
    }
}

/// Doubly linked list of free blocks, mirroring `FreeKVCacheBlockQueue`.
///
/// Block ids are `0..num_blocks`, and the two sentinels live at `num_blocks` (head)
/// and `num_blocks + 1` (tail), exactly like Python's `fake_free_list_head` /
/// `fake_free_list_tail`. A block that is not in the list has both links cleared, so
/// `remove()` on it fails the same way Python's `remove()` does.
///
/// The counter is signed on purpose: Python's `num_free_blocks` is an `int` that can go
/// negative if a caller removes a block twice, and a `usize` would panic where Python
/// merely drifts.
#[derive(Debug, Clone)]
pub struct FreeQueue {
    num_blocks: usize,
    prev: Vec<Option<usize>>,
    next: Vec<Option<usize>>,
    num_free_blocks: i64,
}

impl FreeQueue {
    /// Build the list with every block `0..num_blocks` in ascending order.
    pub fn new(num_blocks: usize) -> Self {
        let n = num_blocks;
        let mut prev = vec![None; n + 2];
        let mut next = vec![None; n + 2];
        for i in 0..n {
            prev[i] = i.checked_sub(1);
            next[i] = if i + 1 < n { Some(i + 1) } else { None };
        }
        let head = n;
        let tail = n + 1;
        if n > 0 {
            next[head] = Some(0);
            prev[0] = Some(head);
            prev[tail] = Some(n - 1);
            next[n - 1] = Some(tail);
        } else {
            next[head] = Some(tail);
            prev[tail] = Some(head);
        }
        Self {
            num_blocks: n,
            prev,
            next,
            num_free_blocks: n as i64,
        }
    }

    /// Number of blocks currently in the free list.
    pub fn num_free_blocks(&self) -> i64 {
        self.num_free_blocks
    }

    /// Block ids from the head to the tail of the list.
    ///
    /// Python's `get_all_free_blocks()` walks while `curr.next_free_block is not None`,
    /// which is what keeps the fake tail out of the result; the loop below does the same
    /// rather than stopping on the sentinel id, so both agree even on a list that a caller
    /// has corrupted with a duplicate `append`. Python raises `RuntimeError` when the head
    /// link is missing; that state is unreachable here because `new` always links both
    /// sentinels.
    pub fn all_free_blocks(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let Some(mut cursor) = self.next[self.num_blocks] else {
            return out;
        };
        while let Some(following) = self.next[cursor] {
            out.push(cursor);
            cursor = following;
        }
        out
    }

    /// `FreeKVCacheBlockQueue.popleft`.
    pub fn popleft(&mut self) -> Result<usize, PyError> {
        let head = self.num_blocks;
        let tail = self.num_blocks + 1;
        let first = match self.next[head] {
            // Python would dereference None here; the list cannot reach that state.
            None => return Err(PyError::Runtime),
            Some(first) => first,
        };
        if first == tail {
            // Python asserts `num_free_blocks == 0` before raising, so a desynchronised
            // list reports the assertion instead of the empty-queue error.
            return Err(if self.num_free_blocks == 0 {
                PyError::Value
            } else {
                PyError::Assert
            });
        }
        let Some(second) = self.next[first] else {
            return Err(PyError::Runtime);
        };
        self.next[head] = Some(second);
        self.prev[second] = Some(head);
        self.prev[first] = None;
        self.next[first] = None;
        self.num_free_blocks -= 1;
        Ok(first)
    }

    /// `FreeKVCacheBlockQueue.popleft_n`.
    pub fn popleft_n(&mut self, n: usize) -> Result<Vec<usize>, PyError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.num_free_blocks < n as i64 {
            return Err(PyError::Assert);
        }
        self.num_free_blocks -= n as i64;

        let head = self.num_blocks;
        let mut cursor = self.next[head];
        let mut ret = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(block) = cursor else {
                return Err(PyError::Assert);
            };
            ret.push(block);
            cursor = self.next[block];
            self.prev[block] = None;
            self.next[block] = None;
        }
        if let Some(block) = cursor {
            self.next[head] = Some(block);
            self.prev[block] = Some(head);
        }
        Ok(ret)
    }

    /// `FreeKVCacheBlockQueue.remove`.
    pub fn remove(&mut self, block: usize) -> Result<(), PyError> {
        let (Some(previous), Some(following)) = (self.prev[block], self.next[block]) else {
            return Err(PyError::Runtime);
        };
        self.next[previous] = Some(following);
        self.prev[following] = Some(previous);
        self.prev[block] = None;
        self.next[block] = None;
        self.num_free_blocks -= 1;
        Ok(())
    }
    /// `FreeKVCacheBlockQueue.append`.
    pub fn append(&mut self, block: usize) -> Result<(), PyError> {
        let tail = self.num_blocks + 1;
        let Some(last) = self.prev[tail] else {
            return Err(PyError::Runtime);
        };
        self.next[last] = Some(block);
        self.prev[block] = Some(last);
        self.next[block] = Some(tail);
        self.prev[tail] = Some(block);
        self.num_free_blocks += 1;
        Ok(())
    }

    /// `FreeKVCacheBlockQueue.prepend_n`.
    pub fn prepend_n(&mut self, blocks: &[usize]) -> Result<(), PyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let head = self.num_blocks;
        let Some(first) = self.next[head] else {
            return Err(PyError::Assert);
        };
        let mut previous = head;
        for &block in blocks {
            self.prev[block] = Some(previous);
            self.next[previous] = Some(block);
            previous = block;
        }
        self.next[previous] = Some(first);
        self.prev[first] = Some(previous);
        self.num_free_blocks += blocks.len() as i64;
        Ok(())
    }

    /// `FreeKVCacheBlockQueue.append_n`.
    pub fn append_n(&mut self, blocks: &[usize]) -> Result<(), PyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let tail = self.num_blocks + 1;
        let Some(mut last) = self.prev[tail] else {
            return Err(PyError::Assert);
        };
        for &block in blocks {
            self.prev[block] = Some(last);
            self.next[last] = Some(block);
            last = block;
        }
        self.next[last] = Some(tail);
        self.prev[tail] = Some(last);
        self.num_free_blocks += blocks.len() as i64;
        Ok(())
    }
}

/// One key of [`BlockHashToBlockMap`] holding one or more blocks in insertion order.
#[derive(Debug, Clone)]
struct Entry {
    key: Vec<u8>,
    /// Block ids in insertion order. Python stores `{block_id: block}` or a bare block;
    /// both behave like an insertion-ordered list for the operations exposed here.
    blocks: Vec<u64>,
}

/// Port of `BlockHashToBlockMap` from `vllm/v1/core/block_pool.py`.
///
/// Python keeps `dict[hash, KVCacheBlock | dict[int, KVCacheBlock]]`; the union exists
/// only to avoid allocating an inner dict for the common single-block case. The
/// observable behaviour is what this crate implements: `get_one_block` returns the first
/// block of the group, `pop` removes one block and drops the key once the group is
/// empty, and a failed `pop` leaves the key in place (moved to the end of the key order,
/// the way a delete-and-reinsert does in Python).
#[derive(Debug, Default)]
pub struct BlockHashToBlockMap {
    entries: Vec<Entry>,
}

impl BlockHashToBlockMap {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct hash keys currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map holds no keys at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn index_of(&self, key: &[u8]) -> Option<usize> {
        self.entries.iter().position(|entry| entry.key == key)
    }

    /// `BlockHashToBlockMap.insert`.
    pub fn insert(&mut self, key: &[u8], block_id: u64) {
        match self.index_of(key) {
            Some(index) => {
                let blocks = &mut self.entries[index].blocks;
                match blocks.iter().position(|id| *id == block_id) {
                    // Python writes into the inner dict, so a repeated id replaces the
                    // stored block and keeps its position.
                    Some(position) => blocks[position] = block_id,
                    None => blocks.push(block_id),
                }
            }
            None => self.entries.push(Entry {
                key: key.to_vec(),
                blocks: vec![block_id],
            }),
        }
    }

    /// `BlockHashToBlockMap.pop`: returns the removed block id, if the key had it.
    pub fn pop(&mut self, key: &[u8], block_id: u64) -> Option<u64> {
        let index = self.index_of(key)?;
        let entry = self.entries.remove(index);
        let mut blocks = entry.blocks;
        let position = blocks.iter().position(|id| *id == block_id);
        let removed = position.map(|position| blocks.remove(position));
        if !blocks.is_empty() {
            // Python deletes the key and re-inserts it, which moves it to the end.
            self.entries.push(Entry {
                key: entry.key,
                blocks,
            });
        }
        removed
    }

    /// `BlockHashToBlockMap.contain`.
    pub fn contain(&self, key: &[u8], block_id: u64) -> bool {
        self.index_of(key)
            .is_some_and(|index| self.entries[index].blocks.contains(&block_id))
    }

    /// `BlockHashToBlockMap.get_one_block`: any block of the group, deterministically
    /// the first one that is still cached.
    pub fn get_one_block(&self, key: &[u8]) -> Option<u64> {
        let index = self.index_of(key)?;
        self.entries[index].blocks.first().copied()
    }

    /// Keys in insertion order, each with its block ids — the `map_dump` trace operation.
    pub fn dump(&self) -> String {
        self.entries
            .iter()
            .map(|entry| {
                let ids =
                    entry.blocks.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
                format!("{}:{}", hex_encode(&entry.key), ids)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_decode(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err(format!("odd-length hex key `{text}`"));
    }
    (0..text.len())
        .step_by(2)
        .map(|offset| {
            u8::from_str_radix(&text[offset..offset + 2], 16)
                .map_err(|_| format!("invalid hex key `{text}`"))
        })
        .collect()
}

/// Replay a recorded operation trace and return the log it produces.
///
/// The trace format is one operation per line; `#` starts a comment and blank lines are
/// ignored. Each line is echoed with the outcome appended as ` -> ok ...` or
/// ` -> err <Python exception>`, and this text must equal the log the Python recorder
/// wrote for the same trace.
pub fn replay(trace: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut queue: Option<FreeQueue> = None;
    let mut map = BlockHashToBlockMap::new();
    let mut pool: Option<BlockPool> = None;
    // `Request`-shaped inputs. `pool_request` appends one; the cache entry points read the one
    // the trace names. Python keeps the hashes at `hash_block_size` granularity plus the token
    // ids a `BlockStored` event carries.
    let mut requests: Vec<(Vec<Vec<u8>>, Vec<i64>)> = Vec::new();
    let mut hashes = HashList::Raw(Vec::new());
    // The harness starts with an empty raw list at a block size of 16, the same defaults the
    // Python driver uses, so a trace can begin with any hash-list operation.
    let mut hash_block_size: usize = 16;

    for (index, raw) in trace.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let tokens = apply(
            &parts,
            &mut queue,
            &mut map,
            &mut pool,
            &mut requests,
            &mut hashes,
            &mut hash_block_size,
        )
        .map_err(|message| format!("line {}: {message}", index + 1))?;
        writeln!(out, "{line} -> {}", tokens.join(" ")).expect("writing to a String");
    }
    Ok(out)
}

fn apply(
    parts: &[&str],
    queue: &mut Option<FreeQueue>,
    map: &mut BlockHashToBlockMap,
    pool: &mut Option<BlockPool>,
    requests: &mut Vec<(Vec<Vec<u8>>, Vec<i64>)>,
    hashes: &mut HashList,
    hash_block_size: &mut usize,
) -> Result<Vec<String>, String> {
    let op = parts[0];
    let ids = |from: usize| -> Result<Vec<usize>, String> {
        parts[from..]
            .iter()
            .map(|token| parse_usize(token))
            .collect::<Result<Vec<_>, _>>()
    };
    let pool_ids = |from: usize| -> Result<Vec<u64>, String> {
        parts[from..]
            .iter()
            .map(|token| parse_u64(token))
            .collect::<Result<Vec<_>, _>>()
    };

    match op {
        "init" => {
            let num_blocks = parse_usize(need(parts, 1, "init <num_blocks>")?)?;
            *queue = Some(FreeQueue::new(num_blocks));
            *map = BlockHashToBlockMap::new();
            Ok(ok())
        }
        "popleft" => match queue {
            Some(queue) => Ok(match queue.popleft() {
                Ok(block) => ok_value(block.to_string()),
                Err(error) => err(error),
            }),
            None => Err("popleft before init".to_string()),
        },
        "popleft_n" => {
            let n = parse_usize(need(parts, 1, "popleft_n <n>")?)?;
            match queue {
                Some(queue) => Ok(match queue.popleft_n(n) {
                    Ok(blocks) => ok_values(blocks.iter().map(usize::to_string)),
                    Err(error) => err(error),
                }),
                None => Err("popleft_n before init".to_string()),
            }
        }
        "append" => {
            let block = parse_usize(need(parts, 1, "append <block_id>")?)?;
            match queue {
                Some(queue) => Ok(result(queue.append(block))),
                None => Err("append before init".to_string()),
            }
        }
        "append_n" | "prepend_n" => {
            let blocks = ids(1)?;
            match queue {
                Some(queue) => {
                    let outcome = if op == "append_n" {
                        queue.append_n(&blocks)
                    } else {
                        queue.prepend_n(&blocks)
                    };
                    Ok(result(outcome))
                }
                None => Err(format!("{op} before init")),
            }
        }
        "remove" => {
            let block = parse_usize(need(parts, 1, "remove <block_id>")?)?;
            match queue {
                Some(queue) => Ok(result(queue.remove(block))),
                None => Err("remove before init".to_string()),
            }
        }
        "peek" => match queue {
            Some(queue) => Ok(ok_values(
                queue.all_free_blocks().iter().map(usize::to_string),
            )),
            None => Err("peek before init".to_string()),
        },
        "num_free" => match queue {
            Some(queue) => Ok(ok_value(queue.num_free_blocks().to_string())),
            None => Err("num_free before init".to_string()),
        },
        "map_insert" => {
            let key = hex_decode(need(parts, 1, "map_insert <key> <block_id>")?)?;
            let block = parse_u64(need(parts, 2, "map_insert <key> <block_id>")?)?;
            map.insert(&key, block);
            Ok(ok())
        }
        "map_pop" => {
            let key = hex_decode(need(parts, 1, "map_pop <key> <block_id>")?)?;
            let block = parse_u64(need(parts, 2, "map_pop <key> <block_id>")?)?;
            Ok(match map.pop(&key, block) {
                Some(block) => ok_value(block.to_string()),
                None => ok_value("-".to_string()),
            })
        }
        "map_contain" => {
            let key = hex_decode(need(parts, 1, "map_contain <key> <block_id>")?)?;
            let block = parse_u64(need(parts, 2, "map_contain <key> <block_id>")?)?;
            Ok(ok_value(
                if map.contain(&key, block) { "1" } else { "0" }.to_string(),
            ))
        }
        "map_get_one" => {
            let key = hex_decode(need(parts, 1, "map_get_one <key>")?)?;
            Ok(match map.get_one_block(&key) {
                Some(block) => ok_value(block.to_string()),
                None => ok_value("-".to_string()),
            })
        }
        "map_len" => Ok(ok_value(map.len().to_string())),
        "map_dump" => {
            let dump = map.dump();
            Ok(if dump.is_empty() {
                ok()
            } else {
                ok_value(dump)
            })
        }
        "pool_init" => {
            let usage = "pool_init <num_blocks> <0|1 caching> <hash_block_size> <0|1 events> <0|1 metrics> <medium>";
            let num_blocks = parse_usize(need(parts, 1, usage)?)?;
            let enable_caching = parse_bool(need(parts, 2, usage)?)?;
            let hash_block_size = parse_i64(need(parts, 3, usage)?)?;
            let enable_events = parse_bool(need(parts, 4, usage)?)?;
            let collect_metrics = parse_bool(need(parts, 5, usage)?)?;
            let medium = need(parts, 6, usage)?;
            match BlockPool::new(
                num_blocks,
                enable_caching,
                hash_block_size,
                enable_events,
                collect_metrics,
                medium,
            ) {
                Ok(created) => {
                    *pool = Some(created);
                    Ok(ok())
                }
                Err(error) => Ok(err(error)),
            }
        }
        "pool_take_events" => match pool {
            Some(pool) => {
                let medium = pool.medium().to_string();
                let events = pool.take_events();
                Ok(if events.is_empty() {
                    ok()
                } else {
                    ok_values(events.iter().map(|event| event.canonical_for_log(&medium)))
                })
            }
            None => Err("pool_take_events before pool_init".to_string()),
        },
        "pool_drain_metrics" => match pool {
            Some(pool) => Ok(ok_value(pool.drain_metric_events().to_string())),
            None => Err("pool_drain_metrics before pool_init".to_string()),
        },
        "pool_new" => {
            let num_blocks = parse_usize(need(parts, 1, "pool_new <n>")?)?;
            match pool {
                Some(pool) => Ok(match pool.get_new_blocks(num_blocks) {
                    Ok(blocks) => ok_values(blocks.iter().map(u64::to_string)),
                    Err(error) => err(error),
                }),
                None => Err("pool_new before pool_init".to_string()),
            }
        }
        "pool_free" => {
            let blocks = pool_ids(1)?;
            match pool {
                Some(pool) => Ok(result(pool.free_blocks(&blocks))),
                None => Err("pool_free before pool_init".to_string()),
            }
        }
        "pool_unpin" => {
            let blocks = pool_ids(1)?;
            match pool {
                Some(pool) => Ok(result(pool.unpin_blocks(&blocks))),
                None => Err("pool_unpin before pool_init".to_string()),
            }
        }
        "pool_touch" => {
            let blocks = pool_ids(1)?;
            match pool {
                Some(pool) => Ok(result(pool.touch(&blocks))),
                None => Err("pool_touch before pool_init".to_string()),
            }
        }
        "pool_evict" => {
            let blocks = pool_ids(1)?;
            match pool {
                Some(pool) => Ok(result(pool.evict_blocks(&blocks))),
                None => Err("pool_evict before pool_init".to_string()),
            }
        }
        "pool_writable" => {
            let block = parse_u64(need(parts, 1, "pool_writable <block_id>")?)?;
            match pool {
                Some(pool) => Ok(ok_value(
                    if pool.is_block_writable(block) {
                        "1"
                    } else {
                        "0"
                    }
                    .to_string(),
                )),
                None => Err("pool_writable before pool_init".to_string()),
            }
        }
        "pool_set_hash" => {
            let usage = "pool_set_hash <block_id> <key> <num_tokens|->";
            let block = parse_u64(need(parts, 1, usage)?)?;
            let key = hex_decode(need(parts, 2, usage)?)?;
            let num_tokens = match need(parts, 3, usage)? {
                "-" => None,
                token => Some(parse_i64(token)?),
            };
            match pool {
                Some(pool) => Ok(result(pool.insert_block_hash(&key, block, num_tokens))),
                None => Err("pool_set_hash before pool_init".to_string()),
            }
        }
        "pool_request" => {
            let usage = "pool_request <hash,hash,...> <token,token,...>";
            let block_hashes = parse_hex_list(need(parts, 1, usage)?)?;
            let all_token_ids = parse_i64_list(need(parts, 2, usage)?)?;
            requests.push((block_hashes, all_token_ids));
            Ok(ok())
        }
        "pool_cache_full" => {
            let usage = "pool_cache_full <request_idx> <num_cached_blocks> <num_full_blocks> \
                         <block_size> <group_id> <mask|-> <block_id>...";
            let request = parse_usize(need(parts, 1, usage)?)?;
            let num_cached_blocks = parse_usize(need(parts, 2, usage)?)?;
            let num_full_blocks = parse_usize(need(parts, 3, usage)?)?;
            let block_size = parse_i64(need(parts, 4, usage)?)?;
            let group_id = parse_u32(need(parts, 5, usage)?)?;
            let mask = match need(parts, 6, usage)? {
                "-" => None,
                flags => Some(parse_bool_list(flags)?),
            };
            let blocks = decode_u64s(&parts[7..])?;
            let Some((block_hashes, all_token_ids)) = requests.get(request) else {
                return Err(format!("pool_request index {request} was never recorded"));
            };
            let view = RequestView {
                block_hashes,
                all_token_ids,
            };
            match pool {
                Some(pool) => Ok(result(pool.cache_full_blocks(
                    &view,
                    &blocks,
                    num_cached_blocks,
                    num_full_blocks,
                    block_size,
                    group_id,
                    mask.as_deref(),
                ))),
                None => Err("pool_cache_full before pool_init".to_string()),
            }
        }
        "pool_cache_partial" => {
            let usage = "pool_cache_partial <request_idx> <block_id> <num_tokens> <group_id> \
                         <block_size> <0|1 replace_existing_hashes>";
            let request = parse_usize(need(parts, 1, usage)?)?;
            let block = parse_u64(need(parts, 2, usage)?)?;
            let num_tokens = parse_i64(need(parts, 3, usage)?)?;
            let group_id = parse_u32(need(parts, 4, usage)?)?;
            let block_size = parse_i64(need(parts, 5, usage)?)?;
            let replace_existing_hashes = parse_bool(need(parts, 6, usage)?)?;
            let Some((block_hashes, all_token_ids)) = requests.get(request) else {
                return Err(format!("pool_request index {request} was never recorded"));
            };
            let view = RequestView {
                block_hashes,
                all_token_ids,
            };
            match pool {
                Some(pool) => Ok(
                    match pool.cache_partial_block(
                        &view,
                        block,
                        num_tokens,
                        group_id,
                        block_size,
                        replace_existing_hashes,
                    ) {
                        Ok(Some(key)) => ok_value(hex_encode(&key)),
                        Ok(None) => ok_value("-".to_string()),
                        Err(error) => err(error),
                    },
                ),
                None => Err("pool_cache_partial before pool_init".to_string()),
            }
        }
        "pool_cached" => {
            let key = hex_decode(need(parts, 1, "pool_cached <key>")?)?;
            match pool {
                Some(pool) => Ok(match pool.get_cached_block(&key) {
                    Some(block) => ok_value(block.to_string()),
                    None => ok_value("-".to_string()),
                }),
                None => Err("pool_cached before pool_init".to_string()),
            }
        }
        "pool_num_free" => match pool {
            Some(pool) => Ok(ok_value(pool.num_free_blocks().to_string())),
            None => Err("pool_num_free before pool_init".to_string()),
        },
        "pool_usage" => match pool {
            // Six decimals on both sides: Python computes the same IEEE double, and the
            // integer zero it returns for a one-block pool formats identically.
            Some(pool) => Ok(ok_value(format!("{:.6}", pool.usage()))),
            None => Err("pool_usage before pool_init".to_string()),
        },
        "pool_order" => match pool {
            Some(pool) => Ok(ok_values(pool.free_order().iter().map(u64::to_string))),
            None => Err("pool_order before pool_init".to_string()),
        },
        "pool_reset" => match pool {
            Some(pool) => Ok(ok_value(
                if pool.reset_prefix_cache() { "1" } else { "0" }.to_string(),
            )),
            None => Err("pool_reset before pool_init".to_string()),
        },
        "pool_dump" => match pool {
            Some(pool) => Ok(ok_value(pool.dump())),
            None => Err("pool_dump before pool_init".to_string()),
        },
        "bhlist_raw" => {
            let size = parse_usize(need(parts, 1, "bhlist_raw <hash_block_size> <hash>...")?)?;
            let raw = decode_hashes(&parts[2..])?;
            *hash_block_size = size;
            *hashes = HashList::Raw(raw);
            Ok(ok())
        }
        "bhlist_init" => {
            let usage = "bhlist_init <hash_block_size> <target_block_size> <hash>...";
            let size = parse_usize(need(parts, 1, usage)?)?;
            let target = parse_usize(need(parts, 2, usage)?)?;
            let raw = decode_hashes(&parts[3..])?;
            // Python assigns the view only when the constructor returns, so a rejected
            // construction leaves the previous state in place.
            match BlockHashListView::new(raw, size, target) {
                Ok(view) => {
                    *hash_block_size = size;
                    *hashes = HashList::View(view);
                    Ok(ok())
                }
                Err(error) => Ok(err(error)),
            }
        }
        "bhlist_len" => Ok(ok_value(hashes.len().to_string())),
        "bhlist_get" => {
            let index = parse_i64(need(parts, 1, "bhlist_get <index>")?)?;
            Ok(match hashes.get(index) {
                Ok(hash) => ok_value(hex_encode(&hash)),
                Err(error) => err(error),
            })
        }
        "bhlist_slice" => {
            let usage = "bhlist_slice <start|-> <stop|-> <step|->";
            let start = parse_optional_i64(need(parts, 1, usage)?)?;
            let stop = parse_optional_i64(need(parts, 2, usage)?)?;
            let step = parse_optional_i64(need(parts, 3, usage)?)?;
            Ok(match hashes.slice(start, stop, step) {
                Ok(sliced) => ok_values(sliced.iter().map(|hash| hex_encode(hash))),
                Err(error) => err(error),
            })
        }
        "bhlist_iter" => Ok(ok_values(
            hashes.values().iter().map(|hash| hex_encode(hash)),
        )),
        "bhlist_resolve" => {
            let usage = "bhlist_resolve <block_size> <0|1 fine_grained> <alignment|->";
            let block_size = parse_usize(need(parts, 1, usage)?)?;
            let fine_grained = parse_bool(need(parts, 2, usage)?)?;
            let alignment = parse_optional_usize(need(parts, 3, usage)?)?;
            Ok(
                match resolve_block_hashes(
                    hashes,
                    *hash_block_size,
                    block_size,
                    fine_grained,
                    alignment,
                ) {
                    Ok(resolved) => {
                        let tokens =
                            ok_values(resolved.values().iter().map(|hash| hex_encode(hash)));
                        *hashes = resolved;
                        tokens
                    }
                    Err(error) => err(error),
                },
            )
        }
        other => Err(format!("unknown operation `{other}`")),
    }
}

fn need<'a>(parts: &[&'a str], index: usize, usage: &str) -> Result<&'a str, String> {
    parts
        .get(index)
        .copied()
        .ok_or_else(|| format!("missing argument, expected `{usage}`"))
}

/// A comma separated list of hex byte strings, as `pool_request` carries its hashes.
fn parse_hex_list(token: &str) -> Result<Vec<Vec<u8>>, String> {
    token.split(',').map(hex_decode).collect()
}

/// A comma separated list of integers.
fn parse_i64_list(token: &str) -> Result<Vec<i64>, String> {
    token.split(',').map(parse_i64).collect()
}

/// A comma separated list of `0`/`1` flags.
fn parse_bool_list(token: &str) -> Result<Vec<bool>, String> {
    token.split(',').map(parse_bool).collect()
}

/// Space separated block ids, the shape every pool operation uses for its tail.
fn decode_u64s(tokens: &[&str]) -> Result<Vec<u64>, String> {
    tokens.iter().map(|token| parse_u64(token)).collect()
}

fn parse_u32(token: &str) -> Result<u32, String> {
    token
        .parse::<u32>()
        .map_err(|_| format!("expected a KV cache group id, got `{token}`"))
}

fn parse_usize(token: &str) -> Result<usize, String> {
    token
        .parse::<usize>()
        .map_err(|_| format!("expected a block id, got `{token}`"))
}

fn parse_u64(token: &str) -> Result<u64, String> {
    token.parse::<u64>().map_err(|_| format!("expected a block id, got `{token}`"))
}

fn parse_i64(token: &str) -> Result<i64, String> {
    token
        .parse::<i64>()
        .map_err(|_| format!("expected a token count, got `{token}`"))
}

fn parse_bool(token: &str) -> Result<bool, String> {
    match token {
        "0" | "false" => Ok(false),
        "1" | "true" => Ok(true),
        _ => Err(format!("expected 0 or 1, got `{token}`")),
    }
}

fn parse_optional_i64(token: &str) -> Result<Option<i64>, String> {
    if token == "-" {
        Ok(None)
    } else {
        parse_i64(token).map(Some)
    }
}

fn parse_optional_usize(token: &str) -> Result<Option<usize>, String> {
    if token == "-" {
        Ok(None)
    } else {
        parse_usize(token).map(Some)
    }
}

fn decode_hashes(tokens: &[&str]) -> Result<Vec<Vec<u8>>, String> {
    tokens.iter().map(|token| hex_decode(token)).collect()
}

fn ok() -> Vec<String> {
    vec!["ok".to_string()]
}

fn ok_value(value: String) -> Vec<String> {
    vec!["ok".to_string(), value]
}

fn ok_values<I: Iterator<Item = String>>(values: I) -> Vec<String> {
    let mut tokens = ok();
    tokens.extend(values);
    tokens
}

fn result(outcome: Result<(), PyError>) -> Vec<String> {
    match outcome {
        Ok(()) => ok(),
        Err(error) => err(error),
    }
}

fn err(error: PyError) -> Vec<String> {
    vec!["err".to_string(), error.python_name().to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_starts_in_block_order() {
        let queue = FreeQueue::new(4);
        assert_eq!(queue.all_free_blocks(), vec![0, 1, 2, 3]);
        assert_eq!(queue.num_free_blocks(), 4);
    }

    #[test]
    fn popleft_takes_from_the_front_and_clears_the_links() {
        let mut queue = FreeQueue::new(3);
        assert_eq!(queue.popleft(), Ok(0));
        assert_eq!(queue.all_free_blocks(), vec![1, 2]);
        assert_eq!(queue.num_free_blocks(), 2);
        // A block that left the list has no links, so remove() rejects it.
        assert_eq!(queue.remove(0), Err(PyError::Runtime));
    }

    #[test]
    fn empty_queue_reports_value_error() {
        let mut queue = FreeQueue::new(0);
        assert_eq!(queue.popleft(), Err(PyError::Value));
        assert_eq!(queue.all_free_blocks(), Vec::<usize>::new());
    }

    #[test]
    fn popleft_n_rejects_more_blocks_than_available() {
        let mut queue = FreeQueue::new(2);
        assert_eq!(queue.popleft_n(3), Err(PyError::Assert));
        assert_eq!(queue.popleft_n(2), Ok(vec![0, 1]));
        assert_eq!(queue.num_free_blocks(), 0);
        assert_eq!(queue.popleft_n(1), Err(PyError::Assert));
        assert_eq!(queue.popleft(), Err(PyError::Value));
    }

    #[test]
    fn append_and_prepend_place_blocks_at_the_right_end() {
        let mut queue = FreeQueue::new(4);
        queue.popleft_n(4).unwrap();
        queue.append(0).unwrap();
        queue.prepend_n(&[3, 2]).unwrap();
        queue.append_n(&[1]).unwrap();
        assert_eq!(queue.all_free_blocks(), vec![3, 2, 0, 1]);
        assert_eq!(queue.num_free_blocks(), 4);
    }

    #[test]
    fn remove_splices_the_neighbours_together() {
        let mut queue = FreeQueue::new(3);
        queue.remove(1).unwrap();
        assert_eq!(queue.all_free_blocks(), vec![0, 2]);
        assert_eq!(queue.num_free_blocks(), 2);
    }

    #[test]
    fn map_keeps_one_block_per_key_until_it_collides() {
        let mut map = BlockHashToBlockMap::new();
        map.insert(b"k", 7);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_one_block(b"k"), Some(7));
        assert!(map.contain(b"k", 7));

        map.insert(b"k", 9);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_one_block(b"k"), Some(7));
        assert_eq!(map.dump(), "6b:7,9".to_string());

        // The group survives while blocks remain, and the key disappears with the last one.
        assert_eq!(map.pop(b"k", 7), Some(7));
        assert_eq!(map.get_one_block(b"k"), Some(9));
        assert_eq!(map.pop(b"k", 7), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.pop(b"k", 9), Some(9));
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn map_insert_of_a_known_block_keeps_its_position() {
        let mut map = BlockHashToBlockMap::new();
        map.insert(b"k", 1);
        map.insert(b"k", 2);
        map.insert(b"k", 1);
        assert_eq!(map.dump(), "6b:1,2".to_string());
    }

    #[test]
    fn replay_logs_the_trace_it_read() {
        let trace = "\
# comment
init 3
num_free
popleft
append 0
peek
popleft_n 9
map_insert a1 5
map_get_one a1
map_pop a1 4
map_dump
";
        let expected = "\
init 3 -> ok
num_free -> ok 3
popleft -> ok 0
append 0 -> ok
peek -> ok 1 2 0
popleft_n 9 -> err AssertionError
map_insert a1 5 -> ok
map_get_one a1 -> ok 5
map_pop a1 4 -> ok -
map_dump -> ok a1:5
";
        assert_eq!(replay(trace).unwrap(), expected);
    }

    #[test]
    fn replay_drives_the_pool_operations() {
        let trace = "\
pool_init 4 1 32 0 0 GPU
pool_order
pool_new 2
pool_set_hash 1 aa 32
pool_cached aa
pool_writable 1
pool_free 1 2
pool_num_free
pool_usage
pool_new 5
pool_reset
pool_dump
";
        let expected = "\
pool_init 4 1 32 0 0 GPU -> ok
pool_order -> ok 1 2 3
pool_new 2 -> ok 1 2
pool_set_hash 1 aa 32 -> ok
pool_cached aa -> ok 1
pool_writable 1 -> ok 0
pool_free 1 2 -> ok
pool_num_free -> ok 3
pool_usage -> ok 0.000000
pool_new 5 -> err ValueError
pool_reset -> ok 1
pool_dump -> ok free=3 map=0 watchers=- events=- metrics=none 0:0:-:-:- 1:0:-:-:- 2:0:-:-:- 3:0:-:-:-
";
        assert_eq!(replay(trace).unwrap(), expected);
    }

    #[test]
    fn replay_drives_the_request_based_cache_entry_points() {
        // Two hashes per request: a 32-token and a 64-token block size, a partial entry inside
        // the block, and the events both paths queue. The null block is passed in the first
        // slot of the second call, so its hash is skipped on both sides.
        let trace = "\
pool_init 4 1 32 1 0 GPU
pool_new 2
pool_request 40112233,41112233,42112233,43112233 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
pool_cache_full 0 0 2 32 1 - 1 2
pool_take_events
pool_cache_partial 0 1 96 1 64 0
pool_take_events
pool_dump
";
        let output = replay(trace).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "pool_init 4 1 32 1 0 GPU -> ok");
        assert_eq!(lines[1], "pool_new 2 -> ok 1 2");
        assert_eq!(lines[2].split_whitespace().next(), Some("pool_request"));
        assert_eq!(lines[3], "pool_cache_full 0 0 2 32 1 - 1 2 -> ok");
        // One `stored` event for the call: both hashes, no parent, the token range, 32-token
        // blocks, group 1, and one empty extra-key entry per block.
        let tokens: String = (0..64).map(|token| token.to_string()).collect::<Vec<_>>().join(",");
        assert_eq!(
            lines[4],
            format!("pool_take_events -> ok stored:40112233,41112233:-:{tokens}:32:1:GPU:~,~")
        );
        // The partial entry sits 96 tokens into block 1, whose cached entry covered 32: the old
        // key is dropped with a BlockRemoved event first, then the new one is stored with the
        // 32 tokens between 64 and 96. The hash at 96 tokens is the request's third one, so its
        // parent is the second.
        assert_eq!(
            lines[5],
            "pool_cache_partial 0 1 96 1 64 0 -> ok 4211223300000001"
        );
        let tail_tokens: String =
            (64..96).map(|token| token.to_string()).collect::<Vec<_>>().join(",");
        assert_eq!(
            lines[6],
            format!(
                "pool_take_events -> ok removed:40112233:1:GPU \
                 stored:42112233:41112233:{tail_tokens}:32:1:GPU:~"
            )
        );
        let dump = lines[7];
        assert!(dump.contains("map=2"), "dump was {dump}");
        assert!(
            dump.contains("1:1:4211223300000001:96:-"),
            "the promoted block keeps its new primary hash: {dump}"
        );
        assert!(
            dump.contains("2:1:4111223300000001:64:-"),
            "the untouched block keeps its entry: {dump}"
        );
    }
}
