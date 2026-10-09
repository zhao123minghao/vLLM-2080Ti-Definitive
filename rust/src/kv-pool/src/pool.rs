// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Port of `BlockPool` from `vllm/v1/core/block_pool.py`.
//!
//! This is the layer on top of [`FreeQueue`](crate::FreeQueue) and
//! [`BlockHashToBlockMap`](crate::BlockHashToBlockMap): reference counts, prefix-cache
//! bookkeeping per block, the eviction order of freed blocks, and the reset path.
//!
//! The `Request`-based entry points are ported as well ([`BlockPool::cache_full_blocks`] and
//! [`BlockPool::cache_partial_block`]), taking a [`RequestView`] instead of a `Request`: the
//! trace carries the hashes a real request would have computed, and the harness models a
//! text-only request, which is why the extra keys stay empty. Hashing itself is out of scope and
//! covered by the prefix-hash parity check.
//!
//! The KV-cache event queue ([`PoolEvent`], `take_events`) and the deterministic half of
//! `KVCacheMetricsCollector` ([`Metrics`]) are part of the port.
//!
//! Everything a caller can observe about the state — the free list, reference counts, the
//! per-block hash metadata, the cached-hash map, the pending reuse watchers — is what
//! [`BlockPool::dump`] prints, which is what the parity harness compares.

use crate::{BlockHashToBlockMap, FreeQueue, HashList, PyError, resolve_block_hashes};

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
    /// `hash_block_size`: the granularity of a request's block hashes.
    hash_block_size: i64,
    blocks: Vec<BlockState>,
    queue: FreeQueue,
    map: BlockHashToBlockMap,
    /// `_reuse_watchers`: block ids whose contents are still being read. The callbacks are
    /// no-ops in the harness, but *when they fire* is state, so the ids are kept.
    reuse_watchers: Vec<u64>,
    /// `enable_kv_cache_events` and the medium the events report.
    enable_kv_cache_events: bool,
    medium: String,
    /// `kv_event_queue`, in emission order.
    kv_event_queue: Vec<PoolEvent>,
    /// `KVCacheMetricsCollector`, or `None` when no collector was passed.
    metrics: Option<Metrics>,
}

/// A `KVCacheEvent` as far as the bookkeeping is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolEvent {
    /// `BlockRemoved(block_hashes=[...], medium=..., group_idx=...)`.
    BlockRemoved { hash: Vec<u8>, group_idx: u32 },
    /// `BlockStored(block_hashes=..., parent_block_hash=..., token_ids=..., block_size=...,
    /// lora_id=..., medium=..., lora_name=..., extra_keys=..., group_idx=..., session_id=...)`.
    ///
    /// `hashes` are the newly cached block hashes (one per cached block in `cache_full_blocks`,
    /// a single one in `cache_partial_block`) and `token_ids` the token range the event covers.
    /// The LoRA id/name and the session id are outside the port: the harness models a text-only
    /// request, so they are `None` on both sides.
    BlockStored {
        hashes: Vec<Vec<u8>>,
        parent: Option<Vec<u8>>,
        token_ids: Vec<i64>,
        block_size: i64,
        group_idx: u32,
        /// `extra_keys` as far as the port is concerned: the harness models a text-only request,
        /// so every entry is `None` and the only observable facts are whether Python passed a
        /// list at all (`cache_full_blocks` passes `extra_keys_list if extra_keys_list else
        /// None` while `cache_partial_block` always passes a one-element list) and how long it
        /// was. A real entry is a LoRA name, a multi-modal identifier, a cache salt or a
        /// prompt-embedding hash; the prefix-hash parity check is what covers those.
        extra_keys: Option<usize>,
    },
    /// `AllBlocksCleared()`.
    AllBlocksCleared,
}

impl PoolEvent {
    /// Canonical form used by the parity log.
    ///
    /// The hash of a `BlockRemoved` is the block hash half of the key (`get_block_hash` strips
    /// the 4-byte group id). `take_events` normally rewrites those hashes into integers
    /// (`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`, on by default), which the parity harness
    /// switches off: it compares the pool's own payload, so the bytes are what both sides
    /// print.
    pub fn canonical_for_log(&self, medium: &str) -> String {
        match self {
            PoolEvent::BlockRemoved { hash, group_idx } => {
                format!("removed:{}:{}:{}", hex_encode(hash), group_idx, medium)
            }
            PoolEvent::BlockStored {
                hashes,
                parent,
                token_ids,
                block_size,
                group_idx,
                extra_keys,
            } => format!(
                "stored:{}:{}:{}:{}:{}:{}:{}",
                join_or_dash(hashes.iter().map(|hash| hex_encode(hash))),
                parent.as_deref().map(hex_encode).unwrap_or_else(|| "-".to_string()),
                join_or_dash(token_ids.iter().map(i64::to_string)),
                block_size,
                group_idx,
                medium,
                match extra_keys {
                    None => "-".to_string(),
                    Some(count) => vec!["~"; *count].join(","),
                },
            ),
            PoolEvent::AllBlocksCleared => format!("cleared:{medium}"),
        }
    }
}

/// The slice of a `Request` the prefix-cache entry points read.
///
/// Python reads `request.block_hashes` (hashes at `hash_block_size` granularity), the token ids
/// the `BlockStored` event carries, and — only to build the extra keys — the multi-modal
/// features, the LoRA request, the cache salt, the prompt embeddings and the session id. The
/// harness models a text-only request, so those last five are absent and every extra key comes
/// out as `None`.
pub struct RequestView<'a> {
    /// `request.block_hashes`.
    pub block_hashes: &'a [Vec<u8>],
    /// `request.all_token_ids`.
    pub all_token_ids: &'a [i64],
}

/// The deterministic half of `KVCacheMetricsCollector`.
///
/// The harness runs the collector with `sample_rate=1.0`, so sampling itself (`random.random()
/// < rate`) has nothing left to decide and every allocated block is tracked. The recorded
/// *durations* are wall-clock values and are not compared; what is compared is which blocks are
/// tracked, how often each was accessed (the history deque caps at 4) and how many eviction
/// events were drained.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    tracked: Vec<(u64, usize)>,
    evictions: usize,
}

impl Metrics {
    fn on_block_allocated(&mut self, block_id: u64) {
        if !self.tracked.iter().any(|(id, _)| *id == block_id) {
            self.tracked.push((block_id, 0));
        }
    }

    fn on_block_accessed(&mut self, block_id: u64) {
        if let Some((_, accesses)) = self.tracked.iter_mut().find(|(id, _)| *id == block_id) {
            // `access_history` is a deque(maxlen=4), so the count saturates.
            *accesses = (*accesses + 1).min(4);
        }
    }

    fn on_block_evicted(&mut self, block_id: u64) {
        if let Some(position) = self.tracked.iter().position(|(id, _)| *id == block_id) {
            self.tracked.remove(position);
            self.evictions += 1;
        }
    }

    fn reset(&mut self) {
        self.tracked.clear();
        self.evictions = 0;
    }

    /// `drain_events()`: return how many eviction events were pending and clear them.
    fn drain(&mut self) -> usize {
        let drained = self.evictions;
        self.evictions = 0;
        drained
    }

    fn dump(&self) -> String {
        let mut tracked = self.tracked.clone();
        tracked.sort_unstable();
        let tracked = tracked
            .iter()
            .map(|(id, accesses)| format!("{id}:{accesses}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "evictions={} tracked={}",
            self.evictions,
            if tracked.is_empty() {
                "-".to_string()
            } else {
                tracked
            },
        )
    }
}

impl BlockPool {
    /// Mirror of `BlockPool(...)`.
    ///
    /// `hash_block_size` is the granularity of the hashes a request carries; the pool itself
    /// only needs it in the `Request`-based entry points, where `resolve_block_hashes` converts
    /// between this granularity and a group's block size. `collect_metrics` stands in for
    /// passing a `KVCacheMetricsCollector` with `sample_rate=1.0`, and `medium` is the string
    /// the KV-cache events report.
    pub fn new(
        num_gpu_blocks: usize,
        enable_caching: bool,
        hash_block_size: i64,
        enable_kv_cache_events: bool,
        collect_metrics: bool,
        medium: &str,
    ) -> Result<Self, PyError> {
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
            hash_block_size,
            blocks,
            queue,
            map: BlockHashToBlockMap::new(),
            reuse_watchers: Vec::new(),
            enable_kv_cache_events,
            medium: medium.to_string(),
            kv_event_queue: Vec::new(),
            metrics: collect_metrics.then(Metrics::default),
        })
    }

    /// `enable_kv_cache_events`.
    pub fn events_enabled(&self) -> bool {
        self.enable_kv_cache_events
    }

    /// The medium the pool reports in its events.
    pub fn medium(&self) -> &str {
        &self.medium
    }

    /// The queued events, in emission order (the raw state, used by the dump).
    pub fn events(&self) -> &[PoolEvent] {
        &self.kv_event_queue
    }

    /// `BlockPool.take_events`: return the queued events and clear the queue.
    pub fn take_events(&mut self) -> Vec<PoolEvent> {
        if !self.enable_kv_cache_events {
            return Vec::new();
        }
        std::mem::take(&mut self.kv_event_queue)
    }

    /// `KVCacheMetricsCollector.drain_events`: how many eviction events were pending.
    pub fn drain_metric_events(&mut self) -> usize {
        self.metrics.as_mut().map(Metrics::drain).unwrap_or(0)
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
            if let Some(metrics) = self.metrics.as_mut() {
                metrics.on_block_allocated(*id);
            }
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
            if let Some(metrics) = self.metrics.as_mut() {
                metrics.on_block_accessed(*id);
            }
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
        // Python reports the eviction to the collector first, whether or not the block was
        // tracked and whether or not it ends up having cache entries.
        if let Some(metrics) = self.metrics.as_mut() {
            metrics.on_block_evicted(id);
        }
        let evicted = self.remove_cached_block_hashes(id);
        let had_entries = !evicted.is_empty();
        self.emit_block_removed_events(&evicted);
        had_entries
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

    /// `BlockPool.cache_full_blocks`: cache the full blocks of `request` that are not cached yet.
    ///
    /// `blocks` are all of the request's blocks, `num_cached_blocks` is how many of them the
    /// prefix cache already served, and only `blocks[num_cached_blocks..num_full_blocks]` are
    /// registered. Null and masked-out blocks are skipped, which is what keeps sliding-window
    /// groups from polluting the hash map. A block that already has a hash is the
    /// partial-to-full promotion case: its old entries are dropped (with `BlockRemoved` events)
    /// before the full-block hash takes over.
    pub fn cache_full_blocks(
        &mut self,
        request: &RequestView<'_>,
        blocks: &[u64],
        num_cached_blocks: usize,
        num_full_blocks: usize,
        block_size: i64,
        kv_cache_group_id: u32,
        block_mask: Option<&[bool]>,
    ) -> Result<(), PyError> {
        if num_cached_blocks >= num_full_blocks {
            return Ok(());
        }
        if num_full_blocks > blocks.len() {
            // Python slices `blocks[...]` (which clamps) but then indexes `blocks[i]` while
            // building the extra keys, so an over-long range is an IndexError there.
            return Err(PyError::Index);
        }
        let new_full_blocks = &blocks[num_cached_blocks..num_full_blocks];
        if block_mask.is_some_and(|mask| mask.len() != new_full_blocks.len()) {
            return Err(PyError::Assert);
        }
        let block_hashes = self.resolve_request_hashes(request, block_size)?;
        // Python slices the resolved hashes as well and indexes that slice per block.
        let new_block_hashes = block_hashes.get(num_cached_blocks..).unwrap_or_default();

        let mut new_hashes: Vec<Vec<u8>> = Vec::new();
        for (i, id) in new_full_blocks.iter().enumerate() {
            let index = *id as usize;
            if self.blocks[index].is_null || block_mask.is_some_and(|mask| !mask[i]) {
                continue;
            }
            let Some(block_hash) = new_block_hashes.get(i) else {
                return Err(PyError::Index);
            };
            let num_hash_tokens = (num_cached_blocks + i + 1) as i64 * block_size;
            let key = make_block_hash_with_group_id(block_hash, kv_cache_group_id);
            if self.blocks[index].hash.is_some() {
                // The only valid case where a "new full block" already has a hash is
                // partial->full promotion of the same cache block.
                let promoted =
                    self.blocks[index].num_tokens.is_some_and(|tokens| tokens < num_hash_tokens);
                if !promoted {
                    return Err(PyError::Assert);
                }
                let removed = self.remove_cached_block_hashes(*id);
                self.emit_block_removed_events(&removed);
            }
            self.insert_block_hash(&key, *id, Some(num_hash_tokens))?;
            new_hashes.push(block_hash.clone());
        }

        if self.enable_kv_cache_events {
            let parent = if num_cached_blocks == 0 {
                None
            } else {
                Some(block_hashes.get(num_cached_blocks - 1).ok_or(PyError::Index)?.clone())
            };
            // Python builds one extra-key entry per registered block and passes the list, or
            // `None` when the list is empty.
            let extra_keys = (!new_hashes.is_empty()).then_some(new_hashes.len());
            self.kv_event_queue.push(PoolEvent::BlockStored {
                hashes: new_hashes,
                parent,
                token_ids: slice_token_ids(
                    request.all_token_ids,
                    num_cached_blocks as i64 * block_size,
                    num_full_blocks as i64 * block_size,
                ),
                block_size,
                group_idx: kv_cache_group_id,
                extra_keys,
            });
        }
        Ok(())
    }

    /// `BlockPool.cache_partial_block`: register a partial prefix-cache entry for a block.
    ///
    /// A partial entry makes an existing block reachable from a fine-grained prefix boundary
    /// inside it. Returns the hash key that was registered, or `None` for a null block — the
    /// one case that returns before the argument assertions run.
    pub fn cache_partial_block(
        &mut self,
        request: &RequestView<'_>,
        id: u64,
        num_tokens: i64,
        kv_cache_group_id: u32,
        block_size: i64,
        replace_existing_hashes: bool,
    ) -> Result<Option<Vec<u8>>, PyError> {
        if self.blocks[id as usize].is_null {
            return Ok(None);
        }
        if self.hash_block_size == 0 {
            return Err(PyError::Zero);
        }
        if block_size % self.hash_block_size != 0 {
            return Err(PyError::Assert);
        }
        if !(replace_existing_hashes
            || (block_size > self.hash_block_size && num_tokens % block_size != 0))
        {
            return Err(PyError::Assert);
        }
        if num_tokens % self.hash_block_size != 0 {
            return Err(PyError::Assert);
        }
        let num_hash_blocks = num_tokens / self.hash_block_size;
        if num_hash_blocks <= 0 || num_hash_blocks > request.block_hashes.len() as i64 {
            return Err(PyError::Assert);
        }
        let block_hash = request.block_hashes[(num_hash_blocks - 1) as usize].clone();
        let key = make_block_hash_with_group_id(&block_hash, kv_cache_group_id);
        let num_hash_tokens = num_hash_blocks * self.hash_block_size;
        let mut already_cached = self.blocks[id as usize].hash.as_deref() == Some(key.as_slice())
            || self.map.contain(&key, id);
        if replace_existing_hashes {
            let removed = self.remove_cached_block_hashes(id);
            self.emit_block_removed_events(&removed);
            already_cached = false;
        } else if !already_cached
            && self.blocks[id as usize].hash.is_some()
            && self.blocks[id as usize]
                .num_tokens
                .is_some_and(|tokens| tokens < num_hash_tokens)
        {
            let removed = self.remove_cached_block_hashes(id);
            self.emit_block_removed_events(&removed);
        }
        self.insert_block_hash(&key, id, Some(num_hash_tokens))?;
        if self.enable_kv_cache_events && !already_cached {
            let parent = if num_hash_blocks > 1 {
                Some(request.block_hashes[(num_hash_blocks - 2) as usize].clone())
            } else {
                None
            };
            let block_start = (num_hash_blocks - 1) * self.hash_block_size;
            self.kv_event_queue.push(PoolEvent::BlockStored {
                hashes: vec![block_hash],
                parent,
                token_ids: slice_token_ids(request.all_token_ids, block_start, num_tokens),
                block_size: num_tokens - block_start,
                group_idx: kv_cache_group_id,
                extra_keys: Some(1),
            });
        }
        Ok(Some(key))
    }

    /// `BlockPool._emit_block_removed_events`: one `BlockRemoved` per dropped key.
    fn emit_block_removed_events(&mut self, keys: &[Vec<u8>]) {
        if !self.enable_kv_cache_events {
            return;
        }
        for key in keys {
            self.kv_event_queue.push(PoolEvent::BlockRemoved {
                hash: block_hash_of(key),
                group_idx: group_id_of(key),
            });
        }
    }

    /// `resolve_block_hashes(request.block_hashes, hash_block_size, block_size)`.
    fn resolve_request_hashes(
        &self,
        request: &RequestView<'_>,
        block_size: i64,
    ) -> Result<Vec<Vec<u8>>, PyError> {
        // The harness only ever asks for positive block sizes; a zero one would divide by zero
        // on the Python side as well.
        if self.hash_block_size <= 0 || block_size <= 0 {
            return Err(PyError::Zero);
        }
        let raw = HashList::Raw(request.block_hashes.to_vec());
        let resolved = resolve_block_hashes(
            &raw,
            self.hash_block_size as usize,
            block_size as usize,
            false,
            None,
        )?;
        Ok(resolved.values())
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
        if let Some(metrics) = self.metrics.as_mut() {
            metrics.reset();
        }
        if self.enable_kv_cache_events {
            self.kv_event_queue.push(PoolEvent::AllBlocksCleared);
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
            format!(
                "events={}",
                join_or_dash(
                    self.kv_event_queue.iter().map(|event| event.canonical_for_log(&self.medium))
                )
            ),
            format!(
                "metrics={}",
                self.metrics.as_ref().map(Metrics::dump).unwrap_or_else(|| "none".to_string())
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

/// `make_block_hash_with_group_id`: the hash bytes with the group id appended big-endian.
fn make_block_hash_with_group_id(hash: &[u8], group_id: u32) -> Vec<u8> {
    let mut key = hash.to_vec();
    key.extend_from_slice(&group_id.to_be_bytes());
    key
}

/// `all_token_ids[start:end]` for the non-negative bounds the cache paths use.
fn slice_token_ids(tokens: &[i64], start: i64, end: i64) -> Vec<i64> {
    let len = tokens.len() as i64;
    let start = start.clamp(0, len);
    let end = end.clamp(0, len);
    if start >= end {
        return Vec::new();
    }
    tokens[start as usize..end as usize].to_vec()
}

/// `get_block_hash`: the key without its trailing 4-byte group id.
fn block_hash_of(key: &[u8]) -> Vec<u8> {
    key[..key.len().saturating_sub(4)].to_vec()
}

/// `get_group_id`: the trailing 4 bytes, big-endian.
fn group_id_of(key: &[u8]) -> u32 {
    let len = key.len();
    if len < 4 {
        return 0;
    }
    u32::from_be_bytes([key[len - 4], key[len - 3], key[len - 2], key[len - 1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool with the defaults the harness uses most of the time.
    fn pool(num_gpu_blocks: usize, enable_caching: bool) -> BlockPool {
        BlockPool::new(num_gpu_blocks, enable_caching, 32, false, false, "GPU").expect("pool")
    }

    /// A pool that queues KV-cache events, which the `Request`-based entry points emit.
    fn eventful_pool(num_gpu_blocks: usize) -> BlockPool {
        BlockPool::new(num_gpu_blocks, true, 32, true, false, "GPU").expect("pool")
    }

    /// `(hash, token ids)` for a text-only request of `num_hashes` blocks of 32 tokens.
    fn request_of(num_hashes: usize) -> (Vec<Vec<u8>>, Vec<i64>) {
        (
            (0..num_hashes)
                .map(|index| vec![0x40 + index as u8, 0x11, 0x22, 0x33])
                .collect(),
            (0..num_hashes as i64 * 32).collect(),
        )
    }

    #[test]
    fn the_pool_reserves_a_null_block() {
        let mut pool = pool(4, true);
        assert_eq!(pool.num_free_blocks(), 3);
        assert_eq!(pool.free_order(), vec![1, 2, 3]);
        assert!(pool.blocks[0].is_null);
        assert_eq!(pool.get_new_blocks(3), Ok(vec![1, 2, 3]));
        // The null block is never handed out.
        assert_eq!(pool.get_new_blocks(1), Err(PyError::Value));
    }

    #[test]
    fn freed_blocks_are_reused_lifo_and_cached_ones_lru() {
        let mut pool = pool(4, true);
        let blocks = pool.get_new_blocks(3).unwrap();
        // 1 has a hash, 2 does not: 2 goes to the front (reused first), 1 to the back.
        pool.insert_block_hash(b"aa", 1, Some(32)).unwrap();
        pool.free_blocks(&blocks).unwrap();
        assert_eq!(pool.free_order(), vec![2, 3, 1]);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn eviction_drops_the_cache_entry_but_keeps_the_ref_count() {
        let mut pool = pool(4, true);
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
        let mut pool = pool(4, true);
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.free_blocks(&blocks).unwrap();
        assert_eq!(pool.num_free_blocks(), 3);
        pool.touch(&blocks).unwrap();
        assert_eq!(pool.num_free_blocks(), 2);
        assert_eq!(pool.blocks[blocks[0] as usize].ref_cnt, 1);
    }

    #[test]
    fn unpin_registers_a_reuse_watcher() {
        let mut pool = pool(4, true);
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
        let mut pool = pool(4, true);
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
        let mut pool = pool(4, true);
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
        let mut pool = pool(3, true);
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.insert_block_hash(b"aa", blocks[0], Some(32)).unwrap();
        let dump = pool.dump();
        assert_eq!(
            dump,
            "free=1 map=1 watchers=- events=- metrics=none 0:0:-:-:- 1:1:6161:32:- 2:0:-:-:-"
        );
    }

    #[test]
    fn eviction_and_reset_queue_their_events() {
        // Key layout as vLLM builds it: block hash plus a 4-byte big-endian group id.
        let mut pool = BlockPool::new(4, true, 32, true, false, "GPU").unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        let key = [0xaa, 0xbb, 0xcc, 0xdd, 0x00, 0x00, 0x00, 0x03];
        pool.insert_block_hash(&key, blocks[0], Some(32)).unwrap();
        assert!(pool.events().is_empty());

        // Handing the block out again evicts its entry, which queues a BlockRemoved with the
        // hash half of the key and its group id.
        pool.free_blocks(&blocks).unwrap();
        let reused = pool.get_new_blocks(3).unwrap();
        assert_eq!(pool.events().len(), 1);
        assert_eq!(
            pool.events()[0].canonical_for_log("GPU"),
            "removed:aabbccdd:3:GPU".to_string()
        );
        assert_eq!(pool.take_events().len(), 1);
        assert!(pool.events().is_empty());

        // A reset queues AllBlocksCleared, and it only succeeds once everything is free.
        pool.free_blocks(&reused).unwrap();
        assert!(pool.reset_prefix_cache());
        assert_eq!(
            pool.take_events()[0].canonical_for_log("GPU"),
            "cleared:GPU".to_string()
        );
    }

    #[test]
    fn events_stay_empty_when_they_are_disabled() {
        let mut pool = pool(4, true);
        let blocks = pool.get_new_blocks(1).unwrap();
        pool.insert_block_hash(b"aa", blocks[0], Some(32)).unwrap();
        pool.free_blocks(&blocks).unwrap();
        pool.get_new_blocks(3).unwrap();
        assert!(pool.events().is_empty());
        assert!(pool.take_events().is_empty());
        assert!(!pool.maybe_evict_cached_block(blocks[0]));
        assert!(pool.events().is_empty());
    }

    #[test]
    fn metrics_track_allocations_accesses_and_evictions() {
        let mut pool = BlockPool::new(4, true, 32, false, true, "GPU").unwrap();
        let blocks = pool.get_new_blocks(1).unwrap();
        let id = blocks[0];
        assert!(
            pool.dump().contains(&format!("{id}:0")),
            "dump was {}",
            pool.dump()
        );

        // Release the block and touch it like a prefix hit would; the history deque caps the
        // recorded accesses at four.
        pool.free_blocks(&blocks).unwrap();
        for _ in 0..6 {
            pool.touch(&[id]).unwrap();
        }
        let dump = pool.dump();
        assert!(dump.contains(&format!("{id}:4")), "dump was {dump}");
        assert!(dump.contains("evictions=0"), "dump was {dump}");

        // Back to zero references, then hand it out again: the collector sees the eviction
        // before the block is allocated anew, which is Python's callback order.
        for _ in 0..6 {
            pool.free_blocks(&[id]).unwrap();
        }
        pool.get_new_blocks(3).unwrap();
        assert_eq!(pool.drain_metric_events(), 1);
        assert_eq!(pool.drain_metric_events(), 0);
    }

    #[test]
    fn cache_full_blocks_caches_the_blocks_after_the_cached_prefix() {
        let mut pool = eventful_pool(4);
        let blocks = pool.get_new_blocks(2).unwrap();
        let (hashes, tokens) = request_of(2);
        let request = RequestView {
            block_hashes: &hashes,
            all_token_ids: &tokens,
        };
        pool.cache_full_blocks(&request, &blocks, 0, 2, 32, 1, None).unwrap();

        // Both blocks own their hash and the token count as of that block, so a second
        // request that covers the same prefix can hit them.
        assert_eq!(
            pool.get_cached_block(&make_block_hash_with_group_id(&hashes[0], 1)),
            Some(blocks[0])
        );
        assert_eq!(
            pool.get_cached_block(&make_block_hash_with_group_id(&hashes[1], 1)),
            Some(blocks[1])
        );
        assert!(pool.dump().contains("1:1:4011223300000001:32:-"));

        // One event for the whole call: both hashes, no parent, the token range of the cached
        // blocks, and one (empty) extra-key entry per block.
        let events = pool.take_events();
        assert_eq!(events.len(), 1);
        let expected = format!(
            "stored:40112233,41112233:-:{}:32:1:GPU:~,~",
            (0..64).map(|token| token.to_string()).collect::<Vec<_>>().join(",")
        );
        assert_eq!(events[0].canonical_for_log("GPU"), expected);
    }

    #[test]
    fn cache_full_blocks_skips_null_and_masked_blocks() {
        let mut pool = eventful_pool(4);
        let blocks = pool.get_new_blocks(2).unwrap();
        let (hashes, tokens) = request_of(2);
        let request = RequestView {
            block_hashes: &hashes,
            all_token_ids: &tokens,
        };
        // The null block is passed in the same slot the first hash belongs to: the hash is not
        // cached, and the event covers one hash only.
        pool.cache_full_blocks(&request, &[0, blocks[1]], 0, 2, 32, 2, None).unwrap();
        assert_eq!(
            pool.get_cached_block(&make_block_hash_with_group_id(&hashes[0], 2)),
            None
        );
        assert_eq!(
            pool.get_cached_block(&make_block_hash_with_group_id(&hashes[1], 2)),
            Some(blocks[1])
        );
        let events = pool.take_events();
        assert!(events[0].canonical_for_log("GPU").starts_with("stored:41112233:-:"));
        assert!(events[0].canonical_for_log("GPU").ends_with(":2:GPU:~"));

        // A mask that drops every block leaves the cache alone and passes `None` extra keys.
        let masked = pool.cache_full_blocks(&request, &blocks, 0, 2, 32, 3, Some(&[false, false]));
        assert_eq!(masked, Ok(()));
        let events = pool.take_events();
        assert_eq!(events.len(), 1);
        let token = events[0].canonical_for_log("GPU");
        assert!(token.starts_with("stored:-:-:"), "token was {token}");
        assert!(token.ends_with(":32:3:GPU:-"), "token was {token}");

        // An over-long range is an IndexError, the way Python's `blocks[i]` lookup is.
        assert_eq!(
            pool.cache_full_blocks(&request, &blocks, 0, 3, 32, 3, None),
            Err(PyError::Index)
        );
    }

    #[test]
    fn cache_partial_block_registers_inside_a_block_and_promotes() {
        let mut pool = eventful_pool(4);
        let blocks = pool.get_new_blocks(1).unwrap();
        let (hashes, tokens) = request_of(4);
        let request = RequestView {
            block_hashes: &hashes,
            all_token_ids: &tokens,
        };
        // A fine-grained entry at 32 tokens: the block has no hash yet, so it becomes primary.
        let key = pool.cache_partial_block(&request, blocks[0], 32, 0, 64, false).unwrap();
        assert_eq!(key, Some(make_block_hash_with_group_id(&hashes[0], 0)));
        assert!(pool.dump().contains("1:1:4011223300000000:32:-"));
        let events = pool.take_events();
        assert!(events[0].canonical_for_log("GPU").ends_with(":32:0:GPU:~"));

        // A deeper entry (96 tokens) promotes the block: the old key is removed with a
        // BlockRemoved event and the new one takes over as the primary hash.
        let key = pool.cache_partial_block(&request, blocks[0], 96, 0, 64, false).unwrap();
        assert_eq!(key, Some(make_block_hash_with_group_id(&hashes[2], 0)));
        let events = pool.take_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].canonical_for_log("GPU"), "removed:40112233:0:GPU");
        let stored = events[1].canonical_for_log("GPU");
        assert!(
            stored.starts_with("stored:42112233:41112233:"),
            "stored was {stored}"
        );
        assert!(stored.ends_with(":32:0:GPU:~"), "stored was {stored}");
        assert_eq!(
            pool.get_cached_block(&make_block_hash_with_group_id(&hashes[0], 0)),
            None
        );

        // Replacing the existing hashes drops them first, and a null block is never registered.
        let (hashes, tokens) = request_of(2);
        let request = RequestView {
            block_hashes: &hashes,
            all_token_ids: &tokens,
        };
        assert_eq!(
            pool.cache_partial_block(&request, 0, 32, 0, 64, false),
            Ok(None)
        );
        assert_eq!(
            pool.cache_partial_block(&request, blocks[0], 32, 0, 64, true),
            Ok(Some(make_block_hash_with_group_id(&hashes[0], 0)))
        );
        let events = pool.take_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].canonical_for_log("GPU"), "removed:42112233:0:GPU");
    }

    #[test]
    fn cache_entry_points_reject_the_same_arguments_as_python() {
        let mut pool = eventful_pool(4);
        let blocks = pool.get_new_blocks(1).unwrap();
        let (hashes, tokens) = request_of(1);
        let request = RequestView {
            block_hashes: &hashes,
            all_token_ids: &tokens,
        };
        // The partial entry has to sit inside an owning block: aligned to it, the call is
        // rejected unless the existing hashes are replaced.
        assert_eq!(
            pool.cache_partial_block(&request, blocks[0], 32, 0, 32, false),
            Err(PyError::Assert)
        );
        assert_eq!(
            pool.cache_partial_block(&request, blocks[0], 64, 0, 64, false),
            Err(PyError::Assert)
        );
        // Beyond the request's own hashes there is no prefix boundary to point at.
        assert_eq!(
            pool.cache_partial_block(&request, blocks[0], 64, 0, 64, true),
            Err(PyError::Assert)
        );
        // A block size that is not a multiple of the hash granularity is rejected.
        assert_eq!(
            pool.cache_partial_block(&request, blocks[0], 32, 0, 48, false),
            Err(PyError::Assert)
        );
    }
}
