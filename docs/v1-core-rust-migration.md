# `v1/core` Rust migration spike

Status: **spike complete for the two CPU paths in scope, parity proven, not recommended as a
performance project.**
Scope: two per-step CPU paths of `vllm/v1/core` that the scheduler runs — the prefix-cache
hashing chain and the KV-cache block bookkeeping. This document records what was measured so
the decision does not have to be re-litigated.

## What was ported

`rust/src/kv-hash` (`vllm-kv-hash`) re-implements:

- `vllm.v1.core.kv_cache_utils.hash_block_tokens(hash_fn, parent_hash, token_ids, extra_keys)`
- `vllm.v1.core.kv_cache_utils.init_none_hash(hash_fn)` / `NONE_HASH`
- the chain driver used by `benchmarks/benchmark_prefix_block_hash.py`

Both algorithms the Python side ships are reproduced **byte for byte**, including the
serialization step, because the hash input is a serialized Python tuple:

| algorithm     | serialization                                    | digest  |
| ------------- | ------------------------------------------------ | ------- |
| `sha256`      | CPython pickle protocol 5 (the vLLM default)      | SHA-256 |
| `sha256_cbor` | canonical CBOR, `cbor2.dumps(..., canonical=True)` | SHA-256 |

`xxhash` / `xxhash_cbor` are **not** ported: they require `xxhash.xxh3_128_digest`, and
no crate providing XXH3-128 is available in this repository's offline cargo registry
(`twox-hash` only implements XXH32/XXH64). Python's seed rule is honoured for the
supported algorithms: `PYTHONHASHSEED` when set, otherwise the fixed
`vllm-none-hash` seed that cryptographic algorithms use.

## Parity evidence

`tools/check_prefix_hash_parity.sh` uses the **real Python implementation** as the
oracle: `tools/gen_prefix_hash_corpus.py` builds a token-block corpus, asks vLLM for the
expected block hashes, then the Rust binary has to reproduce them and the script diffs
the two.

```bash
bash tools/check_prefix_hash_parity.sh --num-blocks 200
# -> prefix_hash_parity_ok total=18
```

18/18 checks pass: 2 algorithms × 3 `extra_keys` shapes (`None`, an int tuple as
multimodal hashes use, a string tuple) × {serialized payload bytes, full 200-block
chained hashes}, plus `NONE_HASH` derivation with and without a pinned
`PYTHONHASHSEED`.

Byte-exact CPython pickle output was the main unknown; the encoder is pinned by golden
bytes in unit tests (7 tests in `cargo test -p vllm-kv-hash`) as well as by the
end-to-end parity run.

## Measured cost

Same corpus (10,000 blocks × 32 tokens, token ids from a 248,320 vocabulary), 5 timed
trials, best-of:

| algorithm     | Python (real vLLM) | Rust      | speedup |
| ------------- | ------------------ | --------- | ------- |
| `sha256`      | 16.06 M tokens/s   | 43.67 M tokens/s | 2.7x |
| `sha256_cbor` | 2.77 M tokens/s    | 45.41 M tokens/s | 16.4x |

Per block (32 tokens): 1.99 µs in Python vs 0.73 µs in Rust.

Two findings worth keeping:

- In Python the **default pickle path is 5.8x faster than the CBOR path**, because
  `pickle.dumps` is C-implemented while `cbor2.dumps(canonical=True)` is not. Switching
  a Python deployment to `--prefix-caching-hash-algo sha256_cbor` costs real CPU; in
  Rust both serializations run at the same ~45 M tokens/s.
- If cross-language parity is ever needed, `sha256_cbor` is the algorithm to standardise
  on (canonical CBOR is spec-defined), but the default `sha256` path is the one that has
  to be reproduced to match an existing deployment's cache keys.

## Block bookkeeping: the state-based half of the port

Hashing is stateless; the block pool is not. `rust/src/kv-pool` (`vllm-kv-pool`) ports the
layers of that state which the scheduler mutates on every step:

* `FreeKVCacheBlockQueue` — the doubly linked list of free blocks, fake head and tail sentinels
  included. Python threads the links through the `KVCacheBlock` objects; the port keeps the same
  links in an index-addressed table, which is the natural Rust shape. The failure modes are part
  of the contract and are reproduced: `ValueError` for `popleft()` on an empty list,
  `AssertionError` for `popleft_n()` beyond the free count, `RuntimeError` for `remove()` of a
  block that is not in the list.
* `BlockHashToBlockMap` — one hash key holding either a single block or an insertion-ordered
  group of blocks. The port matches the observable behaviour, including that the key disappears
  only with its last block and that a failed `pop()` leaves the key in place, at the end of the
  key order, the way delete-and-reinsert does in Python.
* `BlockPool` — the pool built on both, with the semantics that make it interesting:
  reference counts, the *two* reuse orders of `free_blocks` (uncached blocks are reused LIFO,
  cached blocks LRU, which is why the same call can land a block at either end of the free
  list), `touch` pulling a cached block out of the free list for a second request, the
  `unpin`/reuse-watcher path, cache eviction on allocation, per-block hash metadata plus the
  extra-hash set of `cached_block_hashes_by_block`, and the reset path that only succeeds when
  every block but the reserved null block is free.
* `BlockHashListWithBlockSize` and `resolve_block_hashes` — the granularity view the scheduler
  reads hashes through. `hash_block_size` and a group's `block_size` can differ, and because
  every hash chains over its whole prefix, the last `hash_block_size` hash inside a target block
  already fingerprints it: the conversion is a lazy index view, `(idx + 1) * scale_factor - 1`.
  The port reproduces that arithmetic, Python's negative indices and `IndexError`, Python slice
  normalisation (including negative steps and the zero-step `ValueError`), and every branch of
  `resolve_block_hashes` (same size, an existing view, fine-grained partial hits keeping the raw
  hashes, and the divisibility assertion). This matters here rather than in theory: the
  deployment runs a hybrid model whose KV cache groups do not all use `hash_block_size`.
* The pool's KV-cache **event queue** and **metrics collector** — the two optional observers the
  pool drives. `BlockRemoved` (the evicted hash with its 4-byte group id stripped off, the group
  id it carries, and the medium) and `AllBlocksCleared` are queued in emission order and handed
  over by `take_events()`; the collector keeps one access history per allocated block with a cap
  of four samples, counts eviction events, resets together with the pool, and gives its pending
  events up on `drain_events()`. Two parts are deliberately not compared: the rewrite of hash
  bytes into integers that `take_events()` performs under `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`
  (the harness switches it off, so both sides print the pool's own bytes) and the wall-clock
  durations the collector records (`sample_rate=1.0` leaves sampling nothing to decide, and the
  timings are not reproducible).
* The **`Request`-based cache entry points**, `cache_full_blocks` and `cache_partial_block` —
  where a request's hashes actually enter the cache. Both are reproduced: the block-size view
  over `request.block_hashes`, skipping null and masked-out blocks, the
  partial-to-full promotion that drops a block's old entries (with their events) before the full
  block hash takes over, the partial entry that makes an existing block reachable from a
  fine-grained boundary inside it (including `replace_existing_hashes` and the promotion branch
  with its three-part condition), and the single `BlockStored` event each call queues — hash
  list, parent hash, the token range the event covers, block size, group id, medium and the
  extra keys. The `Request` itself is a stand-in: the trace carries the hashes and token ids the
  real one would have computed, and the modelled request is text-only, so LoRA names,
  multi-modal identifiers, cache salts and prompt-embedding hashes never reach an extra key
  (they are covered by the prefix-hash parity check instead).

`tools/check_block_pool_parity.sh` is the hashing method turned into a generator:

```bash
bash tools/check_block_pool_parity.sh
bash tools/check_block_pool_parity.sh --seeds 6 --num-ops 700 --key-space 8 --pool-blocks 24
# -> block_pool_parity_ok traces=12 operations=8602
```

`tools/gen_block_pool_trace.py` drives the **real** Python classes with a seeded random
operation sequence and writes both the trace and the log Python produced, one line per
operation — including `err <ExceptionClass>` for the operations Python rejected. The Rust
binary replays the trace and the two logs are diffed. Sweeping seeds is what makes this a
shadow check rather than a unit test: the interesting part is that the *rejections* line up as
well, which is where a rewrite usually drifts.

Each seed is replayed twice, once with the event queue and the metrics collector off and once
with both on, so the disabled paths stay pinned too (an events-off run always reports an empty
`take_events`, a `0` from `drain_metrics` and `metrics=none`). With events enabled the harness
also evicts one block per call instead of a set of them: `BlockPool.evict_blocks` takes a
`set[int]` and Python iterates it in set order, so the *order* of the events a multi-id call
queues is an implementation detail of CPython's set layout that a port cannot reproduce. The
multi-id form stays in the events-off runs, where no events are emitted at all.

Verified with 6 seeds × 2 modes of ~717 operations (**12 traces, 8,606 operations compared,
691 rejected by Python**: 222 `ValueError`, 313 `AssertionError`, 156 `IndexError`): every log
line matches, including which exception each rejection raises. The 122 recorded requests feed
186 `cache_full_blocks` and 149 `cache_partial_block` calls (51 of them rejected the same way on
both sides), and the events they queue are compared field by field. The events-and-metrics half
contributes 211 `take_events` and 123 `drain_metrics` operations and 272 event tokens — 138
`stored:` events, 263 `removed:` and 9 `cleared:` — and 54 `pool_dump` snapshots, 41 of which
report a non-zero eviction count. The crate's own 31 unit tests pin down what the diff cannot
see — that a popped block really has no links left, that the map drops a key only with its last
block, and that the null block is never allocated.

What this does **not** cover: anything downstream of the pool — the scheduler decisions — and
`Request` itself, which the harness stands in for as described above.

## Why this is not a performance project

Hashing is only part of the Python bookkeeping in `vllm/v1/core`. The other per-step
primitive is the block pool; measured with the repository's own benchmark
(`benchmarks/benchmark_block_pool.py`, 100,000 blocks, 1,000 iterations):

| operation                 | Python     |
| ------------------------- | ---------- |
| `get_new_blocks` (10)     | 3.85 µs    |
| `get_new_blocks` (100)    | 32.8 µs    |
| `free_blocks` (100)       | 15.4 µs    |

That is ~0.32 µs per allocated block and ~0.15 µs per freed block. Even for a large
batch, a decode step's worth of pool and hashing work lands in the **hundreds of
microseconds**, against a GPU step measured in tens of milliseconds. Decode steps hash
nothing at all unless a request's generated tokens complete a new full block, and
Python only hashes blocks that were not hashed before.

So the honest conclusion is: moving `v1/core` to Rust would remove Python from the
engine's control loop, shrink the runtime Python surface, and make the Rust frontend
able to own scheduling decisions directly — but it would **not** show up in throughput,
step latency, or startup time. Startup is dominated by engine initialization and CUDA
graph capture, not by any of this.

## If this is pursued anyway

Recommended order:

1. **Block pool** (`block_pool.py`, `FreeKVCacheBlockQueue`, `KVCacheBlock`,
   `BlockHashListWithBlockSize`): the mutable state the scheduler touches every step. The free
   queue, the hash map, the block-size view, the KV-cache event queue, the metrics collector, the
   `Request`-based cache entry points and `BlockPool` itself are ported and shadow-verified (see
   above), so this step is done apart from the parts of it that belong to `Request` (its hashing
   and its token bookkeeping), which the harness stands in for. This is where a real semantics
   risk lived, not in hashing.
2. **Scheduler decisions** (`sched/scheduler.py`, its own 3,474 lines): must run as a
   shadow first, comparing decisions against the Python scheduler on the same traffic
   before anything is cut over.
3. Move the Rust-owned scheduler behind the `managed-engine` / M3–M4 process work, and
   only then shrink Python to a worker that executes the decided batches.

Note that `v1/core/sched/output.py` is the only file in `v1/core` that imports torch, so
the rest of the package has no compute dependency and can be moved without touching the
torch path.

## Reproducing the numbers

```bash
export PATH="$PWD/.venv/bin:$PATH"
bash tools/check_prefix_hash_parity.sh --num-blocks 200   # also runs the crate's unit tests
bash tools/check_block_pool_parity.sh --seeds 6 --num-ops 700 --key-space 8 --pool-blocks 24

python3 tools/gen_prefix_hash_corpus.py --out-dir /tmp/hashbench \
  --num-blocks 10000 --block-size 32 --vocab-size 248320 --seed 0 --bench-trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256 --corpus /tmp/hashbench/corpus.txt --trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256_cbor --corpus /tmp/hashbench/corpus.txt --trials 5

cd benchmarks && python3 benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000
```

The `vllm-kv-hash` binary is a development tool: it is not installed next to
`vllm/vllm-rs` and no runtime path depends on it.
