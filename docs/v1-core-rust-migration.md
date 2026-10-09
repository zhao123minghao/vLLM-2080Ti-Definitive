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

Hashing is stateless; the block pool is not. `rust/src/kv-pool` (`vllm-kv-pool`) ports the two
pieces of that state which the scheduler mutates on every step:

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

`tools/check_block_pool_parity.sh` is the hashing method turned into a generator:

```bash
bash tools/check_block_pool_parity.sh
bash tools/check_block_pool_parity.sh --seeds 6 --num-ops 1500 --num-blocks 32
# -> block_pool_parity_ok traces=6 operations=9030
```

Verified with 6 traces of 1,505 operations each (**9,030 operations compared, 522 of them
rejected by Python**): every log line matches, including which exception each rejection
raises. The crate's own 9 unit tests pin down what the diff cannot see — that a popped block
really has no links left, and that the map drops a key only with its last block.

`tools/gen_block_pool_trace.py` drives the **real** Python classes with a seeded random
operation sequence and writes both the trace and the log Python produced, one line per
operation — including `err <ExceptionClass>` for the operations Python rejected. The Rust
binary replays the trace and the two logs are diffed. Sweeping seeds is what makes this a
shadow check rather than a unit test: the interesting part is that the *rejections* line up as
well, which is where a rewrite usually drifts.

What this does **not** cover: `BlockPool` itself (block events, partial-block hashing,
reference counting, `BlockHashListWithBlockSize`) and the scheduler decisions on top of it.
The ported pieces are the data structures those layers are built from.

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
   `BlockHashListWithBlockSize`): the mutable state the scheduler touches every step.
   `FreeKVCacheBlockQueue` and `BlockHashToBlockMap` are now ported and shadow-verified (see
   above); the remaining piece is `BlockPool` itself, whose events, partial-block hashing and
   reference counting sit on top of them. This is where a real semantics risk lives, not in
   hashing.
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

python3 tools/gen_prefix_hash_corpus.py --out-dir /tmp/hashbench \
  --num-blocks 10000 --block-size 32 --vocab-size 248320 --seed 0 --bench-trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256 --corpus /tmp/hashbench/corpus.txt --trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256_cbor --corpus /tmp/hashbench/corpus.txt --trials 5

cd benchmarks && python3 benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000
```

The `vllm-kv-hash` binary is a development tool: it is not installed next to
`vllm/vllm-rs` and no runtime path depends on it.
