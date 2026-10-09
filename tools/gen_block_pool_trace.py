#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record an operation trace for the KV-cache block bookkeeping in `vllm/v1/core`.

This is the Python half of `tools/check_block_pool_parity.sh`. It drives the *real*
`FreeKVCacheBlockQueue` (whose linked list is threaded through the blocks' own
`prev_free_block` / `next_free_block` fields) and the real `BlockHashToBlockMap` with a
seeded random operation sequence, and writes two files:

* `trace.txt`    — the operations, one per line;
* `expected.txt` — one log line per operation, `"<operation> -> <outcome>"`.

The Rust side (`rust/src/kv-pool`) replays `trace.txt` and must print exactly
`expected.txt`; `diff` between the two is the correctness argument. Rejected operations
are part of that contract: the log records `err <Python exception class>`, and the Rust
port has to reject the same operation with the same exception kind.

Log format, which both sides implement literally:

    <operation line> -> ok [<value> ...]
    <operation line> -> err <ValueError|AssertionError|RuntimeError>

Values are space separated block ids; `map_dump` prints `hexkey:id,id` groups in key
insertion order; `-` stands for a missing value (`map_pop` / `map_get_one` returning
None). A value-less `ok` means the operation returned nothing useful.

Usage:
    python3 tools/gen_block_pool_trace.py --out-dir /tmp/block-pool-trace
    python3 tools/gen_block_pool_trace.py --out-dir /tmp/block-pool-trace \\
        --num-blocks 32 --num-ops 2000 --key-space 8 --seed 7
    python3 tools/gen_block_pool_trace.py --out-dir /tmp/block-pool-trace \\
        --pool-blocks 48 --pool-caching 1

The trace covers three objects: a standalone `FreeKVCacheBlockQueue`, a standalone
`BlockHashToBlockMap`, and a real `BlockPool` built on both. The pool ops drive its public
allocation, free, touch, unpin, eviction and reset paths, and install cache entries through
the same private helpers Python uses (`_insert_block_hash`, and the cache map for lookups),
so no hashing happens on this side — the trace carries the hashes the allocator would have
computed.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable
from pathlib import Path

# The pool logs on the reset path; keep the tool's own stdout readable (and parsable).
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHashWithGroupId,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
)

# Keys are arbitrary bytes: BlockHashWithGroupId is a NewType over bytes, and the map only
# depends on the key being hashable and comparable. The standalone map and the pool use
# different prefixes so their keys cannot be confused in the trace.
KEY_BYTES = 0xA0
POOL_KEY_BYTES = 0xB0


class Recorder:
    """Drives the Python classes and records both the trace and the expected log."""

    def __init__(
        self,
        num_blocks: int,
        key_space: int,
        rng: random.Random,
        pool_blocks: int = 0,
        pool_caching: bool = True,
    ) -> None:
        self.rng = rng
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self.keys = [bytes([KEY_BYTES + i]) for i in range(key_space)]
        self.queue: FreeKVCacheBlockQueue | None = None
        self.map = BlockHashToBlockMap()
        self.trace: list[str] = []
        self.log: list[str] = []
        self.pool: PoolDriver | None = (
            PoolDriver(self, pool_blocks, pool_caching, key_space, rng) if pool_blocks else None
        )

    # ------------------------------------------------------------------ helpers

    def _record(self, operation: str, tokens: list[str]) -> None:
        self.trace.append(operation)
        self.log.append(f"{operation} -> {' '.join(tokens)}")

    def _record_error(self, operation: str, exc: BaseException) -> None:
        self._record(operation, ["err", type(exc).__name__])

    def _free_ids(self) -> list[int]:
        """Block ids currently in the free list, as the queue itself reports them."""
        if self.queue is None:
            return []
        return [block.block_id for block in self.queue.get_all_free_blocks()]

    def _busy_ids(self) -> list[int]:
        free = set(self._free_ids())
        return [i for i in range(self.num_blocks) if i not in free]

    def _map_dump(self) -> str:
        """`{hash: block}` groups in key insertion order, as `map_dump` reports them."""
        groups = []
        for key, blocks in self.map._cache.items():
            if isinstance(blocks, KVCacheBlock):
                ids = [blocks.block_id]
            else:
                ids = list(blocks.keys())
            hex_key = bytes(key).hex()
            groups.append(f"{hex_key}:{','.join(str(i) for i in ids)}")
        return " ".join(groups)

    # ------------------------------------------------------------------- steps

    def op_init(self) -> None:
        self.queue = FreeKVCacheBlockQueue(self.blocks)
        self.map = BlockHashToBlockMap()
        self._record(f"init {self.num_blocks}", ["ok"])

    def op_popleft(self) -> None:
        assert self.queue is not None
        operation = "popleft"
        try:
            block = self.queue.popleft()
        except Exception as exc:  # noqa: BLE001 - the class name is the expectation
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok", str(block.block_id)])

    def op_popleft_n(self, n: int) -> None:
        assert self.queue is not None
        operation = f"popleft_n {n}"
        try:
            blocks = self.queue.popleft_n(n)
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok", *(str(block.block_id) for block in blocks)])

    def op_append(self, block_id: int) -> None:
        assert self.queue is not None
        operation = f"append {block_id}"
        try:
            self.queue.append(self.blocks[block_id])
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok"])

    def op_append_n(self, block_ids: list[int]) -> None:
        assert self.queue is not None
        operation = "append_n " + " ".join(str(i) for i in block_ids)
        try:
            self.queue.append_n([self.blocks[i] for i in block_ids])
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok"])

    def op_prepend_n(self, block_ids: list[int]) -> None:
        assert self.queue is not None
        operation = "prepend_n " + " ".join(str(i) for i in block_ids)
        try:
            self.queue.prepend_n([self.blocks[i] for i in block_ids])
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok"])

    def op_remove(self, block_id: int) -> None:
        assert self.queue is not None
        operation = f"remove {block_id}"
        try:
            self.queue.remove(self.blocks[block_id])
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok"])

    def op_peek(self) -> None:
        operation = "peek"
        try:
            ids = self._free_ids()
        except Exception as exc:  # noqa: BLE001
            self._record_error(operation, exc)
            return
        self._record(operation, ["ok", *(str(i) for i in ids)])

    def op_num_free(self) -> None:
        assert self.queue is not None
        self._record("num_free", ["ok", str(self.queue.num_free_blocks)])

    def op_map_insert(self, key: bytes, block_id: int) -> None:
        self.map.insert(key, self.blocks[block_id])
        self._record(f"map_insert {key.hex()} {block_id}", ["ok"])

    def op_map_pop(self, key: bytes, block_id: int) -> None:
        block = self.map.pop(key, block_id)
        value = "-" if block is None else str(block.block_id)
        self._record(f"map_pop {key.hex()} {block_id}", ["ok", value])

    def op_map_contain(self, key: bytes, block_id: int) -> None:
        contained = self.map.contain(key, block_id)
        self._record(f"map_contain {key.hex()} {block_id}", ["ok", "1" if contained else "0"])

    def op_map_get_one(self, key: bytes) -> None:
        block = self.map.get_one_block(key)
        value = "-" if block is None else str(block.block_id)
        self._record(f"map_get_one {key.hex()}", ["ok", value])

    def op_map_len(self) -> None:
        self._record("map_len", ["ok", str(len(self.map))])

    def op_map_dump(self) -> None:
        dump = self._map_dump()
        self._record("map_dump", ["ok"] if not dump else ["ok", dump])

    # ----------------------------------------------------------------- schedule

    def run(self, num_ops: int) -> None:
        self.op_init()
        if self.pool is not None:
            self.pool.op_init()
        for _ in range(num_ops):
            self.step()
        # A fixed tail so the final state of both implementations is always compared.
        self.op_peek()
        self.op_num_free()
        self.op_map_dump()
        self.op_map_len()
        if self.pool is not None:
            self.pool.op_order()
            self.pool.op_num_free()
            self.pool.op_usage()
            self.pool.op_dump()

    def step(self) -> None:
        # The pool gets its own share of the steps and is driven from its own state.
        if self.pool is not None and self.rng.random() < 0.6:
            self.pool.step()
            return
        choice = self.rng.choices(
            [
                "popleft",
                "popleft_n",
                "append",
                "append_n",
                "prepend_n",
                "remove",
                "peek",
                "num_free",
                "map_insert",
                "map_pop",
                "map_contain",
                "map_get_one",
                "map_len",
                "map_dump",
            ],
            weights=[8, 6, 8, 3, 3, 5, 3, 2, 8, 5, 4, 5, 1, 1],
        )[0]

        if choice == "popleft":
            self.op_popleft()
        elif choice == "popleft_n":
            # Sometimes ask for more blocks than there are, to exercise the assert.
            free = len(self._free_ids())
            n = self.rng.choice([0, 1, 2, 3, free, free + 1])
            self.op_popleft_n(max(n, 0))
        elif choice in ("append", "append_n", "prepend_n"):
            # Only blocks that are not in the list: Python tolerates a duplicate append,
            # but it corrupts the list in a way that is not interesting to compare.
            candidates = self._busy_ids()
            self.rng.shuffle(candidates)
            if not candidates:
                self.op_peek()
                return
            if choice == "append":
                self.op_append(candidates[0])
            else:
                take = candidates[: self.rng.randint(1, min(3, len(candidates)))]
                if choice == "append_n":
                    self.op_append_n(take)
                else:
                    self.op_prepend_n(take)
        elif choice == "remove":
            candidates = self._free_ids()
            if not candidates:
                self.op_peek()
                return
            self.op_remove(self.rng.choice(candidates))
        elif choice == "peek":
            self.op_peek()
        elif choice == "num_free":
            self.op_num_free()
        else:
            key = self.rng.choice(self.keys)
            # Block ids stay inside the pool: the map stores real KVCacheBlock objects, so
            # the id it reports must be the id written to the trace.
            block_id = self.rng.randrange(self.num_blocks)
            if choice == "map_insert":
                self.op_map_insert(key, block_id)
            elif choice == "map_pop":
                self.op_map_pop(key, block_id)
            elif choice == "map_contain":
                self.op_map_contain(key, block_id)
            elif choice == "map_get_one":
                self.op_map_get_one(key)
            elif choice == "map_len":
                self.op_map_len()
            else:
                self.op_map_dump()


class PoolDriver:
    """Drives a real `BlockPool` and records `pool_*` operations.

    The pool is the layer that owns reference counts, the eviction order and the cached
    hashes, so the operations below are the ones a scheduler actually calls: `get_new_blocks`
    on the way in, `free_blocks` / `unpin_blocks` on the way out, `touch` when a request hits
    an existing prefix, plus the cache lookups and the reset path.

    Cache entries are installed through `_insert_block_hash`, the same private helper Python
    uses, because the public entry point (`cache_full_blocks`) needs a `Request` and computes
    hashes — and hashing is already covered by the prefix-hash parity check. For the same
    reason lookups read the cache map directly instead of going through `get_cached_block`'s
    per-group loop.
    """

    def __init__(
        self,
        recorder: Recorder,
        num_blocks: int,
        enable_caching: bool,
        key_space: int,
        rng: random.Random,
    ) -> None:
        self.rec = recorder
        self.rng = rng
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.keys = [bytes([POOL_KEY_BYTES + i]) for i in range(key_space)]
        self.pool: BlockPool | None = None
        # Blocks handed out and not yet given back, i.e. what a request would hold.
        self.held: list[int] = []
        # Block 0 is the pool's null block; it is never allocated, cached or touched.
        self.block_ids = list(range(1, num_blocks))

    # ------------------------------------------------------------------ helpers

    def _block(self, block_id: int) -> KVCacheBlock:
        assert self.pool is not None
        return self.pool.blocks[block_id]

    def _record(self, operation: str, tokens: list[str]) -> None:
        self.rec._record(operation, tokens)

    def _call(self, operation: str, call: Callable[[], None]) -> bool:
        """Record an operation whose only outcome is the exception class, if any."""
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - the class name is the expectation
            self.rec._record_error(operation, exc)
            return False
        self._record(operation, ["ok"])
        return True

    def _release(self, ids: list[int]) -> None:
        for block_id in ids:
            self.held.remove(block_id)

    def _sample_held(self, limit: int = 3) -> list[int]:
        if not self.held:
            return []
        count = self.rng.randint(1, min(limit, len(self.held)))
        return self.rng.sample(self.held, count)

    def _dump(self) -> str:
        """The pool's observable state, in the format the Rust `pool_dump` prints.

        Two values need a canonical order because Python stores them unordered: the reuse
        watcher ids and the extra hashes of a block are sorted here, and the Rust side keeps
        them sorted as well.
        """
        assert self.pool is not None
        watchers = ",".join(str(i) for i in sorted(self.pool._reuse_watchers)) or "-"
        tokens = [
            f"free={self.pool.get_num_free_blocks()}",
            f"map={len(self.pool.cached_block_hash_to_block)}",
            f"watchers={watchers}",
        ]
        for block in self.pool.blocks:
            extras = sorted(
                block_hash.hex()
                for block_hash in self.pool.cached_block_hashes_by_block.get(block.block_id, set())
            )
            tokens.append(
                ":".join(
                    [
                        str(block.block_id),
                        str(block.ref_cnt),
                        block.block_hash.hex() if block.block_hash is not None else "-",
                        (
                            str(block.block_hash_num_tokens)
                            if block.block_hash_num_tokens is not None
                            else "-"
                        ),
                        ",".join(extras) if extras else "-",
                    ]
                )
            )
        return " ".join(tokens)

    # --------------------------------------------------------------------- ops

    def op_init(self) -> None:
        self.pool = BlockPool(
            num_gpu_blocks=self.num_blocks,
            enable_caching=self.enable_caching,
            hash_block_size=32,
        )
        self.held = []
        self._record(f"pool_init {self.num_blocks} {int(self.enable_caching)}", ["ok"])

    def op_new(self, count: int) -> None:
        assert self.pool is not None
        operation = f"pool_new {count}"
        try:
            blocks = self.pool.get_new_blocks(count)
        except Exception as exc:  # noqa: BLE001
            self.rec._record_error(operation, exc)
            return
        self.held.extend(block.block_id for block in blocks)
        self._record(operation, ["ok", *(str(block.block_id) for block in blocks)])

    def op_free(self, ids: list[int]) -> None:
        assert self.pool is not None
        operation = "pool_free " + " ".join(str(i) for i in ids)
        if self._call(operation, lambda: self.pool.free_blocks([self._block(i) for i in ids])):
            self._release(ids)

    def op_unpin(self, ids: list[int]) -> None:
        assert self.pool is not None
        operation = "pool_unpin " + " ".join(str(i) for i in ids)
        if self._call(
            operation,
            lambda: self.pool.unpin_blocks([self._block(i) for i in ids], lambda _block: None),
        ):
            self._release(ids)

    def op_touch(self, ids: list[int]) -> None:
        assert self.pool is not None
        operation = "pool_touch " + " ".join(str(i) for i in ids)
        self._call(operation, lambda: self.pool.touch([self._block(i) for i in ids]))

    def op_evict(self, ids: list[int]) -> None:
        assert self.pool is not None
        operation = "pool_evict " + " ".join(str(i) for i in ids)
        self._call(operation, lambda: self.pool.evict_blocks(set(ids)))

    def op_set_hash(self, block_id: int, key: bytes, num_tokens: int | None) -> None:
        assert self.pool is not None
        token = "-" if num_tokens is None else str(num_tokens)
        operation = f"pool_set_hash {block_id} {key.hex()} {token}"
        self._call(
            operation,
            lambda: self.pool._insert_block_hash(
                BlockHashWithGroupId(key), self._block(block_id), num_tokens
            ),
        )

    def op_cached(self, key: bytes) -> None:
        assert self.pool is not None
        operation = f"pool_cached {key.hex()}"
        block = self.pool.cached_block_hash_to_block.get_one_block(BlockHashWithGroupId(key))
        self._record(operation, ["ok", "-" if block is None else str(block.block_id)])

    def op_writable(self, block_id: int) -> None:
        assert self.pool is not None
        writable = self.pool.is_block_writable(self._block(block_id))
        self._record(f"pool_writable {block_id}", ["ok", "1" if writable else "0"])

    def op_num_free(self) -> None:
        assert self.pool is not None
        self._record("pool_num_free", ["ok", str(self.pool.get_num_free_blocks())])

    def op_usage(self) -> None:
        assert self.pool is not None
        self._record("pool_usage", ["ok", f"{self.pool.get_usage():.6f}"])

    def op_order(self) -> None:
        assert self.pool is not None
        ids = [block.block_id for block in self.pool.free_block_queue.get_all_free_blocks()]
        self._record("pool_order", ["ok", *(str(i) for i in ids)])

    def op_reset(self) -> None:
        assert self.pool is not None
        # Drop everything first so the reset condition (only the null block is used) holds.
        # Bounded, because a `free` that Python rejects would otherwise loop forever.
        for _ in range(len(self.held) + 1):
            if not self.held:
                break
            self.op_free(list(self.held))
        self._record("pool_reset", ["ok", "1" if self.pool.reset_prefix_cache() else "0"])

    def op_dump(self) -> None:
        self._record("pool_dump", ["ok", self._dump()])

    # ---------------------------------------------------------------- schedule

    def step(self) -> None:
        choice = self.rng.choices(
            [
                "new",
                "free",
                "unpin",
                "touch",
                "evict",
                "set_hash",
                "cached",
                "writable",
                "reset",
                "order",
                "num_free",
                "usage",
                "dump",
            ],
            weights=[10, 8, 4, 5, 3, 8, 7, 3, 1, 3, 2, 2, 2],
        )[0]

        if choice == "new":
            # Sometimes ask for more blocks than the pool has, to exercise the ValueError.
            free = (self.pool.get_num_free_blocks() if self.pool else 0) or 0
            count = self.rng.choice([1, 2, 3, free, free + 1]) if free < 8 else self.rng.randint(1, 4)
            self.op_new(max(count, 1))
        elif choice in ("free", "unpin", "touch"):
            ids = self._sample_held()
            if not ids:
                self.op_order()
                return
            {"free": self.op_free, "unpin": self.op_unpin, "touch": self.op_touch}[choice](ids)
        elif choice == "evict":
            ids = self.rng.sample(self.block_ids, self.rng.randint(1, min(3, len(self.block_ids))))
            self.op_evict(ids)
        elif choice == "set_hash":
            block_id = (
                self.rng.choice(self.held) if self.held else self.rng.choice(self.block_ids)
            )
            num_tokens = self.rng.choice([None, 32, 64])
            self.op_set_hash(block_id, self.rng.choice(self.keys), num_tokens)
        elif choice == "cached":
            self.op_cached(self.rng.choice(self.keys))
        elif choice == "writable":
            block_id = (
                self.rng.choice(self.held) if self.held else self.rng.choice(self.block_ids)
            )
            self.op_writable(block_id)
        elif choice == "reset":
            self.op_reset()
        elif choice == "order":
            self.op_order()
        elif choice == "num_free":
            self.op_num_free()
        elif choice == "usage":
            self.op_usage()
        else:
            self.op_dump()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-ops", type=int, default=400)
    parser.add_argument("--key-space", type=int, default=6)
    parser.add_argument("--pool-blocks", type=int, default=24)
    parser.add_argument("--pool-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_blocks < 1:
        parser.error("--num-blocks must be at least 1")
    if args.key_space < 1:
        parser.error("--key-space must be at least 1")
    if args.pool_blocks < 2:
        parser.error("--pool-blocks must be at least 2 (one block is the null block)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(
        args.num_blocks,
        args.key_space,
        random.Random(args.seed),
        pool_blocks=args.pool_blocks,
        pool_caching=bool(args.pool_caching),
    )
    recorder.run(args.num_ops)

    (args.out_dir / "trace.txt").write_text("\n".join(recorder.trace) + "\n")
    (args.out_dir / "expected.txt").write_text("\n".join(recorder.log) + "\n")
    print(f"operations: {len(recorder.trace)}")
    print(f"errors recorded: {sum(' -> err ' in line for line in recorder.log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
