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
    python3 tools/gen_block_pool_trace.py --out-dir /tmp/block-pool-trace \\
        --pool-events 1 --pool-metrics 1

The trace covers three objects: a standalone `FreeKVCacheBlockQueue`, a standalone
`BlockHashToBlockMap`, and a real `BlockPool` built on both. The pool ops drive its public
allocation, free, touch, unpin, eviction and reset paths, and install cache entries through
the same private helpers Python uses (`_insert_block_hash`, and the cache map for lookups),
so no hashing happens on this side — the trace carries the hashes the allocator would have
computed. `--pool-events` and `--pool-metrics` add the two optional observers: `take_events` /
`drain_metrics` operations and the queued events and tracked blocks the pool dump reports.
Two things are outside the comparison on purpose: the integer rewrite of block hash bytes that
`take_events` performs (switched off at the top of this file) and the collector's wall-clock
durations (the deterministic half is what is compared).
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable
from pathlib import Path

# The pool logs on the reset path; keep the tool's own stdout readable (and parsable).
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
# `take_events` rewrites the pool's block hash bytes into integers by default (see
# `maybe_convert_block_hash`). The harness compares the pool's own payload, and the Rust port
# keeps the bytes, so the cosmetic conversion is switched off instead of being modelled.
os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] = "0"

from vllm.distributed.kv_events import MEDIUM_GPU  # noqa: E402
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
    resolve_block_hashes,
)

# Keys are arbitrary bytes: BlockHashWithGroupId is a NewType over bytes, and the map only
# depends on the key being hashable and comparable. The standalone map and the pool use
# different prefixes so their keys cannot be confused in the trace. Pool keys use vLLM's real
# layout (`make_block_hash_with_group_id`): four hash bytes followed by a 4-byte big-endian
# group id, which is what the KV-cache events strip apart again.
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
        pool_events: bool = False,
        pool_metrics: bool = False,
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
            PoolDriver(
                self,
                pool_blocks,
                pool_caching,
                key_space,
                rng,
                enable_events=pool_events,
                collect_metrics=pool_metrics,
            )
            if pool_blocks
            else None
        )
        self.hashlist = HashListDriver(self, rng)

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
        # Explicit setup so the trace never depends on either side's defaults.
        self.hashlist.op_raw()
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
        self.hashlist.op_len()
        self.hashlist.op_iter()

    def step(self) -> None:
        # The pool and the hash-list view get their own share of the steps and are driven
        # from their own state.
        draw = self.rng.random()
        if self.pool is not None and draw < 0.5:
            self.pool.step()
            return
        if draw < 0.7:
            self.hashlist.step()
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
        enable_events: bool = False,
        collect_metrics: bool = False,
    ) -> None:
        self.rec = recorder
        self.rng = rng
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.enable_events = enable_events
        self.collect_metrics = collect_metrics
        self.keys = [
            bytes([POOL_KEY_BYTES + i, 0x11, 0x22, 0x33]) + (i % 4).to_bytes(4, "big")
            for i in range(key_space)
        ]
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

    def _has_primary_hash(self, block_id: int) -> bool:
        assert self.pool is not None
        return self.pool.blocks[block_id].block_hash is not None

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
            f"events={self._events_dump()}",
            f"metrics={self._metrics_dump()}",
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
        collector = KVCacheMetricsCollector(sample_rate=1.0) if self.collect_metrics else None
        self.pool = BlockPool(
            num_gpu_blocks=self.num_blocks,
            enable_caching=self.enable_caching,
            hash_block_size=32,
            enable_kv_cache_events=self.enable_events,
            metrics_collector=collector,
        )
        self.held = []
        self._record(
            f"pool_init {self.num_blocks} {int(self.enable_caching)} "
            f"{int(self.enable_events)} {int(self.collect_metrics)} {MEDIUM_GPU}",
            ["ok"],
        )

    def op_take_events(self) -> None:
        assert self.pool is not None
        events = self.pool.take_events()
        tokens = [token for event in events for token in self._event_tokens(event)]
        self._record("pool_take_events", ["ok", *tokens])

    def op_drain_metrics(self) -> None:
        assert self.pool is not None
        drained = 0 if self.pool.metrics_collector is None else len(
            self.pool.metrics_collector.drain_events()
        )
        self._record("pool_drain_metrics", ["ok", str(drained)])

    def _events_dump(self) -> str:
        assert self.pool is not None
        tokens = [
            token
            for event in self.pool.kv_event_queue
            for token in self._event_tokens(event)
        ]
        return ",".join(tokens) if tokens else "-"

    def _metrics_dump(self) -> str:
        """Deterministic half of the collector: which blocks are tracked, their capped access
        counts, and how many eviction events are pending. The recorded durations are wall-clock
        values and are deliberately not compared."""
        assert self.pool is not None
        collector = self.pool.metrics_collector
        if collector is None:
            return "none"
        tracked = ",".join(
            f"{block_id}:{len(state.access_history)}"
            for block_id, state in sorted(collector.block_metrics.items())
        )
        return "evictions={} tracked={}".format(len(collector._eviction_events), tracked or "-")

    def _hash_hex(self, block_hash) -> str:
        """Hex of one hash from a KV-cache event.

        `take_events` normally rewrites the pool's bytes into integers, which is switched off
        at import time (see the top of this file). The integer branch is kept as a guard: it is
        what an accidentally re-enabled conversion hits, and `bytes(<64-bit int>)` would
        otherwise try to allocate terabytes.
        """
        if isinstance(block_hash, int):
            raise AssertionError(
                "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES must stay 0: the harness compares raw "
                "block hash bytes"
            )
        return bytes(block_hash).hex()

    def _event_tokens(self, event) -> list[str]:
        """Canonical tokens for one KVCacheEvent.

        The hash bytes of a `BlockRemoved` are already stripped of the group id by the pool
        (`get_block_hash`), and the event carries the group id it recovered, so both come
        straight from the payload instead of being re-derived here.
        """
        if event.__class__.__name__ == "AllBlocksCleared":
            return [f"cleared:{MEDIUM_GPU}"]
        group_idx = getattr(event, "group_idx", None)
        group_idx = 0 if group_idx is None else int(group_idx)
        return [
            f"removed:{self._hash_hex(block_hash)}:{group_idx}:{event.medium}"
            for block_hash in event.block_hashes
        ]

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
                "take_events",
                "drain_metrics",
            ],
            weights=[10, 8, 4, 5, 3, 8, 7, 3, 1, 3, 2, 2, 2, 4, 3],
        )[0]

        if choice == "new":
            # Sometimes ask for more blocks than the pool has, to exercise the ValueError.
            free = (self.pool.get_num_free_blocks() if self.pool else 0) or 0
            count = self.rng.choice([1, 2, 3, free, free + 1]) if free < 8 else self.rng.randint(1, 4)
            self.op_new(max(count, 1))
        elif choice == "take_events":
            self.op_take_events()
        elif choice == "drain_metrics":
            self.op_drain_metrics()
        elif choice in ("free", "unpin", "touch"):
            ids = self._sample_held()
            if not ids:
                self.op_order()
                return
            {"free": self.op_free, "unpin": self.op_unpin, "touch": self.op_touch}[choice](ids)
        elif choice == "evict":
            # `BlockPool.evict_blocks` takes a set, and Python iterates it in set order, so a
            # multi-id call queues its events in an order the Rust port cannot reproduce. With
            # events enabled the log has to be reproducible, so blocks are evicted one at a
            # time there; the runs without events keep the multi-id form.
            limit = 1 if self.enable_events else min(3, len(self.block_ids))
            self.op_evict(self.rng.sample(self.block_ids, self.rng.randint(1, limit)))
        elif choice == "set_hash":
            # With events enabled a block must keep a single cached hash: `_remove_cached_block_hashes`
            # returns the primary one plus an unordered set, and event order is part of the
            # compared log, so the extra-hash case is exercised by the runs without events.
            candidates = [
                block_id
                for block_id in (self.held or self.block_ids)
                if not (self.enable_events and self._has_primary_hash(block_id))
            ]
            if not candidates:
                self.op_num_free()
                return
            block_id = self.rng.choice(candidates)
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


class HashListDriver:
    """Drives `BlockHashListWithBlockSize` and `resolve_block_hashes`.

    These are the two ways the scheduler reads a prefix's hashes at the granularity a KV cache
    group needs. Unlike the pool, the interesting parts here are the index arithmetic
    (`(idx + 1) * scale_factor - 1`), Python's slice normalisation, negative indices, and the
    branches of `resolve_block_hashes` (same size, existing view, fine-grained partial hits,
    and the size assertion).

    `current` is whatever the last operation produced — a raw hash list or a view — which is
    also how the call site sees it, so both representations stay in play.
    """

    def __init__(self, recorder: Recorder, rng: random.Random) -> None:
        self.rec = recorder
        self.rng = rng
        self.hash_block_size = 16
        self.raw: list[bytes] = []
        self.current: list[bytes] | BlockHashListWithBlockSize = []

    def _record(self, operation: str, tokens: list[str]) -> None:
        self.rec._record(operation, tokens)

    def _hashes(self, count: int) -> list[bytes]:
        return [bytes([0xC0 + self.rng.randrange(16), self.rng.randrange(256)]) for _ in range(count)]

    def _hexes(self, values) -> list[str]:
        return [bytes(value).hex() for value in values]

    # --------------------------------------------------------------------- ops

    def op_raw(self) -> None:
        size = self.rng.choice([8, 16, 32])
        hashes = self._hashes(self.rng.randint(1, 8))
        self.hash_block_size = size
        self.raw = hashes
        self.current = hashes
        self._record(
            "bhlist_raw " + " ".join([str(size), *(hash_.hex() for hash_ in hashes)]), ["ok"]
        )

    def op_init(self) -> None:
        size = self.rng.choice([8, 16, 32])
        scale = self.rng.choice([1, 2, 3, 4])
        # Sometimes ask for a target size that is not a multiple, to hit the assert.
        target = size * scale + (1 if self.rng.random() < 0.15 else 0)
        hashes = self._hashes(self.rng.randint(1, 6) * scale)
        operation = "bhlist_init " + " ".join(
            [str(size), str(target), *(hash_.hex() for hash_ in hashes)]
        )
        try:
            view = BlockHashListWithBlockSize(hashes, size, target)
        except Exception as exc:  # noqa: BLE001
            self.rec._record_error(operation, exc)
            return
        self.hash_block_size = size
        self.raw = hashes
        self.current = view
        self._record(operation, ["ok"])

    def op_len(self) -> None:
        self._record("bhlist_len", ["ok", str(len(self.current))])

    def op_get(self) -> None:
        length = len(self.current)
        index = self.rng.choice(
            [0, -1, length - 1, length, length + 3, -length - 2, self.rng.randint(-6, 6)]
        )
        operation = f"bhlist_get {index}"
        try:
            value = self.current[index]
        except Exception as exc:  # noqa: BLE001
            self.rec._record_error(operation, exc)
            return
        self._record(operation, ["ok", bytes(value).hex()])

    def op_slice(self) -> None:
        parts = []
        for _ in range(3):
            if self.rng.random() < 0.4:
                parts.append("-")
            else:
                parts.append(str(self.rng.choice([0, 1, 2, -1, -2, 5, -5, 0])))
        start = None if parts[0] == "-" else int(parts[0])
        stop = None if parts[1] == "-" else int(parts[1])
        step = None if parts[2] == "-" else int(parts[2])
        operation = "bhlist_slice " + " ".join(parts)
        try:
            sliced = self.current[slice(start, stop, step)]
        except Exception as exc:  # noqa: BLE001
            self.rec._record_error(operation, exc)
            return
        self._record(operation, ["ok", *self._hexes(sliced)])

    def op_iter(self) -> None:
        self._record("bhlist_iter", ["ok", *self._hexes(list(self.current))])

    def op_resolve(self) -> None:
        factor = self.rng.choice([1, 1, 2, 3, 4])
        block_size = self.hash_block_size * factor + (1 if self.rng.random() < 0.15 else 0)
        fine = self.rng.random() < 0.5
        alignment = self.rng.choice(
            [None, self.hash_block_size, self.hash_block_size * 2, 1, block_size]
        )
        operation = (
            f"bhlist_resolve {block_size} {int(fine)} "
            f"{'-' if alignment is None else alignment}"
        )
        try:
            resolved = resolve_block_hashes(
                self.current,
                self.hash_block_size,
                block_size,
                supports_fine_grained_hash_lookup=fine,
                alignment_tokens=alignment,
            )
        except Exception as exc:  # noqa: BLE001
            self.rec._record_error(operation, exc)
            return
        self.current = resolved
        self._record(operation, ["ok", *self._hexes(list(resolved))])

    # ---------------------------------------------------------------- schedule

    def step(self) -> None:
        choice = self.rng.choices(
            ["raw", "init", "len", "get", "slice", "iter", "resolve"],
            weights=[2, 4, 3, 8, 8, 3, 10],
        )[0]
        {
            "raw": self.op_raw,
            "init": self.op_init,
            "len": self.op_len,
            "get": self.op_get,
            "slice": self.op_slice,
            "iter": self.op_iter,
            "resolve": self.op_resolve,
        }[choice]()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-ops", type=int, default=400)
    parser.add_argument("--key-space", type=int, default=6)
    parser.add_argument("--pool-blocks", type=int, default=24)
    parser.add_argument("--pool-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pool-events", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pool-metrics", type=int, choices=(0, 1), default=0)
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
        pool_events=bool(args.pool_events),
        pool_metrics=bool(args.pool_metrics),
    )
    recorder.run(args.num_ops)

    (args.out_dir / "trace.txt").write_text("\n".join(recorder.trace) + "\n")
    (args.out_dir / "expected.txt").write_text("\n".join(recorder.log) + "\n")
    print(f"operations: {len(recorder.trace)}")
    print(f"errors recorded: {sum(' -> err ' in line for line in recorder.log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
