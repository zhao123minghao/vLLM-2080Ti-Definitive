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
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    FreeKVCacheBlockQueue,
)

# Keys are arbitrary bytes: BlockHashWithGroupId is a NewType over bytes, and the map only
# depends on the key being hashable and comparable.
KEY_BYTES = 0xA0


class Recorder:
    """Drives the Python classes and records both the trace and the expected log."""

    def __init__(self, num_blocks: int, key_space: int, rng: random.Random) -> None:
        self.rng = rng
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self.keys = [bytes([KEY_BYTES + i]) for i in range(key_space)]
        self.queue: FreeKVCacheBlockQueue | None = None
        self.map = BlockHashToBlockMap()
        self.trace: list[str] = []
        self.log: list[str] = []

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
        for _ in range(num_ops):
            self.step()
        # A fixed tail so the final state of both implementations is always compared.
        self.op_peek()
        self.op_num_free()
        self.op_map_dump()
        self.op_map_len()

    def step(self) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-ops", type=int, default=400)
    parser.add_argument("--key-space", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_blocks < 1:
        parser.error("--num-blocks must be at least 1")
    if args.key_space < 1:
        parser.error("--key-space must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(args.num_blocks, args.key_space, random.Random(args.seed))
    recorder.run(args.num_ops)

    (args.out_dir / "trace.txt").write_text("\n".join(recorder.trace) + "\n")
    (args.out_dir / "expected.txt").write_text("\n".join(recorder.log) + "\n")
    print(f"operations: {len(recorder.trace)}")
    print(f"errors recorded: {sum(' -> err ' in line for line in recorder.log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
