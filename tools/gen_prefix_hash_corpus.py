#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a prefix-cache hashing corpus and Python-computed expectations.

This is the Python half of the `v1/core` migration spike: it produces a corpus of
token blocks plus the block hashes that the *real* vLLM implementation
(`vllm.v1.core.kv_cache_utils.hash_block_tokens`, `init_none_hash`) computes for
them. The Rust re-implementation in `rust/src/kv-hash` must reproduce every one of
these values byte for byte; `tools/check_prefix_hash_parity.sh` drives both sides and
diffs the results.

Usage:
    python3 tools/gen_prefix_hash_corpus.py --out-dir /tmp/prefix-hash-parity \
        --num-blocks 200 --block-size 32 --vocab-size 248320 --seed 0
    python3 tools/gen_prefix_hash_corpus.py --out-dir /tmp/prefix-hash-parity \
        --bench-trials 5
"""

from __future__ import annotations

import argparse
import cbor2
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_none_hash_seed,
    hash_block_tokens,
    init_none_hash,
)

# Algorithms whose parity the Rust spike covers. `xxhash*` needs xxhash.xxh3_128_digest
# and has no offline Rust counterpart in this repository.
SPIKE_ALGOS = ("sha256", "sha256_cbor")

EXTRA_CASES: dict[str, tuple | None] = {
    "none": None,
    "mm": (1234567890123,),
    "tag": ("mmhash",),
}


def serialize(hash_algo: str, value: object) -> bytes:
    """Reproduce the serialization step of the named algorithm."""
    if hash_algo.endswith("_cbor"):
        return cbor2.dumps(value, canonical=True)
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def generate_blocks(
    num_blocks: int, block_size: int, vocab_size: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(block_size)]
        for _ in range(num_blocks)
    ]


def write_corpus(
    path: Path, blocks: list[list[int]], block_size: int
) -> None:
    with path.open("w") as handle:
        handle.write(f"{block_size}\n")
        for block in blocks:
            handle.write(" ".join(str(token) for token in block))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory for generated files.")
    parser.add_argument("--num-blocks", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=248320,
        help="Token id range [0, vocab_size); default matches Qwen3.8-27B.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--payload-blocks",
        type=int,
        default=3,
        help="How many serialized payloads to dump for debugging.",
    )
    parser.add_argument(
        "--bench-trials",
        type=int,
        default=0,
        help="If > 0, also time the Python hashing chain on this corpus.",
    )
    parser.add_argument(
        "--algos",
        default=",".join(SPIKE_ALGOS),
        help="Comma separated algorithms to emit expectations for.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = generate_blocks(args.num_blocks, args.block_size, args.vocab_size, args.seed)
    write_corpus(out_dir / "corpus.txt", blocks, args.block_size)

    for algo in args.algos.split(","):
        hash_fn = get_hash_fn_by_name(algo)
        init_none_hash(hash_fn)
        for case, extra_keys in EXTRA_CASES.items():
            expected: list[str] = []
            payloads: list[str] = []
            parent_hash = None
            for index, block in enumerate(blocks):
                if index < args.payload_blocks:
                    # hash_block_tokens() substitutes NONE_HASH for a falsy parent, so
                    # the dumped payload must do the same to stay faithful.
                    value = (parent_hash or kv_cache_utils.NONE_HASH, tuple(block), extra_keys)
                    payloads.append(serialize(algo, value).hex())
                parent_hash = hash_block_tokens(hash_fn, parent_hash, block, extra_keys)
                expected.append(parent_hash.hex())
            suffix = f"{algo}-{case}"
            (out_dir / f"expected-{suffix}.txt").write_text("\n".join(expected) + "\n")
            (out_dir / f"payload-{suffix}.txt").write_text("\n".join(payloads) + "\n")

        # NONE_HASH is derived from the resolved seed, so ask vLLM for it directly.
        (out_dir / f"none-{algo}.txt").write_text(kv_cache_utils.NONE_HASH.hex() + "\n")
        (out_dir / f"seed-{algo}.txt").write_text(get_none_hash_seed() + "\n")

        if args.bench_trials > 0:
            timings = []
            for _ in range(args.bench_trials):
                parent_hash = None
                start = time.perf_counter()
                for block in blocks:
                    parent_hash = hash_block_tokens(hash_fn, parent_hash, block, None)
                timings.append(time.perf_counter() - start)
            tokens = len(blocks) * args.block_size
            print(
                f"python algo={algo} blocks={len(blocks)} tokens={tokens} "
                f"best={min(timings):.6f}s avg={statistics.mean(timings):.6f}s "
                f"tokens_per_second={tokens / min(timings):.0f}"
            )

    print(f"wrote corpus and expectations to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
