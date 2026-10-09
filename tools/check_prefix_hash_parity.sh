#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Byte-exact parity check between the Rust prefix-cache hashing re-implementation
# (`rust/src/kv-hash`) and the Python original
# (`vllm/v1/core/kv_cache_utils.hash_block_tokens`).
#
# The Python side is the source of truth: `tools/gen_prefix_hash_corpus.py` uses the
# real vLLM functions to produce the corpus and the expected block hashes, then this
# script makes the Rust implementation reproduce them and diffs the two.
#
# Usage:
#   bash tools/check_prefix_hash_parity.sh [--num-blocks N] [--block-size N] [--vocab-size N] [--skip-build] [--skip-tests]
#
# Exits non-zero on any mismatch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The corpus generator imports vLLM and cbor2, so pin the venv the launcher uses instead of
# whatever `python3` the caller's PATH happens to resolve to; an older interpreter from the
# base conda env fails with `ModuleNotFoundError: No module named 'cbor2'`.
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

PYTHON="${PYTHON:-python3}"
RUST_BIN="$REPO_ROOT/rust/target/release/vllm-kv-hash"
WORK_DIR="${WORK_DIR:-}"
NUM_BLOCKS=200
BLOCK_SIZE=32
VOCAB_SIZE=248320
SKIP_BUILD=0
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d /tmp/prefix-hash-parity.XXXXXX)"
fi
mkdir -p "$WORK_DIR"

# Fail in a second with a readable message instead of half a minute later with a traceback
# from inside the corpus generator.
if ! "$PYTHON" -c 'import cbor2' > /dev/null 2>&1; then
  echo "[parity] $PYTHON cannot import cbor2 (needed by tools/gen_prefix_hash_corpus.py);" >&2
  echo "[parity] run from the repo venv or set PYTHON=/path/to/python3" >&2
  exit 1
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[parity] building vllm-kv-hash"
  (cd rust && cargo build --release -p vllm-kv-hash >/dev/null)
fi
if [[ ! -x "$RUST_BIN" ]]; then
  echo "rust binary missing: $RUST_BIN (run without --skip-build)" >&2
  exit 1
fi

# The cross-language diff below proves the two implementations agree on the corpus. The
# crate's own unit tests cover what that diff cannot see: the semantics of a partially
# filled block, the chained-hash state, and the serialization edge cases.
if [[ "$SKIP_TESTS" != "1" ]]; then
  echo "[parity] running vllm-kv-hash unit tests"
  TEST_LOG="$WORK_DIR/cargo-test.log"
  if ! (cd rust && cargo test --release -p vllm-kv-hash) > "$TEST_LOG" 2>&1; then
    echo "[parity] FAIL vllm-kv-hash unit tests" >&2
    grep -aE '^(test |failures:|error)' "$TEST_LOG" | tail -20 >&2
    echo "prefix_hash_parity_failed checks=0 failures=1" >&2
    exit 1
  fi
  # cargo prints one "test result:" line per target and the empty targets report 0 passed,
  # so the last line is not the interesting one; sum them instead.
  tests_passed="$(grep -a '^test result' "$TEST_LOG" | awk '{p += $4} END {print p + 0}')"
  if [[ "$tests_passed" -eq 0 ]]; then
    echo "[parity] FAIL vllm-kv-hash ran no unit tests (use --skip-tests to skip this check)" >&2
    echo "prefix_hash_parity_failed checks=0 failures=1" >&2
    exit 1
  fi
  echo "[parity] PASS vllm-kv-hash unit tests: $tests_passed passed, 0 failed"
fi

checks=0
failures=0

report() {
  local name="$1" ok="$2" detail="${3:-}"
  if [[ "$ok" == "1" ]]; then
    echo "[parity] PASS $name"
  else
    echo "[parity] FAIL $name ${detail}"
    failures=$((failures + 1))
  fi
  checks=$((checks + 1))
}

# extra-case name -> CLI argument for the Rust binary ("" means no --extra)
extra_arg() {
  case "$1" in
    none) echo "" ;;
    mm) echo "1234567890123" ;;
    tag) echo "s:mmhash" ;;
    *) echo "unknown extra case $1" >&2; exit 1 ;;
  esac
}

echo "[parity] generating corpus and Python expectations in $WORK_DIR"
"$PYTHON" tools/gen_prefix_hash_corpus.py \
  --out-dir "$WORK_DIR" \
  --num-blocks "$NUM_BLOCKS" \
  --block-size "$BLOCK_SIZE" \
  --vocab-size "$VOCAB_SIZE" \
  --seed 0 >/dev/null

for algo in sha256 sha256_cbor; do
  # NONE_HASH must match the value init_none_hash() derived in Python.
  "$RUST_BIN" none-hash --algo "$algo" > "$WORK_DIR/rust-none-$algo.txt"
  if diff -q "$WORK_DIR/none-$algo.txt" "$WORK_DIR/rust-none-$algo.txt" >/dev/null; then
    report "none-hash/$algo" 1
  else
    report "none-hash/$algo" 0 "(python=$(cat "$WORK_DIR/none-$algo.txt") rust=$(cat "$WORK_DIR/rust-none-$algo.txt"))"
  fi

  for case_name in none mm tag; do
    extra="$(extra_arg "$case_name")"
    extra_flags=()
    if [[ -n "$extra" ]]; then
      extra_flags=(--extra "$extra")
    fi
    suffix="$algo-$case_name"

    # 1. serialized payload bytes
    "$RUST_BIN" payload --algo "$algo" --corpus "$WORK_DIR/corpus.txt" \
      "${extra_flags[@]}" --limit 3 > "$WORK_DIR/rust-payload-$suffix.txt"
    if diff -q "$WORK_DIR/payload-$suffix.txt" "$WORK_DIR/rust-payload-$suffix.txt" >/dev/null; then
      report "payload/$suffix" 1
    else
      report "payload/$suffix" 0 "$(diff "$WORK_DIR/payload-$suffix.txt" "$WORK_DIR/rust-payload-$suffix.txt" | head -4 | tr '\n' ' ')"
    fi

    # 2. full chained block hashes
    "$RUST_BIN" hash --algo "$algo" --corpus "$WORK_DIR/corpus.txt" \
      "${extra_flags[@]}" > "$WORK_DIR/rust-expected-$suffix.txt"
    if diff -q "$WORK_DIR/expected-$suffix.txt" "$WORK_DIR/rust-expected-$suffix.txt" >/dev/null; then
      report "hash/$suffix ($NUM_BLOCKS blocks)" 1
    else
      report "hash/$suffix ($NUM_BLOCKS blocks)" 0 "$(diff "$WORK_DIR/expected-$suffix.txt" "$WORK_DIR/rust-expected-$suffix.txt" | head -4 | tr '\n' ' ')"
    fi
  done
done

# A pinned PYTHONHASHSEED must change NONE_HASH identically on both sides.
# Python only accepts the value "random" or an integer in [0, 4294967295].
PINNED_SEED="12345"
PINNED_DIR="$WORK_DIR/pinned"
mkdir -p "$PINNED_DIR"
PYTHONHASHSEED="$PINNED_SEED" "$PYTHON" tools/gen_prefix_hash_corpus.py \
  --out-dir "$PINNED_DIR" \
  --num-blocks "$NUM_BLOCKS" \
  --block-size "$BLOCK_SIZE" \
  --vocab-size "$VOCAB_SIZE" \
  --seed 0 >/dev/null
for algo in sha256 sha256_cbor; do
  "$RUST_BIN" none-hash --algo "$algo" --seed "$PINNED_SEED" > "$PINNED_DIR/rust-none-$algo.txt"
  if diff -q "$PINNED_DIR/none-$algo.txt" "$PINNED_DIR/rust-none-$algo.txt" >/dev/null; then
    report "none-hash/$algo (PYTHONHASHSEED)" 1
  else
    report "none-hash/$algo (PYTHONHASHSEED)" 0
  fi
  "$RUST_BIN" hash --algo "$algo" --corpus "$WORK_DIR/corpus.txt" --seed "$PINNED_SEED" \
    > "$PINNED_DIR/rust-expected-$algo.txt"
  if diff -q "$PINNED_DIR/expected-$algo-none.txt" "$PINNED_DIR/rust-expected-$algo.txt" >/dev/null; then
    report "hash/$algo (PYTHONHASHSEED, $NUM_BLOCKS blocks)" 1
  else
    report "hash/$algo (PYTHONHASHSEED, $NUM_BLOCKS blocks)" 0
  fi
done

echo "[parity] artifacts: $WORK_DIR"
if [[ "$failures" != "0" ]]; then
  echo "prefix_hash_parity_failed checks=$checks failures=$failures" >&2
  exit 1
fi
echo "prefix_hash_parity_ok total=$checks"
