#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shadow verification of the KV-cache block bookkeeping: the Rust port in
# `rust/src/kv-pool` must behave exactly like the Python original — `FreeKVCacheBlockQueue`
# and `BlockHashToBlockMap` from `vllm/v1/core/kv_cache_utils.py` and
# `vllm/v1/core/block_pool.py`, and the `BlockPool` built on top of them.
#
# Python is the source of truth. `tools/gen_block_pool_trace.py` drives the real classes
# with a seeded random operation sequence and records both the trace and the log it
# produced; the Rust binary replays the trace and the two logs are diffed line by line,
# including the operations Python rejected and the exception class it raised.
#
# Usage:
#   bash tools/check_block_pool_parity.sh [--seeds N] [--num-ops N] [--num-blocks N]
#                                         [--key-space N] [--pool-blocks N]
#                                         [--pool-caching 0|1] [--skip-build] [--skip-tests]
#
# Exits non-zero on the first mismatch (the diff is printed).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The recorder imports vLLM, so pin the venv the launcher uses instead of whatever
# `python3` the caller's PATH happens to resolve to.
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

PYTHON="${PYTHON:-python3}"
RUST_BIN="$REPO_ROOT/rust/target/release/vllm-kv-pool"
WORK_DIR="${WORK_DIR:-}"
SEEDS=4
NUM_OPS=400
NUM_BLOCKS=16
KEY_SPACE=6
POOL_BLOCKS=24
POOL_CACHING=1
SKIP_BUILD=0
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --num-ops) NUM_OPS="$2"; shift 2 ;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
    --key-space) KEY_SPACE="$2"; shift 2 ;;
    --pool-blocks) POOL_BLOCKS="$2"; shift 2 ;;
    --pool-caching) POOL_CACHING="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d /tmp/block-pool-parity.XXXXXX)"
fi
mkdir -p "$WORK_DIR"

if ! "$PYTHON" -c 'import vllm.v1.core.block_pool' > /dev/null 2>&1; then
  echo "[pool-parity] $PYTHON cannot import vllm.v1.core.block_pool;" >&2
  echo "[pool-parity] run from the repo venv or set PYTHON=/path/to/python3" >&2
  exit 1
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[pool-parity] building vllm-kv-pool"
  (cd rust && cargo build --release -p vllm-kv-pool > /dev/null)
fi
if [[ ! -x "$RUST_BIN" ]]; then
  echo "rust binary missing: $RUST_BIN (run without --skip-build)" >&2
  exit 1
fi

# The unit tests pin down the behaviour the diff cannot see: that a popped block really has
# no links left, and that the map drops a key only with its last block.
if [[ "$SKIP_TESTS" != "1" ]]; then
  echo "[pool-parity] running vllm-kv-pool unit tests"
  TEST_LOG="$WORK_DIR/cargo-test.log"
  if ! (cd rust && cargo test --release -p vllm-kv-pool) > "$TEST_LOG" 2>&1; then
    echo "[pool-parity] FAIL vllm-kv-pool unit tests" >&2
    grep -aE '^(test |failures:|error)' "$TEST_LOG" | tail -20 >&2
    echo "block_pool_parity_failed traces=0 failures=1" >&2
    exit 1
  fi
  # cargo prints one "test result:" line per target and the empty targets report 0 passed.
  tests_passed="$(grep -a '^test result' "$TEST_LOG" | awk '{p += $4} END {print p + 0}')"
  if [[ "$tests_passed" -eq 0 ]]; then
    echo "[pool-parity] FAIL vllm-kv-pool ran no unit tests (use --skip-tests to skip)" >&2
    echo "block_pool_parity_failed traces=0 failures=1" >&2
    exit 1
  fi
  echo "[pool-parity] PASS vllm-kv-pool unit tests: $tests_passed passed, 0 failed"
fi

traces=0
operations=0
failures=0

for seed in $(seq 0 $((SEEDS - 1))); do
  dir="$WORK_DIR/seed-$seed"
  generated="$("$PYTHON" tools/gen_block_pool_trace.py \
    --out-dir "$dir" \
    --num-blocks "$NUM_BLOCKS" \
    --num-ops "$NUM_OPS" \
    --key-space "$KEY_SPACE" \
    --pool-blocks "$POOL_BLOCKS" \
    --pool-caching "$POOL_CACHING" \
    --seed "$seed")"
  ops="$(printf '%s\n' "$generated" | awk -F': ' '/^operations:/ {print $2}')"
  operations=$((operations + ops))

  if "$RUST_BIN" replay --trace "$dir/trace.txt" > "$dir/rust.txt" \
    && diff -q "$dir/expected.txt" "$dir/rust.txt" > /dev/null; then
    echo "[pool-parity] PASS seed=$seed ops=$ops (python == rust)"
  else
    echo "[pool-parity] FAIL seed=$seed ops=$ops" >&2
    diff -u "$dir/expected.txt" "$dir/rust.txt" | head -30 >&2
    failures=$((failures + 1))
  fi
  traces=$((traces + 1))
done

errors="$(grep -ac ' -> err ' "$WORK_DIR"/seed-*/expected.txt | awk -F: '{s += $2} END {print s}' || true)"
echo "[pool-parity] artifacts: $WORK_DIR"
echo "[pool-parity] operations compared: $operations ($errors rejected by Python)"
if [[ "$failures" != "0" ]]; then
  echo "block_pool_parity_failed traces=$traces failures=$failures" >&2
  exit 1
fi
echo "block_pool_parity_ok traces=$traces operations=$operations"
