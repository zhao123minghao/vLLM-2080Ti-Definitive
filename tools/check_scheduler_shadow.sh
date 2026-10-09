#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Scheduler shadow: drives the real `vllm/v1/core/sched/scheduler.py` over a seeded synthetic
# workload and checks the decisions it makes. This is the skeleton of step 3 of the `v1/core`
# migration spike (see `docs/scheduler-shadow.md`): it records the decisions in the format a port
# has to reproduce and enforces the invariants they have to satisfy, but it does not yet compare
# them against a port, because the decision logic is not ported.
#
# The workload runs against a real model *config* directory (no weights are loaded, but the
# architecture and max length come from it), defaulting to the reference route's model.
#
# Usage:
#   bash tools/check_scheduler_shadow.sh [--seeds N] [--num-requests N] [--max-steps N]
#                                        [--model PATH] [--work-dir DIR]
#
# Environment:
#   SCHED_SHADOW_MODEL   model config directory (default: the reference route's model)
#
# Exits non-zero when the scheduler breaks an invariant, and prints it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The recorder imports vLLM and the repository's test fixtures, so pin the venv the launcher uses
# instead of whatever `python3` the caller's PATH happens to resolve to.
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

PYTHON="${PYTHON:-python3}"
MODEL="${SCHED_SHADOW_MODEL:-/data/models/qwen3.8-27b/Qwen3.8-27B-FP8}"
WORK_DIR="${WORK_DIR:-}"
SEEDS=3
NUM_REQUESTS=10
MAX_STEPS=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$MODEL" ]]; then
  echo "[scheduler-shadow] no model config directory at $MODEL" >&2
  echo "[scheduler-shadow] pass --model or set SCHED_SHADOW_MODEL" >&2
  exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d /tmp/scheduler-shadow.XXXXXX)"
fi
mkdir -p "$WORK_DIR"

if ! "$PYTHON" -c 'import vllm.v1.core.sched.scheduler' > /dev/null 2>&1; then
  echo "[scheduler-shadow] $PYTHON cannot import vllm.v1.core.sched.scheduler;" >&2
  echo "[scheduler-shadow] run from the repo venv or set PYTHON=/path/to/python3" >&2
  exit 1
fi

# Three configurations per seed, because the interesting behaviour is on the edges: a pool too
# small for the batch (the scheduler has to preempt), a roomy one with the prefix cache on (cache
# hits and common-prefix blocks), and a middle one at a lower concurrency.
configs=(
  "tight 16 4 0"
  "cached 64 4 1"
  "roomy 48 2 0"
)

runs=0
steps=0
requests=0
failures=0

for config in "${configs[@]}"; do
  read -r name num_blocks max_num_seqs prefix_caching <<< "$config"
  for seed in $(seq 0 $((SEEDS - 1))); do
    dir="$WORK_DIR/$name/seed-$seed"
    generated="$("$PYTHON" tools/gen_scheduler_trace.py \
      --out-dir "$dir" \
      --model "$MODEL" \
      --num-requests "$NUM_REQUESTS" \
      --num-blocks "$num_blocks" \
      --max-num-seqs "$max_num_seqs" \
      --prefix-caching "$prefix_caching" \
      --max-steps "$MAX_STEPS" \
      --seed "$seed" 2>&1)" || true

    summary="$(printf '%s\n' "$generated" | tail -1)"
    if [[ "$summary" != scheduler_shadow_ok* ]]; then
      echo "[scheduler-shadow] FAIL config=$name seed=$seed" >&2
      printf '%s\n' "$generated" | tail -6 >&2
      failures=$((failures + 1))
      continue
    fi
    run_steps="$(printf '%s\n' "$summary" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')"
    run_requests="$(printf '%s\n' "$summary" | sed -n 's/.*requests=\([0-9]*\).*/\1/p')"
    steps=$((steps + run_steps))
    requests=$((requests + run_requests))
    runs=$((runs + 1))
    echo "[scheduler-shadow] PASS config=$name seed=$seed steps=$run_steps requests=$run_requests"
  done
done

if [[ "$runs" == "0" ]]; then
  echo "scheduler_shadow_failed runs=0 failures=$failures" >&2
  exit 1
fi

preemptions="$(grep -ahc 'preempted=[^ -]' "$WORK_DIR"/*/seed-*/expected.txt | awk '{s += $1} END {print s + 0}' || true)"
cached_steps="$(grep -ahc 'cached=[^ -]' "$WORK_DIR"/*/seed-*/expected.txt | awk '{s += $1} END {print s + 0}' || true)"
resumed_steps="$(grep -ahc 'resumed=[^ -]' "$WORK_DIR"/*/seed-*/expected.txt | awk '{s += $1} END {print s + 0}' || true)"
prefix_steps="$(grep -ah 'schedule .* prefix_blocks=[^0]' "$WORK_DIR"/*/seed-*/expected.txt | wc -l || true)"
echo "[scheduler-shadow] artifacts: $WORK_DIR"
echo "[scheduler-shadow] decisions recorded: $steps steps, $cached_steps with cached requests"
echo "[scheduler-shadow] edge cases: $preemptions step(s) with preemptions, $resumed_steps with resumed requests, $prefix_steps with common prefix blocks"

if [[ "$failures" != "0" ]]; then
  echo "scheduler_shadow_failed runs=$runs failures=$failures" >&2
  exit 1
fi
echo "scheduler_shadow_ok runs=$runs steps=$steps requests=$requests preemptions=$preemptions"
