#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Measure which Python distributions a real serving run actually loads.
#
# The hook in tools/dep_audit/sitecustomize.py snapshots, for every Python process of
# the run, both the successfully imported modules and the mapped shared objects from
# /proc/self/maps. The second signal is the only way to see the CUDA libraries that
# torch dlopens (libcublas, libcudnn, libnccl, ...), and it also tells apart a module
# that was really loaded from an optional-import probe that failed.
#
# Usage (all extra arguments are forwarded to launcher.sh; this script starts the route,
# waits for it to become healthy, stops it again and then analyses the snapshots, because a
# non-interactive launcher never returns on its own):
#   bash tools/dep_audit.sh --out-dir /tmp/dep-audit -- <launcher args...>
#   AUDIT_DIR=/tmp/dep-audit/snapshots python3 tools/dep_audit/analyze.py
#
# Compare two runs to see the effect of an import change:
#   bash tools/dep_audit.sh --out-dir /tmp/before -- <launcher args...>
#   bash tools/dep_audit.sh --out-dir /tmp/after  -- <launcher args...>
#   AUDIT_DIR=/tmp/after/snapshots python3 tools/dep_audit/analyze.py
#
# `--port` is read from the forwarded arguments to find the health endpoint; the wait for it
# can be bounded with AUDIT_READY_TIMEOUT (seconds, default 420).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d /tmp/dep-audit.XXXXXX)"
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --) shift; LAUNCH_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) LAUNCH_ARGS+=("$1"); shift ;;
  esac
done

SNAP_DIR="$OUT_DIR/snapshots"
mkdir -p "$SNAP_DIR"
echo "[dep-audit] snapshots: $SNAP_DIR"
echo "[dep-audit] report:    $OUT_DIR/report.json"

PORT=8000
for index in "${!LAUNCH_ARGS[@]}"; do
  if [[ "${LAUNCH_ARGS[$index]}" == "--port" && -n "${LAUNCH_ARGS[$index + 1]:-}" ]]; then
    PORT="${LAUNCH_ARGS[$index + 1]}"
  fi
done
READY_TIMEOUT="${AUDIT_READY_TIMEOUT:-420}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
LAUNCH_LOG="$OUT_DIR/launcher.log"
launcher_pid=""

stop_service() {
  if [[ -n "$launcher_pid" ]]; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$launcher_pid" 2>/dev/null || true
  fi
  pkill -f "vllm-rs serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "sleep 7200" 2>/dev/null || true
  sleep 2
  pkill -9 -f "VLLM::" 2>/dev/null || true
}
trap 'stop_service' EXIT INT TERM

export PATH="$REPO_ROOT/.venv/bin:$PATH"
echo "[dep-audit] starting the route (health: $HEALTH_URL)"
# stdin is held open on purpose: the launcher then leaves its status prompt and keeps the
# service alive until this script stops it. The audit hook is exported for the launcher only,
# so the analyser does not snapshot its own imports.
( exec 3< <(sleep 7200)
  exec env IMPORT_AUDIT_DIR="$SNAP_DIR" \
    IMPORT_AUDIT_LABEL="serve" \
    PYTHONPATH="$REPO_ROOT/tools/dep_audit${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPO_ROOT/launcher.sh" "${LAUNCH_ARGS[@]}" <&3
) > "$LAUNCH_LOG" 2>&1 &
launcher_pid=$!

ready=0
started=$SECONDS
for _ in $(seq 1 $((READY_TIMEOUT / 3))); do
  if curl -sf -m 3 "$HEALTH_URL" > /dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "[dep-audit] launcher exited before the service was healthy; log tail:" >&2
    tail -20 "$LAUNCH_LOG" >&2
    exit 1
  fi
  sleep 3
done
if [[ "$ready" != "1" ]]; then
  echo "[dep-audit] the service was not healthy after ${READY_TIMEOUT}s; log tail:" >&2
  tail -20 "$LAUNCH_LOG" >&2
  exit 1
fi
echo "[dep-audit] healthy after $((SECONDS - started))s; letting the hook take one more dump"
# The hook dumps every 3 s, so this guarantees a snapshot taken after startup finished.
sleep 4
stop_service
launcher_pid=""

snapshots=$(find "$SNAP_DIR" -name 'mods-*.txt' | wc -l)
if [[ "$snapshots" == "0" ]]; then
  echo "[dep-audit] no snapshots were written; the import hook did not run" >&2
  exit 1
fi

AUDIT_DIR="$SNAP_DIR" AUDIT_REPORT="$OUT_DIR/report.json" \
  python3 "$REPO_ROOT/tools/dep_audit/analyze.py"
echo "dep_audit_ok snapshots=$snapshots out_dir=$OUT_DIR"
