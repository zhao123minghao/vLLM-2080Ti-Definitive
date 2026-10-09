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
# Usage (all extra arguments are forwarded to launcher.sh):
#   bash tools/dep_audit.sh --out-dir /tmp/dep-audit -- <launcher args...>
#   AUDIT_DIR=/tmp/dep-audit/snapshots python3 tools/dep_audit/analyze.py
#
# Compare two runs to see the effect of an import change:
#   bash tools/dep_audit.sh --out-dir /tmp/before -- <launcher args...>
#   bash tools/dep_audit.sh --out-dir /tmp/after  -- <launcher args...>
#   AUDIT_DIR=/tmp/after/snapshots python3 tools/dep_audit/analyze.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d /tmp/dep-audit.XXXXXX)"
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --) shift; LAUNCH_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) LAUNCH_ARGS+=("$1"); shift ;;
  esac
done

SNAP_DIR="$OUT_DIR/snapshots"
mkdir -p "$SNAP_DIR"
echo "[dep-audit] snapshots: $SNAP_DIR"
echo "[dep-audit] report:    $OUT_DIR/report.json"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
IMPORT_AUDIT_DIR="$SNAP_DIR" \
IMPORT_AUDIT_LABEL="serve" \
PYTHONPATH="$REPO_ROOT/tools/dep_audit${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO_ROOT/launcher.sh" "${LAUNCH_ARGS[@]}" || true

AUDIT_DIR="$SNAP_DIR" AUDIT_REPORT="$OUT_DIR/report.json" \
  python3 "$REPO_ROOT/tools/dep_audit/analyze.py"
