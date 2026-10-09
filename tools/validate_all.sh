#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-shot validation for local changes, so the checklist in `AGENTS.md` is one command
# instead of five.
#
#   bash tools/validate_all.sh                 # static checks only, a few seconds
#   bash tools/validate_all.sh --with-parity   # + the cross-language parity checks (need cargo)
#   bash tools/validate_all.sh --with-chat     # + chat regression (starts and stops the route)
#   bash tools/validate_all.sh --full          # everything above
#
# Extra arguments that are not options are compiled as Python along with `tools/**/*.py`,
# which turns the "py_compile <changed-python-files>" step of AGENTS.md into this:
#
#   bash tools/validate_all.sh --with-chat vllm/entrypoints/chat_utils.py
#
# Exit codes: 0 when every selected step passed, 1 otherwise, 2 on a usage error.
# The last line is grep-friendly, like the other tools here:
#
#   validate_all_ok steps=N failed=0 skipped=M
#   validate_all_failed steps=N failed=M

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# Keep the tooling on the same interpreter/venv the launcher uses; an older interpreter from
# the base conda env breaks the launcher's inline Python and the profile validator.
export PATH="$REPO_ROOT/.venv/bin:$PATH"

WITH_PARITY=0
WITH_CHAT=0
EXTRA_PY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-parity) WITH_PARITY=1; shift ;;
    --with-chat) WITH_CHAT=1; shift ;;
    --full) WITH_PARITY=1; WITH_CHAT=1; shift ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown argument: $1" >&2; exit 2 ;;
    *) EXTRA_PY+=("$1"); shift ;;
  esac
done

steps=0
failures=0
skipped=0

run_step() {
  local name="$1"
  shift
  steps=$((steps + 1))
  echo "[validate] >>> $name"
  if "$@"; then
    echo "[validate] PASS $name"
  else
    echo "[validate] FAIL $name" >&2
    failures=$((failures + 1))
  fi
}

skip_step() {
  skipped=$((skipped + 1))
  echo "[validate] SKIP $1 ($2)"
}

# ---------------------------------------------------------------- shell syntax (bash -n)
shell_files=()
for candidate in build.sh launcher.sh update.sh build_rust.sh tools/*.sh tools/dep_audit/*.sh; do
  [[ -f "$candidate" ]] && shell_files+=("$candidate")
done
run_step "bash -n (${#shell_files[@]} scripts)" bash -n "${shell_files[@]}"

# ------------------------------------------------------------------------- python compile
py_files=()
while IFS= read -r file; do
  py_files+=("$file")
done < <(find tools -name '*.py' -not -path '*/__pycache__/*' | sort)
for file in "${EXTRA_PY[@]:-}"; do
  [[ -n "$file" ]] && py_files+=("$file")
done
run_step "py_compile (${#py_files[@]} files)" python3 -m py_compile "${py_files[@]}"

# ---------------------------------------------------------------------- profile matrix
run_step "validate_profiles.sh" bash tools/validate_profiles.sh

# ------------------------------------------------------------------------------ whitespace
run_step "git diff --check" git diff --check

# ---------------------------------------------------- optional: cross-language parity
if [[ "$WITH_PARITY" == "1" ]]; then
  if command -v cargo > /dev/null 2>&1; then
    run_step "check_prefix_hash_parity.sh" bash tools/check_prefix_hash_parity.sh
    run_step "check_block_pool_parity.sh" bash tools/check_block_pool_parity.sh
  else
    skip_step "check_prefix_hash_parity.sh" "cargo not on PATH"
    skip_step "check_block_pool_parity.sh" "cargo not on PATH"
  fi
fi

# ------------------------------------------------------- optional: chat regression (live)
if [[ "$WITH_CHAT" == "1" ]]; then
  run_step "chat_smoke.sh --launch --quick" bash tools/chat_smoke.sh --launch --quick
fi

echo "[validate] steps=${steps} failed=${failures} skipped=${skipped}"
if [[ "$failures" != "0" ]]; then
  echo "validate_all_failed steps=$steps failed=$failures"
  exit 1
fi
echo "validate_all_ok steps=$steps failed=0 skipped=$skipped"
