#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-command chat regression for this deployment.
#
#   bash tools/chat_smoke.sh                  # use a server that is already listening
#   bash tools/chat_smoke.sh --launch         # start the reference route, test, stop it
#   bash tools/chat_smoke.sh --launch --quick # four fast cases only
#
# Everything after `--` is forwarded to tools/chat_smoke.py.
#
# `--launch` uses the reference text route by default; override with environment variables:
#   CHAT_SMOKE_MODEL_DIR, CHAT_SMOKE_PROFILE, CHAT_SMOKE_SERVED_NAME, CHAT_SMOKE_PORT,
#   CHAT_SMOKE_GPU_DEVICES, CHAT_SMOKE_TP_SIZE, CHAT_SMOKE_MAX_MODEL_LEN.
#
# The launcher runs in a subshell whose pid this script records; on exit it is killed and a
# few `pkill` fallbacks sweep up any engine or worker process that outlives it. The launcher's
# stdin is held open on purpose: answered or closed stdin makes it leave its status prompt,
# while an open one keeps the server running until this script stops it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 9

# The venv must be first on PATH *before* the launcher runs: the launcher executes inline
# python snippets, and an older interpreter from the base conda env fails them with
# "TypeError: 'type' object is not subscriptable", which surfaces as an invalid config.
export PATH="$REPO_ROOT/.venv/bin:$PATH"

MODEL_DIR="${CHAT_SMOKE_MODEL_DIR:-/data/models/qwen3.8-27b/Qwen3.8-27B-FP8}"
SPEC_MODEL="${CHAT_SMOKE_SPEC_MODEL:-/data/models/qwen3.8-27b/MTP}"
PROFILE="${CHAT_SMOKE_PROFILE:-2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env}"
SERVED_NAME="${CHAT_SMOKE_SERVED_NAME:-Qwen3.8-27B-FP8}"
PORT="${CHAT_SMOKE_PORT:-8000}"
GPU_DEVICES="${CHAT_SMOKE_GPU_DEVICES:-0,1}"
TP_SIZE="${CHAT_SMOKE_TP_SIZE:-2}"
MAX_MODEL_LEN="${CHAT_SMOKE_MAX_MODEL_LEN:-131072}"
START_TIMEOUT="${CHAT_SMOKE_START_TIMEOUT:-420}"

LAUNCH=0
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch) LAUNCH=1; shift ;;
    --) shift; PY_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PY_ARGS+=("$1"); shift ;;
  esac
done

health_url="http://127.0.0.1:${PORT}/health"
launcher_pid=""

stop_service() {
  if [[ -n "$launcher_pid" ]]; then
    kill -TERM "$launcher_pid" 2>/dev/null || true
    sleep 3
    kill -KILL "$launcher_pid" 2>/dev/null || true
    # The pid captured from `$!` is not always the process that survives: launcher.sh can
    # leave a service-manager process behind with the same arguments, reparented to init and
    # immune to the kill above. Sweep that by the exact arguments this run passed, which is
    # only ever done when this script started the service itself.
    pkill -f "launcher[.]sh .*--served-name ${SERVED_NAME} .*--port ${PORT}" 2>/dev/null || true
    sleep 1
    if pgrep -f "launcher[.]sh .*--served-name ${SERVED_NAME} .*--port ${PORT}" > /dev/null 2>&1; then
      pkill -9 -f "launcher[.]sh .*--served-name ${SERVED_NAME} .*--port ${PORT}" 2>/dev/null || true
    fi
  fi
  # Belt and braces for engines that outlive the launcher script itself.
  pkill -f "vllm-rs serve" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "sleep 7200" 2>/dev/null || true
  sleep 2
  pkill -9 -f "VLLM::" 2>/dev/null || true
  if [[ -n "$launcher_pid" ]] && pgrep -f "launcher[.]sh .*--served-name ${SERVED_NAME} .*--port ${PORT}" > /dev/null 2>&1; then
    echo "[chat-smoke] 警告：仍有 launcher 进程残留，请手动用 pgrep -af launcher.sh 检查" >&2
  fi
}

if [[ "$LAUNCH" == "1" ]]; then
  if curl -sf -m 3 "$health_url" > /dev/null 2>&1; then
    echo "[chat-smoke] 已有服务在 ${health_url} 上运行，直接复用它（不再启动新实例）"
  else
    echo "[chat-smoke] 启动参考路由: profile=${PROFILE} model=${MODEL_DIR}"
    log=/tmp/chat_smoke_launch.log
    # stdin is held open on purpose: with answered or closed stdin the launcher leaves its
    # status prompt, while an open one keeps the server running until we stop it.
    ( exec 3< <(sleep 7200)
      exec ./launcher.sh --non-interactive \
        --model-dir "$MODEL_DIR" --speculative-model "$SPEC_MODEL" \
        --served-name "$SERVED_NAME" --profile "$PROFILE" \
        --mode fast --gpu-devices "$GPU_DEVICES" --tp-size "$TP_SIZE" --port "$PORT" \
        --set START_TIMEOUT="$START_TIMEOUT" --set MAX_MODEL_LEN="$MAX_MODEL_LEN" <&3
    ) > "$log" 2>&1 &
    launcher_pid=$!
    trap 'stop_service' EXIT INT TERM

    ready=0
    ready_secs=0
    launch_start=$SECONDS
    for _ in $(seq 1 $((START_TIMEOUT / 3))); do
      if curl -sf -m 3 "$health_url" > /dev/null 2>&1; then
        ready=1
        break
      fi
      # Fail fast instead of waiting out the whole budget on a launcher that already died.
      if ! kill -0 "$launcher_pid" 2>/dev/null; then
        echo "[chat-smoke] launcher 启动失败，日志尾部：" >&2
        tail -20 "$log" >&2
        exit 1
      fi
      sleep 3
    done
    if [[ "$ready" != "1" ]]; then
      echo "[chat-smoke] 等待 health 超时（${START_TIMEOUT}s），日志尾部：" >&2
      tail -20 "$log" >&2
      exit 1
    fi
    # The launcher prints its progress, including "Startup complete after Ns.", to the
    # controlling terminal only; neither $log nor the engine log under run-logs/ ever sees
    # that line. Report this script's own measurement instead of scraping a line that is
    # not in the file.
    ready_secs=$((SECONDS - launch_start))
    echo "[chat-smoke] health OK: 就绪，用时 ${ready_secs}s（引擎日志：run-logs/）"
  fi
fi

if ! curl -sf -m 5 "$health_url" > /dev/null 2>&1; then
  echo "[chat-smoke] 没有服务在 ${health_url}（用 --launch 启动，或先手动启动）" >&2
  exit 2
fi

export PATH="$REPO_ROOT/.venv/bin:$PATH"
python3 "$REPO_ROOT/tools/chat_smoke.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_NAME" "${PY_ARGS[@]}"
rc=$?

if [[ "$LAUNCH" == "1" ]]; then
  echo "[chat-smoke] 停止服务"
  stop_service
  launcher_pid=""
fi

exit $rc
