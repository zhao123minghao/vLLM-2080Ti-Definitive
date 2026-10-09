# Chat regression (`tools/chat_smoke.sh`)

After a kernel, quantization, parser or scheduler change, two questions matter: does chatting
still work, and are the answers still correct? `tools/chat_smoke.sh` answers both with one
command against a live route.

## Running

```bash
bash tools/chat_smoke.sh --launch           # start the reference route, test it, stop it
bash tools/chat_smoke.sh --launch --quick   # only the four fast cases
bash tools/chat_smoke.sh                    # test a server that is already listening
bash tools/chat_smoke.sh --launch -- --json-out /tmp/chat.json
```

`--launch` uses the reference text route (`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only`)
and can be pointed elsewhere with `CHAT_SMOKE_MODEL_DIR`, `CHAT_SMOKE_SPEC_MODEL`,
`CHAT_SMOKE_PROFILE`, `CHAT_SMOKE_SERVED_NAME`, `CHAT_SMOKE_PORT`, `CHAT_SMOKE_GPU_DEVICES`,
`CHAT_SMOKE_TP_SIZE`, `CHAT_SMOKE_MAX_MODEL_LEN`, `CHAT_SMOKE_START_TIMEOUT`. The launcher is
started by this script, and stopped on exit by pid plus an argument-scoped sweep, so no engine,
worker or service-manager process is left behind.

A warm start on the reference route takes about 53 s; the full suite adds roughly one minute of
generation and finishes in under three minutes. When the script starts the service itself it
prints `health OK: 就绪，用时 Ns` (its own measurement — the launcher's
`Startup complete after Ns.` line only goes to the terminal, never to a log file).

Exit codes and the last line (grep-friendly, like the other tools here):

| code | last line | meaning |
| --- | --- | --- |
| 0 | `chat_smoke_ok passed=N failed=0` | everything passed |
| 1 | `chat_smoke_failed passed=N failed=M` | at least one check failed |
| 2 | — | no server on the health endpoint (use `--launch`) |

## What is checked

Content cases run at `temperature=0`. Every prompt asks for a final labelled line, so the
expectation does not depend on prose style (`last_line` is matched against the last non-empty
line after stripping LaTeX/markdown noise):

| case | expected |
| --- | --- |
| 短事实（法国首都） | `巴黎` |
| 中文多步算术（12 苹果 −1/3 +5 −2） | `11` |
| 英文陷阱（bat and ball $1.10） | `0.05`，且末行不是 `0.10` |
| 多步推理（狗在两人间往返跑） | `100` 公里 |
| 数字比较陷阱（9.11 vs 9.8） | `9.8`，且末行不是 `9.11`（流式） |
| 鸡与蛋（三只鸡三天三蛋 → 九只鸡九天） | `27` |
| 幂运算（2^10） | `1024` |
| 多步方程（涨价 20% 再八折 = 96） | `100` |
| 真假话逻辑（只有一人说真话） | `乙` |
| 常识（光年） | `长度` |
| 代码指令跟随（`is_prime`） | 含代码块与 `def is_prime`，且无解释性文字 |
| 多轮上下文（记住 7391 再回忆） | `7391` |

The two traps are deliberate: a model that pattern-matches instead of reasoning answers
`$0.10` and `9.11`. They are the cheapest available signal that a change did not degrade
reasoning quality.

API contract cases pin down this deployment's Rust frontend behaviour:

* the reasoning trace arrives in `message.reasoning` — this frontend does **not** use the
  `reasoning_content` field name, so scripts written against other servers silently see an
  empty trace;
* thinking is **on by default**, and `chat_template_kwargs={"enable_thinking": false}` empties
  the trace (the model may still write a long explanation in `content`, so thinking-off is
  not automatically faster);
* streaming with `stream_options={"include_usage": true}` reports usage, including
  `completion_tokens_details.reasoning_tokens`;
* `max_tokens` counts reasoning tokens, so a tiny budget ends with `finish_reason == "length"`
  and possibly empty `content`;
* multi-turn context is carried across turns.

## Extending

Add a `Case(...)` entry to `CASES` in `tools/chat_smoke.py`: end the prompt with a labelled
line and check it with `last_line`, use `contains`/`forbidden` for free-form answers, and set
`quick=True` for the fast subset. Keep `max_tokens` generous (512-1536 here): with thinking on,
a small budget can be spent entirely on the trace.

Expectations were captured from a verified run on the reference route and should be reviewed —
not silently relaxed — when the served model or its quantization changes.

## Troubleshooting

* **The script prints progress but no launcher output.** The launcher draws its progress on
  the controlling terminal, so it appears even when the script's own stdout is redirected. To
  capture this script's output in a file, redirect the script itself — the launcher lines stay
  on the terminal by design.
* **A launcher survives the run.** `launcher.sh` can leave a service-manager process with the
  same arguments behind, reparented to init. The stop path sweeps
  `launcher.sh ... --served-name <name> ... --port <port>` for that reason and warns if
  anything is still there; check with `pgrep -af launcher.sh` and
  `nvidia-smi --query-gpu=memory.used`.
