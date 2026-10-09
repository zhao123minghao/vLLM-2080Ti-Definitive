# 对话回归（`tools/chat_smoke.sh`）

改完内核、量化、parser 或调度之后，有两件事必须确认：对话还能不能用、答案还对不对。
`tools/chat_smoke.sh` 用一条命令对运行中的路由回答这两个问题。

## 运行方式

```bash
bash tools/chat_smoke.sh --launch           # 启动参考路由 → 做检查 → 停止服务
bash tools/chat_smoke.sh --launch --quick   # 只跑四个快速用例
bash tools/chat_smoke.sh                    # 对已在运行的服务做检查
bash tools/chat_smoke.sh --launch -- --json-out /tmp/chat.json
```

`--launch` 默认使用参考文本路由（`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only`），
可用环境变量改写：`CHAT_SMOKE_MODEL_DIR`、`CHAT_SMOKE_SPEC_MODEL`、`CHAT_SMOKE_PROFILE`、
`CHAT_SMOKE_SERVED_NAME`、`CHAT_SMOKE_PORT`、`CHAT_SMOKE_GPU_DEVICES`、`CHAT_SMOKE_TP_SIZE`、
`CHAT_SMOKE_MAX_MODEL_LEN`、`CHAT_SMOKE_START_TIMEOUT`。launcher 由本脚本启动，退出时先按 pid
停止、再按启动参数兜底清理，因此不会残留 engine、worker 或管理进程。

参考路由热启动约 53s，全量用例额外约一分钟生成时间，整体三分钟内完成。由本脚本启动服务时，
它会打印 `health OK: 就绪，用时 Ns`（这是脚本自己的计时——launcher 的
`Startup complete after Ns.` 只输出到终端，从不写进任何日志文件）。

退出码与最后一行（与本仓库其他工具一样便于 grep）：

| 退出码 | 最后一行 | 含义 |
| --- | --- | --- |
| 0 | `chat_smoke_ok passed=N failed=0` | 全部通过 |
| 1 | `chat_smoke_failed passed=N failed=M` | 至少一项失败 |
| 2 | — | health 端点没有服务（用 `--launch`） |

## 检查内容

内容类用例使用 `temperature=0`。每个 prompt 都要求以带标签的末行收尾，因此期望值与文风无关
（`last_line` 会在去掉 LaTeX/markdown 噪声后匹配最后一个非空行）：

| 用例 | 期望 |
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

其中两个是刻意设置的陷阱：靠模式匹配而非推理的模型会答成 `$0.10` 和 `9.11`。它们是"本次改动
是否损伤推理质量"最廉价的信号。

接口契约用例固定了本部署 Rust 前端的这些行为：

* 思考轨迹在 `message.reasoning` 字段 —— 本前端**不使用** `reasoning_content`，照抄其他服务的
  脚本会静默读到空的思考；
* 思考**默认开启**，`chat_template_kwargs={"enable_thinking": false}` 会清空思考字段
  （但模型仍可能在 `content` 里写长篇解释，所以关思考不一定更快）；
* 流式加 `stream_options={"include_usage": true}` 会返回 usage，其中包含
  `completion_tokens_details.reasoning_tokens`；
* `max_tokens` **计入思考 token**，上限很小时会以 `finish_reason == "length"` 结束，
  `content` 可能为空；
* 多轮上下文正确携带。

## 如何扩展

在 `tools/chat_smoke.py` 的 `CASES` 里加一条 `Case(...)`：prompt 以带标签的末行收尾并用
`last_line` 校验，自由作答的用 `contains`/`forbidden`，需要进快速子集的加 `quick=True`。
`max_tokens` 要给足（这里用 512–1536）：思考开启时，小上限可能被思考全部吃掉。

期望值取自参考路由上一次通过验证的运行结果；当被服务的模型或其量化发生变化时，应当**复核并
更新**这些期望值，而不是悄悄放宽。

## 排障

* **脚本有输出但看不到 launcher 的日志。** launcher 把进度画在控制终端上，即使脚本自身
  stdout 被重定向也会显示在终端。想把脚本输出存文件，重定向脚本本身即可——launcher 的
  那些行按设计只留在终端。
* **运行结束后还有 launcher 进程存活。** `launcher.sh` 可能留下一个参数完全相同的管理进程
  并被 init 收养。停止流程因此会额外清理
  `launcher.sh ... --served-name <name> ... --port <port>`，若仍有残留会给出警告；可用
  `pgrep -af launcher.sh` 与 `nvidia-smi --query-gpu=memory.used` 自查。
