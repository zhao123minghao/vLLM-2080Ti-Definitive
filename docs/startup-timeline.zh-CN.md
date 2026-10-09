# ~53s 的启动时间花在哪（`tools/startup_timeline.py`）

启动耗时是这个项目被问得最多的数字，而"53s"只是一个测量值、没有解释。launcher 的运行日志里其实
已经同时包含答案的两半，`tools/startup_timeline.py` 把它整理成报告：

```bash
python3 tools/startup_timeline.py                  # 默认取最新的 run-logs/*.log
python3 tools/startup_timeline.py --log run-logs/vllm-<...>.log --top-gaps 12 --min-gap 1.5
```

任何启动过服务的工具都会写出这份日志（`tools/chat_smoke.sh --launch`、
`tools/validate_all.sh --with-chat`、`tools/dep_audit.sh`，或直接 `./launcher.sh`），所以不需要额外
操作：启动一次路由，再回头读日志即可。

## 报告包含什么

* **引擎自己测到的耗时。** 像 `init engine (profile, create kv cache, warmup model) took 8.51 s
  (compilation: 0.95 s)` 这类行是精确值；报告中每个阶段保留**最慢的那次**（数据并行下每个 worker
  都会各记一份），并标出出现了几次。
* **挂钟时间线。** 每行日志都带 `MM-DD HH:MM:SS`，文件头带 `Launch time:`，因此报告能给出每个阶段
  首次出现的时间点以及相对上一阶段的增量。
* **中间的空窗。** 引擎没有打点的阶段会表现为沉默；报告列出最大的若干间隔，便于去查而不是靠猜。

## 参考路由上的实测

`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`、`--mode fast`、2×2080 Ti、AOT 缓存
为热（`~/.cache/vllm/torch_compile_cache` 存在）：

| 里程碑 | 时间点 | 增量 |
| --- | --- | --- |
| 前端启动（Rust 前端起来，加载 tokenizer 与 chat template） | +0s | |
| 第一行 Python 引擎日志（`envs.py` 的未知环境变量告警） | +9s | 9s |
| `Initializing a V1 LLM engine`（`core.py:123`） | +18s | 9s |
| 首次 `Loading weights took`（`default_loader.py:430`） | +40s | 22s |
| `Model loading took`（`model_runner.py:431`） | +42s | 2s |
| 复用 AOT 编译产物（`decorators.py:313`） | +43s | 1s |
| `Initial profiling/warmup run took`（`monitor.py`） | +45s | 2s |
| `Graph capturing finished`（`model_runner.py:1088`） | +47s | 2s |
| `init engine (profile, create kv cache, warmup model) took`（`core.py:372`） | +50s | 3s |

同一次运行中引擎自报的耗时：

| 引擎自报耗时 | 数值 |
| --- | --- |
| `init engine (profile, create kv cache, warmup model)`（其中 compilation: 0.95s） | 8.51s |
| `Model loading`（14.51 GiB） | 7.42s |
| `Loading weights` | 3.44s |
| `Initial profiling/warmup run` | 1.38s |
| `Graph capturing finished`（×4） | 1.00s |
| `torch.compile`（命中 AOT 缓存） | 0.85s |
| `JIT kernel warmup` | 0.00s |

对这张表的诚实解读是：**53s 并不花在引擎打了点的那些工作上。** 模型加载加引擎初始化合计约 16s；
最前面的约 18s 属于前端启动与 Python 导入路径（那时引擎还没开始自报），而 `core.py:123` 到首行权重
加载之间的 22s 窗口主要是分布式初始化、KV cache 分配与 worker 协同——这些引擎都没有计时。这也解释了
为什么之前做的懒加载优化不会体现为启动提速：它影响的是那 9s 的导入段，而不是 22s 的协同段。

要继续推进，下一步应该是**加打点**而不是继续分析日志：在引擎里给 KV cache 分配、分布式初始化和
worker 同步打上时间戳，并给前端自己的子阶段计时。`tools/startup_timeline.py` 无需改动即可识别这些行。

## 报告里会明说的口径限制

* 时间戳是**秒级精度**，因此小增量属于噪声。
* 日志由**两个进程**写入（Rust 前端与 Python 引擎），因此两行之间的间隔可能是"某个进程在等另一个
  进程"，而不是"某个阶段慢"。
* 像 `Loading safetensors checkpoint shards: 9% ...` 这样的进度行**完全没有时间戳**；以这类行为主的
  阶段只能靠相邻行夹出区间。
* launcher 自己的 `Startup complete after Ns.` 只打印到终端、不写日志，所以不在报告里。上面那次运行它是
  54s，而引擎侧是 +50s，差值是健康检查与前端握手。

## 冷启动对比

把 AOT 与 Triton 缓存**一起**挪开（两者耦合：只挪 `triton-cache/` 会失败并报
`Cubin file saved by TritonBundler not found`），启动一次路由，再用同一命令读新日志。差异会体现在
明显变大的 `torch.compile` 行以及更慢的 `init engine` 上——依赖审计文档里那个 106s 的冷启动就是
这样拆开的。
