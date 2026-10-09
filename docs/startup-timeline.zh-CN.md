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
* **本 fork 新增的 `startup:` 标记。** worker 拉起原本就是这样一个空窗，所以 executor 现在为每个 worker
  记一行（spawn、entry、`init_worker`、`init_device`、`load_model`、ready），外加 KV cache 阶段耗时，
  都相对于父进程 fork 它的那一刻。Linux 下 `CLOCK_MONOTONIC` 是系统级的，因此 fork 前取的时间戳在子进程
  里可比。

## 参考路由上的实测

`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`、`--mode fast`、2×2080 Ti、AOT 缓存
为热（`~/.cache/vllm/torch_compile_cache` 存在）。launcher 总计：**51s**。

| 阶段 | 区间 | 说明 |
| --- | --- | --- |
| Rust 前端启动、拉起 Python 引擎、第一行 Python 日志 | 0 – 9s | 只有前端；引擎尚未自报 |
| Python 导入与引擎构造 | 9 – 17s | 结束于 `Initializing a V1 LLM engine`（`core.py:123`） |
| 引擎核心初始化：worker 拉起、模型加载 | 17 – 41s | 见下表标记 |
| 内存画像、KV cache 配置、warmup、图捕获 | 41 – 49s | 画像 5.75s，分配 0.03s |
| 引擎就绪 → launcher 就绪（健康检查、握手） | 49 – 51s | launcher 的数字只写终端 |

第三个区间内，本 fork 新增的 `startup:` 标记把日志原先掩盖的部分展开了：

| 标记 | 时间点 | 增量 |
| --- | --- | --- |
| worker rank=0 spawned | +24.0s | |
| rank=0 entered `worker_main` | +24.0s | 距其 spawn 0.3s |
| rank=0 `init_worker`（模型模块导入）完成 | +25.0s | 1.2s |
| worker rank=1 spawned | +31.0s | **距 rank 0 的 spawn 6.7s** |
| rank=1 entered `worker_main` | +31.0s | **距其 spawn 7.0s** |
| rank=0 / rank=1 `init_device` 完成 | +34.0s | 8.4s / 1.7s |
| rank=0 / rank=1 `load_model` 完成 | +41.0s | 6.9s / 6.9s |
| 两个 worker 就绪 | +41.0s | 距首次 spawn 9.98s |
| available-memory 画像运行 | +46.0s | 5.75s |
| KV cache 配置 + 分配 | +46.0s | 0.03s |

同一次运行中引擎自报的耗时：

| 引擎自报耗时 | 数值 |
| --- | --- |
| `init engine (profile, create kv cache, warmup model)` | 8.52s |
| `Model loading`（14.51 GiB） | 6.84s |
| `Loading weights` | 3.32s |
| `Initial profiling/warmup run` | 1.38s |
| `Graph capturing finished`（×4） | 1.00s |
| `torch.compile`（命中 AOT 缓存） | 0.87s |
| `JIT kernel warmup` | 0.00s |

## 这些数字说明了什么

* **引擎自己的初始化不是大头。** 模型加载 6.8s + `init engine` 8.5s + 内存画像 5.8s 合计约 21s／51s，
  其余是进程拉起。
* **两个 worker 是先后起来的，第二个要为第一个的工作买单。** rank 0 在 spawn 后 0.3s 就进入
  `worker_main`，rank 1 花了 7.0s；而 rank 1 是在 rank 0 spawn 后 6.7s 才被 fork 的——那时 rank 0 已经在跑
  CUDA／分布式初始化并加载权重。这是**争用**而非固定的 fork 开销；可用的杠杆是要么让 worker 在各自的重
  初始化开始前都先起来，要么换一个不复制父进程页表的启动方式（`VLLM_WORKER_MULTIPROC_METHOD`）。
* **KV cache 分配基本免费。** 配置加分配是 0.03s；旁边的 5.75s 是**内存画像运行**（用前向算峰值），
  两件事不一样。
* **引擎核心存在之前就有约 17s**：前端加解释器启动 9s，Python 导入与引擎构造 8s。这是懒加载优化的目标
  区间，占总量的五分之一——而不是盯着 TUI 时感觉的一半。

这些在加打点之前都看不到，所以标记已经进树：上面这些阶段正是日志过去会从
`Initializing a V1 LLM engine` 直接跳到首行权重加载的那一段。

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
