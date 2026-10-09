# 非交互启动

先执行 `./build.sh`，再传入扁平化的 profile 路径。Profile 只包含路线参数；检查点、GPU 拓扑、端口和 mode 都由 launcher 管理。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` 可省略，默认是 `fast`。没有 `MODE` 的 profile 会保留 launcher 的选择；profile 中显式的 `MODE=normal` 或 `MODE=fast` 可以覆盖它。目录中不再区分 `fast/` 和 `normal/`。

可以使用 `--model-dir`、`--speculative-model`、`--profile`、`--mode`、`--gpu-devices`、`--tp-size`、`--pp-size`、`--port`、`--start-timeout` 和 `--print-config`。高级 launcher/runtime 参数使用 `--set KEY=VALUE`。不要把 Prefix Cache、Mamba cache、GPU、端口或模型路径写入 profile，验证器会拒绝这些字段。

Profile 库是验证矩阵，不代表每个文件在每台机器上都能运行。只有外部审计完成启动、4K/128、32K/512、并发和图文正确性验证后，路线才会被 promote。

## 重复启动保护

当已有托管服务在运行，或目标端口已被监听时，launcher 会拒绝再次启动。缺少该校验时，
启动阶段的健康检查会打到已有监听者上，从而误报 `START OK`。请先停止运行中的服务、
换端口，或设置 `ALLOW_DUPLICATE_START=1` 强制启动。

## 编译预热

解码图会在某条路线的首次启动时编译并捕获，因此首次启动通常比后续启动多花约 30 s。
编译缓存写入后，相同路线、上下文长度、KV 精度和投机解码设置的后续启动会直接复用。

用 `--prewarm-compile-only` 主动支付这笔开销：launcher 会按解析出的路线启动，等待
`/health` 以确保图捕获与编译产物落盘，然后停止服务，并报告 AOT 缓存是否被复用：

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 \
  --prewarm-compile-only
```

命令以 `PREWARM OK` 结束且不会留下运行中的服务，之后该路线的正式启动即可复用已编译
的图。

## Rust 前端

Rust OpenAI 前端（由 `./build_rust.sh` 构建的 `vllm-rs`）可以代替 Python API
server 服务同样的路线。它监管一个 Python headless 引擎，并把 HTTP、分词器、对
话模板、解析器和 metrics 层留在 Rust 侧，因此启动时不再需要 Python 前端进程。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 --frontend rust
```

服务类参数（`served-model-name`、`reasoning-parser`、`language-model-only`、
`enable-prompt-tokens-details`、`disable-log-stats` 以及 chat template 和
tool parser 选项）交给 Rust 前端，其余引擎参数原样转发给被监管的 Python 引擎。
`FRONTEND` 默认是 `auto`，按路线自动选择并在启动摘要里打印原因：

- 路线满足 `MESSAGE_TYPE=text-only` 或 `text+image`、未使用投机解码或使用 MTP/DFlash、
- 路线满足 `MESSAGE_TYPE=text-only` 或 `text+image`、未使用投机解码或使用 MTP/DFlash
  （包括这些路线默认开启的 per-request spec-decode metrics）、无工具调用、
  未启用 `KV_DISK_CACHE_DIR`、未启用 scale-out 端点，且 `vllm-rs` 二进制存在时，
  选 `rust`；
- 其它情况选 `python`，因为工具调用、SSD 前缀缓存和 scale-out 端点尚未在那里验证。

`--frontend rust` 可对门限拒绝的路线强制启用（专家用法），`--frontend python`
强制回到原行为。用 `DRY_RUN=1` 逐个跑 launcher 统计 27 个自带 profile，现在
**27 个全部满足** `auto` 门限，也就是说 `auto` 与 `python` 的差别只剩工具调用、
SSD 前缀缓存和 scale-out 路线。

不要只看 profile 文件统计 `auto` 覆盖率：门限还会看到 launcher 派生默认值
（例如 DFlash 的 metrics 设置）。

per-request spec-decode metrics：响应 `metrics` 块（含 `metrics.speculative_decoding`）
已在 Rust 前端实现，completions 与 chat-completions、流式与非流式都支持，因此
DFlash 默认开启的 `detailed` 指标不再被丢弃。双 RTX 2080 Ti + Qwen3.8-27B-FP8
实测：

- MTP 路线（`mtp4-fp16kv-1x148K`，`131072` 上下文）：chat、流式 completions、
  流式 chat 三种响应的 `metrics` 块都与 Python 前端逐字段完全一致
  （`mean_acceptance_length=3.4`、`draft_acceptance_rate=0.6`、
  `acceptance_histogram=[0,2,0,2,1]`、`per_step_accepted=[1,3,1,4,3]`）。
- DFlash 路线（`nomtp-fp16kv-1x176K` 加 `SPECULATIVE_METHOD=dflash`、
  `SPECULATIVE_TOKENS=7`、`65536` 上下文）：`FRONTEND=auto` 选中 `rust`，生成文本与
  `metrics` 块均与 Python 前端完全一致（`mean_acceptance_length=2.666666666666667`、
  `draft_acceptance_rate=0.23809523809523808`、`acceptance_histogram=[1,3,1,0,0,1,0,0]`
  即 `num_spec_tokens + 1` 个桶），两侧 `GPU KV cache size` 也都是 67,195 tokens。

Rust 前端尚未计算 per-request 时间字段（`time_to_first_token_ms`、
`generation_time_ms`、`queue_time_ms`、`mean_itl_ms`、`tokens_per_second`）：
非流式响应里序列化为 `null`，流式块里省略，与 Python 前端在这些指标不可用时的
输出完全一致。

多模态验证上下文：双 RTX 2080 Ti + Qwen3.8-27B-FP8，`nomtp-fp16kv-1x121K-text-image`
（121,856 上下文）与 `mtp4-fp8kv-1x186K-text-image`（加 `--set MAX_MODEL_LEN=131072`
与 MTP 草稿）都通过 Rust 前端正确回答了颜色问题：64x64 纯红 PNG 得到 `Red`，
同一提示换成纯蓝 PNG 得到 `Blue`（`finish_reason=stop`，`prompt_tokens=129`，说明
图像已展开为视觉 token）。launcher 会传
`--limit-mm-per-prompt '{"image":64,"video":0,"audio":0}'`，worker 日志显示
profiled 1 image item；audio 与 video 仍未验证。

投机方法验证上下文：双 RTX 2080 Ti + Qwen3.8-27B-FP8，MTP
（`mtp4-fp16kv-1x148K`）与 DFlash2 草稿（`65536` 上下文，67,195 KV tokens）
都已通过 Rust 前端完成真实生成。同一 DFlash 路线在 `131072` 上下文下会被引擎的
KV 容量准入拦下；这是该路线在本机上的容量限制，不是前端差异（Python 前端走同一
校验，launcher 的编译预热兜底与准入诊断对两者一致）。

本仓库实测（双 RTX 2080 Ti、Qwen3.8-27B-FP8、1x147K FP16 KV、TP=2、编译缓存已热）：
`--frontend rust` 在 56 s 后通过 `/health`，Python 前端是 62 s；`GPU KV cache
size`（176,947 tokens）一致，`/v1/completions` 可用，`/metrics` 提供
`vllm:kv_cache_usage_perc`。

## 启动排障

若启动卡住，当日志出现“60 秒内没有可用的共享内存广播块”时 launcher 会打印提示。
该信息表示某个 worker 卡住或极慢（例如首次编译），导致广播队列始终未被消费。

反复强制终止还会残留 `/dev/shm/psm_*`（torch 共享内存段）。launcher 只报告数量、
不会删除它们，因为可能属于其它 torch 程序。请仅在没有任何 vLLM/torch 服务运行时清理。

## 实验性 SSD 前缀缓存

Launcher 可以启用 vLLM 内置的 `OffloadingConnector`，通过 CPU 暂存层和文件系统层将已完成的前缀 KV 块写入磁盘，不需要常驻的独立缓存进程。目录必须位于持久化存储上；重启后保持模型路径、KV 精度、块布局和并行配置一致。CPU 暂存空间占用主机内存，不会增加 GPU KV 容量。每次启动前，launcher 会读取目标模型和本地草稿模型的全部文件，按内容摘要选择独立的磁盘命名空间。因此在原路径替换权重后不会读到旧 KV；代价是启动时额外完整读取一次检查点。旧命名空间需要手动清理以回收空间。切换到不兼容的运行时仍需清理缓存。磁盘缓存要求检查点位于本地目录；launcher 启动和服务运行期间不要修改检查点文件。

文件系统层不会自动淘汰旧文件，也没有磁盘配额；请使用空间充足的专用卷，监控用量，并在服务停止后清理过期缓存。`KV_DISK_CPU_BYTES` 只限制主机内存暂存空间。启用磁盘缓存时 launcher 固定 `PYTHONHASHSEED=0`，并拒绝其他值，以保证重启后的块哈希稳定。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,2,3,4 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/mnt/nvme/vllm-kv-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

去掉 `--print-config` 即可启动。启用后 launcher 会要求打开前缀缓存，并设置稳定的 `PYTHONHASHSEED`。停止服务前应等待 `vllm:kv_offload_tiering_active_cascade_jobs` 回到零。重启后可查看 `vllm:external_prefix_cache_hits_total`，以及文件系统层的 `vllm:kv_offload_tiering_chunk_hits_total`、`vllm:kv_offload_tiering_read_bytes_total`。

在 4xT10、Qwen3.8-27B AWQ-INT4、FP16 KV 上，整机重启后，相同的 1,929-token 提示词从 SSD 恢复了 1,568 tokens，输出与冷启动一致。文件系统层记录了 5 次命中和约 257 MB 读取。此路线仍属实验性，暂无正式吞吐成绩。
在双 RTX 2080 Ti 上，同模型、FP16 KV 的 1,933-token 提示词在 vLLM 服务退出并重启后命中 1,568 tokens，输出保持一致。双 2080 Ti 测试未单独进行整机重启。

还在双 RTX 2080 Ti（GPU 1、5）上验证了仓库正式 `fast` profile。Qwen3.8-27B
W8A16 FP8 权重 / FP8 KV、MTP4、262K 纯文本路线使用 4 GiB CPU 暂存空间，服务重启后
在 5,356 个提示 token 中命中 3,232 个。W4A16 NVFP4 权重 / FP8 KV、DFlash2、262K
图文路线使用 8 GiB CPU 暂存空间，以纯文本请求测试，服务重启后在 5,355 个提示 token
中命中 3,296 个；文件系统层记录 21 个 chunk 命中、读取 5,244,911,616 字节。两条路线的
输出均与首次请求一致，都是 `READY`。这些测试未包含整机重启或吞吐基准。在这台主机上，
MTP4 FP16 KV 148K 路线启用 SSD offload 时，需要通过 launcher 设置 `GPU_UTIL=0.965`
才能保留其 148,480-token 容量；这是该主机的启动覆盖值，正式 profile 未修改。
