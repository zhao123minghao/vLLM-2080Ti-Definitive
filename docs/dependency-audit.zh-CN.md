# 依赖审计：一次服务运行到底加载了什么

本文解决的是部署痛点：运行环境有 **205 个发行包 / 7.1 GiB** 的 `site-packages`，
真正麻烦的不是体积本身，而是每次新增组件都要重新解析并保持一致的那一大片版本面。
本文记录我们的路由在真实运行时**实际加载**了哪些包、哪些完全没加载、以及哪些导入链
仍然会拽进沉重的可选依赖栈。

以下全部用 `tools/dep_audit.sh` 实测得出。该脚本包裹一次 launcher 启动，对每个 Python
进程记录两项信号：

- `sys.modules` 快照 —— **只统计成功导入的模块**，因此失败的可选依赖探测
  （`try: import jax` 之类）不会被当成"使用了"；
- `/proc/self/maps` 中已映射的动态库 —— 这是唯一能看到 torch `dlopen` 的 CUDA 库
  （它们从不作为 Python 模块出现，却占据大部分体积）的方法。

被审计的路由：`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`
（Qwen3.8-27B-FP8 + MTP，TP=2，纯文本）。

## 实测结果

| | 发行包数 | 体积 |
| --- | --- | --- |
| 已安装 | 205 | 7.1 GiB |
| 本次路由真实加载 | **123** | 5.5 GiB |
| 完全未加载 | **82** | 1.2 GiB |

在两批惰性化改造（见下文）之前，本路由加载 130 个发行包 / 6.0 GiB，因此两批改造共
减少了 **385 MiB** 已加载模块（`opencv-python-headless` 152、`llvmlite` 160、
`mistral_common` 与 `numba`，以及 `humming`），导入的顶层模块数从 334 降至 327。

加载体积最大的项：`torch` 1044 MiB、`triton` 687 MiB、`nvidia-cublas` 570、
`nvidia-cudnn-cu13` 471、`nvidia-cutlass-dsl-libs-cu12` + `cu13` 428、`nvidia-cufft` 274、
`nvidia-cusolver` 234、`nvidia-nccl-cu13` 226、`nvidia-cusparselt-cu13` 223、
`nvidia-cuda-nvrtc` 218、`tilelang` 179、`flashinfer-python` 90。

已安装但完全未加载的最大项：`tokenspeed-triton` 293 MiB、`nvidia-nvvm` 163、
`nvidia-cuda-nvcc` 147、`pynvvideocodec` 84、`cmake` 73、`instanttensor` 20、
`nvidia-cuda-cccl` 18、`cryptography` 14、`nccl4py` 14、`hf-xet` 12。

`nvidia_*` 系列共 3.0 GiB，与系统 CUDA 13.2（7.2 GiB）重复：本次审计中
**`/usr/local/cuda` 下的库一个都没被映射**，说明真正在用的是 pip wheel 里的 CUDA 库。
改用系统 CUDA 需要针对它重建 torch —— 本文暂不涉及，但它是目前最大的一块体积。

关于"未加载"清单的注意事项：`/proc/self/maps` 看不到以**子进程**方式执行的工具。
`nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-cccl` 提供 Triton 与
`torch.utils.cpp_extension` JIT 编译所需的 `nvcc`、`ptxas` 与头文件，即使没有库常驻内存
也可能仍是必需的。只有在完成一次完整的冷编译（含清空 `triton-cache`）验证后才能删。

## 导入归因

由审计 hook 记录每个重量级可选顶层模块的"导入者"（修复前，第 1 次审计）：

| 模块 | 导入者 |
| --- | --- |
| `cv2` | `vllm/multimodal/media/connector.py:32`（模块级） |
| `av` | `vllm/multimodal/media/audio.py:16` |
| `torchcodec` | `vllm/multimodal/media/__init__.py:3` |
| `torchvision` | `transformers.image_processing_base:25` |
| `torchaudio` | `transformers.processing_utils:38` |
| `flashinfer` | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `cutlass`、`cudnn` | `flashinfer.cute_dsl.availability`、`flashinfer.gemm` |
| `tilelang`、`tvm` | 无 Python 栈帧 —— 由原生扩展导入 |
| `numba`、`llvmlite` | `_has_module("numba")` 试探导入，随后 `vllm.model_executor.models.kimi_k25` |
| `humming` | `_has_module("humming")` 试探导入，随后 `vllm.utils.humming` |
| `llguidance` | `_has_module("llguidance")` 试探导入 |
| `pandas` | `vllm/model_executor/layers/utils.py`、`layers/quantization/input_quant_fp8.py` |
| `boto3` | `vllm/model_executor/model_loader/sharded_state_loader.py:23` |
| `ray` | `vllm/v1/engine/utils.py:34` |
| `redis` | `torch._inductor.runtime.autotune_cache:48` |
| `z3` | `torch.fx.experimental.symbolic_shapes` |

## 已做的修改

在 `vllm/utils/import_utils.py` 新增 `LazyModule`：首次访问属性时才导入模块，并保留
`PlaceholderModule` 的"该装哪个 extra"错误提示。

| 文件 | 修改 |
| --- | --- |
| `vllm/tilelang_utils/__init__.py` | CUDA 上不再提前 `import tilelang`；`tilelang` / `T` 通过模块 `__getattr__` 解析，`tilelang_jit` 把装饰延迟到首次调用（ROCm 路径本来就是这个模式） |
| `vllm/model_executor/kernels/mhc/tilelang_kernels.py` | 只导入 `tilelang_jit`；`T` / `tilelang` 仅在 `TYPE_CHECKING` 下导入 |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 两个 FlashInfer 通信器的导入改为按对应开关条件导入 |
| `vllm/multimodal/video.py` | `cv2` 改为 `LazyModule` |
| `vllm/multimodal/video_decoders/opencv.py` | `cv2` / `cv2.videoio_registry` 改为 `LazyModule` |
| `vllm/multimodal/audio.py`、`vllm/multimodal/media/audio.py` | `av`、`soundfile` 改为 `LazyModule` |
| `vllm/utils/import_utils.py` | 新增 `_SPEC_ONLY_PROBE` = {`numba`, `humming`, `llguidance`, `tilelang`}：这些可选依赖的存在性检查改用 `find_spec`，不再用会真正加载它们的试探导入 |
| `vllm/parser/engine/registered_adapters.py` | 适配器改为按需构建：模块 `__getattr__` 把 `<Engine>ReasoningAdapter` / `<Engine>ToolAdapter` 映到所属引擎模块并缓存，导入注册表不再导入十二个 parser 引擎 |
| `vllm/model_executor/layers/quantization/utils/humming_utils.py` | humming dtype 表改为首次查表时构建（`_LazyDtypeMap`） |
| `vllm/model_executor/models/kimi_k25.py` | 基于 numba 的 fused vision processor 改为在使用它的方法内导入 |

已验证的效果：`cv2` 的加载进程数从 4 降到 2，第二批后降到 **0**；
`_has_module("numba")` 不再导入 numba；本路由上 `mistral_common`、`cv2`、`numba`、
`llvmlite`、`humming` 均已为 **0** 个进程。总量没继续下降的原因见下面各条路径。

### 吞吐校验（受控 A/B）

导入类改动很容易被归咎为吞吐下降，因此把本批改动 stash 掉、在同一路由上背靠背各测一次：

| 条件 | 4K/128 prefill | 4K/128 decode | 32K/512 prefill | 32K/512 decode | 启动 |
| --- | --- | --- | --- | --- | --- |
| 移除本批改动 | 1554.03, 1543.24 | 107.39, 107.24 | 1334.93 | 100.52 | 53 s |
| 含本批改动（run 1） | 1573.19, 1563.83 | 107.86, 107.41 | 1361.20 | 101.79 | 53 s |
| 含本批改动（run 2） | 1566.48, 1558.42 | 107.62, 107.54 | 1344.06 | 100.97 | 53 s |

即惰性导入没有可测量的代价；含改动的那两次反而快约 1%，仍在运行间波动范围内。方法上值得
注意：触发这次 A/B 的"基线"是同路由但在数小时之前测到的 1583–1599 tok/s，那 1.6% 的差距最终
被证明是环境漂移而不是代码。这条泳道的数字必须在同一时段内比较。

测量说明：launcher 的交互式性能通道在非交互运行下拿不到样本（它会阻塞在之后的提示上，样本
留在 stdio 缓冲里），因此改为保持 stdin 打开启动服务，并用 launcher 完全相同的参数直接调用
`tools/profile_request.py`（completions 端点、纯填充 prompt、固定 `" the"` 补全 token、每个样本
使用唯一 prompt variant）。

## 剩余工作（按收益排序）

1. **FlashInfer 注意力/采样集群 —— 已实测，开关不是杠杆。**
   `VLLM_ALLREDUCE_USE_FLASHINFER` 默认为 1（`vllm/envs.py:1947`），看起来是
   `flashinfer` + `cutlass`（cu12 + cu13，428 MiB）+ `cudnn` 前端被导入的原因。但同一
   提交上的 A/B（flag=1 vs flag=0）表明它是个 **no-op**：两次运行加载的顶层模块完全
   相同（均 334 个），且都打印
   `FlashInfer All Reduce is disabled because it is not supported for world_size=2`，
   随后是 `Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order)`。
   flashinfer 仍被加载，是因为本路由确实把它用在两处：
   `Using FLASHINFER attention backend out of ['FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`
   与 `Using FlashInfer for top-p & top-k sampling`。

   因此这一集群（约 555 MiB：flashinfer + 两个 cutlass-dsl 版本 + cudnn-frontend）
   的真正杠杆是**注意力/采样后端的选择**，它会改变 SM75 上的数值与速度——必须实测，
   不能直接切换。需要注意：SM75 的投机解码 prefill 路径（用于 FULL CUDA Graph 重放的
   FA2 graph 包装器，`flashinfer.py::_get_sm75_spec_prefill_wrapper`）就实现在该后端内部，
   所以在本路由上 FlashInfer 是**必需件**而不是可选项：去掉它意味着要重写 SM75 的
   MTP 注意力实现，而不是改一个配置。

   副发现：任何进入编译缓存键的 flag 都会带来一次冷编译。flag=0 那次启动耗时 91s，
   而 flag=1 为 53s，日志里出现 4 次 `Compiling a graph`，没有命中 AOT 缓存。
2. **`transformers` → `torchvision` + `torchaudio`。** 只要这两个包被安装，
   `transformers.image_processing_base` / `processing_utils` 就会导入它们。只有在
   纯文本部署里**不安装**它们（相关代码路径都有保护），或避免导入这些 transformers
   模块才能消除。
3. ~~**`mistral_common` → `cv2`**~~ **已完成。** 适配器注册表改为按需解析
   `<Engine>ReasoningAdapter` / `<Engine>ToolAdapter`，只导入被选中的那个 parser 引擎。
   实测：`mistral_common` 与 `cv2` 不再被加载（它们正是纯文本路由会导入 OpenCV 的原因）。
   `torchvision`/`torchaudio`（约 60 MiB）仍会加载，因为只要装了它们 `transformers` 就会导入——
   这属于依赖裁剪决策，不是代码问题。
4. **`torchcodec`** —— `vllm/multimodal/media/audio.py` 顶层
   `try: from torchcodec.decoders import AudioDecoder`。
5. ~~**`numba` / `llvmlite`**~~ **已完成。** `kimi_k25.py` 把基于 numba 的 fused vision
   processor 改为在选中它的方法内导入，因此本路由不再加载 `numba`/`llvmlite`
   （实测：2 个进程 → 0 个）。
6. ~~**依赖剪裁与干净环境验证**~~ **已完成** —— 见下节“环境剪裁”。
7. **`tilelang` + `tvm` 仍被加载（2 个进程）。** 在所有 vLLM 侧的导入者被去除后，它们仍然
   在**没有 Python 栈帧**的情况下被导入，即来自原生扩展。剩下的候选就是 `flashinfer`，
   而实验 1 已证明它在本路由上是必需件，因此要关闭这条链需要改动第三方包。此处只做记录，
   不列入计划。

## 复现方式

```bash
bash tools/dep_audit.sh --out-dir /tmp/dep-before -- \
  --non-interactive --model-dir /data/models/qwen3.8-27b/Qwen3.8-27B-FP8 \
  --speculative-model /data/models/qwen3.8-27b/MTP --served-name Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 --port 8000 \
  --set MAX_MODEL_LEN=131072
```

脚本会打印报告路径；每个进程的原始快照保留在 `<out-dir>/snapshots/`，因此可以对两次运行
做逐模块对比。

`bash tools/validate_all.sh --with-audit` 会对参考路由跑同一套审计（自己启停服务），随后执行
下面的最小集检查——依赖改动应该通过这一条命令验收。

## 环境剪裁（实测最小依赖集）

审计说明了运行时加载什么；据此可以真正地剪裁部署环境并验证结果。

| 步骤 | 发行包数 | site-packages |
| --- | --- | --- |
| 剪裁前 | 205 | 7.1 GiB |
| 移除 31 个从未加载的发行包后 | **174** | **5.8 GiB** |

移除集合即审计出的“从未加载”清单在 ≥1 MiB 阈值下的部分（约 1.3 GiB）：`tokenspeed-triton`、
`nvidia-nvvm`、`nvidia-cuda-nvcc`、`llvmlite`、`opencv-python-headless`、`pynvvideocodec`、
`cmake`、`mistral_common`、`pycountry`、`nvidia-cuda-cccl`、`numba`、`nccl4py`、
`cryptography`、`hf-xet`、`nvidia-cuda-nvdisasm`、`anthropic`、`outlines_core`、
`tokenspeed-mla`、`nvtx`、`quack-kernels`、`rignore`、`torch_c_dlpack_ext`、`sentry-sdk`、
`fastar`、`httptools`、`supervisor`、`watchfiles`、`humming-kernels`、`mcp`、
`nvidia-cuda-crt`、`dnspython`。仍安装但未使用的总量仅 22.5 MiB，其中 `pip` 自身占 10.6 MiB。

剪裁后环境的验证：

* 路由启动、健康检查、smoke 测试全部通过，**零**导入错误；
* 加载集合不变（123 个发行包 / 5663 MiB）——只移除了从未被碰到的包；
* **冷编译通过**：把 AOT 编译缓存挪开后，launcher 的 `--prewarm-compile-only` 报
  `AOT reused=no`，从零重编译并达到 `Startup complete after 106s`（热启动 52–53s）。
  这正是“被移除的 CUDA 工具（`nvcc`、`ptxas`、`cmake`、`ninja`）并不需要”的证据：系统
  CUDA 提供了它们。此外独立的 Triton 冷编译与 `torch.compile` 冷编译也各自通过。

`requirements/definitive-2080ti-measured.txt` 记录这份实测集合（122 个 pin），
`tools/dep_audit/check_minimal_set.py` 可对着新的审计复检：若某个被加载的发行包不在清单里
就报错，对失效条目给出警告，并列出剩余的可剪裁项。

两条值得保留的操作经验：

* **要在 launcher 使用的环境里剪裁，而不是 pip 默认指向的那个。** 这里 `.venv/` 有自己的
  `site-packages`，而 `.venv/bin/python3` 实际解析到共享的 conda 解释器，后者的环境里是
  另一套集合。用 `pip --python` 装 pip 再执行 `python -m pip uninstall`，作用在了 conda
  环境而不是 `.venv` 上。请使用 `.venv/bin/python3 -m pip …`（没有 pip 的 venv 可用
  `python -m ensurepip --upgrade` 补上），并在动手前保存 `pip freeze --all` 快照。
* **AOT 编译缓存与 Triton cubin 缓存是耦合的。** 只挪开 `triton-cache/` 而保留
  `~/.cache/vllm/torch_compile_cache` 并不会触发重编译，启动会失败于
  `RuntimeError: Cubin file saved by TritonBundler not found at .../triton-cache/<hash>/....cubin`。
  两者必须一起清空或一起挪开。
