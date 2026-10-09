# Dependency audit: what a serving run actually loads

The deployment problem this solves: the runtime environment has **205 distributions /
7.1 GiB** of `site-packages`, and the painful part is not the size itself but the
version surface that has to be resolved and kept consistent whenever anything is added.
This document records which of those distributions a real run of our route loads, which
it does not, and which import chains still pull heavy optional stacks.

Everything below is measured with `tools/dep_audit.sh`, which wraps a launcher run and
records, per Python process:

- a snapshot of `sys.modules` — **successful imports only**, so an optional-import probe
  that fails (`try: import jax`) is not counted as usage;
- the mapped shared objects from `/proc/self/maps` — the only way to see the CUDA
  libraries torch `dlopen`s, which never appear as Python modules but dominate the size.

Route audited: `2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env` (Qwen3.8-27B-FP8
+ MTP, TP=2, text only).

## Measured result

| | distributions | size |
| --- | --- | --- |
| installed | 205 | 7.1 GiB |
| loaded by the route | **123** | 5.5 GiB |
| not loaded | **82** | 1.2 GiB |

Before the two lazy-import batches (see below) the audited route loaded 130
distributions / 6.0 GiB, so the batches removed **385 MiB** of loaded modules
(`opencv-python-headless` 152, `llvmlite` 160, `mistral_common` and `numba`, plus
`humming`), and the number of imported top-level modules went 334 -> 327.

Largest loaded items: `torch` 1044 MiB, `triton` 687 MiB, `nvidia-cublas` 570,
`nvidia-cudnn-cu13` 471, `nvidia-cutlass-dsl-libs-cu12` + `cu13` 428, `nvidia-cufft` 274,
`nvidia-cusolver` 234, `nvidia-nccl-cu13` 226, `nvidia-cusparselt-cu13` 223,
`nvidia-cuda-nvrtc` 218, `tilelang` 179, `flashinfer-python` 90.

Largest items that are installed but never loaded: `tokenspeed-triton` 293 MiB,
`nvidia-nvvm` 163, `nvidia-cuda-nvcc` 147, `pynvvideocodec` 84, `cmake` 73,
`instanttensor` 20, `nvidia-cuda-cccl` 18, `cryptography` 14, `nccl4py` 14, `hf-xet` 12.

The `nvidia_*` wheels account for 3.0 GiB and duplicate the system CUDA 13.2 install
(7.2 GiB): **nothing from `/usr/local/cuda` is mapped** in the audited run, so the CUDA
libraries in use are the pip wheels. Using the system CUDA instead would mean rebuilding
torch against it — out of scope here, but it is the largest single size item on the table.

Caveat on the "not loaded" list: `/proc/self/maps` cannot see tools that are executed as
subprocesses. `nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-cccl` provide `nvcc`,
`ptxas` and headers used by Triton / `torch.utils.cpp_extension` JIT compilation, so they
may still be required even though no library stays mapped. Remove them only after a full
cold compile (including a cleared `triton-cache`) has been verified.

## Import attribution

Recorded by an audit hook that reports the importing module for each heavyweight optional
top-level module (pre-fix, `dep-audit` run 1):

| module | imported by |
| --- | --- |
| `cv2` | `vllm/multimodal/media/connector.py:32` (module level) |
| `av` | `vllm/multimodal/media/audio.py:16` |
| `torchcodec` | `vllm/multimodal/media/__init__.py:3` |
| `torchvision` | `transformers.image_processing_base:25` |
| `torchaudio` | `transformers.processing_utils:38` |
| `flashinfer` | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `cutlass`, `cudnn` | `flashinfer.cute_dsl.availability`, `flashinfer.gemm` |
| `tilelang`, `tvm` | no Python frame — a native extension imports them |
| `numba`, `llvmlite` | `_has_module("numba")` trial import, then `vllm.model_executor.models.kimi_k25` |
| `humming` | `_has_module("humming")` trial import, then `vllm.utils.humming` |
| `llguidance` | `_has_module("llguidance")` trial import |
| `pandas` | `vllm/model_executor/layers/utils.py`, `layers/quantization/input_quant_fp8.py` |
| `boto3` | `vllm/model_executor/model_loader/sharded_state_loader.py:23` |
| `ray` | `vllm/v1/engine/utils.py:34` |
| `redis` | `torch._inductor.runtime.autotune_cache:48` |
| `z3` | `torch.fx.experimental.symbolic_shapes` |

## Changes made

New helper `LazyModule` in `vllm/utils/import_utils.py`: a module proxy that imports on
first attribute access and keeps `PlaceholderModule`'s extra-aware error message.

| file | change |
| --- | --- |
| `vllm/tilelang_utils/__init__.py` | dropped the eager `import tilelang` on CUDA; `tilelang` / `T` resolve through a module `__getattr__`, and `tilelang_jit` defers the decoration to the first call (the ROCm path already worked this way) |
| `vllm/model_executor/kernels/mhc/tilelang_kernels.py` | imports only `tilelang_jit`; `T` / `tilelang` are `TYPE_CHECKING`-only |
| `vllm/distributed/device_communicators/cuda_communicator.py` | the two FlashInfer communicator imports are now conditional on the flags that select them |
| `vllm/multimodal/video.py` | `cv2` is a `LazyModule` |
| `vllm/multimodal/video_decoders/opencv.py` | `cv2` / `cv2.videoio_registry` are `LazyModule`s |
| `vllm/multimodal/audio.py`, `vllm/multimodal/media/audio.py` | `av`, `soundfile` are `LazyModule`s |
| `vllm/utils/import_utils.py` | `_SPEC_ONLY_PROBE` = {`numba`, `humming`, `llguidance`, `tilelang`}: availability checks for these use `find_spec` instead of a trial import, which used to load the packages themselves |
| `vllm/parser/engine/registered_adapters.py` | the adapter classes are built on demand: a module `__getattr__` maps `<Engine>ReasoningAdapter` / `<Engine>ToolAdapter` to the owning engine module and memoises the pair, so importing the registry no longer imports twelve parser engines |
| `vllm/model_executor/layers/quantization/utils/humming_utils.py` | the humming dtype maps are built on first lookup (`_LazyDtypeMap`) instead of at import time |
| `vllm/model_executor/models/kimi_k25.py` | the numba-based fused vision processor is imported inside the method that selects it |

Verified effect: `cv2` dropped from 4 to 2 processes, then to **0** after the second batch;
`_has_module("numba")` no longer imports numba; and on the audited route
`mistral_common`, `cv2`, `numba`, `llvmlite` and `humming` now load in **zero** processes.
The remaining paths below explain why the totals did not move further.

### Throughput check (controlled A/B)

Import changes are easy to blame for a throughput dip, so the batch was measured both ways
by stashing it, on the same route, back to back:

| condition | 4K/128 prefill | 4K/128 decode | 32K/512 prefill | 32K/512 decode | startup |
| --- | --- | --- | --- | --- | --- |
| without the batch | 1554.03, 1543.24 | 107.39, 107.24 | 1334.93 | 100.52 | 53 s |
| with the batch (run 1) | 1573.19, 1563.83 | 107.86, 107.41 | 1361.20 | 101.79 | 53 s |
| with the batch (run 2) | 1566.48, 1558.42 | 107.62, 107.54 | 1344.06 | 100.97 | 53 s |

So the lazy imports cost nothing measurable; the changed runs are ~1% ahead, which is inside
run-to-run variation. Note the method: an earlier uncontrolled baseline for the same route
read 1583-1599 tok/s prefill, measured hours earlier, and that apparent 1.6% gap is what
triggered the A/B — it turned out to be environmental drift, not the code. Reported numbers
for this lane should always be compared within one session.

Measurement note: the launcher's interactive performance lane cannot deliver samples in a
non-interactive run (it blocks on a later prompt and the samples stay in the stdio buffer),
so the server was started with stdin held open and `tools/profile_request.py` was called
directly with the launcher's exact flags (completions endpoint, pure filler prompt, fixed
`" the"` completion token, unique prompt variant per sample).

## Remaining work, in payoff order

1. **FlashInfer attention/sampler cluster — measured, and the flag is not the lever.**
   `VLLM_ALLREDUCE_USE_FLASHINFER` defaults to 1 (`vllm/envs.py:1947`), which looked like
   the cause of the `flashinfer` + `cutlass` (cu12 + cu13, 428 MiB) + `cudnn` frontend
   imports. An A/B on this route (flag=1 vs flag=0, same commit) shows it is a **no-op**:
   both runs load the same 334 top-level modules, and both log
   `FlashInfer All Reduce is disabled because it is not supported for world_size=2`
   followed by `Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order)`.
   FlashInfer stays loaded because this route selects it for two real jobs:
   `Using FLASHINFER attention backend out of ['FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`
   and `Using FlashInfer for top-p & top-k sampling`.

   So the real lever for this cluster (~555 MiB: flashinfer + both cutlass-dsl editions +
   cudnn-frontend) is the **attention / sampler backend choice**, which changes numerics
   and speed on SM75 — it has to be benchmarked, not flipped. Note that the SM75
   speculative-decode prefill path (the FA2 graph wrapper used for FULL CUDA Graph replay,
   `flashinfer.py::_get_sm75_spec_prefill_wrapper`) is implemented *inside* that backend,
   so on this route FlashInfer is load-bearing rather than incidental: dropping it means
   replacing the SM75 MTP attention implementation, not changing a setting.

   Side finding: a flag that participates in the compile-cache key costs a one-time cold
   compile. The flag=0 run started in 91 s against 53 s for flag=1, with
   `Compiling a graph` appearing four times instead of hitting the AOT cache.
2. **`transformers` → `torchvision` + `torchaudio`.** Imported by
   `transformers.image_processing_base` / `processing_utils` whenever the packages are
   installed. Only avoidable by not installing them for a text-only deployment (the code
   paths are guarded), or by avoiding those transformers modules.
3. ~~**`mistral_common` → `cv2`**~~ **Done.** The adapter registry resolves
   `<Engine>ReasoningAdapter` / `<Engine>ToolAdapter` on demand, so only the selected
   parser engine is imported. Measured: `mistral_common` and `cv2` no longer load at all
   (they were the reason a text-only route imported OpenCV). `torchvision`/`torchaudio`
   (~60 MiB) still load because `transformers` imports them whenever they are installed —
   that one is a requirements decision, not a code change.
4. **`torchcodec`** — a module-level `try: from torchcodec.decoders import AudioDecoder` in
   `vllm/multimodal/media/audio.py`.
5. ~~**`numba` / `llvmlite`**~~ **Done.** `kimi_k25.py` imports the numba-based fused
   vision processor inside the method that selects it, so `numba`/`llvmlite` no longer
   load on this route (measured: 2 processes -> 0).
6. ~~**Requirement pruning and clean-environment validation**~~ **Done** — see
   "Pruning the environment" below.
7. **`tilelang` + `tvm` still load (2 processes).** After every vLLM-side importer was
   removed, they are imported with **no Python frame**, i.e. from a native extension. The
   remaining candidate is `flashinfer`, which experiment 1 showed is load-bearing on this
   route, so closing this needs a change inside a third-party package. Listed rather than
   planned.

## Reproducing

```bash
bash tools/dep_audit.sh --out-dir /tmp/dep-before -- \
  --non-interactive --model-dir /data/models/qwen3.8-27b/Qwen3.8-27B-FP8 \
  --speculative-model /data/models/qwen3.8-27b/MTP --served-name Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 --port 8000 \
  --set MAX_MODEL_LEN=131072
```

The wrapper prints the report path; the raw per-process snapshots stay under
`<out-dir>/snapshots/` so two runs can be diffed module by module.

`bash tools/validate_all.sh --with-audit` runs this same audit against the reference route
(launching and stopping the service itself) and then the minimal-set check below, which is how
a dependency change is supposed to be validated end to end.

## Pruning the environment (measured minimal set)

The audit says what a run loads; that made it possible to actually prune the deployment
environment and verify the result.

| step | distributions | site-packages |
| --- | --- | --- |
| before | 205 | 7.1 GiB |
| after removing 31 never-loaded distributions | **174** | **5.8 GiB** |

The removed set is the never-loaded list at a ≥1 MiB threshold (~1.3 GiB):
`tokenspeed-triton`, `nvidia-nvvm`, `nvidia-cuda-nvcc`, `llvmlite`,
`opencv-python-headless`, `pynvvideocodec`, `cmake`, `mistral_common`, `pycountry`,
`nvidia-cuda-cccl`, `numba`, `nccl4py`, `cryptography`, `hf-xet`, `nvidia-cuda-nvdisasm`,
`anthropic`, `outlines_core`, `tokenspeed-mla`, `nvtx`, `quack-kernels`, `rignore`,
`torch_c_dlpack_ext`, `sentry-sdk`, `fastar`, `httptools`, `supervisor`, `watchfiles`,
`humming-kernels`, `mcp`, `nvidia-cuda-crt`, `dnspython`. Everything still installed but
unused is 22.5 MiB in total, of which `pip` alone is 10.6 MiB.

Validation of the pruned environment:

* route startup, health check and smoke test pass with **zero** import errors;
* the loaded set is unchanged (123 distributions, 5663 MiB) — only untouched packages went;
* **cold compile passes**: with the AOT compile cache moved aside, the launcher's
  `--prewarm-compile-only` reports `AOT reused=no`, recompiles from scratch and reaches
  `Startup complete after 106s` (52-53 s warm). That is what proves the removed CUDA *tools*
  (`nvcc`, `ptxas`, `cmake`, `ninja`) were unnecessary: the system CUDA toolkit provides
  them. A standalone cold Triton compile and a cold `torch.compile` compile each passed as
  well.

`requirements/definitive-2080ti-measured.txt` records the verified set (122 pins) and
`tools/dep_audit/check_minimal_set.py` re-checks it against a fresh audit: it fails when a
loaded distribution is missing from the file, warns about stale entries, and lists the
remaining pruning candidates.

Two operational findings worth keeping:

* **Prune the environment the launcher uses, not the one pip defaults to.** Here `.venv/`
  has its own `site-packages`, while `.venv/bin/python3` resolves to a shared conda
  interpreter whose environment holds a *different* set. Installing pip through
  `pip --python` and then running `python -m pip uninstall` acted on the conda environment,
  not on `.venv`. Use `.venv/bin/python3 -m pip …` (a venv without pip can get one from
  `python -m ensurepip --upgrade`) and keep a `pip freeze --all` snapshot before removing
  anything.
* **The AOT compile cache and the Triton cubin cache are coupled.** Moving `triton-cache/`
  aside while leaving `~/.cache/vllm/torch_compile_cache` in place does not trigger a
  recompile: startup fails with `RuntimeError: Cubin file saved by TritonBundler not found
  at .../triton-cache/<hash>/....cubin`. Move or clear both together.
