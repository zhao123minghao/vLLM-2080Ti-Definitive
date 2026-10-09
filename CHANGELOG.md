# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## Unreleased

### KV-cache block bookkeeping: Rust shadow verification

The second step of the `v1/core` migration spike (after prefix-cache hashing) ports the block
state the scheduler mutates every step, and proves it equivalent to Python by replaying a
recorded operation trace:

- New `rust/src/kv-pool` (`vllm-kv-pool`): `FreeKVCacheBlockQueue` as an index-addressed
  doubly linked list with the same fake head/tail sentinels, `BlockHashToBlockMap` with
  the same single-or-group semantics, `BlockPool` on top of both — reference counts, the
  two reuse orders of `free_blocks` (uncached blocks LIFO, cached blocks LRU), `touch`,
  `unpin` with its reuse watchers, eviction on allocation, the per-block hash metadata and
  the reset path — and `BlockHashListWithBlockSize` plus `resolve_block_hashes`, the
  block-size-aware view the scheduler reads hashes through (index arithmetic, Python slice
  normalisation, negative indices, and the fine-grained-partial-hit branch). The failure
  modes are part of the contract: `ValueError`, `AssertionError`, `RuntimeError` and
  `IndexError` are reproduced as the exception Python would have raised, and the free-block
  counter is signed because Python's `int` can drift negative.
- New `tools/check_block_pool_parity.sh` plus `tools/gen_block_pool_trace.py`: the recorder
  drives the **real** Python classes with a seeded random operation sequence and writes the
  trace together with the log Python produced, including the exception class of every
  rejected operation; the Rust binary replays the trace and the two logs are diffed.
  Verified with 6 traces of ~1,217 operations: **7,300 operations compared, 3,640 of them pool
  operations and 1,428 hash-list operations, 810 rejections, every line identical**, plus 23
  crate unit tests for what a log diff cannot see.

### One-command validation

The checklist in `AGENTS.md` was five commands that had to be remembered; it is now one, and the
three gaps in it are closed:

- New `tools/validate_all.sh` runs the four static steps (`bash -n` over every shell script,
  `py_compile` over `tools/**/*.py` plus any file passed as an argument, the profile matrix, and
  `git diff --check`) and takes `--with-parity` and `--with-chat` for the two checks that need
  more than a checkout; `--with-parity` now runs both parity scripts. Last line is grep-friendly
  (`validate_all_ok steps=N failed=0 skipped=M`), and the parity steps are skipped with a
  message when `cargo` is absent instead of failing.
- Every tool under `tools/` now pins the repository venv on its own `PATH`. The parity script
  was the one tool that relied on the caller's environment, so from a shell without `.venv` it
  resolved `python3` to the base conda interpreter and died with
  `ModuleNotFoundError: No module named 'cbor2'` — reproduced and fixed, with a one-second
  check that reports the situation instead of a traceback from inside the corpus generator.
- `tools/check_prefix_hash_parity.sh` now also runs the `vllm-kv-hash` crate's own unit tests
  (`cargo test --release -p vllm-kv-hash`, `--skip-tests` to opt out). The cross-language diff
  proves the two implementations agree on a corpus; the unit tests cover what that diff cannot
  see, such as partially filled blocks and chained-hash state. Verified: 18 parity checks plus
  7 unit tests pass (`validate_all_ok steps=5 failed=0`), and the step fails outright if the
  crate ends up running no tests at all.
- The dependency audit is now part of the checklist: `AGENTS.md` points at
  `tools/dep_audit.sh` plus `tools/dep_audit/check_minimal_set.py` for dependency, requirement
  and environment changes, which previously only lived in a document.
- `README.md` / `README.zh-CN.md` gained a Validation section listing the entry points and
  linking the chat-regression, dependency-audit and `v1/core` migration documents, which were
  otherwise only reachable through `AGENTS.md` or the `docs/` directory listing.
- `profiles/local/` — the user-local scratch area that `update.sh` preserves and that
  `tools/validate_profiles.sh` deliberately excludes from the official matrix — is now
  git-ignored, so a local route cannot be committed by accident.

### Chat regression smoke

Correctness of the served chat path is now checkable with one command, which matters after a
kernel, quantization, parser or scheduler change:

- New `tools/chat_smoke.sh` (starts the reference route, waits for health, runs the checks,
  stops it again; `--quick` for the fast subset, or run it against an already listening
  server) and `tools/chat_smoke.py` (stdlib only, exit 0/1, last line
  `chat_smoke_ok passed=N failed=0`). The stop path kills the recorded pid *and* sweeps any
  `launcher.sh` service-manager process that re-parents itself to init with the same
  arguments, so a run leaves no engine, worker or manager behind.
- Full suite verified end-to-end on the reference route: `chat_smoke_ok passed=16 failed=0`
  (warm start 51-54 s, whole run under three minutes, GPUs back to 0 MiB afterwards).
- Twelve content probes at `temperature=0`, all verified correct on the reference route:
  capital of France, a Chinese multi-step arithmetic, the bat-and-ball problem ($0.05), the
  running-dog problem (100 km), 9.11 vs 9.8 (9.8), chickens-and-eggs (27), 2^10 (1024), a
  multi-step price equation (100), a truth-teller logic puzzle (乙), the light-year unit
  (长度), `is_prime` code-only instruction following, and a two-turn memory check (7391).
  The bat-and-ball and decimal-comparison cases are deliberate traps that catch
  pattern-matching instead of reasoning.
- Four API-contract probes pin down this deployment's Rust frontend: the trace arrives in
  `message.reasoning` (**not** `reasoning_content`), thinking is on by default and
  `enable_thinking=false` empties the trace, streaming with `stream_options.include_usage`
  reports `completion_tokens_details.reasoning_tokens`, and `max_tokens` counts reasoning
  tokens (tiny budget -> `finish_reason == "length"` with possibly empty content).
- Documented in `docs/chat-regression.md` and listed in `AGENTS.md` as a validation step for
  model/kernel/quantization/parser changes. Two launcher quirks found while building it are
  documented as troubleshooting: the launcher draws its `Startup complete after Ns.` progress
  on the controlling terminal only (never in a log file), and it can leave a service-manager
  process behind.

### Dependency audit and lazy imports for heavy optional packages

The deployment environment carries 205 distributions / 7.1 GiB of `site-packages`, so a
real serving run was audited to separate what is declared from what is loaded:

- New `tools/dep_audit.sh` (with `tools/dep_audit/{sitecustomize,analyze}.py`) wraps a
  launcher run and records, per Python process, both the successfully imported modules
  and the shared objects mapped in `/proc/self/maps`. The second signal is the only way
  to see the CUDA libraries torch `dlopen`s, and the pair distinguishes a real import
  from a failed optional-import probe.
- Measured on the 27B text route: **130 of 205 distributions are loaded** (6.0 GiB of
  7.1 GiB); 75 are never loaded (891 MiB). The `nvidia_*` wheels are 3.0 GiB and
  duplicate the system CUDA 13.2 install — nothing under `/usr/local/cuda` is mapped.
- New `LazyModule` in `vllm/utils/import_utils.py` (import on first attribute access,
  `PlaceholderModule` error message preserved) and applied to the heavy optional paths:
  `cv2` in `vllm/multimodal/video.py` and `vllm/multimodal/video_decoders/opencv.py`,
  `av`/`soundfile` in the audio loaders, the CUDA `tilelang` import in
  `vllm/tilelang_utils`, and the FlashInfer communicator imports in
  `cuda_communicator.py` (now conditional on the flags that select them).
- Availability checks for `numba`, `humming`, `llguidance` and `tilelang` no longer
  trial-import the package: `_has_module` uses `find_spec` for those, because probing
  them pulled numba/llvmlite and the TileLang/TVM chain into startup for code paths that
  never ran.
- Verified: `cv2` loads in 2 processes instead of 4 on the audited route, and
  `_has_module("numba")` no longer imports numba.
- A/B on the audited route (same commit) shows `VLLM_ALLREDUCE_USE_FLASHINFER=0` is a
  **no-op**: identical module sets and the same `['CUSTOM', 'PYNCCL']` dispatch, because
  the engine already logs `FlashInfer All Reduce is disabled because it is not supported
  for world_size=2`. FlashInfer stays loaded because this route selects it for the
  attention backend and for top-k/top-p sampling, so the ~555 MiB cluster is a
  backend-choice question (FlashInfer vs Triton/FlexAttention), not a flag. Changing that
  flag also invalidated the AOT graph cache and cost a cold recompile (91 s vs 53 s).
- Second batch of lazy imports, measured with the same audit: loaded distributions
  130 -> **123** and loaded size 6.0 -> **5.5 GiB** (**-385 MiB**), imported top-level
  modules 334 -> 327. On the audited route `mistral_common`, `cv2`, `numba`, `llvmlite`
  and `humming` now load in **zero** processes: the parser adapter registry builds
  `<Engine>ReasoningAdapter` / `<Engine>ToolAdapter` on demand instead of importing twelve
  parser engines, `humming_utils` builds its dtype maps on first lookup, and `kimi_k25`
  imports its numba-based fused vision processor inside the method that selects it.
  `tilelang`/`tvm` still load with no Python frame (native import; `flashinfer` is the
  remaining candidate) and `torchvision`/`torchaudio` still load via `transformers` — both
  are documented in `docs/dependency-audit.md`.
- Controlled A/B for the second batch (stashed back to back, same route): without it 4K/128
  prefill 1554.03/1543.24 tok/s and decode 107.39/107.24, with it 1573.19/1563.83 and
  107.86/107.41 (run 1) and 1566.48/1558.42 and 107.62/107.54 (run 2); 32K/512 prefill
  1334.93 -> 1361.20/1344.06. Startup was 53 s in every condition, so the lazy imports
  cost nothing measurable. An earlier uncontrolled baseline of 1583-1599 tok/s for the same
  route (measured hours earlier) was environmental drift, not a regression.
- Environment pruning, validated end to end. The never-loaded distributions from the audit
  (≥1 MiB each) were removed from the deployment venv: **205 -> 174 installed
  distributions, 7.1 -> 5.8 GiB** of `site-packages`; what remains installed but unused is
  22.5 MiB in total. Removed: `tokenspeed-triton`, `nvidia-nvvm`, `nvidia-cuda-nvcc`,
  `llvmlite`, `opencv-python-headless`, `pynvvideocodec`, `cmake`, `mistral_common`,
  `pycountry`, `nvidia-cuda-cccl`, `numba`, `nccl4py`, `cryptography`, `hf-xet`,
  `nvidia-cuda-nvdisasm`, `anthropic`, `outlines_core`, `tokenspeed-mla`, `nvtx`,
  `quack-kernels`, `rignore`, `torch_c_dlpack_ext`, `sentry-sdk`, `fastar`, `httptools`,
  `supervisor`, `watchfiles`, `humming-kernels`, `mcp`, `nvidia-cuda-crt`, `dnspython`.
  The route still starts (health + smoke, zero import errors), the loaded set is unchanged
  (123 distributions / 5663 MiB), and a full **cold compile passes**: with the AOT cache
  moved aside, `--prewarm-compile-only` reports `AOT reused=no` and reaches
  `Startup complete after 106s` (52-53 s warm), which is what shows the removed CUDA tools
  (`nvcc`/`ptxas`/`cmake`/`ninja`) were supplied by the system CUDA toolkit instead.
- New `requirements/definitive-2080ti-measured.txt` (122 pinned distributions, derived from
  measurement, with the caveats that matter) and `tools/dep_audit/check_minimal_set.py`,
  which fails if a loaded distribution is missing from that file, warns about stale entries
  and lists remaining pruning candidates.
- Operational findings documented in `docs/dependency-audit.md`: prune the venv the launcher
  uses rather than the environment `pip` defaults to (`.venv/bin/python3 -m pip …`, snapshot
  with `pip freeze --all` first), and the AOT compile cache is coupled to the Triton cubin
  cache — moving only `triton-cache/` makes startup fail with `Cubin file saved by
  TritonBundler not found`.
- See `docs/dependency-audit.md` for the full attribution table and the remaining
  work, in payoff order.

### `v1/core` migration spike: byte-exact prefix-cache block hashing in Rust

Feasibility work for moving `vllm/v1/core` out of Python, starting with the
per-step prefix-cache hashing chain:

- New crate `rust/src/kv-hash` (`vllm-kv-hash`) re-implements
  `hash_block_tokens` / `init_none_hash` from `vllm.v1.core.kv_cache_utils`,
  reproducing both shipped algorithms byte for byte, including the
  serialization step: `sha256` (CPython pickle protocol 5, the vLLM default) and
  `sha256_cbor` (canonical CBOR). `xxhash` / `xxhash_cbor` are not ported: they
  need XXH3-128, which no offline cargo crate in this repository provides.
- `tools/check_prefix_hash_parity.sh` + `tools/gen_prefix_hash_corpus.py` make
  the **real Python implementation** the oracle: 18/18 checks pass (2 algorithms
  × 3 `extra_keys` shapes × {serialized payload bytes, 200-block chained
  hashes}, plus `NONE_HASH` with and without a pinned `PYTHONHASHSEED`).
- Same-corpus measurement (10,000 blocks × 32 tokens, 248,320 vocab, best of 5):
  hashing runs at 16.06 M tokens/s in Python vs 43.67 M tokens/s in Rust for
  `sha256` (2.7x, 1.99 µs vs 0.73 µs per 32-token block) and 2.77 M vs 45.41 M
  tokens/s for `sha256_cbor` (16.4x). Python's default pickle path is 5.8x
  faster than its CBOR path.
- Verdict: **not** a performance project. A decode step's hashing plus block
  pool work lands in the hundreds of microseconds (block pool measured at
  0.32 µs per allocated and 0.15 µs per freed block with this repository's own
  benchmark), against GPU steps measured in tens of milliseconds. See
  `docs/v1-core-rust-migration.md` for the measured numbers, the remaining
  semantics risk (block pool, scheduler decisions) and the recommended order if
  the migration is pursued for architecture reasons instead.
- Fixed `benchmarks/benchmark_block_pool.py`, which had drifted from the local
  `BlockPool` signature (missing `hash_block_size`) and could not run; it takes
  `--block-size` now.

The `vllm-kv-hash` binary is a development tool and is not installed next to
`vllm/vllm-rs`.

### Platform probe: one NVML session per process

Platform detection initialized and shut down NVML repeatedly while
`import vllm`:

- `vllm/platforms/cuda.py::with_nvml_context` now initializes NVML once per
  process instead of an `nvmlInit()`/`nvmlShutdown()` pair per decorated call,
  and `cuda_platform_plugin()` is cached (it was probed twice per import and
  called `nvmlDeviceGetCount()` twice per probe).

Measured on the 2x2080Ti host with interleaved A/B (three rounds, alternating
patched/upstream in one session, no background `nvidia-smi` polling): cold
`import vllm` drops from 13.34–13.47 s to 5.09–5.21 s, i.e. about **8.2 s per
process**. The saving does **not** show up in the managed-engine startup on the
same route: `init engine` 8.35 s vs 8.15 s and the launcher finishes in ~58 s
either way, because the three engine processes start in parallel and the cold
driver initialization is overlapped. Treat this as an improvement to the import
path (CLI tools, tests, single-process frontends) rather than a launcher startup
win. Note: a 1 Hz `nvidia-smi` poller on the host used to inflate and destabilize
these measurements (single-shot spread 3–15 s); with it stopped, repeated runs
vary by ~ 0.02 s.

### Engine import path: stop loading the serving stack

The headless Python engine no longer imports the OpenAI/FastAPI serving layer:

- `vllm.entrypoints.cli.main` imports only the subcommand module for the
  requested command (`serve` for managed engines) instead of every subcommand,
  and `VLLM_SUBCMD_PARSER_EPILOG`/`cli_env_setup` moved to
  `vllm/entrypoints/cli/types.py` (still re-exported from
  `entrypoints/serve/utils/api_utils.py`), so importing the CLI no longer
  imports the FastAPI-based serving utilities.
- `vllm/entrypoints/cli/serve.py` imports the FastAPI API-server entry and the
  DP supervisor only in the branches that need them.
- `cli_args` no longer imports the tool-parser registry (and with it the OpenAI
  SDK) to render the `--tool-call-parser` metavar; the registry is consulted
  only when help output is requested.
- The light helpers that the CLI and parser layers need (`chat template option
  types`, `validate_chat_template`, `get_tool_call_id_type`,
  `make_tool_call_id`) moved to `vllm/entrypoints/chat_common.py`, and
  `chat_utils`, `cli_args`, `parser/abstract_parser`, and
  `entrypoints/generate/base/protocol.py` import them from there.

Measured on the 2x2080Ti 27B route: the engine's import chain drops from 5584 to
5403 modules. Wall-clock import time is 17.7 s → 16.6 s (mean of five runs per
state), which is within run-to-run spread, so treat it as "not measurable"
rather than a win. Decoupling one importer only hands the same heavy leaves
(OpenAI SDK, PIL, multimodal stack) to the next one: after `chat_utils`, the
carrier became `vllm.entrypoints.openai.chat_completion.protocol`, imported
eagerly by `vllm/parser/abstract_parser.py`. The engine path therefore still
needs the structural fix, not another import repoint:

- `vllm/v1/engine/**` should not import the parser/renderer/serving trees at
  module level (the Rust frontend owns parsing and rendering), or
- the OpenAI request/response *protocol types* should be split from the runtime
  modules that pull the SDK.

Remaining measured costs per engine process: `vllm.utils.torch_utils` 6.5 s
(CUDA availability probing at import time, reorderable but not removable),
`import torch` 1.2–2.1 s, `transformers` 1.15 s, `kernels.mhc` +0.9 s (tilelang,
unused on these routes), `humming` ≈0.5 s (mxfp4 linear backends, unused on
FP8/TurboQuant routes).

### Launcher duplicate-start guard

The launcher now refuses to start a service while a managed service is already
running, or while the target port already has a listener. Previously the
startup health poll could succeed against the existing listener and report a
false `START OK`, leaving two services competing for the same GPUs. Set
`ALLOW_DUPLICATE_START=1` to bypass the guard.

### Launcher compile prewarm mode

`--prewarm-compile-only` starts the resolved route, waits for `/health` so the
graph capture and compile cache are written, stops the server, and reports
whether the AOT cache was reused. The first start of a route pays the decode
graph compilation; running the prewarm once after changing profile, context, or
KV precision lets later starts reuse the compiled graph.

### Launcher frontend selection (Rust frontend)

`--frontend rust` runs the Rust OpenAI frontend (`vllm-rs serve`) instead of the
Python API server. The Rust frontend supervises the Python headless engine and
owns the HTTP, tokenizer, chat-template, parsing, and metrics layers, which
removes the Python frontend process from startup. Serving arguments stay with
the frontend, all other engine arguments are forwarded verbatim.

`FRONTEND` defaults to `auto`: it selects the Rust frontend for `text-only` and
`text+image` routes that use no speculative decoding or MTP/DFlash, without tool
calling, SSD prefix caching, or scale-out endpoints, and keeps the Python
frontend otherwise, printing the reason in the launch summary. All 27 shipped
profiles satisfy that gate today (measured by running the launcher with
`DRY_RUN=1`), so `auto` and `python` now differ only for tool calling, SSD
prefix caching, and scale-out routes. Multimodal serving is validated on the
Rust frontend with colour-matched image requests, and MTP plus images was
validated as well. See `docs/non-interactive-launch.md` for the measured
comparison (56 s versus 62 s on the dual 2080 Ti 27B route).

### Rust frontend per-request metrics

The Rust frontend now serializes the response `metrics` block, including
`metrics.speculative_decoding`. The engine already sent
`spec_decode_metrics`; the protocol type was opaque and the frontend dropped it.
It is now typed end to end (engine-core protocol, llm, text, chat, gRPC
`FinishInfo`, and the OpenAI response types) for the completions and
chat-completions APIs, streaming and non-streaming. Non-streaming responses keep
the Python frontend's explicit `null` timing fields, while streaming chunks omit
`null` fields like the Python frontend's `exclude_none=True` chunks.

Measured on dual RTX 2080 Ti with Qwen3.8-27B-FP8: the MTP route
(`mtp4-fp16kv-1x148K`, `131072` context) returned a `metrics` block identical to
the Python frontend across chat, streaming completions, and streaming chat; the
DFlash route (`SPECULATIVE_METHOD=dflash`, `SPECULATIVE_TOKENS=7`, `65536`
context) is now selected by `FRONTEND=auto` with byte-identical text and metrics
(`mean_acceptance_length=2.666666666666667`,
`acceptance_histogram=[1,3,1,0,0,1,0,0]`) and the same 67,195-token KV cache.
Per-request timing metrics are not computed yet and stay `null`/omitted, exactly
as the Python frontend reports them when unavailable.

### Launcher startup diagnostics

When a start fails, the launcher now inspects the engine log and explains known
stall signatures, starting with the shared-memory broadcast stall, and reports
how many stale `/dev/shm/psm_*` segments exist without deleting them.

## v0.2.2-post3 - 2026-10-01

Patch release following `v0.2.2-post2`.

### Optional local SSD prefix KV persistence (#243)

Adds experimental, opt-in prefix KV persistence on local SSD through the
launcher, with checkpoint fingerprints and stable hashes for reuse across
restarts. Cache recovery was validated after a host reboot on 4xT10 and after
service restarts on dual RTX 2080 Ti MTP4 and DFlash2 routes.

### Hybrid Mamba prefix-cache correctness (#241)

Restricts fine-grained Mamba prefix hits to backends that can restore their
recurrent state, including for Mooncake Store; GDN/FlashQLA uses block-aligned
hits while FlashKDA retains supported fine-grained reuse. The growing-prefix
reproducer passed 40/40 checks on both 4xT10 and dual RTX 2080 Ti routes.

### Launcher interaction and GPU topology (#244)

Adds confirmed Esc cancellation during startup, wraparound menu navigation,
separate histories of up to 10 local target and draft directories, and
topology-aware TP grouping and PP stage ordering with P2P checks.

## v0.2.2-post2 - 2026-09-26

Patch release following `v0.2.2-post1`. This release further strengthens
DFlash2 prefix-cache replay and target-state handling.

### DFlash2 prefix cache and target state

- Retains DFlash2 target replay state across prefix-cache reuse and alternating
  prefill/decode requests, so reusable target state is not lost between turns.
- Restores the configured target Mamba speculative state pages for DFlash2 and
  DSpark while keeping draft KV pages in their independent pools.
- Accounts for the resident replay and hashed-boundary pages together with the
  target speculative pages in Mamba align mode, keeping target block tables and
  KV capacity consistent during speculative replay.
- Adds regression coverage for prefix-cache replay and DFlash target state
  sizing and accounting.

## v0.2.2-post1 - 2026-09-26

Patch release following `v0.2.2`, based on the latest `main` runtime.

### DFlash and hybrid Mamba correctness

- Keeps requests that already emitted decode tokens active while a later request
  continues chunked prefill, preventing unnecessary decode pauses.
- Stops reserving speculative Mamba state pages for DFlash2 and DSpark target
  KV, while retaining the resident replay state required by align mode.
- Preserves hashed Mamba align boundary state across decode and releases only
  obsolete unhashed blocks, keeping DFlash target prefix-cache entries reusable.
- Preserves restored-token accounting through the following DFlash decode step.

### Profiles and validation

- Adds four validated 2xT10 Qwen3.8-27B INT4 W4A16 MTP3 routes covering FP8 KV,
  TQ4NC, TQK8V4, text-only, and text-image serving.
- Adds focused scheduler, KV sizing, and prefix-cache regression coverage for
  the repaired paths.

## v0.2.2 - 2026-09-25

Incremental release from the validated `v0.2.1` SM75 serving snapshot
(`b296055efd`) through the current `main` branch. This release focuses on
correctness and reproducibility for Qwen3.8 hybrid Mamba serving on Turing
GPUs, and introduces the published runtime identity `vllm-def-cu130`.

### Prefix caching and speculative decoding

- Reuses DFlash2 target KV across prefix-cache hits and retires replaced
  hashed Mamba states at aligned prefill boundaries.
- Fixes FP8 KV hybrid prefix hits when the Mamba block size equals the hash
  block size, including the 1648-token SM75 route.
- Preserves request-row boundaries and hardens FULL-graph padding during
  speculative replay.
- Keeps SM75 DFlash2 tensor-parallel reductions in one scale domain.

### MTP, GDN, and TurboQuant correctness

- Prevents late prefills from pausing active MTP decode and preserves exact
  verifier KV when a late prefill joins an MTP batch.
- Fixes the hybrid-Mamba prefill barrier livelock and adds explicit alignment
  limits and tail-cohort coverage.
- Stabilizes TurboQuant verifier logits across late prefills and fails closed
  when adaptive query boundaries cannot be honored.
- Partitions stateful Qwen GDN forward execution and fixes SM75 GDN/DFlash
  speculative quality regressions.

### SM75 runtime and launcher

- Prevents unsupported CUTLASS FP8 dispatch on SM75 by selecting the supported
  fallback path.
- Adds safer FULL-graph padding and trace controls for the SM75 runtime.
- Makes the launcher load the model's generation configuration by default.
- Synchronizes renamed validated profile references and refreshes the shipped
- Adds the validated 4xT10 W8A16 FP8KV DFlash2 text-only route at 2x220K.
- Keeps FlashInfer MTP draft metadata correct across metadata rebuilds and
  preserves eager MTP execution for the SM75 compatibility path.

## v0.2.1 - 2026-09-19

Stable release for the CUDA 13 / PyTorch 2.13 SM75 line.

- Promotes the validated SM75 serving routes and profile set from the RC.

## v0.2.1-RC1 - 2026-09-18

Release candidate for the CUDA 13 / PyTorch 2.13 SM75 line.

- Adds validated DFlash2 routes, fused Triton grouped convolution, stable SM75
  FlashInfer FA2 graph buffers, and live metadata refresh for FULL graph replay.
- Fixes concurrent speculative graph sizing, automatic prefill alignment,
  independent target/draft KV pools, and concurrent MTP FA2 execution.
- Generalizes qualified PCIe custom all-reduce to 2–16 GPUs and improves the
  CUDA 13 Torch/Triton mirror build path.
- Converges profiles on validated routes, requires explicit KV precision, and
  removes unsupported concurrency presets.
- Moves profiles to a flat hardware/model/weight route layout and makes `fast`
  the launcher default startup mode.

## v0.2.1-pre4 - 2026-09-14

Nightly-alignment candidate based on upstream `b23433088b`
(`v0.29.1rc0-33`).

- Adopts upstream replacements for previously local runtime fixes and retains
  only the SM75-specific FlashQLA, TurboQuant, parser, and rendezvous paths.
- Keeps `humming-kernels[cu13]==0.1.13` for INT6/AutoRound loading.

## v0.2.1-pre3 - 2026-08-27

Validated true-concurrency candidate for dual RTX 2080 Ti.

- Adds packed-varlen FlashQLA, prefill batch alignment, bounded TurboQuant
  wrapper caching, and long-context workspace handling.
- Improves reasoning/tool parsing, Mamba/offload boundaries, GPU selection, and
  single-node process rendezvous.
- Validates the documented NVFP4 concurrency route and the associated parser,
  serving, TurboQuant, offload, and FlashQLA paths.

## v0.2.1-pre2 - 2026-08-21

First validated preview based on upstream vLLM `v0.27.1`.

- Adds Qwen3.8 FP8/NVFP4 profiles, TP/PP launcher selection, stricter preflight
  checks, and the required SM75 FlashInfer/FlashQLA compatibility paths.
- Ports the validated reasoning, tool-call, CUDA Graph, and TurboQuant fixes to
  the newer runtime architecture.
- Validates the documented dual-2080-Ti TP=2 routes; broader PP and additional
  long-context routes remained experimental.

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based
on upstream vLLM `v0.27.1`.

Thanks to upstream vLLM and its contributors. Fork release and validation work
is maintained by [@weicj](https://github.com/weicj), with contributions from
the project community.
