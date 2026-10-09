# Non-Interactive Launch

Build the runtime with `./build.sh`, then pass a flat profile path. Profile
files contain route parameters only; checkpoint, GPU topology, port, and mode
remain launcher options.

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

`--mode` is optional and defaults to `fast`. A profile without `MODE` keeps
the launcher selection; an explicit profile `MODE=normal` or `MODE=fast` may
override it. There are no `fast/` or `normal/` profile directories.

Useful options include `--model-dir`, `--speculative-model`, `--profile`,
`--mode`, `--gpu-devices`, `--tp-size`, `--pp-size`, `--port`,
`--start-timeout`, and `--print-config`. Use `--set KEY=VALUE` for advanced
launcher/runtime settings. Do not add Prefix Cache, Mamba cache, GPU, port, or
model-path fields to a profile; the validator rejects them.

The profile library is a validation matrix, not a promise that every filename
fits every machine. Capacity and performance are promoted only after the
external audit records startup, 4K/128, 32K/512, concurrency, and image
correctness evidence.
## Duplicate Start Guard

The launcher refuses to start a second service when a managed service is
already running, or when the target port already has a listener. Without that
check the startup health poll would succeed against the existing listener and
report a false `START OK`. Stop the running service, choose another port, or
set `ALLOW_DUPLICATE_START=1` to start anyway.

## Compile Prewarm

Decode graphs are compiled and captured during the first start of a route, and
that first start can take roughly 30 s longer than the following ones. Once the
compile cache is written, later starts on the same route, context length, KV
precision, and spec-decode setting reuse it.

Use `--prewarm-compile-only` to pay that cost deliberately: the launcher starts
the resolved route, waits for `/health` so graph capture and compile artifacts
are written, stops the server, and reports whether the AOT cache was reused:

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 \
  --prewarm-compile-only
```

The command exits with `PREWARM OK` and leaves no service running, so the next
real start on that route reuses the compiled graph.
## Rust Frontend

The Rust OpenAI frontend (`vllm-rs`, built by `./build_rust.sh`) can serve the
same routes instead of the Python API server. It supervises a managed Python
headless engine and keeps the HTTP, tokenizer, chat-template, parsing, and
metrics layers in Rust, so the Python frontend process disappears from startup.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env \
  --mode fast --gpu-devices 0,1 --tp-size 2 --frontend rust
```

Serving arguments (`served-model-name`, `reasoning-parser`,
`language-model-only`, `enable-prompt-tokens-details`, `disable-log-stats`,
chat-template and tool-parser options) are handed to the Rust frontend; every
other engine argument is forwarded verbatim to the managed Python engine.

`FRONTEND` defaults to `auto`, which picks the frontend per route and prints
the reason in the launch summary:

- `rust` when the route is `MESSAGE_TYPE=text-only` or `text+image`, uses no
speculative decoding or MTP/DFlash (including the per-request spec-decode
metrics those routes enable), has no tool calling, no `KV_DISK_CACHE_DIR`, no
scale-out endpoints, and the `vllm-rs` binary exists.
- `python` otherwise, because tool calling, SSD prefix caching, and the
scale-out endpoints are not validated on the Rust frontend yet.

`--frontend rust` forces the Rust frontend for a route the gate rejects (expert
use), and `--frontend python` forces the previous behaviour. Counting the 27
shipped profiles by running the launcher with `DRY_RUN=1`, all 27 satisfy the
`auto` gate today, so `auto` and `python` differ only for tool calling, SSD
prefix caching, and scale-out routes.

Do not count `auto` coverage from the profile files alone: the gate also sees
launcher-derived defaults such as the DFlash metrics setting.

Per-request spec-decode metrics: the response `metrics` block (with
`metrics.speculative_decoding`) is implemented for both the completions and the
chat-completions APIs, streaming and non-streaming, so the `detailed` metrics
that DFlash enables by default are no longer dropped. Measured on dual RTX 2080
Ti with Qwen3.8-27B-FP8:

- MTP route (`mtp4-fp16kv-1x148K`) at `131072` context: chat, streaming
  completions and streaming chat responses all returned a `metrics` block
  byte-identical to the Python frontend
  (`mean_acceptance_length=3.4`, `draft_acceptance_rate=0.6`,
  `acceptance_histogram=[0,2,0,2,1]`, `per_step_accepted=[1,3,1,4,3]`).
- DFlash route (`nomtp-fp16kv-1x176K` plus `SPECULATIVE_METHOD=dflash`,
  `SPECULATIVE_TOKENS=7`, `65536` context): `FRONTEND=auto` selected `rust`, the
  generated text and the `metrics` block matched the Python frontend exactly
  (`mean_acceptance_length=2.666666666666667`,
  `draft_acceptance_rate=0.23809523809523808`, `acceptance_histogram=[1,3,1,0,0,1,0,0]`,
  i.e. `num_spec_tokens + 1` buckets), and both frontends reported the same
  `GPU KV cache size` of 67,195 tokens.

The Rust frontend does not compute the per-request timing fields
(`time_to_first_token_ms`, `generation_time_ms`, `queue_time_ms`, `mean_itl_ms`,
`tokens_per_second`) yet. They are serialized as `null` in non-streaming
responses and omitted in streaming chunks, which is exactly what the Python
frontend emits for the same request shapes when its per-request timing metrics
are unavailable.

Multimodal validation context: on dual RTX 2080 Ti with Qwen3.8-27B-FP8, both
`nomtp-fp16kv-1x121K-text-image` (121,856 context) and
`mtp4-fp8kv-1x186K-text-image` with `--set MAX_MODEL_LEN=131072` plus the MTP
draft answered colour questions correctly through the Rust frontend: a 64x64
red PNG produced `Red` and the same prompt with a blue PNG produced `Blue`
(`finish_reason=stop`, `prompt_tokens=129`, so the image was expanded into
vision tokens). The launcher passes
`--limit-mm-per-prompt '{"image":64,"video":0,"audio":0}'` and the worker
logs one profiled image item; audio and video stay unvalidated.

Validation context for the speculative methods: dual RTX 2080 Ti,
Qwen3.8-27B-FP8, MTP (`mtp4-fp16kv-1x148K`) and DFlash2 draft
(`65536` context, 67,195 KV tokens) both served real completions through the
Rust frontend. The same DFlash route at `131072` context fails the engine's KV
capacity admission with the draft loaded; that is a capacity limit of the
route on this host, not a frontend difference - the Python frontend hits the
same check, and the launcher's compile-prewarm fallback and admission
diagnostics apply to both.

Route check on this repository (dual RTX 2080 Ti, Qwen3.8-27B-FP8, 1x147K FP16
KV, TP=2, warm compile cache): `--frontend rust` answered `/health` after 56 s
versus 62 s for the Python frontend, with an identical `GPU KV cache size`
(176,947 tokens), a working `/v1/completions`, and `vllm:kv_cache_usage_perc`
on `/metrics`.

## Startup Troubleshooting

If a start stalls, the launcher prints a hint when the engine log reports that
no shared-memory broadcast block became available within 60 s. That message
means a worker was stuck or extremely slow (for example a first-run compile),
so the broadcast queue was never drained.

Repeated forced kills also leave stale `/dev/shm/psm_*` segments behind (torch
shared memory). The launcher reports the count but does not delete them,
because other torch programs may own them. Clear them only while no vLLM or
torch service is running.

## Experimental SSD Prefix Cache

The launcher can opt into vLLM's built-in `OffloadingConnector` with a CPU
staging tier and a filesystem tier. This stores completed prefix KV blocks on
disk without a separate cache daemon. Keep the directory on persistent storage
and use the same model path, KV precision, block layout, and parallel settings
after restarting. The CPU staging allocation uses host RAM; it is not GPU KV
capacity. On each launch, the launcher hashes every file in the target and
local draft checkpoints before loading the model. The content digest selects
a separate disk namespace, so replacing weights at the same path cannot load
old KV. This adds one full checkpoint read to startup; old namespaces must be
removed manually to reclaim space. Clear the cache after changing to an
incompatible runtime. Disk caching requires local checkpoint directories;
do not modify checkpoint files while the launcher starts or the service runs.

The filesystem tier does not evict old files or enforce a disk quota. Use a
dedicated volume with sufficient free space, monitor its usage, and clear stale
cache data while the service is stopped. `KV_DISK_CPU_BYTES` limits only host
RAM staging. The launcher fixes `PYTHONHASHSEED=0` when disk caching is enabled
and rejects another value, so block hashes stay stable across restarts.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,2,3,4 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/mnt/nvme/vllm-kv-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

Remove `--print-config` to start the service. This option requires prefix
caching and sets a stable `PYTHONHASHSEED`. Before stopping the service, wait
for `vllm:kv_offload_tiering_active_cascade_jobs` to return to zero. Compare
`vllm:external_prefix_cache_hits_total` and the filesystem tier's
`vllm:kv_offload_tiering_chunk_hits_total` and
`vllm:kv_offload_tiering_read_bytes_total` after restarting.

On 4xT10 with a Qwen3.8-27B AWQ-INT4 checkpoint and FP16 KV, an identical
1,929-token prompt recovered 1,568 tokens from SSD after a full host reboot;
the answer matched the cold run. The FS tier reported five hits and 257 MB
read. This route remains experimental and has no published throughput claim.
On dual RTX 2080 Ti, the same model and FP16 KV recovered 1,568 of 1,933
prompt tokens after the vLLM service was stopped and restarted, with the same
answer. The dual-2080-Ti run did not include a host reboot.

The shipped `fast` profiles were also checked on dual RTX 2080 Ti (GPUs 1,5)
with the Qwen3.8-27B checkpoints. The W8A16 FP8-weight / FP8-KV MTP4 262K
text-only profile recovered 3,232 of 5,356 prompt tokens after a service
restart with 4 GiB of CPU staging. The W4A16 NVFP4-weight / FP8-KV DFlash2
262K text-and-image profile recovered 3,296 of 5,355 text prompt tokens after
a service restart with 8 GiB of CPU staging; the filesystem tier reported 21
chunk hits and 5,244,911,616 bytes read. Both returned the same `READY` answer
as their cold runs. These checks did not include a host reboot or a throughput
benchmark. On this host, the MTP4 FP16-KV 148K profile with SSD offloading
needed `GPU_UTIL=0.965` to retain its advertised 148,480-token capacity;
this is a launcher override for that host, not a change to the shipped profile.
