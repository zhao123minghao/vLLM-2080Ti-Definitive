# Where the ~53 s startup goes (`tools/startup_timeline.py`)

Startup time is the number this project gets asked about most, and "53 s" is a measurement
without an explanation. The launcher run log already contains both halves of the answer, so
`tools/startup_timeline.py` turns it into a report:

```bash
python3 tools/startup_timeline.py                  # newest run-logs/*.log
python3 tools/startup_timeline.py --log run-logs/vllm-<...>.log --top-gaps 12 --min-gap 1.5
```

Every tool that starts the service writes such a log (`tools/chat_smoke.sh --launch`,
`tools/validate_all.sh --with-chat`, `tools/dep_audit.sh`, or a plain `./launcher.sh`), so there
is nothing extra to run: start the route once, then read the log back.

## What it reports

* **What the engine measured itself.** Lines like
  `init engine (profile, create kv cache, warmup model) took 8.51 s (compilation: 0.95 s)` are
  exact; the report keeps the slowest occurrence of each phase (in a data-parallel run every
  worker logs its own copy) and how many times it appeared.
* **A wall-clock timeline.** Every log line carries `MM-DD HH:MM:SS` and the file header carries
  `Launch time:`, so the report shows when each stage was first seen and the delta from the
  previous one.
* **The gaps in between.** Phases the engine does not instrument show up as silence; the report
  lists the largest gaps so they can be investigated instead of guessed at.
* **The `startup:` markers this fork logs.** The worker bring-up used to be exactly such a gap, so
  the executor now logs one line per worker (spawn, entry, `init_worker`, `init_device`,
  `load_model`, ready) plus the KV cache phase timings, each relative to the moment the parent
  forked that worker. `CLOCK_MONOTONIC` is system-wide on Linux, which is what makes a timestamp
  taken before `fork()` comparable inside the child.

## Measured on the reference route

`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`, `--mode fast`, 2×2080 Ti, warm AOT
cache (`~/.cache/vllm/torch_compile_cache` present). Launcher total: **51 s**.

| phase | window | notes |
| --- | --- | --- |
| Rust frontend start, spawn of the Python engine, first Python log line | 0 – 9 s | frontend only; the engine has not reported itself yet |
| Python imports and engine construction | 9 – 17 s | ends at `Initializing a V1 LLM engine` (`core.py:123`) |
| engine core init: worker bring-up, model load | 17 – 41 s | see the markers below |
| memory profile, KV cache config, warmup, graph capture | 41 – 49 s | profile run 5.75 s, allocation 0.03 s |
| engine ready → launcher ready (health check, handshake) | 49 – 51 s | the launcher's number is terminal-only |

Within that third window, the `startup:` markers this fork logs show what the log used to hide:

| marker | offset | delta |
| --- | --- | --- |
| worker rank=0 spawned | +24.0 s | |
| rank=0 entered `worker_main` | +24.0 s | 0.3 s after its spawn |
| rank=0 `init_worker` (model module import) done | +25.0 s | 1.2 s |
| worker rank=1 spawned | +31.0 s | **6.7 s after rank 0 was spawned** |
| rank=1 entered `worker_main` | +31.0 s | **7.0 s after its spawn** |
| rank=0 / rank=1 `init_device` done | +34.0 s | 8.4 s / 1.7 s |
| rank=0 / rank=1 `load_model` done | +41.0 s | 6.9 s / 6.9 s |
| all 2 workers ready | +41.0 s | 9.98 s since the first spawn |
| available-memory profile run | +46.0 s | 5.75 s |
| KV cache config + allocation | +46.0 s | 0.03 s |

What the engine measured for itself in that run:

| engine-reported duration | time |
| --- | --- |
| `init engine (profile, create kv cache, warmup model)` | 8.52 s |
| `Model loading` (14.51 GiB) | 6.84 s |
| `Loading weights` | 3.32 s |
| `Initial profiling/warmup run` | 1.38 s |
| `Graph capturing finished` (×4) | 1.00 s |
| `torch.compile` (AOT cache hit) | 0.87 s |
| `JIT kernel warmup` | 0.00 s |

## What the numbers say

* **The engine's own init is not the bulk.** Model loading 6.8 s plus `init engine` 8.5 s plus the
  memory profile 5.8 s account for ~21 s of 51 s; the rest is process bring-up.
* **The two workers come up one after another, and the second one pays for the first one's work.**
  Rank 0 entered `worker_main` 0.3 s after being spawned, rank 1 took 7.0 s — and rank 1 was only
  forked 6.7 s after rank 0, while rank 0 was already running its CUDA/distributed init and
  loading weights. This is contention, not a fixed fork cost; the lever is either to bring the
  workers up before the heavy per-worker init starts, or to use a start method that does not copy
  the parent's page tables (`VLLM_WORKER_MULTIPROC_METHOD`).
* **KV cache allocation is free.** Config plus allocation is 0.03 s; the 5.75 s next to it is the
  *memory profile run* (forward passes to measure peak usage), which is a different thing.
* **~17 s happen before the engine core exists**: 9 s in the frontend plus interpreter start, 8 s
  of Python imports and engine construction. This is the window the lazy-import work targets, and
  it is a fifth of the total — not the half it feels like when watching the TUI.

None of this is visible without the markers, which is why they are in the tree: the phases above
are the ones the log used to skip from `Initializing a V1 LLM engine` straight to the first
weight-loading line.

## Caveats this report states explicitly

* Timestamps have **one-second resolution**, so small deltas are noise.
* **Two processes write the log** (the Rust frontend and the Python engine), so a gap between
  two lines can be one process waiting for the other rather than one stage being slow.
* Progress lines such as `Loading safetensors checkpoint shards: 9% ...` have **no timestamp**
  at all; a phase dominated by them is only bracketed by its neighbouring lines.
* The launcher's own `Startup complete after Ns.` is printed to the terminal and never logged,
  so it is not in this report. In the run above it was 54 s against the engine's +50 s: the rest
  is the health check and the frontend handshake.

## Comparing a cold start

Move the AOT and Triton caches aside together (they are coupled: moving only `triton-cache/`
fails with `Cubin file saved by TritonBundler not found`), run the route, and read the new log
with the same command. The difference shows up as a much larger `torch.compile` row plus a
slower `init engine`, which is how the 106 s cold start in the dependency-audit document breaks
down.
