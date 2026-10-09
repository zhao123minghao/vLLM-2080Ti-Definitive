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

## Measured on the reference route

`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`, `--mode fast`, 2×2080 Ti, warm AOT
cache (`~/.cache/vllm/torch_compile_cache` present):

| milestone | at | delta |
| --- | --- | --- |
| frontend up (Rust frontend starts, loads tokenizer and chat template) | +0 s | |
| first Python engine line (`envs.py` warning about an unknown vLLM env var) | +9 s | 9 s |
| `Initializing a V1 LLM engine` (`core.py:123`) | +18 s | 9 s |
| first `Loading weights took` (`default_loader.py:430`) | +40 s | 22 s |
| `Model loading took` (`model_runner.py:431`) | +42 s | 2 s |
| AOT compile artifacts reused (`decorators.py:313`) | +43 s | 1 s |
| `Initial profiling/warmup run took` (`monitor.py`) | +45 s | 2 s |
| `Graph capturing finished` (`model_runner.py:1088`) | +47 s | 2 s |
| `init engine (profile, create kv cache, warmup model) took` (`core.py:372`) | +50 s | 3 s |

What the engine measured for itself in that run:

| engine-reported duration | time |
| --- | --- |
| `init engine (profile, create kv cache, warmup model)` (compilation: 0.95 s) | 8.51 s |
| `Model loading` (14.51 GiB) | 7.42 s |
| `Loading weights` | 3.44 s |
| `Initial profiling/warmup run` | 1.38 s |
| `Graph capturing finished` (×4) | 1.00 s |
| `torch.compile` (AOT cache hit) | 0.85 s |
| `JIT kernel warmup` | 0.00 s |

The honest reading of that table: **the work the engine instruments is not where the 53 s
goes.** Model loading plus engine init is ~16 s; the first ~18 s belong to the frontend spawn
and the Python import path before the engine even reports itself, and the 22 s window between
`core.py:123` and the first weight-loading line is mostly distributed setup, KV cache
allocation and worker coordination, none of which the engine times. That is also why the
lazy-import work measured earlier does not show up as a startup win: it moves the 9 s import
phase, not the 22 s coordination phase.

To go further, the next step is instrumentation rather than more profiling of the log:
timestamp the KV cache allocation, the distributed init and the worker barrier in the engine,
and the frontend's own sub-phase timings. `tools/startup_timeline.py` will pick those lines up
with no changes.

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
