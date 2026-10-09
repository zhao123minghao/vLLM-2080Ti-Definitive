# Scheduler shadow: the skeleton

The scheduler is step 3 of the `v1/core` migration spike. It is the largest piece of the port
(`vllm/v1/core/sched/scheduler.py` is 3,474 lines) and the only one whose correctness cannot be
argued from state alone: it decides *when* things happen. That is why it has to run as a shadow —
record the decisions the Python scheduler makes, and compare them against the port — before
anything is cut over.

This document records the skeleton of that shadow: the recorder, the trace format, and the
invariants a recorded decision has to satisfy. The decision logic itself is **not** ported yet, so
there is no comparison to run; what exists is the oracle and the contract.

## What is recorded

`tools/gen_scheduler_trace.py` drives the **real** `Scheduler` (through the repository's own
fixtures in `tests/v1/core/utils.py`, so the setup is the one the CPU tests use) over a seeded
synthetic workload and writes two files:

| file           | holds                                                                            |
| -------------- | -------------------------------------------------------------------------------- |
| `trace.txt`    | the workload and the model replies the scheduler was fed                          |
| `expected.txt` | the decisions the scheduler produced, one line per scheduling step and per reply   |

`trace.txt` is what a port has to be given:

```
request r0 48 1
request r1 48 1
...
step r0=1 r1=1 r2=1
step r2=1 r3=1 r4=1
```

Each `request` line is a request the scheduler accepted (prompt length and `max_tokens`); each
`step` line is one synthetic model reply, listing the requests the model runner reported tokens
for. Nothing about *what the scheduler should do* is in this file.

`expected.txt` is what the port has to produce:

```
schedule 0 -> ok new=r0,r1,r2 cached=- resumed=- tokens=r0:48,r1:16,r2:16 total=80 preempted=- prefix_blocks=2 free=18
update 0 -> ok finished=r0,r1 waiting=r3,r4,r5,r6,r7 running=r2 free=20
schedule 1 -> ok new=r3,r4 cached=r2 resumed=- tokens=r2:1,r3:48,r4:48 total=97 preempted=- prefix_blocks=0 free=13
```

A `schedule` line is the `SchedulerOutput` as far as a port has to reproduce it: which requests
were scheduled for the first time and which were cached (and which of those were resumed after a
preemption), how many tokens each request got, the total, the preemptions, the common-prefix
blocks each KV-cache group shares, and the pool's free-block count after the decision. An
`update` line is the scheduler's reaction to the model reply: what finished, and the waiting and
running queues afterwards.

The workload is chosen to reach the edges rather than the middle: prompt lengths land either side
of a block boundary (so chunked prefill produces multi-step prefills and partially filled blocks),
`max_tokens` of 1 makes requests finish immediately, half the prompt groups are shared (so the
prefix cache hits), and the driver runs a pool too small for the batch as well as a roomy one (so
the scheduler has to preempt and then resume).

## The invariants

A trace is only written when every decision in it passed these checks. They are what a port is
compared against first — a port that gets the decisions wrong usually breaks one of them, and it
can be told why.

| check                     | what it rules out                                                     |
| ------------------------- | --------------------------------------------------------------------- |
| `total-matches-sum`       | a total that disagrees with the per-request token counts               |
| `scheduled-matches-tokens`| scheduling a request without asking for tokens, or the reverse         |
| `no-duplicate-request`    | the same request twice in one step                                     |
| `positive-token-count`    | an empty scheduling entry                                              |
| `bounded-token-count`     | scheduling more tokens than the request still has to compute           |
| `free-blocks-in-range`    | a free-block count outside `[0, num_blocks]`                           |
| `finished-once`           | a finished request being scheduled again                               |
| `queue-drains`            | the workload ending with requests still queued                         |
| `finished-count`          | requests disappearing without finishing                                |
| `blocks-released`         | blocks leaked: the pool must end with only the null block used         |

The last one is the strongest: whatever the scheduler decides, every block it took has to be back
in the pool when the workload is over, with the prefix cache enabled and evictions in the middle
of the run. `bounded-token-count` is checked against the state *before* `schedule()` was called,
because `schedule()` advances `num_computed_tokens` itself.

## Running it

```bash
bash tools/check_scheduler_shadow.sh                     # 3 seeds × 3 configurations
bash tools/check_scheduler_shadow.sh --seeds 5 --num-requests 16
SCHED_SHADOW_MODEL=/path/to/model bash tools/check_scheduler_shadow.sh
```

Verified with 3 seeds × 3 configurations (**9 runs, 150 scheduling steps, 90 requests, no
invariant violations**), including 136 steps with cached requests, 3 steps with preemptions
(and 3 with requests resumed afterwards) and 6 steps with common prefix blocks. The sweeps run on
CPU and never touch a GPU: the scheduler is driven directly, no engine and no weights.

## What this is not

* **Not a comparison.** Nothing re-derives `expected.txt`. The next step is a Rust reader for
  `trace.txt` that has to reproduce the file byte for byte, using the already shadow-verified
  block pool (`rust/src/kv-pool`) for the block bookkeeping and the same invariant checks as a
  first line of defence.
* **Not a throughput or latency measurement.** The workload is synthetic and the model runner is
  a stub that returns one fixed token per request; sampling, logprobs, structured output and
  speculative decoding are all out of scope.
* **Not a prefix-cache benchmark.** The shared prompts are there to make the scheduler take its
  cache-hit and common-prefix paths, not to measure cache effectiveness.

## Environment

The scheduler needs a real model *config* directory: architecture and `max_model_len` come from
it, no weights are loaded and no GPU is used. It defaults to the reference route's model
(`/data/models/qwen3.8-27b/Qwen3.8-27B-FP8`); `--model` or `SCHED_SHADOW_MODEL` overrides it, and
`tools/validate_all.sh --with-scheduler` skips the step when the directory is missing rather than
failing.
