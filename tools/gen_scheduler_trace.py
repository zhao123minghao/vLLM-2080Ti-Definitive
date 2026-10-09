#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record what the real vLLM `Scheduler` decides for a synthetic workload.

Step 3 of the `v1/core` migration spike (see `docs/v1-core-rust-migration.md`) is the scheduler:
`vllm/v1/core/sched/scheduler.py` decides every step which requests run, how many tokens they
get, which request is preempted and how many blocks the pool hands out. A port of that logic has
to reproduce those decisions exactly, so this tool drives the **real** scheduler on a seeded
workload and records both halves:

* `trace.txt`    — the workload the scheduler was fed: one line per request
                   (`request <id> <prompt_tokens> <max_tokens>`) and per model reply
                   (`step <id>=<tokens> ...`).
* `expected.txt` — what the scheduler decided: one line per scheduling step (first-time versus
                   cached requests, tokens per request, preemptions, common-prefix blocks, the
                   pool's free-block count) and one per applied model output (finished requests).

This is the *skeleton* of the shadow, not the comparison: the decision logic is not ported yet, so
nothing re-derives these lines today. What the tool does deliver is the recorded protocol plus
the invariant checks below, which any implementation has to satisfy — a trace is only written
when every check passed. The next step is a Rust reader for `trace.txt` that has to produce
`expected.txt` byte for byte.

Usage:
    python3 tools/gen_scheduler_trace.py --out-dir /tmp/scheduler-trace
    python3 tools/gen_scheduler_trace.py --out-dir /tmp/scheduler-trace \\
        --num-requests 16 --num-blocks 48 --max-num-seqs 4 --prefix-caching 1 --seed 3
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# The scheduler logs every decision it makes; the tool's own output has to stay readable.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

# `tests/v1/core/utils.py` holds the repository's own scheduler fixture and is imported from here
# rather than re-implemented, so the workload runs against the same setup the CPU tests use. The
# repository root has to be importable for that, even when the tool is started by path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

# The scheduler needs a model *config*, but a real one: everything except the path (architecture,
# max length) comes from it. A local directory is therefore enough and no weights are loaded.
DEFAULT_MODEL = "/data/models/qwen3.8-27b/Qwen3.8-27B-FP8"

# What the synthetic model runner reports for every scheduled request. Fixed, because this tool
# records scheduling decisions rather than sampling behaviour.
GENERATED_TOKEN = 1000

# The invariant families a recorded decision has to satisfy. A trace is only written when all of
# them pass, so what lands in `--out-dir` is a self-consistent recording; they are also the
# contract a port of the decision logic is compared against first.
INVARIANTS = (
    "total-matches-sum",
    "scheduled-matches-tokens",
    "no-duplicate-request",
    "positive-token-count",
    "bounded-token-count",
    "free-blocks-in-range",
    "finished-once",
    "queue-drains",
    "finished-count",
    "blocks-released",
)


@dataclass(frozen=True)
class WorkloadConfig:
    """The workload and the scheduler configuration it runs against."""

    num_requests: int
    num_blocks: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    prefix_caching: bool
    max_steps: int
    seed: int
    model: str


def build_scheduler(config: WorkloadConfig):
    """Build the real scheduler, seeded so the workload stays reproducible."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    return create_scheduler(
        model=config.model,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=config.prefix_caching,
        num_blocks=config.num_blocks,
        block_size=config.block_size,
        max_model_len=config.max_num_batched_tokens,
    )


def build_requests(config: WorkloadConfig, rng: random.Random) -> list[Request]:
    """The workload: prompt lengths around the block size, some prompts shared.

    Prompts land either side of a block boundary so chunked prefill produces multi-step prefills
    and partial blocks, `max_tokens` of 1 makes requests finish immediately, and the shared-prompt
    groups are what give the prefix cache something to hit. The pool is small relative to the
    concurrency, which is what makes the scheduler preempt.
    """
    requests: list[Request] = []
    while len(requests) < config.num_requests:
        batch = min(rng.randint(1, 3), config.num_requests - len(requests))
        index = len(requests)
        requests.extend(
            create_requests(
                batch,
                num_tokens=rng.choice([16, 32, 33, 48, 64]),
                max_tokens=rng.choice([1, 2, 3, 4, 8]),
                block_size=config.block_size,
                same_prompt=rng.random() < 0.5,
                req_ids=[f"r{index + offset}" for offset in range(batch)],
            )
        )
    return requests


def joined(values) -> str:
    """`,` separated, or `-` for nothing: the shape the decision log uses everywhere."""
    rendered = ",".join(str(value) for value in values)
    return rendered if rendered else "-"


def trace_header(requests: list[Request]) -> list[str]:
    return [
        f"request {request.request_id} {len(request.prompt_token_ids or [])} {request.max_tokens}"
        for request in requests
    ]


def check_step(
    scheduler,
    scheduler_output,
    num_blocks: int,
    computed_before: dict[str, int],
    finished_before: set[str],
    violations: list[str],
) -> None:
    """Check one recorded decision against the invariants above."""
    counts = scheduler_output.num_scheduled_tokens
    new_ids = [request.req_id for request in scheduler_output.scheduled_new_reqs]
    cached = scheduler_output.scheduled_cached_reqs
    scheduled = [*new_ids, *cached.req_ids]

    if scheduler_output.total_num_scheduled_tokens != sum(counts.values()):
        violations.append(
            "total-matches-sum: total="
            f"{scheduler_output.total_num_scheduled_tokens} sum={sum(counts.values())}"
        )
    if sorted(scheduled) != sorted(counts):
        violations.append(f"scheduled-matches-tokens: {joined(scheduled)} vs {joined(counts)}")
    if len(set(scheduled)) != len(scheduled):
        violations.append(f"no-duplicate-request: {joined(scheduled)}")
    for request_id, count in counts.items():
        if count < 1:
            violations.append(f"positive-token-count: {request_id}={count}")
            continue
        request = scheduler.requests.get(request_id)
        if request is None:
            violations.append(f"scheduled-matches-tokens: unknown request {request_id}")
            continue
        # A step never schedules more tokens than the request still has to compute: prefill is
        # chunked, not over-scheduled. `schedule()` advances `num_computed_tokens` itself, so the
        # count from before the call is what the decision has to fit into.
        remaining = request.num_tokens - computed_before.get(
            request_id, request.num_computed_tokens
        )
        if count > remaining:
            violations.append(f"bounded-token-count: {request_id}={count} > {remaining}")
        if request_id in finished_before:
            violations.append(f"finished-once: {request_id} scheduled after finishing")

    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    if not 0 <= free <= num_blocks:
        violations.append(f"free-blocks-in-range: free={free} of {num_blocks}")


def record_step(
    scheduler,
    config: WorkloadConfig,
    step: int,
    trace: list[str],
    log: list[str],
    violations: list[str],
    finished_before: set[str],
) -> bool:
    """Run one scheduling step and record it. Returns whether the scheduler is still busy."""
    computed_before = {
        request_id: request.num_computed_tokens
        for request_id, request in scheduler.requests.items()
    }
    scheduler_output = scheduler.schedule()
    if scheduler_output.total_num_scheduled_tokens == 0:
        # Nothing to schedule: the queue is empty.
        return False

    check_step(
        scheduler,
        scheduler_output,
        config.num_blocks,
        computed_before,
        finished_before,
        violations,
    )
    cached = scheduler_output.scheduled_cached_reqs
    log.append(
        "schedule {} -> ok new={} cached={} resumed={} tokens={} total={} preempted={} "
        "prefix_blocks={} free={}".format(
            step,
            joined(request.req_id for request in scheduler_output.scheduled_new_reqs),
            joined(cached.req_ids),
            joined(sorted(cached.resumed_req_ids)),
            joined(
                f"{request_id}:{count}"
                for request_id, count in sorted(scheduler_output.num_scheduled_tokens.items())
            ),
            scheduler_output.total_num_scheduled_tokens,
            joined(sorted(scheduler_output.preempted_req_ids or ())),
            joined(scheduler_output.num_common_prefix_blocks),
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        )
    )

    # The reply the model runner would have produced: one token per scheduled request, with a
    # fixed value, so what the log reports is the scheduler's reaction and not a sampling choice.
    request_ids = sorted(scheduler_output.num_scheduled_tokens)
    trace.append("step " + " ".join(f"{request_id}=1" for request_id in request_ids))
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={request_id: order for order, request_id in enumerate(request_ids)},
            sampled_token_ids=[[GENERATED_TOKEN] for _ in request_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    finished = sorted(scheduler.finished_req_ids - finished_before)
    finished_before.update(finished)
    log.append(
        "update {} -> ok finished={} waiting={} running={} free={}".format(
            step,
            joined(finished),
            joined(sorted(request.request_id for request in scheduler.waiting)),
            joined(request.request_id for request in scheduler.running),
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        )
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--num-blocks", type=int, default=48)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefix-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_requests < 1:
        parser.error("--num-requests must be at least 1")
    if args.num_blocks < 4:
        parser.error("--num-blocks must be at least 4 (one block is the null block)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = WorkloadConfig(
        num_requests=args.num_requests,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        prefix_caching=bool(args.prefix_caching),
        max_steps=args.max_steps,
        seed=args.seed,
        model=args.model,
    )

    rng = random.Random(config.seed)
    scheduler = build_scheduler(config)
    requests = build_requests(config, rng)
    trace = trace_header(requests)
    log: list[str] = []
    violations: list[str] = []
    finished_before: set[str] = set()

    for request in requests:
        scheduler.add_request(request)

    steps = 0
    while steps < config.max_steps and record_step(
        scheduler, config, steps, trace, log, violations, finished_before
    ):
        steps += 1

    # The workload is done when the queue is empty: every request finished and every block went
    # back to the pool. Cached blocks stay *in* the free list, so the count returns to the null
    # block alone even with the prefix cache enabled.
    unfinished = sorted(
        request.request_id for request in [*scheduler.running, *scheduler.waiting]
    )
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    if unfinished:
        violations.append(f"queue-drains: still queued {joined(unfinished)}")
    # `scheduler.finished_req_ids` only holds the ids of the current step (it is cleared every
    # step), so the accumulated set is what covers the whole workload.
    if len(finished_before) != len(requests):
        violations.append(f"finished-count: {len(finished_before)} of {len(requests)}")
    if free != config.num_blocks - 1:
        violations.append(f"blocks-released: free={free}, expected {config.num_blocks - 1}")

    (args.out_dir / "trace.txt").write_text("\n".join(trace) + "\n")
    (args.out_dir / "expected.txt").write_text("\n".join(log) + "\n")

    if violations:
        print(f"scheduler_shadow_failed violations={len(violations)}")
        for violation in violations[:10]:
            print(f"  {violation}")
        return 1
    print(f"requests: {len(requests)}")
    print(f"steps: {steps}")
    print(f"checks: {len(INVARIANTS)} invariant families, 0 violations")
    print(f"scheduler_shadow_ok steps={steps} requests={len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
