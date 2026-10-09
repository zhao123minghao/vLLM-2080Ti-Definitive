#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Break a launcher run log down into a startup timeline.

"Startup takes ~53 s" is measured, but not explained. The launcher run log
(`run-logs/vllm-<served-name>-<timestamp>.log`) carries two kinds of evidence:

* the engine's **own reported durations**, e.g. `init engine (profile, create kv cache, warmup
  model) took 8.51 s (compilation: 0.95 s)` and `Model loading took ... 7.42 seconds`, which are
  exact and need no guessing;
* a **wall-clock timeline**, because every log line carries `MM-DD HH:MM:SS` and the file header
  carries the launch time. Milestone rows show when each stage was first seen, and the gap table
  shows where the run was silent.

Caveats this tool states rather than hides: the timestamps have one-second resolution, and the
log is written by more than one process (the Rust frontend and the Python engine), so a "gap"
between two lines may be one process waiting on another rather than one stage being slow. The
engine's own durations are the authoritative numbers; the timeline says where to look.

Usage:
    python3 tools/startup_timeline.py                       # newest run-logs/*.log
    python3 tools/startup_timeline.py --log run-logs/x.log
    python3 tools/startup_timeline.py --top-gaps 12 --min-gap 1.5

Exit codes: 0 for a report (even when the log is short), 2 when no log can be read.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# `INFO 10-09 21:42:08 [envs.py:2334] ...`, optionally behind a `(RustFrontend pid=N)` prefix.
STAMP = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
LAUNCH_TIME = re.compile(r"^Launch time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# The engine reports its own phase costs; keep the number and the label around it.
REPORTED = re.compile(
    r"(?P<label>[^\[]*?)\s*(?:took|finished in)\s+(?P<seconds>\d+(?:\.\d+)?)\s*"
    r"(?:s\b|sec\b|secs\b|second\b|seconds\b)"
)
# A log line's message, with the process prefix and the timestamp removed.
PREFIX = re.compile(
    r"^\(?[A-Za-z_][A-Za-z0-9_]* ?[A-Za-z0-9_]* ?(?:pid=\d+)?\)? ?"
    r"(?:INFO|WARNING|ERROR|DEBUG|TRACE) \d{2}-\d{2} \d{2}:\d{2}:\d{2} "
)

# Milestones are matched in the order they happen; the first match wins. Patterns are kept
# deliberately specific: the launcher's own TUI stage heuristic matches loose words like
# `init`, which is fine for a progress bar but useless as a timestamp.
MILESTONES: list[tuple[str, str]] = [
    ("frontend up", r"process\.rs:.*starting managed Python headless engine"),
    ("tokenizer loaded", r"\[hf(/mod)?\.rs:"),
    ("first engine line", r"\[(core|envs|config|arg_utils|registry)\.py:"),
    ("engine init", r"Initializing a V1 LLM engine"),
    ("weights loading", r"Loading safetensors checkpoint shards|default_loader\.py:.*Loading weights"),
    ("weights loaded", r"Model loading took"),
    ("compile path", r"torch\.compile took|Directly load AOT compilation"),
    ("profiling/warmup", r"Initial profiling/warmup run took"),
    ("graph capture", r"Graph capturing finished"),
    ("engine ready", r"init engine \(profile, create kv cache, warmup model\) took"),
]


@dataclass
class Entry:
    at: datetime
    offset: float
    message: str


def read_entries(path: Path) -> tuple[datetime, list[Entry]]:
    """Return the launch time and every timestamped line with its offset in seconds."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    launch: datetime | None = None
    for line in text[:40]:
        match = LAUNCH_TIME.match(line)
        if match:
            launch = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            break

    entries: list[Entry] = []
    for line in text:
        match = STAMP.search(line[:80])
        if not match:
            continue
        month, day, hour, minute, second = (int(group) for group in match.groups())
        at = datetime(launch.year if launch else 2000, month, day, hour, minute, second)
        entries.append(Entry(at=at, offset=0.0, message=PREFIX.sub("", line.strip())))
    if launch is None and entries:
        launch = entries[0].at
    for entry in entries:
        entry.offset = (entry.at - launch).total_seconds() if launch else 0.0
    return (launch or datetime.now()), entries


def first_matches(entries: list[Entry]) -> list[tuple[str, Entry | None]]:
    found: list[tuple[str, Entry | None]] = []
    for label, pattern in MILESTONES:
        regex = re.compile(pattern)
        found.append((label, next((e for e in entries if regex.search(e.message)), None)))
    return found


def reported_durations(entries: list[Entry]) -> list[tuple[str, float, int, Entry]]:
    """The engine's own `took N seconds` lines, collapsed per phase.

    Both workers of a data-parallel run log their own copy, so a phase can appear several
    times; the slowest occurrence is kept (that is the one that gates startup) together with
    how many times the phase was reported.
    """
    grouped: dict[str, tuple[float, int, Entry]] = {}
    for entry in entries:
        match = REPORTED.search(entry.message)
        if not match:
            continue
        label = match.group("label").strip(" ,;:-[]")
        if not label:
            continue
        # Drop the `module.py:123]` part the logger prepends.
        label = re.sub(r"^[A-Za-z0-9_./-]+\.py:\d+\]\s*", "", label)
        label = label.strip() or "(unnamed)"
        seconds = float(match.group("seconds"))
        best = grouped.get(label)
        if best is None or seconds > best[0]:
            grouped[label] = (seconds, 1 if best is None else best[1] + 1, entry)
        else:
            grouped[label] = (best[0], best[1] + 1, best[2])
    return sorted(
        ((label, seconds, count, entry) for label, (seconds, count, entry) in grouped.items()),
        key=lambda row: row[3].offset,
    )


def gap_rows(entries: list[Entry], min_gap: float, top: int) -> list[tuple[float, Entry]]:
    rows: list[tuple[float, Entry]] = []
    previous = entries[0].at if entries else None
    for entry in entries[1:]:
        assert previous is not None
        gap = (entry.at - previous).total_seconds()
        if gap >= min_gap:
            rows.append((gap, entry))
        previous = entry.at
    # Sort on the gap only: two gaps of the same size would otherwise compare Entry objects.
    return sorted(rows, key=lambda row: row[0], reverse=True)[:top]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, help="launcher run log (default: the newest one)")
    parser.add_argument("--top-gaps", type=int, default=10, help="how many gaps to print")
    parser.add_argument(
        "--min-gap",
        type=float,
        default=2.0,
        help="only report gaps of at least this many seconds (default 2)",
    )
    args = parser.parse_args()

    path = args.log
    if path is None:
        candidates = glob.glob("run-logs/*.log")
        if not candidates:
            print("[timeline] no run log found; pass --log or start the service first", file=sys.stderr)
            return 2
        path = Path(max(candidates, key=os.path.getmtime))
    if not path.exists():
        print(f"[timeline] {path} does not exist", file=sys.stderr)
        return 2

    launch, entries = read_entries(path)
    if not entries:
        print(f"[timeline] {path} has no timestamped lines", file=sys.stderr)
        return 2

    print(f"log:      {path}")
    print(f"launch:   {launch:%Y-%m-%d %H:%M:%S}")
    raw_lines = len(path.read_text(encoding='utf-8', errors='replace').splitlines())
    print(
        f"lines:    {len(entries)} timestamped of {raw_lines} "
        f"({raw_lines - len(entries)} without a timestamp, e.g. weight-loading progress)"
    )
    print("note:     one-second resolution, and two processes write this log; the engine's own")
    print("          durations below are exact, the timeline shows where to look")

    durations = reported_durations(entries)
    if durations:
        print("\nreported by the engine (slowest occurrence, which is what gates startup):")
        for label, seconds, count, entry in durations:
            suffix = f"  (x{count})" if count > 1 else ""
            print(f"  +{entry.offset:6.0f}s  {seconds:7.2f}s  {label}{suffix}")

    print("\nstartup markers logged by the engine (`startup:` lines):")
    markers = [entry for entry in entries if entry.message.lstrip("[").split("] ")[-1].startswith("startup:")]
    if not markers:
        print("  none (this log predates the startup instrumentation, or the run is not a fork run)")
    previous = None
    for entry in markers:
        text = entry.message.split("startup:", 1)[1].strip()
        delta = "" if previous is None else f"  (+{entry.offset - previous:.2f}s)"
        print(f"  +{entry.offset:6.2f}s  {text}{delta}")
        previous = entry.offset

    print("\nmilestones (first line seen):")
    previous_offset = 0.0
    reached = 0
    for label, entry in first_matches(entries):
        if entry is None:
            print(f"  {label:<20}          -  (not in this log)")
            continue
        reached += 1
        print(f"  {label:<20} +{entry.offset:6.0f}s  +{entry.offset - previous_offset:5.0f}s  {entry.message[:70]}")
        previous_offset = entry.offset

    engine_ready = next((e for label, e in first_matches(entries) if label == "engine ready"), None)
    total = engine_ready.offset if engine_ready else entries[-1].offset
    first_engine = next(
        (e for label, e in first_matches(entries) if label == "first engine line"), None
    )
    print("\nphases without an engine-side duration:")
    if first_engine is not None:
        print(
            f"  +0s..+{first_engine.offset:.0f}s  frontend spawn: the Rust frontend starts, loads"
            " the tokenizer and chat template, then execs the Python engine"
        )
        engine_init = next((e for label, e in first_matches(entries) if label == "engine init"), None)
        if engine_init is not None:
            window = engine_init.offset - first_engine.offset
            print(
                f"  +{first_engine.offset:.0f}s..+{engine_init.offset:.0f}s  imports and engine"
                f" construction before the engine reports its own timings ({window:.0f}s; this is"
                " where the lazy-import work lands)"
            )
    if engine_ready is not None:
        print(
            f"  +{engine_ready.offset:.0f}s        engine ready (the launcher's own"
            " `Startup complete` is a little later: health check plus frontend handshake, and it"
            " is printed to the terminal, not logged)"
        )

    rows = gap_rows(entries, args.min_gap, args.top_gaps)
    if rows:
        print(f"\nlargest gaps between log lines (>= {args.min_gap:g}s):")
        for gap, entry in rows:
            print(f"  {gap:6.0f}s ending at +{entry.offset:6.0f}s  {entry.message[:70]}")

    print(f"\nstartup_timeline_ok log={path} milestones={reached} reported={len(durations)} total={total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
