#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that a measured dependency set still covers what a route loads.

Input is the report written by `tools/dep_audit/analyze.py` (see `tools/dep_audit.sh`)
plus a pinned requirement file such as `requirements/definitive-2080ti-measured.txt`.

The check is deliberately asymmetric:

* a distribution that the run **loaded** but the file does not list is an **error** — the
  set is no longer sufficient, so a fresh install from it would be missing something;
* a distribution that the file lists but the run never loaded is a **warning** — the entry
  may have become stale after a code change;
* a distribution that is installed but is neither listed nor loaded is **information** —
  it is a pruning candidate, with its size.

Usage:
    python3 tools/dep_audit/check_minimal_set.py \
        --report /tmp/dep-audit/report.json \
        --requirements requirements/definitive-2080ti-measured.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)")


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def load_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match is None:
            continue
        pins[_norm(match.group(1))] = match.group(2)
    return pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("requirements/definitive-2080ti-measured.txt"),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    pins = load_pins(args.requirements)

    loaded = {_norm(d["name"]): d for d in report.get("loaded_dists", [])}
    installed = {
        _norm(d["name"]): d
        for d in report.get("loaded_dists", []) + report.get("unloaded_dists", [])
    }

    missing = sorted(name for name in loaded if name not in pins and name != "vllm")
    stale = sorted(name for name in pins if name not in loaded)
    prunable = sorted(
        (
            (installed[name]["size"], installed[name]["name"])
            for name in installed
            if name not in pins and name not in loaded
        ),
        reverse=True,
    )

    mb = 1024 * 1024
    print(f"requirements: {args.requirements} ({len(pins)} pins)")
    print(
        f"audited run : {report.get('loaded')} loaded / {report.get('installed')} installed"
        f" ({report.get('loaded_bytes', 0) / mb:.0f} MiB loaded)"
    )

    if missing:
        print(f"\nERROR: loaded but not listed ({len(missing)}):")
        for name in missing:
            print(f"  {installed[name]['name']}=={installed[name]['version']}")
    else:
        print("\nOK: every loaded distribution is covered by the requirement file")

    if stale:
        print(f"\nWARNING: listed but never loaded ({len(stale)}), possibly stale entries:")
        for name in stale:
            print(f"  {name}=={pins[name]}")

    if prunable:
        total = sum(size for size, _ in prunable)
        print(f"\nINFO: installed but neither listed nor loaded ({len(prunable)}, "
              f"{total / mb:.1f} MiB) — pruning candidates:")
        for size, name in prunable[:20]:
            print(f"  {size / mb:8.2f} MiB  {name}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
