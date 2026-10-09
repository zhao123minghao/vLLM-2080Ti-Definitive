# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#!/usr/bin/env python3
"""Classify installed distributions as truly loaded / not loaded by a real run.

Signals per process snapshot in IMPORT_AUDIT_DIR:
  mods-<pid>.txt : effective `sys.modules` top-level names (successful imports only)
  so-<pid>.txt   : shared objects mapped in /proc/self/maps (catches dlopen'd CUDA libs)

A distribution counts as loaded if either a module it provides was imported, or one of
its own files (e.g. nvidia/cublas/lib/libcublas.so.13) was mapped. Paths are compared
after realpath() because the venv is a symlink farm.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from importlib.metadata import distributions, packages_distributions
from pathlib import Path

AUDIT = Path(os.environ.get("AUDIT_DIR", "/tmp/depaudit2/so"))
OUT = Path(os.environ.get("AUDIT_REPORT", "/tmp/depaudit2/report.json"))

modules: set[str] = set()
mapped: set[str] = set()
per_process: dict[str, dict[str, object]] = {}

for path in sorted(AUDIT.glob("mods-*.txt")):
    pid = path.stem.split("-")[1]
    text = path.read_text().splitlines()
    argv = next((ln[7:] for ln in text if ln.startswith("# argv=")), "")
    mods = {ln.strip() for ln in text if ln.strip() and not ln.startswith("#")}
    modules |= mods
    per_process.setdefault(pid, {})["modules"] = len(mods)
    per_process[pid]["argv"] = argv[:120]

for path in sorted(AUDIT.glob("so-*.txt")):
    pid = path.stem.split("-")[1]
    libs = {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}
    mapped |= {os.path.realpath(lib) for lib in libs}
    per_process.setdefault(pid, {})["libs"] = len(libs)

pkg2dist: dict[str, list[str]] = defaultdict(list)
for module, dists in packages_distributions().items():
    pkg2dist[module].extend(dists)

imported_dists = set()
for module in modules:
    for dist in pkg2dist.get(module, []):
        imported_dists.add(dist.lower())


def size_of(dist) -> int:
    total = 0
    for file in dist.files or []:
        try:
            path = dist.locate_file(file)
        except Exception:
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def mapped_files(dist) -> int:
    hits = 0
    for file in dist.files or []:
        if not str(file).endswith((".so", ".so.1")) and ".so." not in str(file):
            continue
        try:
            path = os.path.realpath(dist.locate_file(file))
        except Exception:
            continue
        if path in mapped:
            hits += 1
    return hits


rows = []
for dist in distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    used_module = name.lower() in imported_dists
    so_hits = mapped_files(dist)
    rows.append(
        {
            "name": name,
            "version": dist.version,
            "loaded": used_module or so_hits > 0,
            "by_module": used_module,
            "so_files_mapped": so_hits,
            "size": size_of(dist),
        }
    )

loaded = [r for r in rows if r["loaded"]]
unloaded = sorted((r for r in rows if not r["loaded"]), key=lambda r: -r["size"])
mb = 1024 * 1024
report = {
    "processes": len(per_process),
    "top_level_modules": sorted(m for m in modules if m not in sys.stdlib_module_names),
    "system_cuda_libs_mapped": sorted(
        lib for lib in mapped if "/usr/local/cuda" in lib
    )[:40],
    "installed": len(rows),
    "loaded": len(loaded),
    "unloaded": len(unloaded),
    "loaded_bytes": sum(r["size"] for r in loaded),
    "unloaded_bytes": sum(r["size"] for r in unloaded),
    "unloaded_dists": unloaded,
    "loaded_dists": sorted(loaded, key=lambda r: -r["size"]),
    "per_process": per_process,
}
OUT.write_text(json.dumps(report, indent=2))

print(f"进程快照={len(per_process)} 已安装={len(rows)} 真实加载={len(loaded)} 未加载={len(unloaded)}")
print(f"加载体积={report['loaded_bytes'] / mb:.0f} MiB  未加载体积={report['unloaded_bytes'] / mb:.0f} MiB")
print("\n未加载的发行包（按体积排序，前 30）：")
for row in unloaded[:30]:
    print(f"  {row['size'] / mb:9.1f} MiB  {row['name']}=={row['version']}")
print("\n系统 CUDA（/usr/local/cuda）被映射的库数:", len(report["system_cuda_libs_mapped"]))
for lib in report["system_cuda_libs_mapped"][:6]:
    print("   ", lib)
print(f"\nreport: {OUT}")
