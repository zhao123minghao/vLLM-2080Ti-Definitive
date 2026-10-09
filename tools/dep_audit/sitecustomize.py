# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Exact dependency audit hook (v2).
#
# Two signals per Python process, dumped every 3 s and at exit so killed processes
# still leave data:
#   1. the real `sys.modules` snapshot -- only successfully imported modules count,
#      unlike an audit hook which also records failed optional-import probes;
#   2. the mapped shared objects from /proc/self/maps -- the only way to see the CUDA
#      libraries torch `dlopen`s (libcublas, libcudnn, libnccl, ...), which never
#      appear as Python modules but dominate the on-disk size.
import atexit
import os
import sys
import threading

_DIR = os.environ.get("IMPORT_AUDIT_DIR")


def _so_paths():
    libs = set()
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                fields = line.split(None, 5)
                if len(fields) < 6:
                    continue
                path = fields[5].strip()
                if ".so" in path:
                    libs.add(path.replace("\\040", " "))
    except Exception:
        pass
    return sorted(libs)


if _DIR:
    def _dump():
        try:
            pid = os.getpid()
            origin = os.environ.get("IMPORT_AUDIT_LABEL", "unknown")
            top = sorted({m.partition(".")[0] for m in list(sys.modules) if m})
            with open(os.path.join(_DIR, "mods-%d.txt" % pid), "w") as handle:
                handle.write("# pid=%d label=%s\n# argv=%s\n" % (pid, origin, sys.argv))
                handle.write("\n".join(top) + "\n")
            with open(os.path.join(_DIR, "so-%d.txt" % pid), "w") as handle:
                handle.write("\n".join(_so_paths()) + "\n")
        except Exception:
            pass

    def _loop():
        while True:
            _dump()
            threading.Event().wait(3.0)

    atexit.register(_dump)
    threading.Thread(target=_loop, daemon=True).start()
