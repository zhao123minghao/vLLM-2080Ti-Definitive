# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if TYPE_CHECKING:
    import tilelang
    import tilelang.language as T


def __getattr__(name: str) -> Any:
    """Resolve the `tilelang` / `T` module globals on first access (PEP 562).

    Importing TileLang eagerly made every importer pay for a large optional
    dependency, including call sites that only register MHC kernels and never run
    them. TileLang parses a decorated kernel body on its first call, so binding the
    names lazily is sufficient.
    """
    if name in ("tilelang", "T"):
        _ensure_tilelang_imported()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_tilelang_imported() -> None:
    """Bind the `tilelang` and `T` module globals, importing them if needed.

    On ROCm, this runs on the first kernel call instead of at import time.

    Raises:
        ImportError: If TileLang is not installed.
    """
    global T, tilelang

    if tilelang is not None:
        return
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang as tilelang_module
    import tilelang.language as tilelang_language

    tilelang = tilelang_module
    T = tilelang_language


@cache
def _get_pass_configs() -> dict[Any, Any]:
    _ensure_tilelang_imported()
    pass_configs: dict[Any, Any] = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if current_platform.is_cuda():
        pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10
    return pass_configs


def tilelang_jit(kernel_function: Callable[..., Any]) -> Callable[..., Any]:
    """Apply `tilelang.jit`, deferring the decoration until the first call.

    Deferring keeps importing the caller's module free of TileLang: the dependency is
    only needed once a kernel actually runs, and vLLM warms these kernels up
    explicitly at startup, so compilation still happens outside the serving loop.

    The kernel body parsed by TileLang references `T` as an unqualified global, so
    this rebinds `T`/`tilelang` in the decorated function's own module globals once
    they become available.
    """
    compiled_kernel: Callable[..., Any] | None = None

    @functools.wraps(kernel_function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled_kernel
        if compiled_kernel is None:
            _ensure_tilelang_imported()
            kernel_function.__globals__["tilelang"] = tilelang
            kernel_function.__globals__["T"] = T
            compiled_kernel = tilelang.jit(pass_configs=_get_pass_configs())(
                kernel_function
            )
        return compiled_kernel(*args, **kwargs)

    return wrapper
