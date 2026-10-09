# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from importlib import import_module
from typing import Any

from vllm.parser.engine.adapters import make_adapters

# Parser engine class name -> module providing it. `make_adapters` names the adapter
# pair after the engine class, so this mapping is all that is needed to resolve a name.
_PARSER_ENGINES: dict[str, str] = {
    "DeepSeekV32Parser": "vllm.parser.deepseek_v32",
    "DeepSeekV4Parser": "vllm.parser.deepseek_v4",
    "DeepSeekV41Parser": "vllm.parser.deepseek_v41",
    "Gemma4Parser": "vllm.parser.gemma4",
    "Glm47MoeParser": "vllm.parser.glm47_moe",
    "InklingParser": "vllm.parser.inkling",
    "KimiK2Parser": "vllm.parser.kimi_k2",
    "MinimaxM2Parser": "vllm.parser.minimax_m2",
    "MistralParser": "vllm.parser.mistral",
    "NemotronV3Parser": "vllm.parser.nemotron_v3",
    "Qwen3Parser": "vllm.parser.qwen3",
    "SeedOssParser": "vllm.parser.seed_oss",
}

_REASONING_SUFFIX = "ReasoningAdapter"
_TOOL_SUFFIX = "ToolAdapter"
_ADAPTER_CACHE: dict[str, tuple[type, type]] = {}


def _adapters_for(engine_name: str) -> tuple[type, type]:
    """Build (and memoise) the `(reasoning, tool)` adapter pair of one engine."""
    adapters = _ADAPTER_CACHE.get(engine_name)
    if adapters is None:
        module = import_module(_PARSER_ENGINES[engine_name])
        adapters = make_adapters(getattr(module, engine_name))
        _ADAPTER_CACHE[engine_name] = adapters
    return adapters


def __getattr__(name: str) -> Any:
    """Resolve an adapter name by importing only the engine that owns it.

    Importing every parser engine up front also imported their optional dependencies
    (for example `mistral_common`, which pulls in OpenCV) even for a route that uses a
    single parser, so engines are now loaded on demand.
    """
    for engine_name in _PARSER_ENGINES:
        if name == f"{engine_name}{_REASONING_SUFFIX}":
            value = _adapters_for(engine_name)[0]
            break
        if name == f"{engine_name}{_TOOL_SUFFIX}":
            value = _adapters_for(engine_name)[1]
            break
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(
        {
            *globals(),
            *(
                f"{engine}{suffix}"
                for engine in _PARSER_ENGINES
                for suffix in (_REASONING_SUFFIX, _TOOL_SUFFIX)
            ),
        }
    )
