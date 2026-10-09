# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Light-weight chat/serving helpers.

These helpers used to live in :mod:`vllm.entrypoints.chat_utils`, which imports
the OpenAI SDK, PIL, and the multimodal processing stack. The CLI argument
parser and the parser layer only need the option types, the chat-template
validator, and the tool-call ID helpers, so they live here instead and
``chat_utils`` re-exports every name for backwards compatibility.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vllm.exceptions import VLLMValidationError
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.config import ModelConfig

# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# After resolving "auto"
ChatTemplateContentFormat = Literal["string", "openai"]


def validate_chat_template(chat_template: Path | str | None):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise VLLMValidationError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}",
                    parameter="chat_template",
                )

    else:
        raise VLLMValidationError(
            f"{type(chat_template)} is not a valid chat template type",
            parameter="chat_template",
        )


_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3")


def get_tool_call_id_type(model_config: "ModelConfig") -> str:
    """Return the tool-call ID type for a given model configuration."""
    hf_overrides = getattr(model_config, "hf_overrides", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    )
    if any(model_type in _KIMI_MODEL_TYPES for model_type in model_types) or (
        isinstance(hf_overrides, dict)
        and hf_overrides.get("model_type") in _KIMI_MODEL_TYPES
    ):
        return "kimi_k2"
    return "random"


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{random_uuid()}"
