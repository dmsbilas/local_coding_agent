"""LLM system prompts used by agents. Edit these without touching agent logic."""

from prompts.basic import (
    CODE_GEN_SYSTEM_PROMPT as BASIC_CODE_GEN_SYSTEM_PROMPT,
    VALIDATE_SYSTEM_PROMPT as BASIC_VALIDATE_SYSTEM_PROMPT,
)
from prompts.react import (
    CODE_GEN_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    VALIDATE_SYSTEM_PROMPT,
)

__all__ = [
    "BASIC_CODE_GEN_SYSTEM_PROMPT",
    "BASIC_VALIDATE_SYSTEM_PROMPT",
    "CODE_GEN_SYSTEM_PROMPT",
    "PLAN_SYSTEM_PROMPT",
    "VALIDATE_SYSTEM_PROMPT",
]
