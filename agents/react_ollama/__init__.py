"""ReAct coding agent via local Ollama."""

from agents.react_ollama.agent import (
    MAX_REVISIONS,
    MAX_STEPS,
    MODEL_NAME,
    PLAN_MODE,
    app,
    new_thread_id,
)

__all__ = [
    "MAX_REVISIONS",
    "MAX_STEPS",
    "MODEL_NAME",
    "PLAN_MODE",
    "app",
    "new_thread_id",
]
