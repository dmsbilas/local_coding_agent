"""Default agent: ReAct coding agent via AI Doer (gpt-4o)."""

from agents.react_ai_doer.agent import (
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
