"""Minimal generate → validate → write agent (local Ollama)."""

from agents.basic.agent import MAX_REVISIONS, MODEL_NAME, app, new_thread_id

__all__ = ["MAX_REVISIONS", "MODEL_NAME", "app", "new_thread_id"]
