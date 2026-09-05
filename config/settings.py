"""
Central settings for the agent boilerplate.

All tunables come from environment variables (or `.env` via python-dotenv).
Change defaults here when forking this repo into a new agent project.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root (parent of the `config/` package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Output / sandbox
# ---------------------------------------------------------------------------
DEFAULT_SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox").strip() or "sandbox"
SANDBOX_ROOT = (PROJECT_ROOT / DEFAULT_SANDBOX_DIR).resolve()
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
# AI Doer (OpenAI-compatible cloud) — used by agents.react_ai_doer
AI_DOER_BASE_URL = os.getenv("AI_DOER_BASE_URL", "https://ai-doer.com/v1/api")
AI_DOER_API_KEY = os.getenv("AI_DOER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Local Ollama — used by agents.basic and agents.react_ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5-coder:7b")

# Planning: "reactive" or "plan_and_execute"
PLAN_MODE = os.getenv("PLAN_MODE", "reactive").strip().lower()

# ---------------------------------------------------------------------------
# Safety bounds (also act as stopping conditions)
# ---------------------------------------------------------------------------
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "6"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
MAX_TOOL_RETRIES = int(os.getenv("MAX_TOOL_RETRIES", "3"))
MAX_VALIDATOR_ATTEMPTS = int(os.getenv("MAX_VALIDATOR_ATTEMPTS", "2"))
MAX_REPEATED_ACTIONS = int(os.getenv("MAX_REPEATED_ACTIONS", "3"))

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "1.0"))
RETRY_MAX_SECONDS = float(os.getenv("RETRY_MAX_SECONDS", "8.0"))

# ---------------------------------------------------------------------------
# Paths the agent must never overwrite or use as an output root
# ---------------------------------------------------------------------------
PROTECTED_NAMES = {
    "main.py",
    "requirements.txt",
    "run.sh",
    "readme.md",
    "architecture.md",
    ".gitignore",
    ".env",
    ".env.example",
}

BLOCKED_OUTPUT_DIRS = {
    ".git",
    ".venv",
    ".env",
    "__pycache__",
    "node_modules",
    "config",
    "prompts",
    "tools",
    "utils",
    "agents",
    "docs",
}


def require_ai_doer_api_key() -> None:
    """Raise if AI_DOER_API_KEY is missing (call before cloud LLM invokes)."""
    if not AI_DOER_API_KEY:
        raise ValueError(
            "AI_DOER_API_KEY is not set. Add it to your .env file "
            "(create a key at https://ai-doer.com → Integrations)."
        )
