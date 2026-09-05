"""Safe path helpers: output-dir inference and sandbox containment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config.settings import (
    BLOCKED_OUTPUT_DIRS,
    DEFAULT_SANDBOX_DIR,
    PROJECT_ROOT,
    PROTECTED_NAMES,
)


def strip_dot_slash(path: str) -> str:
    """Remove leading ./ segments without treating '../' as characters to strip."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def normalize_output_dir(raw: str | None) -> str:
    """Return a safe project-relative output dir, or the default sandbox name."""
    if not raw:
        return DEFAULT_SANDBOX_DIR

    cleaned = strip_dot_slash(str(raw))
    if cleaned in {"", ".", ".."} or cleaned.startswith("../") or "/../" in f"/{cleaned}/":
        return DEFAULT_SANDBOX_DIR

    if Path(cleaned).is_absolute():
        return DEFAULT_SANDBOX_DIR

    first = cleaned.split("/", 1)[0].lower()
    if first in BLOCKED_OUTPUT_DIRS or cleaned.lower() in PROTECTED_NAMES:
        return DEFAULT_SANDBOX_DIR

    target = (PROJECT_ROOT / cleaned).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return DEFAULT_SANDBOX_DIR

    if target == PROJECT_ROOT:
        return DEFAULT_SANDBOX_DIR

    return cleaned


def infer_output_dir_from_text(text: str) -> str:
    """Detect a user-mentioned target folder; otherwise default to sandbox/.

    Examples:
      - "create a todo app in projects/todo"
      - "put the files under ./my_app"
      - "write to folder demo_cli"
    If nothing matches → DEFAULT_SANDBOX_DIR.
    """
    if not text:
        return DEFAULT_SANDBOX_DIR

    patterns = [
        r"(?i)\b(?:in|into|under|to|at)\s+(?:the\s+)?(?:folder|directory|dir|path)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\b(?:folder|directory|dir|path)\s*[:=]\s*[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\b(?:in|into|under|to|at)\s+\./([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*)",
        r"(?i)\b(?:in|into|under|to|at)\s+[\"'`]?(\.?/?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)+)[\"'`]?",
        r"(?i)\b(?:in|into|under|to|at)\s+[\"'`](\.?/?[A-Za-z0-9_./-]+)[\"'`]",
        r"(?i)\bsave\s+(?:it|them|files?|code)?\s*(?:in|into|under|to)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\bwrite\s+(?:it|them|files?|code)?\s*(?:in|into|under|to)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\bcreate\s+(?:(?:a|the)\s+)?(?:project|app|files?)?\s*(?:in|into|under|at)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
    ]

    stopwords = {
        "python", "javascript", "typescript", "java", "rust", "go",
        "code", "file", "files", "script", "project", "app",
        "the", "a", "an", "this", "that", "here", "there",
        "disk", "repo", "repository",
    }

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = strip_dot_slash(match.group(1))
            if not candidate:
                continue
            if "/" not in candidate and candidate.lower() in stopwords:
                continue
            normalized = normalize_output_dir(candidate)
            if normalized != DEFAULT_SANDBOX_DIR or candidate.lower() in {
                DEFAULT_SANDBOX_DIR,
                "sandbox",
            }:
                return normalized
            if any(k in pattern for k in ("folder", "directory", "dir", "path")):
                if candidate.lower() not in stopwords:
                    return normalize_output_dir(candidate)

    return DEFAULT_SANDBOX_DIR


def first_user_text(messages: list[Any]) -> str:
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type in {"human", "user"}:
            return str(getattr(msg, "content", "") or "")
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            return str(msg.get("content") or "")
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] in {"user", "human"}:
            return str(msg[1])
    return ""


def output_root(output_dir: str | None) -> Path:
    """Absolute root directory where generated files are written."""
    rel = normalize_output_dir(output_dir)
    root = (PROJECT_ROOT / rel).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_under_output(relative_path: str, output_dir: str | None) -> Path | str:
    """Map a relative file path into the chosen output root."""
    root = output_root(output_dir)
    root_name = root.name
    cleaned = strip_dot_slash(relative_path)

    while True:
        lower = cleaned.lower()
        stripped = False
        for prefix in (root_name, DEFAULT_SANDBOX_DIR, "sandbox"):
            if lower == prefix:
                cleaned = ""
                stripped = True
                break
            if lower.startswith(prefix + "/"):
                cleaned = strip_dot_slash(cleaned[len(prefix) + 1 :])
                stripped = True
                break
        if not stripped:
            break

    if (
        not cleaned
        or cleaned in {".", ".."}
        or cleaned.startswith("../")
        or "/../" in f"/{cleaned}/"
    ):
        return (
            f"Error: path '{relative_path}' is empty or escapes the output folder. "
            "Use a path relative to the output directory."
        )

    target = (root / cleaned).resolve()
    try:
        target.relative_to(root)
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return (
            f"Error: path '{relative_path}' escapes the allowed output area. "
            "Use a path relative to the output directory."
        )
    return target
