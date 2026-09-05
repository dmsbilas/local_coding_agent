"""Parse LLM text into JSON objects or fenced source code."""

from __future__ import annotations

import json
import re
from typing import Any


def looks_like_json_blob(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return (
            '"fixed_code"' in stripped
            or '"relative_path"' in stripped
            or '"approved"' in stripped
        )


def parse_json_object(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def code_from_fences(text: str) -> str | None:
    blocks = re.findall(r"```(?!json)(\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return None
    bodies = [body.strip() for _, body in blocks if body.strip()]
    if not bodies:
        return None
    candidate = max(bodies, key=len)
    return None if looks_like_json_blob(candidate) else candidate


def sanitize_filename_stem(path: str, protected_names: set[str]) -> str:
    """If path basename is protected, rename stem with `_app` suffix."""
    from pathlib import Path

    p = Path(path)
    if p.name.lower() in protected_names:
        stem = p.stem + "_app"
        return str(p.with_name(stem + (p.suffix or ".py")))
    return path
