"""File read/write tools bound to the project sandbox / output directory."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from config.settings import DEFAULT_SANDBOX_DIR, PROJECT_ROOT, PROTECTED_NAMES
from tools.paths import normalize_output_dir, resolve_under_output, strip_dot_slash
from tools.schemas import ReadFileInput, WriteFileInput
from utils.parsing import code_from_fences, looks_like_json_blob, sanitize_filename_stem


def write_file_impl(relative_path: str, content: str, output_dir: str | None) -> str:
    target = resolve_under_output(relative_path, output_dir)
    if isinstance(target, str):
        return target

    if target.parent == PROJECT_ROOT and target.name.lower() in PROTECTED_NAMES:
        return f"Error: refusing to overwrite protected file '{relative_path}'."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    try:
        shown = str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(target)
    return f"Successfully wrote {len(content)} bytes to {shown}"


def read_file_impl(relative_path: str, output_dir: str | None) -> str:
    target = resolve_under_output(relative_path, output_dir)
    if isinstance(target, str):
        return target

    if not target.exists():
        return f"Error: file '{relative_path}' does not exist in the output folder."
    if not target.is_file():
        return f"Error: '{relative_path}' is not a file."

    return target.read_text(encoding="utf-8")


@tool(args_schema=WriteFileInput)
def write_file(relative_path: str, content: str) -> str:
    """Write source content inside the chosen output folder (sandbox if none)."""
    return write_file_impl(relative_path, content, DEFAULT_SANDBOX_DIR)


@tool(args_schema=ReadFileInput)
def read_file(relative_path: str) -> str:
    """Read a file safely from the chosen output folder (sandbox if none)."""
    return read_file_impl(relative_path, DEFAULT_SANDBOX_DIR)


TOOLS: dict[str, BaseTool] = {
    "write_file": write_file,
    "read_file": read_file,
}


def _sanitize_path(path: str, output_dir: str | None = None) -> str:
    """Normalize a file path relative to the output folder (no output-dir prefix)."""
    path = strip_dot_slash(path)
    root_name = normalize_output_dir(output_dir)
    while True:
        lower = path.lower()
        stripped = False
        for prefix in (root_name, DEFAULT_SANDBOX_DIR, "sandbox"):
            if lower == prefix:
                return ""
            if lower.startswith(prefix + "/"):
                path = strip_dot_slash(path[len(prefix) + 1 :])
                stripped = True
                break
        if not stripped:
            break
    return sanitize_filename_stem(path, PROTECTED_NAMES)


def normalize_files(raw_files: Any, output_dir: str | None = None) -> list[dict[str, str]]:
    """Normalize validator files into paths relative to the chosen output folder."""
    if not isinstance(raw_files, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        path = item.get("relative_path") or item.get("path") or item.get("filename")
        content = item.get("content") or item.get("code")
        if not isinstance(path, str) or not isinstance(content, str):
            continue

        path = _sanitize_path(path, output_dir)
        content = content.strip()
        fenced = code_from_fences(content)
        if fenced:
            content = fenced

        if not path or not content or looks_like_json_blob(content):
            continue

        resolved = resolve_under_output(path, output_dir)
        if isinstance(resolved, str):
            continue

        cleaned.append({"relative_path": path, "content": content})

    return cleaned
