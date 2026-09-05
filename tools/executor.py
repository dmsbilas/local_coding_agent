"""Parse, validate, and execute model-produced tool calls."""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from pydantic import ValidationError

from config.settings import MAX_TOOL_RETRIES
from tools.file_tools import TOOLS
from tools.schemas import ToolCall
from utils.retry import invoke_with_retry


def parse_tool_call(raw: Any) -> ToolCall | None:
    """Parse and validate a model-produced tool call.

    Handles dict/object, JSON text, fenced JSON, or surrounding prose with one object.
    Invalid output returns None instead of being executed blindly.
    """
    if isinstance(raw, ToolCall):
        return raw

    if isinstance(raw, dict):
        candidate = raw
    else:
        text = str(raw or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                candidate = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    try:
        return ToolCall.model_validate(candidate)
    except ValidationError:
        return None


def validate_tool_arguments(call: ToolCall) -> dict[str, Any]:
    """Validate arguments against the exact tool schema before execution."""
    tool_obj = TOOLS.get(call.name)
    if tool_obj is None:
        raise ValueError(f"Unknown tool '{call.name}'.")

    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return call.arguments

    validated = args_schema.model_validate(call.arguments)
    return validated.model_dump()


def execute_tool_call(call: ToolCall) -> str:
    """Execute one validated tool call with bounded retries."""
    tool_obj = TOOLS.get(call.name)
    if tool_obj is None:
        return f"Tool error: unknown tool '{call.name}'."

    try:
        arguments = validate_tool_arguments(call)
    except ValidationError as exc:
        return f"Tool error: malformed arguments for {call.name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Tool error: {exc}"

    try:
        return invoke_with_retry(
            lambda: tool_obj.invoke(arguments),
            attempts=MAX_TOOL_RETRIES,
            operation=f"tool {call.name}",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Tool error after retries: {exc}"


def execute_tool_calls(calls: list[ToolCall], *, parallel: bool = False) -> list[str]:
    """Execute tool calls sequentially or in parallel.

    Parallel is only appropriate when calls are independent. File writes can have
    ordering/conflict concerns, so the default for coding agents is sequential.
    """
    if not calls:
        return []

    if not parallel or len(calls) == 1:
        return [execute_tool_call(call) for call in calls]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = [executor.submit(execute_tool_call, call) for call in calls]
        return [future.result() for future in futures]
