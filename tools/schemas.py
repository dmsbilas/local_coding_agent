"""Pydantic schemas for tool / function calling."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WriteFileInput(BaseModel):
    """Schema presented to a function/tool caller for writes."""

    relative_path: str = Field(
        ...,
        description=(
            "File path relative to the chosen output folder, e.g. "
            "'hello_world.py' or 'src/main.py'."
        ),
        min_length=1,
    )
    content: str = Field(
        ...,
        description="Complete source-file contents; never JSON or tool-call text",
    )


class ReadFileInput(BaseModel):
    relative_path: str = Field(
        ...,
        description="File path relative to the chosen output folder.",
        min_length=1,
    )


class ToolCall(BaseModel):
    """Normalized representation of a model-produced tool call."""

    name: Literal["write_file", "read_file"]
    arguments: dict[str, Any]
