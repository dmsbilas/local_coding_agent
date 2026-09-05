"""Agent tools: file IO, schemas, and safe execution."""

from tools.executor import (
    ToolCall,
    execute_tool_call,
    execute_tool_calls,
    parse_tool_call,
    validate_tool_arguments,
)
from tools.file_tools import (
    TOOLS,
    normalize_files,
    read_file,
    read_file_impl,
    write_file,
    write_file_impl,
)
from tools.paths import (
    first_user_text,
    infer_output_dir_from_text,
    normalize_output_dir,
    output_root,
    resolve_under_output,
    strip_dot_slash,
)
from tools.schemas import ReadFileInput, WriteFileInput

__all__ = [
    "TOOLS",
    "ToolCall",
    "ReadFileInput",
    "WriteFileInput",
    "execute_tool_call",
    "execute_tool_calls",
    "first_user_text",
    "infer_output_dir_from_text",
    "normalize_files",
    "normalize_output_dir",
    "output_root",
    "parse_tool_call",
    "read_file",
    "read_file_impl",
    "resolve_under_output",
    "strip_dot_slash",
    "validate_tool_arguments",
    "write_file",
    "write_file_impl",
]
