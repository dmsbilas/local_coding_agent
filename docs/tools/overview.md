# Tools (`tools/`)

Tools are the agent’s **side effects**: reading/writing files under a safe output directory, plus the machinery to parse and execute model-produced tool calls.

## Module map

| File | Responsibility |
|------|----------------|
| `schemas.py` | Pydantic models: `WriteFileInput`, `ReadFileInput`, `ToolCall` |
| `paths.py` | Output-dir inference, normalization, path containment |
| `file_tools.py` | `write_file` / `read_file` LangChain tools + `write_file_impl` / `normalize_files` |
| `executor.py` | `parse_tool_call` → `validate_tool_arguments` → `execute_tool_call(s)` |
| `__init__.py` | Public re-exports |

## Path safety (`paths.py`)

| Function | Purpose |
|----------|---------|
| `strip_dot_slash` | Normalize `./` prefixes |
| `normalize_output_dir` | Safe relative dir or default `sandbox` |
| `infer_output_dir_from_text` | Detect “in folder X” from user text |
| `first_user_text` | Extract first human message from state |
| `output_root` | Absolute output directory (created if missing) |
| `resolve_under_output` | Map a relative path into the output root or return an error string |

Blocked output roots include `.git`, `.venv`, `agents`, `config`, `prompts`, `tools`, `utils`, `docs`, etc. (see `config.settings.BLOCKED_OUTPUT_DIRS`).

## File tools (`file_tools.py`)

| Symbol | Purpose |
|--------|---------|
| `write_file_impl(path, content, output_dir)` | Core write used by agent nodes |
| `read_file_impl(path, output_dir)` | Core read |
| `write_file` / `read_file` | `@tool`-wrapped versions (default sandbox) |
| `TOOLS` | Registry dict for the executor |
| `normalize_files` | Clean validator `files[]` payloads |

## Executor (`executor.py`)

Used when the model emits tool-call JSON (or when you add function-calling later).

1. **Parse** — dict, raw JSON, fenced JSON, or prose containing one object  
2. **Validate** — against the tool’s `args_schema`  
3. **Execute** — with `invoke_with_retry` and `MAX_TOOL_RETRIES`  
4. **Batch** — `execute_tool_calls(..., parallel=False)` by default (safer for writes)

## Adding a tool

```python
# 1. schemas.py — add Input model and extend ToolCall.name Literal
# 2. implement _my_tool_impl(...)
# 3. @tool(args_schema=MyInput) def my_tool(...): ...
# 4. TOOLS["my_tool"] = my_tool
```

Document the new tool in this file and mention it from any agent that uses it.
