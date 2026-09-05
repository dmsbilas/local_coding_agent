# Utils (`utils/`)

Small shared helpers with **no agent-specific state**.

## Module map

| File | Exports | Purpose |
|------|---------|---------|
| `retry.py` | `invoke_with_retry`, `backoff`, `is_rate_limit_error`, `is_timeout_error` | Bounded retries for LLM/tool calls |
| `parsing.py` | `parse_json_object`, `code_from_fences`, `looks_like_json_blob`, `sanitize_filename_stem` | Turn messy LLM text into JSON or source |
| `__init__.py` | Re-exports | Convenient imports |

## Retry policy

- Retries only **transient** errors (rate limit / timeout markers).
- Delay: exponential from `RETRY_BASE_SECONDS`, capped by `RETRY_MAX_SECONDS`, plus light jitter.
- Attempt count comes from the caller (usually `MAX_TOOL_RETRIES`).

## Parsing notes

- `parse_json_object` accepts raw JSON, fenced ```json blocks, or a `{…}` substring.
- `code_from_fences` picks the largest non-JSON fenced block (skips ```json).
- `looks_like_json_blob` prevents writing validator JSON into `.py` files.
