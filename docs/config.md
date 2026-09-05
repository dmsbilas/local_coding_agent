# Configuration (`config/`)

All tunables live in `config/settings.py` and are loaded from the environment / `.env` via `python-dotenv`.

## Module map

| File | Role |
|------|------|
| `config/__init__.py` | Re-exports common settings |
| `config/settings.py` | Definitions, defaults, `require_ai_doer_api_key()` |

## Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `AI_DOER_API_KEY` | _(empty)_ | `react_ai_doer` — required |
| `AI_DOER_BASE_URL` | `https://ai-doer.com/v1/api` | `react_ai_doer` |
| `MODEL_NAME` | `gpt-4o` | `react_ai_doer` |
| `AGENT` | `react_ai_doer` | `main.py` agent picker |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `basic`, `react_ollama` |
| `OLLAMA_MODEL_NAME` | `qwen2.5-coder:7b` | `basic`, `react_ollama` |
| `SANDBOX_DIR` | `sandbox` | All agents (default output folder name) |
| `PLAN_MODE` | `reactive` | ReAct agents (`reactive` \| `plan_and_execute`) |
| `MAX_REVISIONS` | `6` | All agents |
| `MAX_STEPS` | `12` | ReAct agents |
| `MAX_TOOL_RETRIES` | `3` | LLM/tool retries |
| `MAX_VALIDATOR_ATTEMPTS` | `2` | Malformed validator JSON retries |
| `MAX_REPEATED_ACTIONS` | `3` | Loop detection |
| `LLM_TIMEOUT_SECONDS` | `120` | AI Doer client timeout |
| `RETRY_BASE_SECONDS` | `1.0` | Backoff base |
| `RETRY_MAX_SECONDS` | `8.0` | Backoff cap |

## Important symbols

| Symbol | Meaning |
|--------|---------|
| `PROJECT_ROOT` | Repo root (parent of `config/`) |
| `SANDBOX_ROOT` | Absolute default output directory |
| `PROTECTED_NAMES` | Filenames refused at repo root |
| `BLOCKED_OUTPUT_DIRS` | First path segments that cannot be output roots (`agents`, `config`, `.git`, …) |
| `require_ai_doer_api_key()` | Raises if cloud key missing |

## Forking tip

When starting a new agent product from this boilerplate, change defaults in `settings.py` first, then add product-specific env vars in the same file so `main.py` and agents stay free of scattered `os.getenv` calls.
