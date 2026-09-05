# Project structure

Every top-level package has one job. Keep new code in the matching folder so this stays a usable boilerplate.

## Tree

```
local_coding_agent/
├── main.py              # CLI only — no agent logic
├── run.sh               # Convenience launcher (.venv)
├── requirements.txt
├── .env / .env.example
├── README.md
│
├── config/              # Settings & env
│   ├── __init__.py
│   └── settings.py
│
├── prompts/             # LLM system prompts (strings only)
│   ├── __init__.py
│   ├── basic.py
│   └── react.py
│
├── tools/               # Callable tools + path safety
│   ├── __init__.py
│   ├── schemas.py       # Pydantic input models
│   ├── paths.py         # Output-dir inference & containment
│   ├── file_tools.py    # write_file / read_file
│   └── executor.py      # Parse → validate → execute tool calls
│
├── utils/               # Generic helpers (no domain logic)
│   ├── __init__.py
│   ├── retry.py
│   └── parsing.py
│
├── agents/              # Compiled LangGraph apps
│   ├── __init__.py      # Re-exports default agent
│   ├── basic/
│   ├── react_ollama/
│   └── react_ai_doer/
│
├── docs/                # Human documentation
└── sandbox/             # Default generated-code directory
```

## Responsibility map

| Package | Owns | Must not contain |
|---------|------|------------------|
| `config/` | Env vars, paths, limits | Prompts, graph nodes |
| `prompts/` | System prompt text | File IO, LLM clients |
| `tools/` | Schemas, disk IO, tool execution | Graph routing |
| `utils/` | Retry, JSON/code parsing | Agent-specific state |
| `agents/` | State, nodes, routers, compiled `app` | Raw prompt duplication (import from `prompts/`) |
| `main.py` | User I/O, streaming print, agent selection | Business logic |

## Adding a new agent (checklist)

1. `mkdir agents/my_agent && touch agents/my_agent/__init__.py agents/my_agent/agent.py`
2. Import prompts from `prompts/`, tools from `tools/`, settings from `config/`
3. Export `app`, `new_thread_id`, and any limits `main.py` needs
4. Register in `main.py` → `_load_agent()`
5. Add `docs/agents/my_agent.md`

## Adding a new tool (checklist)

1. Add Pydantic schema in `tools/schemas.py` (or a new module)
2. Implement function in `tools/file_tools.py` (or `tools/<name>.py`)
3. Register in `TOOLS`
4. Extend `ToolCall.name` Literal if using the executor
5. Document in `docs/tools/overview.md`
