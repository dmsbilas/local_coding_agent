# Documentation index

Complete reference for this AI agent boilerplate.

## Guides

| Document | Description |
|----------|-------------|
| [getting-started.md](getting-started.md) | Install, configure, run |
| [structure.md](structure.md) | Folder tree and responsibility of each package |
| [architecture.md](architecture.md) | LangGraph flows, ReAct loop, system context |
| [config.md](config.md) | Every setting and environment variable |

## Packages

| Area | Document |
|------|----------|
| Prompts | [prompts/overview.md](prompts/overview.md) |
| Tools | [tools/overview.md](tools/overview.md) |
| Agents — basic | [agents/basic.md](agents/basic.md) |
| Agents — react_ollama | [agents/react_ollama.md](agents/react_ollama.md) |
| Agents — react_ai_doer | [agents/react_ai_doer.md](agents/react_ai_doer.md) |
| CLI (`main.py`) | [main.md](main.md) |
| Utils | [utils.md](utils.md) |

## Extending the boilerplate

1. Add a prompt file under `prompts/` and export it from `prompts/__init__.py`.
2. Add a tool under `tools/` (schema + implementation + register in `TOOLS`).
3. Create `agents/<name>/agent.py` that builds a `StateGraph` and exports `app`, `new_thread_id`, and limit constants.
4. Wire the agent in `main.py` (`_load_agent`) and document it under `docs/agents/`.
