# AI Agent Boilerplate

A reusable project layout for building LangGraph coding agents. Prompts, tools, agents, config, and the CLI entrypoint are separated so you can fork this repo and ship a new agent without untangling a single mega-file.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set AI_DOER_API_KEY
./run.sh               # or: python main.py
```

Default agent: **`agents.react_ai_doer`** (AI Doer / gpt-4o).

Switch agents:

```bash
AGENT=react_ai_doer python main.py   # cloud (default)
AGENT=react_ollama python main.py    # local Ollama ReAct
AGENT=basic python main.py           # minimal Ollama loop
```

## Project structure

```
local_coding_agent/
├── main.py                 # CLI entrypoint (thin)
├── run.sh                  # Runs main.py with .venv
├── requirements.txt
├── .env.example
│
├── config/                 # Environment & settings
│   └── settings.py
│
├── prompts/                # System prompts only (no logic)
│   ├── basic.py
│   └── react.py
│
├── tools/                  # Agent tools (file IO, schemas, executor)
│   ├── schemas.py
│   ├── paths.py
│   ├── file_tools.py
│   └── executor.py
│
├── utils/                  # Shared helpers (retry, parsing)
│   ├── retry.py
│   └── parsing.py
│
├── agents/                 # LangGraph agent graphs
│   ├── basic/              # generate → validate → write
│   ├── react_ollama/       # full ReAct + local Ollama
│   └── react_ai_doer/      # full ReAct + AI Doer (default)
│
├── docs/                   # Full documentation for every piece
│   ├── structure.md
│   ├── getting-started.md
│   ├── architecture.md
│   ├── config.md
│   ├── agents/
│   ├── tools/
│   └── prompts/
│
└── sandbox/                # Default write target for generated code
```

## How to use this as a boilerplate

1. **Copy / fork** this repo.
2. **Edit prompts** in `prompts/` — change behavior without touching graph code.
3. **Add tools** in `tools/` — register them in `TOOLS` and (if needed) extend `ToolCall.name`.
4. **Add an agent** under `agents/<your_agent>/` — import prompts + tools + config.
5. **Point `main.py`** at your agent (or set `AGENT=...`).
6. **Document** the new pieces under `docs/`.

## Documentation index

| Doc | What it covers |
|-----|----------------|
| [docs/getting-started.md](docs/getting-started.md) | Install, env, first run |
| [docs/structure.md](docs/structure.md) | Every folder and file |
| [docs/architecture.md](docs/architecture.md) | Graphs, ReAct loop, data flow |
| [docs/config.md](docs/config.md) | All settings / env vars |
| [docs/prompts/](docs/prompts/) | Prompt modules |
| [docs/tools/](docs/tools/) | Tools & path safety |
| [docs/agents/](docs/agents/) | Each agent implementation |

## License / notes

Generated code lands in `sandbox/` (or a user-named folder). Boilerplate packages (`agents/`, `config/`, `prompts/`, `tools/`, `utils/`, `docs/`) are blocked as output roots.
