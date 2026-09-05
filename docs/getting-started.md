# Getting started

## Prerequisites

- Python 3.11+ recommended
- For **react_ai_doer**: an [AI Doer](https://ai-doer.com) API key (`aiob_…`)
- For **basic** / **react_ollama**: [Ollama](https://ollama.com) running locally + `pip install langchain-ollama`

## Install

```bash
cd local_coding_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least:

```env
AI_DOER_API_KEY=aiob_your_key_here
AGENT=react_ai_doer
```

## Run

```bash
./run.sh
# or
python main.py
```

Describe what you want built. The agent plans (optional), generates, validates, revises, then writes files under `sandbox/` or a folder you named in the prompt (e.g. “put it in `projects/todo`”).

## Choose an agent

| `AGENT` value | Backend | Graph |
|---------------|---------|-------|
| `react_ai_doer` (default) | AI Doer cloud | Full ReAct + output dir |
| `react_ollama` | Local Ollama | Same ReAct graph |
| `basic` | Local Ollama | generate → validate → write |

```bash
AGENT=basic python main.py
```

## Common issues

**`ModuleNotFoundError`** — activate `.venv` or run `.venv/bin/python main.py`.

**`AI_DOER_API_KEY is not set`** — copy `.env.example` → `.env` and paste a real key.

**Ollama agents fail on import** — `pip install langchain-ollama` and run `ollama serve`.
