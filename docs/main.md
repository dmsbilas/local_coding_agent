# CLI entrypoint (`main.py`)

Thin interface between the human and a compiled LangGraph `app`. **No graph logic lives here.**

## Responsibilities

1. Select agent via `AGENT` env (`react_ai_doer` | `react_ollama` | `basic`)
2. Read user task from stdin
3. Stream node updates (`stream_mode="updates"`) and print a readable status
4. Summarize final `status` / `write_results` / escalation

## Key functions

| Function | Role |
|----------|------|
| `_load_agent()` | Dynamic import of the chosen agent package |
| `read_user_input()` | Prompt + handle Ctrl-C / EOF |
| `_print_partial_state()` | Pretty-print messages, plan, status counters |
| `run_agent()` | Interactive loop until `quit` |

## Initial graph state

```python
{
  "messages": [("user", user_input)],
  "iteration": 0,
  "approved": False,
  "files": [],
  "feedback": "",
  "status": "starting",
  "step_count": 0,
  "revision_count": 0,
  "action_history": [],
  "observations": [],
  "plan": [],
  "plan_index": 0,
  "output_dir": "",
}
```

## Companion script

`run.sh` — runs `.venv/bin/python main.py` so the wrong system Python is not used by accident.

## Extending

When you add `agents/my_agent`, add a branch in `_load_agent()` that imports `app`, `new_thread_id`, and the constants the CLI prints (`MODEL_NAME`, limits, `PLAN_MODE`).
