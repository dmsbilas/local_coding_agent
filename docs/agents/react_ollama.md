# Agent: `react_ollama`

**Path:** `agents/react_ollama/agent.py`  
**Backend:** Local Ollama (`ChatOllama`)  
**CLI:** `AGENT=react_ollama python main.py`  
**Extra dependency:** `langchain-ollama`

## Purpose

Full ReAct + planning + escalation stack identical in structure to `react_ai_doer`, but running against a local model. Use this when you want the same control flow offline.

## Graph

```
set_output_dir → create_plan → reason → ACT → observe ⟲ → END
```

ACT nodes: `generate_code` | `validate_code` | `write_files` | `escalate` | `fail`

## Differences from `react_ai_doer`

| Concern | `react_ollama` | `react_ai_doer` |
|---------|----------------|-----------------|
| LLM client | `ChatOllama` | `ChatOpenAI` → AI Doer |
| Auth | None (local) | `AI_DOER_API_KEY` |
| Model env | `OLLAMA_MODEL_NAME` | `MODEL_NAME` |
| Timeout kwargs | Ollama defaults | `LLM_TIMEOUT_SECONDS` + `stream=False` |

## State fields

Same as `react_ai_doer` (includes `output_dir`, `plan`, `step_count`, `action_history`, `escalation_reason`, …). See [react_ai_doer.md](react_ai_doer.md).

## Public API

`app`, `new_thread_id()`, `MAX_REVISIONS`, `MAX_STEPS`, `MODEL_NAME`, `PLAN_MODE`
