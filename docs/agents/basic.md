# Agent: `basic`

**Path:** `agents/basic/agent.py`  
**Backend:** Local Ollama (`OLLAMA_MODEL_NAME`)  
**CLI:** `AGENT=basic python main.py`  
**Extra dependency:** `langchain-ollama`

## Purpose

Smallest reference agent for learning / experiments. Fixed loop with no ReAct reasoner, no planning, no escalation node.

## Graph

```
generate_code → validate_code ⟲ → write_files → END
                     │
                     └─ fail_max_revisions → END
```

## State fields

| Field | Meaning |
|-------|---------|
| `messages` | Chat history (LangGraph `MessagesState`) |
| `status` | `generating` / `validating` / `revising` / `writing` / `done` / `failed` |
| `approved` | Validator approval flag |
| `feedback` | Latest validator feedback |
| `files` | `[{relative_path, content}, …]` ready to write |
| `iteration` | Validation round counter |
| `write_results` | Strings returned from write attempts |

## Imports from boilerplate

- Prompts: `prompts.basic`
- Tools: `tools.file_tools.normalize_files`, `write_file_impl`
- Config: `MAX_REVISIONS`, `DEFAULT_SANDBOX_DIR`, Ollama settings
- Utils: `utils.parsing.parse_json_object`

## Output

Always writes under `SANDBOX_DIR` (default `sandbox/`).

## Public API

`app`, `new_thread_id()`, `MAX_REVISIONS`, `MODEL_NAME`
