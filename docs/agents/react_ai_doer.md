# Agent: `react_ai_doer` (default)

**Path:** `agents/react_ai_doer/agent.py`  
**Backend:** AI Doer OpenAI-compatible API (`gpt-4o` by default)  
**CLI:** `python main.py` or `AGENT=react_ai_doer python main.py`  
**Required env:** `AI_DOER_API_KEY`

## Purpose

Production-style coding agent: infer output directory, optional plan, ReAct loop, auto-write after validation, escalate when stuck.

## Graph

```
set_output_dir → create_plan → reason
                    │
        ┌───────────┴───────────┐
        ▼                       │
   generate_code / validate_code / write_files / escalate / fail
        │                       │
        └──── observe ──────────┘
                 │
            (loop or END)
```

## Node reference

| Node | What it does |
|------|----------------|
| `set_output_dir` | Parses user text for a folder; else `sandbox/` |
| `create_plan` | If `PLAN_MODE=plan_and_execute`, fills `plan[]` |
| `reason` | Sets `current_action` from state / plan / limits |
| `generate_code` | Invokes LLM with `CODE_GEN_SYSTEM_PROMPT` |
| `validate_code` | Invokes LLM with `VALIDATE_SYSTEM_PROMPT`; retries bad JSON |
| `write_files` | Calls `write_file_impl` for each approved file |
| `observe` | Appends `last_observation`; advances `plan_index` |
| `escalate` | Human handoff with reason |
| `fail` | Terminal non-recoverable error |

## Routers

| Router | From | Decisions |
|--------|------|-----------|
| `after_reason` | `reason` | which ACT node / escalate / fail |
| `after_act` | ACT nodes | observe / escalate / fail |
| `after_observe` | `observe` | reason / escalate / fail / end |

## State fields

| Field | Meaning |
|-------|---------|
| `output_dir` | Project-relative write root |
| `plan` / `plan_index` | Optional plan steps |
| `step_count` | Reason iterations |
| `revision_count` | Failed validation rounds |
| `action_history` | Recent actions (loop detection) |
| `observations` / `last_observation` | Compact ReAct memory |
| `current_action` | Decision from `reason` |
| `write_results` | Disk write status lines |
| `escalation_reason` | Why the agent stopped for a human |

## Boilerplate imports

- `config.settings` — API, limits, `require_ai_doer_api_key`
- `prompts.react` — all system prompts
- `tools.paths` / `tools.file_tools` — dirs + writes
- `utils.retry` / `utils.parsing` — resilience + JSON

## Public API

`app`, `new_thread_id()`, `MAX_REVISIONS`, `MAX_STEPS`, `MODEL_NAME`, `PLAN_MODE`
