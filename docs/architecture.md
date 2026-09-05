# Architecture

## System context (default: react_ai_doer)

```
┌──────────────┐     stream      ┌─────────────────────────┐     HTTPS      ┌─────────────────┐
│   main.py    │ ──────────────▶ │  agents/react_ai_doer   │ ─────────────▶ │  AI Doer API    │
│              │ ◀────────────── │  StateGraph (ReAct)     │ ◀───────────── │  gpt-4o         │
│ • prompt     │   node updates  │                         │                │ ai-doer.com     │
│ • print      │                 └───────────┬─────────────┘                └─────────────────┘
│ • summary    │                             │
└──────────────┘                             ▼
                                   ┌─────────────────────┐
                                   │ MemorySaver         │
                                   │ (per thread_id)     │
                                   └─────────┬───────────┘
                                             ▼
                                   ┌─────────────────────┐
                                   │ Disk writes         │
                                   │ user folder or      │
                                   │ sandbox/            │
                                   └─────────────────────┘
```

Shared layers used by all agents:

```
agents/*  →  prompts/*  (system text)
          →  tools/*    (file IO + path safety)
          →  config/*   (env / limits)
          →  utils/*    (retry + parsing)
```

---

## Agent A — `basic`

**Idea:** Smallest autonomous loop. No separate Reason node, no plan, no escalate split.

```
generate_code → validate_code ⟲ → write_files → END
                     │
                     └─ fail_max_revisions → END
```

| Node | Role |
|------|------|
| `generate_code` | LLM writes/revises code (no disk) |
| `validate_code` | LLM returns JSON `{approved, feedback, files}` |
| `write_files` | Writes approved files under `sandbox/` |
| `fail_max_revisions` | Stops after `MAX_REVISIONS` |

---

## Agent B/C — ReAct (`react_ollama`, `react_ai_doer`)

**Idea:** Explicit REASON → ACT → OBSERVE with optional planning and escalation.

```
set_output_dir → create_plan → reason ──▶ ACT (generate | validate | write | escalate | fail)
                                  ▲              │
                                  │              ▼
                                  └────────── observe → END (when done)
```

| Node | Role |
|------|------|
| `set_output_dir` | Infer write folder from user text (else `sandbox/`) |
| `create_plan` | If `PLAN_MODE=plan_and_execute`, ask LLM for steps |
| `reason` | Choose next action from state (reactive or plan-guided) |
| `generate_code` | ACT: produce/revise code |
| `validate_code` | ACT: structured validation + file payload |
| `write_files` | ACT: auto-write approved files |
| `observe` | Record observation; advance plan index |
| `escalate` / `fail` | Safe terminal stops |

### Stopping conditions

- `MAX_STEPS` / `MAX_REVISIONS` exceeded → escalate  
- Same action repeated `MAX_REPEATED_ACTIONS` times → escalate  
- Non-recoverable LLM/tool failure → fail  
- Successful write → status `done` → END  

### Planning modes

| `PLAN_MODE` | Behavior |
|-------------|----------|
| `reactive` (default) | Each reason step picks generate/validate/write from latest observation |
| `plan_and_execute` | One-shot plan JSON; steps guide actions but state still wins |

---

## Data written to disk

- Paths are always resolved under `PROJECT_ROOT / output_dir`
- Boilerplate packages and protected filenames cannot be overwritten
- Validator `files[]` are sanitized via `tools.file_tools.normalize_files`
