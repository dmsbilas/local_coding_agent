# Understanding `agent2.py`

A thorough guide to the **ReAct-based** autonomous coding agent — the version wired to `main.py`.

---

## 1. Big picture

`agent2.py` is an interview-ready, production-style agent that:

1. Accepts a user task once (via `main.py`).
2. Optionally creates a **plan** (`plan_and_execute` mode).
3. **Reasons** about what to do next.
4. **Acts** — generates code, validates it, or writes files.
5. **Observes** the result and loops until done, escalated, or failed.

Files are written **only** after the validator LLM fully approves the code.

### Flow diagram

```
                    ┌─────────────┐
                    │ create_plan │  (optional, PLAN_MODE=plan_and_execute)
                    └──────┬──────┘
                           ▼
              ┌────────────────────────┐
              │        reason          │◄──────────────────┐
              └───────────┬────────────┘                   │
                          │                                │
         ┌────────────────┼────────────────┐               │
         ▼                ▼                ▼               │
  generate_code    validate_code     write_files          │
         │                │                │               │
         └────────────────┼────────────────┘               │
                          ▼                                │
                    ┌──────────┐                           │
                    │ observe  │───────────────────────────┘
                    └────┬─────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
          END        escalate         fail
        (done)      (human help)   (hard error)
```

### How this differs from `agent.py`

| | `agent.py` | `agent2.py` |
|---|-----------|-------------|
| Pattern | Fixed generate → validate loop | ReAct: reason → act → observe |
| Planning | None | `reactive` or `plan_and_execute` |
| Tools | `write_file` only | `write_file` + `read_file` with Pydantic schemas |
| Retries | Minimal | LLM + tool retries with backoff |
| Failure | `fail_max_revisions` | `escalate` + `fail` + loop detection |

---

## 2. Key concepts

### ReAct (Reason → Act → Observe)

| Step | Node | What happens |
|------|------|----------------|
| **Reason** | `reason` | Picks next action from state (`generate_code`, `validate_code`, `write_files`, `escalate`) |
| **Act** | `generate_code` / `validate_code` / `write_files` | Runs the chosen action |
| **Observe** | `observe` | Records what happened; feeds back into the next reason step |

### Tool / function calling

Tools are defined with explicit **Pydantic schemas** (`WriteFileInput`, `ReadFileInput`).  
Execution goes through `execute_tool_call()` which validates arguments before running.

### Planning modes

| `PLAN_MODE` | Behavior |
|-------------|----------|
| `reactive` (default) | Decide one action at a time from current state |
| `plan_and_execute` | `create_plan` builds a short step list first; `reason` follows it |

### Stopping conditions

The agent stops when:

- **Success** — files written, `status=done`
- **Escalation** — max steps/revisions, repeated actions, or recoverable failure → `status=escalated`
- **Failure** — hard error → `status=failed`

---

## 3. Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `MODEL_NAME` | `qwen2.5-coder:7b` | Model for generate + validate |
| `PLAN_MODE` | `reactive` | `reactive` or `plan_and_execute` |
| `MAX_REVISIONS` | `6` | Max validate→revise cycles |
| `MAX_STEPS` | `12` | Max ReAct reason steps |
| `MAX_TOOL_RETRIES` | `3` | Retries per tool/LLM call |
| `MAX_VALIDATOR_ATTEMPTS` | `2` | Retries for malformed validator JSON |
| `MAX_REPEATED_ACTIONS` | `3` | Loop detection threshold |
| `LLM_TIMEOUT_SECONDS` | `120` | Ollama request timeout |
| `RETRY_BASE_SECONDS` | `1.0` | Backoff base |
| `RETRY_MAX_SECONDS` | `8.0` | Backoff cap |

---

## 4. Tools

### `write_file(relative_path, content)`

Writes UTF-8 source under `PROJECT_ROOT`. Blocks path traversal and protected files (`agent.py`, `main.py`, etc.).

### `read_file(relative_path)`

Reads a file safely from the project folder. Useful for future extensions; writes use the validated `files` payload.

### Tool execution pipeline

```
ToolCall (Pydantic)
    → _validate_tool_arguments()   # schema check
    → execute_tool_call()          # invoke with retries
    → execute_tool_calls()         # sequential (default) or parallel
```

`write_files` uses `execute_tool_calls()` with sequential writes to avoid ordering conflicts.

---

## 5. State: `AgentState`

```python
class AgentState(MessagesState, total=False):
    status: str
    approved: bool
    feedback: str
    files: list[dict[str, str]]
    iteration: int
    step_count: int
    revision_count: int
    action_history: list[str]
    observations: list[str]
    last_observation: str
    current_action: str
    plan: list[str]
    plan_index: int
    write_results: list[str]
    escalation_reason: str
```

| Field | Role |
|-------|------|
| `messages` | Full chat history |
| `status` | Current stage (`reasoning`, `generating`, `validating`, `writing`, `done`, `failed`, `escalated`, …) |
| `approved` | Validator decision |
| `feedback` | Last rejection text |
| `files` | Approved `{relative_path, content}` list ready to write |
| `iteration` | Validator round counter |
| `step_count` | ReAct step counter (vs `MAX_STEPS`) |
| `revision_count` | Revision counter (vs `MAX_REVISIONS`) |
| `action_history` | Recent actions for loop detection |
| `observations` | Last 10 observations |
| `last_observation` | Latest result summary |
| `current_action` | Action chosen by `reason` |
| `plan` / `plan_index` | Plan-and-execute tracking |
| `write_results` | Output strings from `write_file` |
| `escalation_reason` | Why human help was requested |

---

## 6. Nodes

### `create_plan`

- Runs only when `PLAN_MODE=plan_and_execute` and no plan exists.
- LLM returns JSON `{"steps": ["...", "..."]}`.
- On failure, falls back to reactive mode (empty plan).

### `reason`

Chooses `current_action` based on:

- Approved + files → `write_files`
- `revision_count >= MAX_REVISIONS` → `escalate`
- `step_count >= MAX_STEPS` → `escalate`
- Plan mode → follow plan step (generate vs validate)
- Reactive mode → generate first, then validate, revise on feedback

Increments `step_count` and appends to `action_history`.

### `generate_code`

- Prepends `CODE_GEN_SYSTEM_PROMPT`.
- Injects current plan step if planning.
- Calls `invoke_llm()` (with retries).
- Sets `status` to `generating` or `revising`.
- On LLM failure → `status=failed`, `escalation_reason` set.

### `validate_code`

- Calls validator LLM with `VALIDATE_SYSTEM_PROMPT`.
- Parses JSON: `{approved, feedback, files}`.
- Retries up to `MAX_VALIDATOR_ATTEMPTS` on malformed JSON.
- If approved → stores normalized `files`, `status=writing`.
- If rejected → injects `[VALIDATOR]` `HumanMessage`, increments `revision_count`.

### `write_files`

- Builds `ToolCall` list from approved `files`.
- Runs `execute_tool_calls(..., parallel=False)`.
- Sets `status=done` if all writes succeed, else `failed`.

### `observe`

- Appends `last_observation` to `observations` (keeps last 10).
- Advances `plan_index` in plan-and-execute mode.

### `escalate`

- Terminal node for “needs human” cases.
- Sets `status=escalated` with `escalation_reason`.

### `fail`

- Terminal node for hard failures.
- Sets `status=failed`.

---

## 7. Routers

### `after_reason`

Routes from `reason` to the chosen act node, or to `escalate`/`fail` if:

- Status is already terminal
- Same action repeated `MAX_REPEATED_ACTIONS` times
- `step_count > MAX_STEPS`

### `after_act`

After generate/validate/write:

- `done` → `observe`
- `escalating` → `escalate`
- `failed` → `fail`
- else → `observe`

### `after_observe`

- `status=done` → `END`
- `escalated` / `failed` → terminal nodes
- `step_count >= MAX_STEPS` → `escalate`
- else → `reason` (ReAct loop continues)

---

## 8. Error handling

### `invoke_with_retry`

Wraps LLM and tool calls. On rate-limit or timeout errors:

1. Exponential backoff with jitter
2. Retry up to `MAX_TOOL_RETRIES`
3. Raise `RuntimeError` if all attempts fail

### Malformed validator JSON

Validator gets a correction prompt and retries (bounded by `MAX_VALIDATOR_ATTEMPTS`).  
If still unparseable → treated as rejection with feedback to regenerate.

### Loop detection

`_repeated_action_count()` — if the same action appears `MAX_REPEATED_ACTIONS` times in a row → escalate.

---

## 9. Prompts

### `CODE_GEN_SYSTEM_PROMPT`

Expert engineer in a ReAct loop. Must implement the request, obey `[VALIDATOR]` feedback, use markdown fences, not write files itself.

### `VALIDATE_SYSTEM_PROMPT`

Strict reviewer. Returns only JSON with `approved`, `feedback`, `files`.  
`approved=true` requires at least one valid source file.

### `PLAN_SYSTEM_PROMPT`

Returns JSON `{"steps": [...]}` — short bounded plan for plan-and-execute mode.

---

## 10. How `main.py` drives `agent2`

1. Imports `app`, `new_thread_id`, limits, and `PLAN_MODE` from `agent2`.
2. Reads one task description.
3. Creates a fresh `thread_id` per run.
4. Streams `app.stream(..., stream_mode="updates")` and prints each node:
   - Node name (`reason`, `generate_code`, `validate_code`, …)
   - AI messages and `[VALIDATOR]` feedback
   - Plan steps (if created)
   - Status line: steps, revisions, action, approved
5. After stream ends, prints final summary:
   - `done` → list written files
   - `escalated` → show reason
   - `failed` → show reason

No `accept` prompt. The agent is fully autonomous.

---

## 11. End-to-end example

**User:** “Write a Python script that prints hello world”

1. `create_plan` — skipped in reactive mode (or creates 2–3 steps in plan mode)
2. `reason` → `generate_code`
3. `generate_code` → code in markdown fence
4. `observe` → records observation
5. `reason` → `validate_code`
6. `validate_code` → `approved=false`, feedback: “add `if __name__` guard”
7. `observe` → records rejection
8. `reason` → `generate_code` (revise)
9. `generate_code` → fixed code
10. `validate_code` → `approved=true`, `files: [{hello_world.py, ...}]`
11. `reason` → `write_files`
12. `write_files` → disk write
13. `observe` → `status=done` → `END`
14. CLI prints success

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| **ReAct** | Reason → Act → Observe agent loop |
| **Reactive** | Choose next action from current state only |
| **Plan-and-execute** | Create a plan first, then follow it |
| **Validator** | Second LLM pass that approves or rejects code |
| **Escalate** | Stop and ask for human help |
| **ToolCall** | Pydantic model for a normalized tool invocation |
| **Protected names** | Files the agent must not overwrite |

---

## 13. Suggested reading order

1. Big picture + flow diagram  
2. `AgentState` fields  
3. `reason` → act nodes → `observe`  
4. Routers (`after_reason`, `after_act`, `after_observe`)  
5. Tool execution + retries  
6. `main.py` streaming loop  
7. Run: `.env/bin/python main.py`
