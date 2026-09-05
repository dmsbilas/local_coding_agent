# Understanding `agent_ai_doer.py`

Deep guide to the **AI Doer–powered** autonomous coding agent.  
Same LangGraph ReAct core as `agent2.py`, but every LLM call goes to **AI Doer’s OpenAI-compatible API** (`gpt-4o`).

This document is organized around the concepts interviewers and agent frameworks care about most:

1. Tool / function calling mechanics  
2. The ReAct pattern (Reason → Act → Observe)  
3. Planning strategies (plan-and-execute vs reactive)  
4. Stopping conditions (done / stuck / escalate)  
5. Error handling & retries  

---

## 0. Big picture

```
set_output_dir → create_plan → reason ⟷ (generate | validate | write) → observe
                                      ↓
                               escalate / fail → END
```

| Phase | What happens |
|-------|----------------|
| `set_output_dir` | Pick write folder (user-mentioned path, else `sandbox/`) |
| `create_plan` | Optional bounded plan (`PLAN_MODE=plan_and_execute`) |
| `reason` | Choose next action from state (no LLM) |
| Act | `generate_code` / `validate_code` (AI Doer) or `write_files` (local tools) |
| `observe` | Record result; loop back to `reason` |
| Terminal | `done`, `escalated`, or `failed` → `END` |

**No human “accept” step.** When the validator sets `approved=true` with valid files, `write_files` runs automatically.

### AI Doer call path (short)

```python
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=AI_DOER_API_KEY,
    base_url="https://ai-doer.com/v1/api",
    temperature=0.2,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=0,                 # retries owned by invoke_with_retry()
    model_kwargs={"stream": False},
)
```

Under the hood that is `POST /chat/completions` with `Authorization: Bearer aiob_…`.

Nodes that call AI Doer: `create_plan` (optional), `generate_code`, `validate_code`.  
Nodes that do **not**: `reason`, `observe`, `write_files`, `escalate`, `fail`.

---

## 1. Tool / function calling mechanics

Agents become useful when the model can request **actions** (tools) instead of only emitting text. This codebase treats tools as **first-class, schema-validated functions** — not free-form JSON the model invents in chat.

### 1.1 Schema design

Each tool has:

1. A **Pydantic input model** (`WriteFileInput`, `ReadFileInput`) — field names, types, descriptions, constraints (`min_length=1`).
2. A `@tool(args_schema=...)` wrapper — LangChain registers the schema so callers (or models) know the contract.
3. A **normalized `ToolCall` model** — `{ name: Literal["write_file","read_file"], arguments: dict }` — so every call is the same shape before execution.

```python
class WriteFileInput(BaseModel):
    relative_path: str = Field(..., min_length=1, description="...")
    content: str = Field(..., description="Complete source-file contents...")

class ToolCall(BaseModel):
    name: Literal["write_file", "read_file"]
    arguments: dict[str, Any]

@tool(args_schema=WriteFileInput)
def write_file(relative_path: str, content: str) -> str:
    ...
```

**Why schemas matter**

| Without schema | With schema |
|----------------|-------------|
| Model invents wrong keys (`path` vs `relative_path`) | Validation fails early with a clear error |
| Empty / huge / wrong-type args sneak through | Pydantic rejects before side effects |
| Hard to document for humans & models | Descriptions become the tool “API docs” |

Registry:

```python
TOOLS: dict[str, BaseTool] = {"write_file": write_file, "read_file": read_file}
```

Unknown tool names never execute.

### 1.2 Execution pipeline

```
raw model/tool payload
    → _parse_tool_call()          # tolerate JSON, fences, surrounding prose
    → ToolCall.model_validate()   # name must be allowed
    → _validate_tool_arguments()  # args_schema.model_validate(...)
    → invoke_with_retry(tool.invoke)
    → string observation back into the agent loop
```

`write_files` (post-approval) uses the same safety path via `_write_file_impl`, with the run’s `output_dir` from state — so disk writes still go through path sandboxing and protected-file checks.

### 1.3 Sequential vs parallel tool calls

```python
def execute_tool_calls(calls: list[ToolCall], *, parallel: bool = False) -> list[str]:
    if not parallel or len(calls) == 1:
        return [execute_tool_call(c) for c in calls]   # sequential (default)

    with ThreadPoolExecutor(...) as pool:
        return [f.result() for f in futures]           # parallel
```

| Mode | When to use | Risk |
|------|-------------|------|
| **Sequential** (default) | File writes, ordered steps, dependent tools | Slower, but safe |
| **Parallel** | Independent reads / checks with no shared mutation | Race conditions, conflicting writes |

**Rule of thumb:** anything that **mutates shared state** (disk, DB, tickets) → sequential. Pure reads → parallel is fine.

Approved multi-file writes in this agent are **always sequential** so `a/__init__.py` can exist before `a/app.py` depends on it, and two tools never fight over the same path.

### 1.4 Handling malformed tool-call output

Models often return broken tool payloads: prose + JSON, fenced blocks, wrong types, missing fields.

This agent **never executes blind**:

1. `_parse_tool_call` tries dict → JSON string → fenced JSON → first `{...}` substring.  
2. If parse fails → `None` (no execution).  
3. If parse succeeds but `ValidationError` on schema → return `"Tool error: malformed arguments..."` as the **observation**.  
4. That string is what the next Reason step sees — so the agent can correct course instead of crashing.

Same idea for **validator structured output** (not a tool, but the same failure class):

- Expect JSON `{approved, feedback, files}`  
- On garbage: retry up to `MAX_VALIDATOR_ATTEMPTS` with an explicit “reply with ONLY JSON” correction  
- Still bad → treat as `approved=false` with actionable feedback (do not write files)

**Interview takeaway:** tool calling is not “trust the model.” It is **parse → validate → execute → observe**, with soft failures preferred over hard crashes.

---

## 2. The ReAct pattern (Reason → Act → Observe)

ReAct is the base loop almost every agent framework wraps (LangGraph, AutoGPT-style runners, tool agents, etc.).

```
┌──────────┐     ┌──────────┐     ┌───────────┐
│  REASON  │────▶│   ACT    │────▶│  OBSERVE  │──┐
└──────────┘     └──────────┘     └───────────┘  │
     ▲                                           │
     └───────────────────────────────────────────┘
```

### 2.1 How this repo maps the pattern

| ReAct step | Node(s) | LLM? | Job |
|------------|---------|------|-----|
| **Reason** | `reason` | No | Pick `generate_code` / `validate_code` / `write_files` / `escalate` from state |
| **Act** | `generate_code`, `validate_code`, `write_files` | Yes for gen/validate | Do the chosen work |
| **Observe** | `observe` | No | Append `last_observation`; advance plan index |

Routers glue the loop:

- `after_reason` → which Act node  
- `after_act` → `observe` (or escalate/fail on hard errors)  
- `after_observe` → `reason` again, or `END` if done  

### 2.2 Why separate Reason from Act?

If Act also “decides,” you get tangled prompts and harder debugging. Here:

- **Reason** is deterministic Python over structured state (`approved`, `feedback`, `step_count`, plan index…).  
- **Act** focuses on one job (write code, validate JSON, write disk).  
- **Observe** forces every action to leave a trail for the next decision.

That separation is what makes stopping conditions and planning modes possible without rewriting the LLM prompts every time.

### 2.3 Concrete coding loop

1. Reason → `generate_code`  
2. Act → AI Doer emits source in fences  
3. Observe → store result  
4. Reason → `validate_code`  
5. Act → AI Doer returns approve/reject JSON  
6. If reject → Observe → Reason → `generate_code` again with `[VALIDATOR]` feedback  
7. If approve → Reason → `write_files` → Observe → END  

---

## 3. Planning strategies

Controlled by `PLAN_MODE`:

| Value | Behavior |
|-------|----------|
| `reactive` (default) | No upfront plan. Each Reason step reacts to the latest observation. |
| `plan_and_execute` | `create_plan` asks AI Doer for a short JSON step list once, then Reason follows it as guidance. |

### 3.1 Reactive (step-by-step)

**How Reason chooses (simplified):**

- Approved + files → `write_files`  
- Else if feedback and not approved → `generate_code` (revise)  
- Else if just generated → `validate_code`  
- First turn → `generate_code`  
- Else → `validate_code`

**When reactive is appropriate**

- Short coding tasks with clear success criteria  
- Environments where the next step depends heavily on the last tool/validator result  
- You want lower latency / fewer LLM calls up front  

**Weakness:** can thrash on large multi-stage work without a global roadmap.

### 3.2 Plan-and-execute

`create_plan` (only if `PLAN_MODE=plan_and_execute` and no plan yet):

1. Call AI Doer with `PLAN_SYSTEM_PROMPT`  
2. Parse `{"steps": ["...", "..."]}`  
3. Cap length to `MAX_STEPS`  
4. On failure → empty plan and **fall back to reactive** (do not spin forever)

During Reason, the current plan step (by `plan_index`) biases generate vs validate. After each Observe, `plan_index` advances.

**When plan-and-execute is appropriate**

- Multi-file features with natural phases (“scaffold → implement → tests → polish”)  
- Tasks where humans want visibility into intended steps  
- Longer runs where reactive wandering is costly  

**Weakness:** a bad plan can mislead; this agent mitigates by still checking **state** every Reason turn (approval / revision limits override the plan).

### 3.3 Hybrid reality

Even in plan-and-execute mode, **stopping conditions and approval still win**. The plan is guidance, not a rigid script — that is the healthy pattern for production agents.

---

## 4. Stopping conditions

Knowing when to stop is a common blind spot and a strong interview differentiator. Infinite loops burn money and trust.

### 4.1 Success — agent is done

| Signal | Meaning |
|--------|---------|
| Validator `approved=true` + normalized `files` | Quality gate passed |
| `write_files` all return `Successfully…` | Side effects committed |
| `status=done` → `after_observe` → `END` | Graph terminates |

Files are written **without human confirmation** once approved.

### 4.2 Stuck in a loop — detect and break

| Detector | Config | Action |
|----------|--------|--------|
| Same Act chosen repeatedly | `MAX_REPEATED_ACTIONS` (default 3) | `after_reason` → `escalate` |
| Too many ReAct steps | `MAX_STEPS` (default 12) | Reason / observe → `escalate` |
| Too many revise cycles | `MAX_REVISIONS` (default 6) | Reason → `escalate` |

`action_history` records every Reason decision; `_repeated_action_count` looks at the trailing streak of identical actions.

There is also `_state_signature` (hash of action + feedback + counters) for detecting “same world, same choice” style stalls.

### 4.3 Escalate to a human

`escalate` is a **first-class terminal**, not an afterthought:

```text
🧑‍💻 Human escalation required.
Reason: <feedback or escalation_reason>
Steps: N; revisions: M
```

Use escalation when:

- Revision budget exhausted without approval  
- Step budget exhausted  
- Repeated identical actions  
- Recoverable LLM/tool failures that still block progress (`status=escalating`)  

**Fail** (`status=failed`) is for non-recoverable Act failures (e.g. code generation exhausted retries).

### 4.4 Interview framing

A mature agent answers three questions every loop:

1. **Am I done?** (success criteria met)  
2. **Am I stuck?** (no progress / repetition / budget)  
3. **Should a human take over?** (unsafe, ambiguous, or budget blown)  

If you only implement (1), demos look fine until production.

---

## 5. Error handling & retries

External systems fail. Agents must treat that as normal.

### 5.1 Classification

```python
_is_rate_limit_error  # 429, "rate limit", "too many requests", quota…
_is_timeout_error     # TimeoutError, "timed out", "deadline exceeded"
```

Only **transient** classes retry. Validation errors and unknown tools do **not** get infinite backoff — they return an error observation immediately.

### 5.2 Bounded retry with exponential backoff

```python
def _backoff(attempt: int) -> float:
    delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** attempt))
    return delay + random.uniform(0, delay * 0.1)  # jitter
```

`invoke_with_retry` wraps both LLM calls (`invoke_llm`) and tool invokes (`execute_tool_call`), capped by `MAX_TOOL_RETRIES`.

`ChatOpenAI(..., max_retries=0)` on purpose — **one** retry owner avoids double-retry stampedes.

### 5.3 Failure matrix

| Failure | Handler | Outcome |
|---------|---------|---------|
| Missing `AI_DOER_API_KEY` | `_require_api_key` | Fast hard error before calls |
| Rate limit / timeout | `invoke_with_retry` | Sleep + retry; then escalate/fail |
| Malformed validator JSON | `MAX_VALIDATOR_ATTEMPTS` loop | Correction prompt; then reject (no write) |
| Malformed tool JSON / args | parse + Pydantic | Error string observation |
| Tool / path escape / protected file | `_write_file_impl` | Error string; may mark write `failed` |
| Planning JSON broken | `create_plan` except | Fall back to reactive |
| Generation exhausted retries | `generate_code` | `status=failed` → fail terminal |

### 5.4 Malformed responses vs tool failures

These are different:

- **Malformed response** — model returned text that does not match the contract (JSON / tool schema). Fix with **re-prompt + validate**, not by executing garbage.  
- **Tool failure** — schema was fine; the side effect failed (IO, API 500). Fix with **retry / backoff / escalate**.  

Confusing the two is a common agent bug (e.g. retrying a permanently invalid schema forever).

### 5.5 Timeouts

`LLM_TIMEOUT_SECONDS` (default 120) is passed into `ChatOpenAI`. Combined with backoff caps (`RETRY_MAX_SECONDS`), worst-case wait stays bounded — critical for CLI agents that otherwise appear hung.

---

## 6. Output directory (where files land)

| User request | `output_dir` |
|--------------|--------------|
| Mentions a folder (`in projects/todo`, `under ./my_app`, `folder: demo`) | That folder (under repo root) |
| No folder mentioned | `sandbox/` (`SANDBOX_DIR`) |

`set_output_dir` runs first. Writes still refuse:

- Paths outside the project  
- Blocked roots (`.git`, `.venv`, …)  
- Overwriting repo-root agent files (`main.py`, `agent_ai_doer.py`, …)

---

## 7. Configuration cheat sheet

| Variable | Default | Topic it controls |
|----------|---------|-------------------|
| `AI_DOER_API_KEY` | required | Auth |
| `AI_DOER_BASE_URL` | `https://ai-doer.com/v1/api` | LLM endpoint |
| `MODEL_NAME` | `gpt-4o` | Model |
| `PLAN_MODE` | `reactive` | Planning strategy |
| `MAX_STEPS` | `12` | Stopping (ReAct budget) |
| `MAX_REVISIONS` | `6` | Stopping (revise budget) |
| `MAX_REPEATED_ACTIONS` | `3` | Stopping (loop detect) |
| `MAX_TOOL_RETRIES` | `3` | Retries |
| `MAX_VALIDATOR_ATTEMPTS` | `2` | Malformed JSON retries |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeouts |
| `RETRY_BASE_SECONDS` / `RETRY_MAX_SECONDS` | `1` / `8` | Backoff |
| `SANDBOX_DIR` | `sandbox` | Default output folder |

---

## 8. How to run

```bash
source .venv/bin/activate
# .env must contain AI_DOER_API_KEY=aiob_...
python main.py
# or: .venv/bin/python main.py
# or: ./run.sh
```

`main.py` streams each node (`set_output_dir`, `reason`, `generate_code`, …) and prints success under the chosen `output_dir`, or escalation/failure reasons.

---

## 9. End-to-end example

**User:** `Create a hello world Python script in demos/hello`

1. `set_output_dir` → `demos/hello`  
2. `create_plan` → skipped (reactive)  
3. Reason → Act `generate_code` (AI Doer)  
4. Observe → Reason → Act `validate_code` (AI Doer JSON)  
5. If rejected → Reason → generate again with `[VALIDATOR]` feedback  
6. If approved → Reason → Act `write_files` → `demos/hello/...` on disk **automatically**  
7. Observe → END (`status=done`)  

If the agent keeps failing validation for 6 revisions, or repeats the same action 3 times → **escalate** with a human-readable reason instead of looping forever.

---

## 10. Suggested study order (interview prep)

1. **ReAct** section — draw the loop from memory  
2. **Stopping conditions** — explain done vs stuck vs escalate  
3. **Tool calling** — schema → validate → sequential vs parallel → malformed handling  
4. **Planning** — when you’d pick reactive vs plan-and-execute  
5. **Retries** — transient vs permanent failures; why one retry owner  
6. Trace one live run with `.venv/bin/python main.py` and map each printed node to Reason/Act/Observe  

That set is exactly what this agent is built to demonstrate.
