# Understanding `agent_ai_doer.py`

Guide to the **AI Doer–powered** ReAct coding agent — the version wired to `main.py`.

Same LangGraph workflow as `agent2.py`, but every LLM call goes to **AI Doer’s OpenAI-compatible API** (`gpt-4o`) instead of local Ollama.

---

## 1. Big picture

`agent_ai_doer.py` is an autonomous coding agent that:

1. Accepts a user task once (via `main.py`).
2. Optionally creates a **plan** (`plan_and_execute` mode).
3. **Reasons** about the next action.
4. **Acts** — generates code, validates it, or writes files (via AI Doer for LLM steps).
5. **Observes** the result and loops until done, escalated, or failed.

Files are written **only** after the validator LLM fully approves the code.

### Flow diagram

```
                    ┌─────────────┐
                    │ create_plan │  (optional; calls AI Doer if PLAN_MODE=plan_and_execute)
                    └──────┬──────┘
                           ▼
              ┌────────────────────────┐
              │        reason          │◄──────────────────┐
              └───────────┬────────────┘                   │
                          │                                │
         ┌────────────────┼────────────────┐               │
         ▼                ▼                ▼               │
  generate_code    validate_code     write_files          │
  (AI Doer)        (AI Doer)         (local disk)          │
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
```

### vs `agent2.py`

| | `agent2.py` | `agent_ai_doer.py` |
|---|-------------|---------------------|
| LLM backend | Local Ollama (`ChatOllama`) | AI Doer cloud (`ChatOpenAI`) |
| Default model | `qwen2.5-coder:7b` | `gpt-4o` |
| Auth | None (local) | `AI_DOER_API_KEY` (Bearer) |
| Base URL | `http://localhost:11434` | `https://ai-doer.com/v1/api` |
| Graph / ReAct / tools | Same | Same |

---

## 2. How AI Doer is called

### 2.1 API setup (from AI Doer docs)

| Item | Value |
|------|--------|
| Base URL | `https://ai-doer.com/v1/api` |
| Endpoint used | `POST /chat/completions` |
| Auth | `Authorization: Bearer <AI_DOER_API_KEY>` |
| Key prefix | `aiob_…` (create on Integrations at ai-doer.com) |
| Compatibility | OpenAI SDK — set `base_url` to the URL above |

Equivalent curl (what the SDK does under the hood):

```bash
curl https://ai-doer.com/v1/api/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{ "role": "user", "content": "Hello!" }],
    "stream": false
  }'
```

### 2.2 Client wiring in code

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_NAME,                    # default: gpt-4o
    api_key=AI_DOER_API_KEY,
    base_url=AI_DOER_BASE_URL,           # https://ai-doer.com/v1/api
    temperature=0.2,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=0,                       # our own invoke_with_retry()
    model_kwargs={"stream": False},      # AI Doer streams by default; we need full JSON
)
```

`langchain_openai.ChatOpenAI` talks to any OpenAI-compatible server. Pointing `base_url` at AI Doer is enough — no Ollama involved.

### 2.3 Single entry point: `invoke_llm()`

Every generation / validation / planning call goes through:

```python
def invoke_llm(messages, operation: str):
    _require_api_key()   # fail fast if AI_DOER_API_KEY missing
    return invoke_with_retry(
        lambda: llm.invoke(messages),
        attempts=MAX_TOOL_RETRIES,
        operation=operation,
    )
```

That means:

1. Key must be set in `.env` / environment.
2. Transient errors (rate limit, timeout) get exponential backoff retries.
3. Permanent failures escalate instead of looping forever.

### 2.4 Which nodes call AI Doer?

| Node | Calls AI Doer? | Purpose |
|------|----------------|---------|
| `create_plan` | Yes (if `plan_and_execute`) | Short JSON plan |
| `reason` | **No** | Pure Python decision logic |
| `generate_code` | **Yes** | Write / revise source code |
| `validate_code` | **Yes** | Approve or reject with JSON |
| `write_files` | **No** | Local `write_file` tool only |
| `observe` / `escalate` / `fail` | **No** | State bookkeeping |

---

## 3. Code generation path in detail

When `reason` chooses `generate_code`:

1. Build messages:
   - System: `CODE_GEN_SYSTEM_PROMPT` (ReAct engineer rules)
   - Full chat history (`state["messages"]`)
   - Optional: current plan step as a user message
2. `invoke_llm(messages, "code generation")` → AI Doer `gpt-4o`
3. Append the AI reply to state
4. Set `status` to `generating` or `revising`
5. Next: `observe` → `reason` → usually `validate_code`

The model is instructed to:

- Emit complete runnable source in markdown fences
- Treat `[VALIDATOR]` feedback as mandatory
- **Not** write files itself (disk write happens only after approval)

---

## 4. Validation path in detail

When `reason` chooses `validate_code`:

1. Call AI Doer with `VALIDATE_SYSTEM_PROMPT` + history
2. Expect **only** JSON:

```json
{
  "approved": true,
  "feedback": "",
  "files": [
    { "relative_path": "hello_world.py", "content": "print('hi')\n" }
  ]
}
```

3. If malformed → retry up to `MAX_VALIDATOR_ATTEMPTS`
4. If `approved=true` + valid files → later `write_files`
5. If rejected → inject `[VALIDATOR]` feedback → generate again

---

## 5. Configuration

Put secrets in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_DOER_API_KEY` | _(required)_ | Bearer key (`aiob_…`) |
| `AI_DOER_BASE_URL` | `https://ai-doer.com/v1/api` | API root |
| `MODEL_NAME` | `gpt-4o` | Model id from `GET /models` |
| `PLAN_MODE` | `reactive` | `reactive` or `plan_and_execute` |
| `MAX_REVISIONS` | `6` | Validate→revise cap |
| `MAX_STEPS` | `12` | ReAct step cap |
| `MAX_TOOL_RETRIES` | `3` | LLM/tool retries |
| `MAX_VALIDATOR_ATTEMPTS` | `2` | Bad JSON retries |
| `MAX_REPEATED_ACTIONS` | `3` | Loop detection |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout |

---

## 6. Tools & output location

| Tool | Role |
|------|------|
| `write_file` | Write approved source under the run’s output directory |
| `read_file` | Read files from that directory |

**Where files go (automatic — no human accept):**

1. If the user **mentions a folder** (e.g. `in projects/todo`, `under ./my_app`, `folder: demo_cli`) → write there (under the repo root).
2. If **no folder** is mentioned → write under `sandbox/` (or `SANDBOX_DIR`).

`set_output_dir` runs first and records `output_dir` in state. After the validator sets `approved=true`, `write_files` persists immediately.

Paths cannot escape the project, overwrite repo-root agent files (`main.py`, `agent_ai_doer.py`, …), or use blocked roots (`.git`, `.venv`, …).

`write_files` writes sequentially via `_write_file_impl`.

---

## 7. State, routers, stopping (same as agent2)

State tracks: `messages`, `approved`, `feedback`, `files`, `iteration`, `step_count`, `revision_count`, `action_history`, `observations`, `plan`, `write_results`, `escalation_reason`.

Routers:

- `after_reason` → act node or escalate/fail  
- `after_act` → observe / escalate / fail  
- `after_observe` → reason again or END  

Stops on: success (`done`), escalation, hard failure, max steps/revisions, or repeated actions.

---

## 8. How `main.py` drives this agent

```bash
source .venv/bin/activate
# ensure .env has AI_DOER_API_KEY=aiob_...
python main.py
# or: .venv/bin/python main.py
# or: ./run.sh
```

`main.py` imports `app` from `agent_ai_doer`, streams each node, and prints:

- Generated / validator messages  
- Plan (if any)  
- Final written files, or escalation/failure reason  

No manual `accept` step.

---

## 9. End-to-end example

**User:** “Write a Python hello world script”

1. `create_plan` — skipped in reactive mode  
2. `reason` → `generate_code`  
3. **AI Doer** returns code in a fence  
4. `observe` → `reason` → `validate_code`  
5. **AI Doer** returns `approved=true` + `files`  
6. `reason` → `write_files` → disk write  
7. `observe` → `END` → CLI shows success  

Billing: each AI Doer chat completion uses your BDT credit balance (`GET /balance` on the API).

---

## 10. Suggested reading order

1. Section **2** (AI Doer call path)  
2. Flow diagram + generate / validate sections  
3. Config / `.env`  
4. Compare with [`agent2.md`](agent2.md) for shared ReAct details  
5. Run: `.venv/bin/python main.py`
