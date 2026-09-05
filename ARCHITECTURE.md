# System Architecture: Local AI Coding Agent

## 1. Overview

A LangGraph-powered **autonomous** coding CLI that uses a local Ollama instance (default model `qwen2.5-coder:7b`). The user describes desired code once via the terminal; the agent plans (optionally), reasons, generates code, validates it with the LLM, revises until approved, writes files to disk, and finishes — with no manual accept step.

The active agent implementation is **`agent2.py`** (ReAct + planning + tool framework). **`agent.py`** remains as a simpler fixed-loop alternative.

**Repository root:** `local_coding_agent/`

---

## 2. Project Structure

```
local_coding_agent/
├── .venv/                # Python virtual environment (venv)
├── agent.py              # Simple autonomous agent (generate → validate loop)
├── agent2.py             # ReAct agent — wired to main.py
├── agent.md              # Guide for agent.py
├── agent2.md             # Guide for agent2.py
├── main.py               # CLI entry point (streams agent2 graph)
├── ARCHITECTURE.md       # This file
└── .gitignore
```

| File | Purpose |
|------|---------|
| `agent2.py` | ReAct LangGraph: plan → reason → act → observe; tools, retries, escalation |
| `agent.py` | Simpler graph: `generate_code` → `validate_code` → write / revise / fail |
| `main.py` | Reads one task, streams `agent2.app`, prints progress and final result |
| `agent2.md` | Detailed explanation of `agent2.py` |
| `agent.md` | Detailed explanation of `agent.py` |

---

## 3. Component Diagram

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  main.py    │──────▶│   agent2.py      │──────▶│  ChatOllama     │
│             │ stream│                  │ invoke│  (localhost:    │
│  • read UI  │◀──────│  StateGraph      │       │   11434)        │
│  • stream   │ state │  (ReAct loop)    │       │  qwen2.5-coder  │
│  • summary  │       │                  │       │  :7b            │
└─────────────┘       └────────┬─────────┘       └─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ MemorySaver         │
                    │ (per-thread state)  │
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ write_file /        │
                    │ read_file tools     │
                    │ (project sandbox)   │
                    └─────────────────────┘
```

---

## 4. Data Flow (`agent2.py`)

### 4.1 Successful path

```
User describes task
    → main.py streams agent2.app
    → create_plan (optional, PLAN_MODE=plan_and_execute)
    → reason (pick action)
    → generate_code (LLM writes/revises code in chat)
    → observe
    → reason → validate_code (LLM returns JSON: approved / feedback / files)
    → observe
    → [if not approved] reason → generate_code → validate_code … (bounded by MAX_REVISIONS)
    → [if approved] reason → write_files (write_file tool, sequential)
    → observe → status=done → END
    → main.py prints written file paths
```

### 4.2 ReAct loop

Each step follows:

| Phase | Node | Role |
|-------|------|------|
| Reason | `reason` | Choose next action from state |
| Act | `generate_code` / `validate_code` / `write_files` | Execute action |
| Observe | `observe` | Record result; advance plan index |

### 4.3 Exit paths

| Condition | Status | Action |
|-----------|--------|--------|
| Validator approves + files written | `done` | CLI prints success + paths |
| Max revisions / steps / repeated actions | `escalated` | CLI suggests human review |
| LLM or tool hard failure | `failed` | CLI prints error reason |
| User types `quit` | — | Exit CLI |

There is **no** user feedback or `accept` loop in `main.py`.

---

## 5. LangGraph State Machine (`agent2.py`)

### 5.1 State schema

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

`MessagesState` appends messages returned from each node.

### 5.2 Nodes

| Node | Action |
|------|--------|
| `create_plan` | Optional bounded plan (`plan_and_execute` mode) |
| `reason` | ReAct: select `generate_code`, `validate_code`, `write_files`, or `escalate` |
| `generate_code` | LLM produces/revises source (markdown fences) |
| `validate_code` | LLM returns approval JSON + file payloads or feedback |
| `write_files` | Persist approved files via `write_file` tool |
| `observe` | Store observation; advance plan |
| `escalate` | Human escalation terminal node |
| `fail` | Hard failure terminal node |

### 5.3 Routers

| From | Router | Targets |
|------|--------|---------|
| `reason` | `after_reason` | `generate_code`, `validate_code`, `write_files`, `escalate`, `fail` |
| act nodes | `after_act` | `observe`, `escalate`, `fail` |
| `observe` | `after_observe` | `reason`, `escalate`, `fail`, `END` |

### 5.4 Graph wiring

```
create_plan → reason ⟷ (generate | validate | write) → observe
                              ↓
                         escalate / fail → END
                              ↓
                    observe → END (when done)
```

Entry point: **`create_plan`**.  
Checkpointer: **`MemorySaver()`** — fresh `thread_id` per task from `new_thread_id()`.

---

## 6. Configuration

Environment variables (via `dotenv`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL_NAME` | `qwen2.5-coder:7b` | Model identifier |
| `PLAN_MODE` | `reactive` | `reactive` or `plan_and_execute` |
| `MAX_REVISIONS` | `6` | Max validate→revise cycles |
| `MAX_STEPS` | `12` | Max ReAct reason steps |
| `MAX_TOOL_RETRIES` | `3` | LLM/tool retry attempts |
| `MAX_VALIDATOR_ATTEMPTS` | `2` | Retries for malformed validator JSON |
| `MAX_REPEATED_ACTIONS` | `3` | Loop detection threshold |
| `LLM_TIMEOUT_SECONDS` | `120` | Ollama request timeout |
| `RETRY_BASE_SECONDS` | `1.0` | Exponential backoff base |
| `RETRY_MAX_SECONDS` | `8.0` | Backoff cap |

Temperature is hardcoded to `0.2` in `ChatOllama` for deterministic output.

---

## 7. Tools & safety

| Tool | Purpose |
|------|---------|
| `write_file` | Write approved source under `PROJECT_ROOT` |
| `read_file` | Read project files safely (for future use) |

Safety:

- Path traversal blocked (`../` escapes rejected)
- Protected files cannot be overwritten: `agent.py`, `main.py`, `architecture.md`, `agent.md`, `.gitignore`
- Tool arguments validated with Pydantic before execution
- Writes run sequentially to avoid ordering conflicts

---

## 8. External Dependencies

| Package | Role |
|---------|------|
| `langgraph` | `StateGraph`, `END`, conditional edges, checkpointing |
| `langchain-core` | Messages, tools |
| `langchain-ollama` | `ChatOllama` bridge |
| `pydantic` | Tool schemas, `ToolCall` validation |
| `python-dotenv` | Load `.env` |

**Required service:** Ollama running locally with the model pulled:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

---

## 9. Execution

```bash
.venv/bin/python main.py
```

- One task per run; fresh `thread_id` per invocation
- Progress streamed node-by-node (`stream_mode="updates"`)
- `recursion_limit` scaled to `MAX_STEPS` for ReAct depth

---

## 10. `agent.py` vs `agent2.py`

| Aspect | `agent.py` | `agent2.py` |
|--------|-----------|-------------|
| Pattern | Fixed generate → validate | ReAct reason → act → observe |
| Planning | No | Optional `plan_and_execute` |
| Tools | `write_file` direct | Pydantic tools + execution framework |
| Retries | Validator JSON only | LLM + tool backoff |
| Failure | `fail_max_revisions` | `escalate` + `fail` + loop detection |
| Used by `main.py` | No (legacy) | **Yes** |

Use `agent.py` for a minimal learning example; use `agent2.py` for the full autonomous CLI experience.

---

## 11. Further reading

- [`agent2.md`](agent2.md) — line-by-line concepts and walkthrough for `agent2.py`
- [`agent.md`](agent.md) — guide for the simpler `agent.py`
