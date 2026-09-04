# System Architecture: Local AI Coding Agent

## 1. Overview

A LangGraph-powered agentic CLI that iteratively generates, reviews, and refines code using a local Ollama instance (default model `qwen2.5-coder:7b`). The user describes desired code via the terminal; the agent loops through generation and review stages, pausing for feedback until the user accepts the output or exits.

**Repository root:** `/Users/abuhaidersiddiq/codes/playground/local_coding_agent`

---

## 2. Project Structure

```
local_coding_agent/
├── .env/                 # Python virtual environment (venv)
│   ├── bin/              # Executables (python, pip)
│   └── lib/              # Installed site-packages
├── __pycache__/          # Compiled Python bytecode cache
├── .gitignore            # Git ignore rules
├── .git/                 # Git repository metadata
├── agent.py              # LangGraph state graph, nodes, and edge routers
└── main.py               # CLI entry point – interactive user loop
```

| File | Size | Purpose |
|---|---|---|
| `agent.py` | ~4.8 KB | Defines the `AgentState` schema, LLM wiring (`ChatOllama`), three workflow nodes (`generate_code`, `review_code`, `ask_user`), two routing functions, and compiles the `StateGraph`. |
| `main.py` | ~2.7 KB | Interactive CLI driver: reads stdin prompts, calls `app.invoke()` with threaded checkpoint state, handles accept/feedback/quit logic. |
| `.env/` | venv dir | Isolated Python 3.14 virtualenv with all dependencies pre-installed. |

---

## 3. Component Diagram

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  main.py    │──────▶│   agent.py       │──────▶│  ChatOllama     │
│             │  invoke│                  │  llm   │  (Localhost:114 │
│  • read UI  │◀──────│  StateGraph      │invoke │   34) / qwen2.5 │
│  • loop     │  state │  (CompiledApp)  │        │  -coder:7b      │
│  • print    │        │                  │        │                 │
└─────────────┘       └────────┬─────────┘       └─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ MemorySaver         │
                    │ (in-memory check-   │
                    │  pointer per thread)│
                    └─────────────────────┘
```

---

## 4. Data Flow

### 4.1 Initial Code Generation Cycle

```
User ──► Describe code ──► main.py ──► app.invoke({"messages": [("user", "...")]})
                                                                       │
                                            Entry point: generate_code
                                                                       │
                                              Prepend system prompt + msgs
                                                       │
                                               llm.invoke()
                                                        │
                                                   Assistant responds
                                                        │
                                          Return {messages, status="generating"}
                                                        │
                                   Router should_continue → "review_code"
                                                        │
                                       Prepend review system prompt + msgs
                                                │
                                           llm.invoke()
                                                    │
                                             Reviewer responds
                                                    │
                                Router continue_or_end → "ask_user"
                                                        │
                                          Return {messages, status="pending_feedback"}
```

### 4.2 Feedback Loop

```
User types feedback ──► "[USER] <feedback>" ──► main.py invokes app again
                                                         │
                                                 Same thread_id (persisted state)
                                                         │
                                              Entry: generate_code (restarts)
                                                         │
                                               Model sees full history
                                                        │
                                                  Refined code generated
                                                        │
                                     Auto-review → ask_user → back to CLI loop
```

### 4.3 Exit Paths

| Condition           | Action                |
|---------------------|-----------------------|
| User types `accept` | Print success, break  |
| User types `quit`   | Print bye, exit(0)    |
| KeyboardInterrupt   | Catch, print bye, exit(0) |

---

## 5. LangGraph State Machine

### 5.1 State Schema

```python
class AgentState(MessagesState, total=False):
    """Extends MessagesState with custom status tracking."""
    status: str  # "generating" | "reviewing" | "fixed" | "done"
```

`MessagesState` (from LangGraph) provides an auto-reducing `messages` channel that appends each new message as it's returned from nodes.

### 5.2 Nodes

| Node | Input | Action | Output |
|---|---|---|---|
| `generate_code` | `state["messages"]` | Prepends `CODE_GEN_SYSTEM_PROMPT`, calls `llm.invoke()` | Append assistant response, set `status="generating"` |
| `review_code` | `state["messages"]` | Prepends `system_review` (JSON-output instructions), calls `llm.invoke()` | Append reviewer response, set `status="reviewing"` |
| `ask_user` | none meaningful | Does not modify messages | Set `status="pending_feedback"` |

### 5.3 Conditional Edges

| From | Router Function | Logic | Registered Targets |
|---|---|---|---|
| `generate_code` | `should_continue` | If last message content starts with `"[USER]"` → `"generate_code"`; else → `"review_code"` | `{"review_code", "__end__"}` |
| `review_code` | `continue_or_end` | Always returns `"ask_user"` | `{"ask_user", "__end__"}` |

### 5.4 Graph Wiring

```
[generate_code] ──conditional(should_continue)──▶ [review_code] ──conditional(continue_or_end)──▶ [ask_user]
      ▲                                                                          │
      │__________________________________________________________________________│
                         (implicit: feedback re-invokes graph from entry point)
```

Entry point: **`generate_code`**.  
Checkpointer: **`MemorySaver()`** — thread-scoped in-memory message history.

---

## 6. Configuration

All configuration is sourced from the environment (via `dotenv`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL where the local Ollama server listens |
| `MODEL_NAME` | `qwen2.5-coder:7b` | HuggingFace-style model identifier passed to Ollama |
| `temperature` | `0.2` | Hardcoded in `ChatOllama` init — low for deterministic code output |

No `.env` file is shipped with the repo; create one if you need to override defaults.

---

## 7. External Dependencies

| Package | Version | Role |
|---|---|---|
| `langgraph` | latest | Core orchestration: `StateGraph`, `END`, conditional edges |
| `langchain-core` | latest | Foundation abstractions (messages, prompts) used by langgraph |
| `langchain-ollama` | latest | Bridge between LangChain and Ollama HTTP API |
| `ollama` (Python SDK) | latest | Lightweight Ollama client library (transitive dep) |
| `langgraph-checkpoint` | latest | Base checkpointer protocol |
| `langgraph-prebuilt` | latest | Pre-built agent utilities |
| `langgraph-sdk` | latest | SDK extras for distributed features |
| `python-dotenv` | latest | Load `.env` into `os.environ` |
| `pydantic` | latest | Data validation (LangGraph internal dependency) |

**Required external service:** A running Ollama instance (`ollama serve`) with the model pulled (`ollama pull qwen2.5-coder:7b`).

---

## 8. Execution Environment

- **Host OS:** macOS
- **Shell:** zsh
- **Python version:** 3.14.x (managed inside `.env/` virtualenv)
- **Runner command:** `.env/bin/python main.py`
- **Threading:** All sessions share `thread_id = "default-thread"`. Each `app.invoke()` call resumes state from the same thread, accumulating message history across iterations.

---

## 9. Identified Issues & Known Gaps

### Issue 1 — Generated code never displayed to the user

The `print_code(code)` helper function exists in `main.py` but is **never called** inside `run_agent()`. The extracted `content` variable holds the assistant's response but is only logged via the status line:

```python
print(f"\n📋 Status: {result.get('status', 'unknown')}")
```

The user never sees the actual generated or reviewed code. This is a cosmetic bug that blocks usability.

### Issue 2 — `should_continue` router has type mismatch and unregistered edge

```python
def should_continue(state: AgentState) -> Literal["review_code", "__end__"]:
    ...
    return "generate_code"    # ← NOT in registered targets!
```

- The return annotation claims `Literal["review_code", "__end__"]` but the body can return `"generate_code"`.
- The registered edge map `{"review_code": "review_code", "__end__": "__end__"}` does not include `"generate_code"`.
- Furthermore, this branch is effectively unreachable: `should_continue` is evaluated right after `generate_code` executes, so the last message is always the LLM's response (which never starts with `"[USER]"`). When user feedback arrives, it triggers a **fresh** `app.invoke()` that hits the entry point rather than transitioning within the existing graph.

### Issue 3 — `continue_or_end` always hardcodes `"ask_user"`

```python
def continue_or_end(state: AgentState) -> Literal["ask_user", "__end__"]:
    return "ask_user"
```

This is currently acceptable because the review node should always be followed by asking for user feedback. However, the hardcoded behavior means there is no mechanism for the reviewer to signal that the code is correct and no further iteration is needed (i.e., the graph can never short-circuit to `"__end__"` based on the review result).

---

## 10. Recommendation Summary

| Priority | Area | Suggestion |
|---|---|---|
| **High** | CLI output | Call `print_code(content)` after `app.invoke()` results so users see the generated code. |
| **Medium** | Router types | Fix `should_continue`'s return type literal and add `"generate_code": "generate_code"` to the target map if the self-loop is intentional. |
| **Medium** | Short-circuit review | Make `continue_or_end` inspect the review response to allow early `"__end__"` termination when no issues are found. |
| **Low** | Structured output | Use `llm.with_structured_output(...)` in `review_code` instead of raw-text JSON prompting for guaranteed schema compliance. |
| **Low** | Error handling | Wrap `app.invoke()` in try/except to handle missing Ollama connection gracefully. |
| **Low** | Session isolation | Consider deriving `thread_id` from user input or time-stamping it to support parallel conversations. |
