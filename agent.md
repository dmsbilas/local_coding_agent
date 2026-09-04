# Understanding `agent.py`

A thorough guide to the **autonomous** LangGraph coding agent — written so you can learn *why* each piece exists, not only *what* it does.

---

## 1. Big picture

`agent.py` is the **brain** of this project. `main.py` is only the CLI (read one task description, stream progress, print results). All AI logic lives here.

### What problem does it solve?

1. User describes code they want **once**.
2. The LLM **generates** code (does not write to disk yet).
3. The same LLM **validates** that code against the original request.
4. If not approved → validator feedback is fed back → generate again.
5. Loop continues until the validator is happy **or** `MAX_REVISIONS` is hit.
6. Only when approved → **write files** to the project folder → finish.

There is **no** user `accept` step. Approval is entirely LLM-driven.

### Mental model

```
                         ┌──────────────────────────────────────┐
                         │                                      │
                         ▼                                      │
user request ──► generate_code ──► validate_code                │
                                      │                         │
                     ┌────────────────┼────────────────┐        │
                     │                │                │        │
                     ▼                ▼                ▼        │
              write_files    generate_code again   fail_max_    │
                     │       (with [VALIDATOR]      revisions   │
                     │        feedback)                 │       │
                     │                │                 │       │
                     └───────┬────────┴────────┬────────┘       │
                             │                 │                │
                             ▼                 ▼                │
                            END               END               │
                                                                │
                     (not approved & iteration < MAX) ──────────┘
```

---

## 2. Key concepts

### LangGraph

| Concept | Meaning |
|---------|---------|
| **State** | Shared data flowing through the graph (`messages`, `approved`, `files`, …) |
| **Node** | A function that reads state and returns updates |
| **Edge** | Always go to a fixed next node |
| **Conditional edge** | A router chooses the next node from state |
| **Checkpointer** | Saves state per `thread_id` |
| **END** | Terminal node — the run finishes |

### Messages

Conversation is a list of messages (`HumanMessage`, `AIMessage`).  
`MessagesState` **appends** when a node returns `{"messages": [...]}`.

Validator feedback is injected as a `HumanMessage` starting with `[VALIDATOR]` so the next generate pass must address it.

### Tools

`write_file` is a LangChain `@tool`, but in this design it is **called by the `write_files` node** after approval — not by the model inventing tool-call JSON during generation. That avoids the old bug where JSON tool text was saved as “source code.”

### Why JSON parsing helpers?

The **validator** must return structured JSON (`approved`, `feedback`, `files`). Local models sometimes wrap it in fences or add prose. Helpers extract a real `dict` reliably.

---

## 3. File map

| Section | Purpose |
|---------|---------|
| Config | Project root, Ollama, `MAX_REVISIONS` |
| `write_file` | Safe disk writes |
| Parsing helpers | JSON / fences / path / file-list cleanup |
| Prompts | Generate + validate system prompts |
| `AgentState` | Graph state schema |
| Nodes | `generate_code`, `validate_code`, `write_files`, `fail_max_revisions` |
| Router | `after_validate` |
| Graph compile | Wire nodes → `app` |
| `new_thread_id` | Fresh run id for each user task |

---

## 4. Config

```python
PROJECT_ROOT = Path(__file__).resolve().parent
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "6"))
```

| Variable | Role |
|----------|------|
| `PROJECT_ROOT` | Sandbox root — all writes must stay under here |
| `OLLAMA_BASE_URL` | Local Ollama HTTP API |
| `MODEL_NAME` | Coding model (same model generates *and* validates) |
| `MAX_REVISIONS` | Cap on validate→revise loops (default 6) |

```python
llm = ChatOllama(..., temperature=0.2)
```

Low temperature → more deterministic code and validation JSON.

### Protected names

```python
_PROTECTED_NAMES = {"agent.py", "main.py", "architecture.md", "agent.md", ".gitignore"}
```

The agent must not overwrite its own infrastructure files.

---

## 5. `write_file` tool

```python
@tool
def write_file(relative_path: str, content: str) -> str:
```

### Safety checks

1. Resolve `PROJECT_ROOT / relative_path`.
2. `relative_to(PROJECT_ROOT)` — reject `../` escapes.
3. Refuse protected basenames.
4. `mkdir` parents, write UTF-8, return a success/error string.

Used only from `write_files` after the validator approves.

---

## 6. Parsing helpers

### `_looks_like_json_blob(text)`

Detects JSON (or JSON-ish text with keys like `"approved"`). Used so we never treat a JSON blob as runnable source.

### `_parse_json_object(text)`

Best-effort parse:

1. Unwrap \`\`\`json fences if present  
2. `json.loads` whole string  
3. Else find `{...}` and parse that  

Returns a `dict` or `None`.

### `_code_from_fences(text)`

Pulls the largest non-`json` markdown code fence. Rejects if the body looks like JSON.

### `_sanitize_path(path)`

Strips `./`, renames protected names to `stem_app.ext`.

### `_normalize_files(raw_files)`

Turns validator `files` into a clean list of `{relative_path, content}`:

- Accepts keys `relative_path` / `path` / `filename` and `content` / `code`
- Sanitizes paths
- Unwraps accidental fences
- Drops empty or JSON-looking content

---

## 7. Prompts

### `CODE_GEN_SYSTEM_PROMPT`

Tells the generator:

1. Implement the user request fully  
2. Treat `[VALIDATOR]` messages as mandatory feedback  
3. **Do not** write to disk — validation/write steps handle that  
4. Use markdown fences; name files intuitively  
5. No fake tool-call JSON  

### `VALIDATE_SYSTEM_PROMPT`

Tells the validator to return **only** JSON:

**Not approved:**
```json
{
  "approved": false,
  "feedback": "Actionable list of fixes",
  "files": []
}
```

**Approved (ready to write):**
```json
{
  "approved": true,
  "feedback": "",
  "files": [
    {
      "relative_path": "hello_world.py",
      "content": "<pure source code>"
    }
  ]
}
```

`approved=true` is allowed only when code is correct **and** `files` contains real source.

---

## 8. State: `AgentState`

```python
class AgentState(MessagesState, total=False):
    status: str       # generating | validating | revising | writing | done | failed
    approved: bool
    feedback: str
    files: list       # [{relative_path, content}, ...]
    iteration: int    # how many validate rounds so far
    write_results: list
```

| Field | Meaning |
|-------|---------|
| `messages` | Full conversation (from `MessagesState`) |
| `status` | Human-readable stage for the CLI |
| `approved` | Validator decision |
| `feedback` | Last rejection reasons |
| `files` | Approved payloads ready to write |
| `iteration` | Validate counter (compared to `MAX_REVISIONS`) |
| `write_results` | Strings returned by `write_file` |

---

## 9. Nodes

### `generate_code`

1. Prepend `CODE_GEN_SYSTEM_PROMPT`  
2. `llm.invoke(messages)`  
3. Append AI reply  
4. Set `status` to `"generating"` (first pass) or `"revising"`  
5. Set `approved=False`  

Does **not** touch the filesystem.

### `validate_code`

1. Increment `iteration`  
2. Call LLM with `VALIDATE_SYSTEM_PROMPT` + history  
3. Parse JSON (retry once if unparseable)  
4. Normalize `files`  
5. If `approved` but files empty/invalid → force `approved=False` with feedback  

**If approved:**  
- Store cleaned `files`  
- Append summary `AIMessage`  
- `status="writing"`  

**If not approved:**  
- Append raw validator text + `[VALIDATOR] …` `HumanMessage` with feedback  
- Clear `files`  
- Next hop will be `generate_code` again  

### `write_files`

For each approved `{relative_path, content}`:

```python
write_file.invoke({"relative_path": path, "content": content})
```

Sets `status` to `"done"` if every write succeeded, else `"failed"`.  
Stores `write_results` for the CLI summary.

### `fail_max_revisions`

Reached when still not approved after `MAX_REVISIONS` rounds.  
Emits a failure message with the last feedback; `status="failed"`. No files written.

---

## 10. Router: `after_validate`

```python
def after_validate(state):
    if state.get("approved") and state.get("files"):
        return "write_files"
    if int(state.get("iteration") or 0) >= MAX_REVISIONS:
        return "fail_max_revisions"
    return "generate_code"
```

| Condition | Next |
|-----------|------|
| Approved + valid files | `write_files` |
| Hit revision limit | `fail_max_revisions` |
| Otherwise | `generate_code` (revise) |

---

## 11. Graph wiring

```python
workflow.set_entry_point("generate_code")
workflow.add_edge("generate_code", "validate_code")
workflow.add_conditional_edges("validate_code", after_validate, {...})
workflow.add_edge("write_files", END)
workflow.add_edge("fail_max_revisions", END)

app = workflow.compile(checkpointer=memory)
```

| From | To |
|------|----|
| start | `generate_code` |
| `generate_code` | `validate_code` (always) |
| `validate_code` | `write_files` / `generate_code` / `fail_max_revisions` |
| `write_files` | `END` |
| `fail_max_revisions` | `END` |

```python
def new_thread_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"
```

Each user task gets a **fresh** thread so runs don’t share polluted history.

---

## 12. How `main.py` drives the agent

1. Read one description (or `quit`).  
2. `app.stream(...)` with `stream_mode="updates"` so each node prints live.  
3. Show AI output and `[VALIDATOR]` feedback as they appear.  
4. After the graph ends, read final state:
   - `status == "done"` → print written file paths  
   - `status == "failed"` → report stop without writes  

No accept/feedback prompts. The user only starts the job.

`recursion_limit` is raised so generate↔validate loops don’t hit LangGraph’s default cap.

---

## 13. End-to-end example

**User:** “Write a Python script that prints the current time”

1. **generate_code** → proposes `show_time.py` with code in a fence  
2. **validate_code** → maybe `approved=false`, feedback: “use timezone-aware datetime”  
3. **generate_code** → revises using `[VALIDATOR]` note  
4. **validate_code** → `approved=true` + `files: [{relative_path: "show_time.py", content: "..."}]`  
5. **write_files** → creates `show_time.py` on disk  
6. **END** → CLI prints success  

If the validator never approves within 6 rounds → `fail_max_revisions` → END with error.

---

## 14. Design decisions

| Decision | Reason |
|----------|--------|
| Autonomous validate loop | No human accept; agent owns quality gate |
| Same local model for gen + validate | Simple setup; works offline with Ollama |
| Write only after approve | Avoids saving half-baked or JSON blobs |
| Programmatic `write_file` in node | Reliable disk IO; model doesn’t need tool-calling for saves |
| `[VALIDATOR]` human messages | Clear signal for the generator to revise |
| `MAX_REVISIONS` | Prevents infinite loops |
| Path sandbox + protected names | Safety |
| Fresh `thread_id` per task | Clean slate each project |

---

## 15. Glossary

| Term | Meaning |
|------|---------|
| **Generate** | LLM writes/revises code in chat |
| **Validate** | LLM judges code; returns approve or feedback JSON |
| **Revision** | One generate←validate cycle after rejection |
| **Approve** | Validator sets `approved=true` and supplies `files` |
| **write_files** | Node that persists approved source to disk |
| **Fence** | Markdown \`\`\`code\`\`\` block |
| **Checkpointer** | Persists graph state for a thread |
| **END** | Graph finished |

---

## 16. Suggested learning order

1. Big picture + diagram  
2. `AgentState` fields  
3. `generate_code` → `validate_code` → `after_validate`  
4. Validator JSON shape + `_normalize_files`  
5. `write_files` + path safety  
6. Skim `main.py` streaming loop  
7. Trace one real run with `.env/bin/python main.py`

Once those click, the file is: **generate, judge, revise, then write.**
