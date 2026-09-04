# Understanding `agent.py`

A thorough guide to the LangGraph coding agent — written so you can learn *why* each piece exists, not only *what* it does.

---

## 1. Big picture

`agent.py` is the **brain** of this project. `main.py` is only the CLI (read input, print output, call `app.invoke()`). All AI logic lives here.

### What problem does it solve?

1. User describes code they want.
2. A local LLM (Ollama + `qwen2.5-coder:7b`) **generates** code.
3. The same LLM **reviews** that code.
4. The CLI asks for feedback; user can refine or **accept**.
5. On accept, code is saved as a **real source file** (not JSON), with an **intuitive filename**.

### Mental model

Think of the agent as a **state machine** (a graph):

```
                    ┌─────────────┐
                    │  tools      │  (runs write_file)
                    └──────┬──────┘
           tool_calls ▲    │
                      │    │ after_tools
           ┌──────────┴────▼──────────┐
user ───►  │     generate_code        │
           └──────────┬───────────────┘
                      │ no tool_calls
                      ▼
           ┌──────────────────────────┐
           │      review_code         │──tool_calls──► tools ──► ask_user
           └──────────┬───────────────┘
                      │ no tool_calls
                      ▼
           ┌──────────────────────────┐
           │       ask_user           │  (pause; main.py takes over)
           └──────────────────────────┘
```

On **accept** (outside the graph), `main.py` calls `save_accepted_code()`, which may call the LLM **again** to clean the code and pick a filename, then writes the file.

---

## 2. Key concepts (learn these first)

### LangGraph

LangGraph lets you build agents as graphs:

| Concept | Meaning |
|---------|---------|
| **State** | Shared data that flows through the graph (`messages`, `status`) |
| **Node** | A function that reads state and returns updates |
| **Edge** | Connection from one node to the next |
| **Conditional edge** | A router function that *chooses* the next node |
| **Checkpointer** | Saves state per `thread_id` so follow-up `invoke()` calls remember history |

### Messages

Conversation is a list of messages (`HumanMessage`, `AIMessage`, `ToolMessage`).  
`MessagesState` automatically **appends** new messages when a node returns `{"messages": [...]}`.

### Tools

A **tool** is a Python function the LLM can request.  
`llm.bind_tools([write_file])` teaches the model the tool schema.  
If the model emits `tool_calls`, `ToolNode` runs the real Python function.

### Why so much JSON / fence parsing?

Local models often:

- Paste a fake `write_file` JSON into chat instead of a real tool call
- Wrap review output in JSON
- Put code inside markdown \`\`\` fences

The helper functions exist so **accept** still saves clean source code even when the model is messy.

---

## 3. File structure (map of sections)

| Lines (approx.) | Section | Purpose |
|-----------------|---------|---------|
| 1–18 | Imports | Libraries |
| 19–36 | Config | Project root + Ollama LLM |
| 39–70 | `write_file` tool | Safe disk writes |
| 72–170 | Extract helpers | Pull real code out of messy text |
| 172–351 | Accept / finalize | Clean code + filename + save |
| 353–365 | Code-gen prompt | Instructions for generation |
| 368–374 | `AgentState` | Graph state schema |
| 377–424 | Nodes | generate / review / ask_user |
| 427–450 | Routers | Decide next node |
| 453–486 | Graph build | Wire everything into `app` |

---

## 4. Imports and startup

```python
import json, os, re
from pathlib import Path
from typing import Literal
```

- `json` — parse tool/review/finalize JSON  
- `os` — read `OLLAMA_BASE_URL`, `MODEL_NAME`  
- `re` — extract markdown code fences  
- `Path` — safe paths, project-root checks  
- `Literal` — type-safe router return values  

```python
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()
```

- `load_dotenv()` loads a `.env` file into environment variables (optional overrides).
- `ChatOllama` talks to your local Ollama server.
- `StateGraph` + `MessagesState` are the graph foundation.
- `ToolNode` executes tool calls.
- `MemorySaver` keeps conversation memory in RAM keyed by `thread_id`.

---

## 5. Project root and LLM setup

```python
PROJECT_ROOT = Path(__file__).resolve().parent
```

`__file__` is `agent.py`’s path. `.resolve().parent` is the project folder.  
**Every write must stay under this directory** (security sandbox).

```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)
```

- Defaults assume Ollama is running locally with `qwen2.5-coder:7b` pulled.
- `temperature=0.2` → more deterministic, less “creative” nonsense (good for code).

There are effectively **two** LLM handles later:

| Handle | Used for |
|--------|----------|
| `llm` | Finalize-on-accept (no tools) |
| `llm_with_tools` | Generate + review (can call `write_file`) |

---

## 6. The `write_file` tool

```python
@tool
def write_file(relative_path: str, content: str) -> str:
```

`@tool` turns this into a LangChain tool. The **docstring matters** — the model reads it to know when/how to call the tool.

### Path safety (critical)

```python
target = (PROJECT_ROOT / relative_path).resolve()
try:
    target.relative_to(PROJECT_ROOT)
except ValueError:
    return "Error: path escapes..."
```

Example attack blocked: `relative_path="../secret.txt"`  
After resolve, that would be *outside* `PROJECT_ROOT`, so `relative_to` raises and we refuse to write.

### Write

```python
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(content, encoding="utf-8")
```

Creates folders like `new/` if needed, then writes UTF-8 text.

```python
TOOLS = [write_file]
llm_with_tools = llm.bind_tools(TOOLS)
```

`bind_tools` attaches the tool schema to the model so it can emit structured `tool_calls`.

---

## 7. Protected filenames

```python
_PROTECTED_NAMES = {"agent.py", "main.py", "architecture.md", ".gitignore"}
```

If the model tries to save as `agent.py`, we rewrite to something like `agent_app.py` so the agent cannot destroy itself.

---

## 8. Helper functions — cleaning messy model output

These exist because **local models are unreliable about formats**.

### `_looks_like_json_blob(text)`

Returns `True` if text looks like JSON (or JSON-ish with keys like `"fixed_code"`).  
Used to **refuse** saving JSON as if it were Python/JS source.

### `_code_from_fences(text)`

Finds markdown fences:

````markdown
```python
print("hi")
```
````

Prefers non-`json` fences, picks the **largest** block, rejects JSON bodies.

### `_parse_json_object(text)`

Best-effort parse:

1. Strip whitespace  
2. Unwrap \`\`\`json fences if present  
3. `json.loads` whole string  
4. Else find `{...}` substring and parse that  

Returns a `dict` or `None`.

### `_code_from_tool_json(text)`

Handles the common failure mode where the model **prints** this instead of calling the tool:

```json
{"name": "write_file", "arguments": {"relative_path": "hello.py", "content": "print(1)"}}
```

We unwrap and return `("print(1)", "hello.py")` — real source + path.

Also supports flat form: `{"relative_path": "...", "content": "..."}`.

### `_code_from_review_json(text)`

Older review format had `{"issues": [...], "fixed_code": "..."}`.  
Pulls `fixed_code` if it is real source, not nested JSON.

### `extract_accepted_code(messages)`

Walks the conversation (newest first) and returns `(code, path)` using this priority:

1. Real structured `tool_calls` for `write_file`
2. Tool JSON pasted as AI text
3. Reviewer `fixed_code`
4. Largest markdown code fence

**Never** returns a raw JSON/tool-call blob as “code”.

---

## 9. Accept path — finalize with the LLM

When the user types `accept` in `main.py`, it calls `save_accepted_code(messages)`.

### Why call the LLM again?

Because chat history may still contain:

- Explanations mixed with code  
- Wrong formats  
- Unclear filenames  

A dedicated **finalize** pass asks the model for one clean payload:

```json
{
  "relative_path": "hello_world.py",
  "content": "def main():\n    print(\"Hello\")\n..."
}
```

### `_conversation_digest(messages)`

Builds a short `USER:` / `ASSISTANT:` transcript (last ~12 messages, AI text capped at 4000 chars) so the finalize prompt stays small.

### `_FINALIZE_SYSTEM`

System instructions for that finalize call:

- Return **only** JSON with `relative_path` + `content`  
- Meaningful filename (`hello_world.py`, not `generated.py`)  
- `content` must be **pure source code**  
- Do not overwrite `agent.py` / `main.py`

### `_finalize_with_llm(messages)`

1. Build digest + optional locally extracted hint  
2. Call `llm.invoke` (no tools) up to **2** times  
3. Parse JSON; if invalid or content looks like JSON → retry with correction message  
4. Strip accidental fences from `content`  
5. Rename if path hits `_PROTECTED_NAMES`  
6. Return `(content, path)` or `(None, None)`

### `save_accepted_code(messages, relative_path=None)`

```text
finalize with LLM
    ↓ fail?
local extract_accepted_code
    ↓ fail?
return Error
    ↓
guard protected names + refuse JSON blobs
    ↓
write_file.invoke(...)
```

This is the **reliable save path**. Generation-time tool calls are optional; accept always tries to persist clean code.

---

## 10. Generation system prompt

`CODE_GEN_SYSTEM_PROMPT` tells the coding model:

1. Ask if the request is ambiguous  
2. Write complete, runnable code  
3. Comment non-obvious logic  
4. Follow language best practices  
5. Prefer markdown fences for readability  
6. Suggest intuitive filenames; don’t overwrite agent files  
7. If using `write_file`, put **only source** in `content`

This prompt is prepended as a `system` message inside `generate_code`.

---

## 11. State: `AgentState`

```python
class AgentState(MessagesState, total=False):
    status: str  # "generating" | "reviewing" | "fixed" | "done"
```

- Inherits `messages` from `MessagesState` (append-only list).  
- Adds `status` so routers know whether tools were requested during **generate** or **review**.

`total=False` means extra keys are optional TypedDict-style fields.

---

## 12. Nodes (the work units)

### `generate_code(state)`

```python
messages = [
    {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
    *state["messages"],
]
response = llm_with_tools.invoke(messages)
return {"messages": [response], "status": "generating"}
```

- Prepends the coding system prompt  
- Calls the tool-enabled LLM  
- Appends the AI reply to state  
- Marks status as `"generating"`

The reply might be plain text **or** include `tool_calls`.

### `review_code(state)`

Same pattern with a **reviewer** system prompt:

- Look for bugs, missing error handling, style issues  
- Reply with a short review + corrected code in a markdown fence  
- Fence must contain **only** source code  

Sets `status="reviewing"`.

### `ask_user(state)`

```python
return {"messages": [], "status": "pending_feedback"}
```

Does **not** add messages. It only marks that the graph is waiting.  
`main.py` prints output and prompts for feedback / accept.

---

## 13. Routers (traffic cops)

### `after_generate`

| Condition | Next node |
|-----------|-----------|
| Last AI message has `tool_calls` | `tools` |
| Otherwise | `review_code` |

### `after_review`

| Condition | Next node |
|-----------|-----------|
| Has `tool_calls` | `tools` |
| Otherwise | `ask_user` |

### `after_tools`

| `status` | Next node | Why |
|----------|-----------|-----|
| `"reviewing"` | `ask_user` | Don’t loop forever after a review write |
| else (`"generating"`) | `generate_code` | Classic ReAct: observe tool result, think again |

---

## 14. Building and compiling the graph

```python
workflow = StateGraph(AgentState)

workflow.add_node("generate_code", generate_code)
workflow.add_node("review_code", review_code)
workflow.add_node("ask_user", ask_user)
workflow.add_node("tools", ToolNode(TOOLS))

workflow.set_entry_point("generate_code")
```

Every run starts at `generate_code`.

```python
workflow.add_conditional_edges("generate_code", after_generate, {...})
workflow.add_conditional_edges("review_code", after_review, {...})
workflow.add_conditional_edges("tools", after_tools, {...})
```

The third argument maps **router return strings** → **node names**.

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

- `app` is what `main.py` imports.  
- `checkpointer=memory` means each `thread_id` keeps message history across `invoke()` calls (feedback loops remember prior turns).

---

## 15. End-to-end example

**User:** “Write a hello world in Python”

1. `main.py` → `app.invoke({"messages": [("user", "...")]}, config)`  
2. **generate_code** produces code (maybe with fences)  
3. No tool calls → **review_code**  
4. Review adds a fenced final version → **ask_user**  
5. CLI shows messages; user types `accept`  
6. `save_accepted_code`:  
   - `_finalize_with_llm` → e.g. `hello_world.py` + clean `print("Hello...")`  
   - `write_file` writes the file under `PROJECT_ROOT`  
7. CLI prints success

**If user types feedback instead of accept:**  
`main.py` invokes again with `[USER] ...`; same thread resumes; generate → review → ask_user again.

---

## 16. How this file talks to `main.py`

| Export | Used by CLI for |
|--------|-----------------|
| `app` | Run the generate/review graph |
| `save_accepted_code` | Persist on accept |
| `extract_accepted_code` | (available; save uses it as fallback) |
| `write_file` / `PROJECT_ROOT` | Internals / tests |

`main.py` responsibilities only:

- Read stdin  
- Print new AI/tool messages  
- On accept → `save_accepted_code`  
- On feedback → another `app.invoke`

---

## 17. Design decisions (why it looks this way)

| Decision | Reason |
|----------|--------|
| Local Ollama | Runs offline; no cloud API required |
| Generate then review | Two-pass quality without a huge single prompt |
| `write_file` tool | Model *can* save during generation |
| Finalize on accept | Model often fails tool calling; accept must still save clean code |
| Path sandbox | Prevent writing outside the project |
| Protected names | Don’t let the agent overwrite its own source |
| Reject JSON blobs | Earlier bug: `hello.py` contained tool-call JSON |
| MemorySaver | Feedback iterations need full chat context |

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **Node** | One step function in the graph |
| **Router** | Chooses the next node from state |
| **Tool** | Python function the LLM can request |
| **ToolNode** | Executes those requests |
| **Checkpointer** | Persists state between invokes |
| **Fence** | Markdown \`\`\`code\`\`\` block |
| **Finalize** | Extra LLM call on accept to produce clean file + name |
| **ReAct loop** | Reason → Act (tool) → Observe → Reason again |

---

## 19. Suggested learning order

1. Read **Big picture** + diagram  
2. Skim **Nodes** and **Routers**  
3. Read **`write_file`** + path safety  
4. Trace one run with a simple “hello world” request  
5. Study **extract helpers** + **finalize** (the accept path)  
6. Open `main.py` and match each CLI step to `app` / `save_accepted_code`

Once those click, the whole file is just “graph + tools + careful save.”
