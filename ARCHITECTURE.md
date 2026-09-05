# System Architecture: Local AI Coding Agent

## 1. Overview

This repo contains **three generations** of LangGraph coding agents. Each builds on the previous idea: generate code with an LLM, validate it, revise until good enough, then write files — without a human `accept` step.

| Agent | Role | LLM backend | Wired to `main.py`? |
|-------|------|-------------|---------------------|
| `agent.py` | Minimal autonomous loop | Local Ollama (`qwen2.5-coder:7b`) | No (learning / baseline) |
| `agent2.py` | Full ReAct + tools + planning + escalation | Local Ollama | No (ReAct reference) |
| `agent_ai_doer.py` | Same ReAct stack as agent2 + smart output dirs | **AI Doer** (`gpt-4o`) | **Yes** |

**Repository root:** `local_coding_agent/`

---

## 2. Project Structure

```
local_coding_agent/
├── .venv/                 # Python virtual environment
├── .env / .env.example    # Secrets & config (API keys, limits)
├── sandbox/               # Default write target when no folder is named
├── agent.py               # Gen-1: fixed generate → validate loop
├── agent2.py              # Gen-2: ReAct + tools + planning (Ollama)
├── agent_ai_doer.py       # Gen-3: ReAct + AI Doer cloud + output_dir
├── agent.md / agent2.md / agent_ai_doer.md
├── main.py                # CLI — currently imports agent_ai_doer
├── ARCHITECTURE.md        # This file
├── requirements.txt
└── run.sh
```

---

## 3. System context (CLI + active agent)

```
┌──────────────┐     stream      ┌────────────────────┐     HTTPS      ┌─────────────────┐
│   main.py    │ ──────────────▶ │  agent_ai_doer.py  │ ─────────────▶ │  AI Doer API    │
│              │ ◀────────────── │  StateGraph (ReAct)│ ◀───────────── │  gpt-4o         │
│ • prompt     │   node updates  │                    │  chat/completions│ ai-doer.com/v1/api│
│ • print      │                 └─────────┬──────────┘                 └─────────────────┘
│ • summary    │                           │
└──────────────┘                           ▼
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

To run local Ollama agents instead, point imports at `agent.app` / `agent2.app` and start `ollama serve`.

---

## 4. Architecture A — `agent.py` (fixed loop)

### 4.1 Idea

Smallest autonomous coding agent: always **generate → validate → (revise | write | fail)**.  
No separate Reason node, no plan, no escalate/fail split, no tool-call framework.

### 4.2 Graph

```
                    ┌────────────────┐
                    │ generate_code  │◀──────────────────────┐
                    │   (Ollama)     │                       │
                    └───────┬────────┘                       │
                            │ always                         │
                            ▼                                │
                    ┌────────────────┐                       │
                    │ validate_code  │                       │
                    │   (Ollama)     │                       │
                    └───────┬────────┘                       │
                            │                                │
              ┌─────────────┼─────────────────┐              │
              │             │                 │              │
              ▼             ▼                 ▼              │
       write_files   generate_code     fail_max_revisions    │
       (approved)    (not approved,      (iteration ≥ MAX)   │
              │       iteration < MAX)         │              │
              │             │                  │              │
              ▼             └──────────────────┘              │
             END                    │                         │
                                    └─────────────────────────┘
                                         (revise again)
```

Entry: `generate_code`.  
Router: `after_validate` → `write_files` | `generate_code` | `fail_max_revisions`.

### 4.3 Component view

```
main / caller
    │
    ▼
┌──────────── agent.py ────────────┐
│  ChatOllama (localhost:11434)    │
│  generate_code → validate_code   │
│  writewrite_file (direct invoke)     │
│  MemorySaver                     │
└──────────────────────────────────┘
```

### 4.4 Strengths / limits (snapshot)

| Strengths | Limits |
|-----------|--------|
| Easy to read and teach | No ReAct separation (decide vs do) |
| Few nodes, few failure modes | No planning mode |
| Works fully offline with Ollama | Weak stopping story (only max revisions) |
| Good baseline for interviews | No escalate vs fail; no loop detection |
| | No tool schema / parallel execution framework |

---

## 5. Architecture B — `agent2.py` (ReAct + tools + planning)

### 5.1 Idea

Interview-ready agent that demonstrates:

- ReAct (Reason → Act → Observe)  
- Tool schemas + sequential/parallel execution  
- Reactive vs plan-and-execute  
- Stopping conditions + human escalation  
- Retries / rate limits / timeouts  

Still runs on **local Ollama**.

### 5.2 Graph

```
                         ┌─────────────┐
                         │ create_plan │  (skip unless PLAN_MODE=plan_and_execute)
                         └──────┬──────┘
                                ▼
                   ┌────────────────────────┐
          ┌───────▶│        reason          │◀──────────────┐
          │        └───────────┬────────────┘               │
          │                    │ after_reason               │
          │     ┌──────────────┼──────────────┐             │
          │     ▼              ▼              ▼             │
          │ generate_code  validate_code  write_files        │
          │  (Ollama)       (Ollama)      (tools)           │
          │     │              │              │             │
          │     └──────────────┼──────────────┘             │
          │                    ▼ after_act                  │
          │             ┌────────────┐                      │
          │             │  observe   │──────────────────────┘
          │             └─────┬──────┘     after_observe → reason
          │                   │
          │         ┌─────────┼─────────┐
          │         ▼         ▼         ▼
          │        END    escalate     fail
          │              (human)    (hard error)
          │
          └── also: escalate/fail can be reached from after_reason / after_act
```

### 5.3 Component view

```
┌──────────────┐     ┌──────────────── agent2.py ────────────────┐
│   caller     │────▶│  create_plan / reason / observe           │
└──────────────┘     │  generate_code / validate_code (Ollama)   │
                     │  write_files → ToolCall pipeline             │
                     │    parse → Pydantic validate → retry       │
                     │    sequential (default) or parallel        │
                     │  escalate / fail terminals                │
                     │  MemorySaver                              │
                     └──────────────────┬────────────────────────┘
                                        ▼
                               ChatOllama @ localhost
```

### 5.4 Strengths / limits (snapshot)

| Strengths | Limits |
|-----------|--------|
| Clear ReAct teaching model | Still local-model quality dependent |
| Tool calling done “properly” | No smart output-folder inference |
| Planning modes | Writes usually to project/sandbox conventions only |
| Rich stopping + escalation | Heavier than agent.py to learn first |
| Production-ish retries | Offline-only unless you change the LLM client |

---

## 6. Architecture C — `agent_ai_doer.py` (ReAct + AI Doer + output dirs)

### 6.1 Idea

Same ReAct / tools / planning / stopping stack as `agent2.py`, plus:

1. **AI Doer cloud LLM** (`ChatOpenAI` → `https://ai-doer.com/v1/api`, model `gpt-4o`)  
2. **`set_output_dir`** — user-mentioned folder, else `sandbox/`  
3. **Automatic writes** after approval (no human permission)  
4. Docs focused on tool calling, ReAct, planning, stopping, retries  

This is what **`main.py` imports today**.

### 6.2 Graph

```
┌────────────────┐
│ set_output_dir │  infer folder from user text → else sandbox/
└───────┬────────┘
        ▼
┌────────────────┐
│  create_plan   │  optional (plan_and_execute)
└───────┬────────┘
        ▼
┌────────────────┐
│     reason     │◀─────────────────────────────────────┐
└───────┬────────┘                                      │
        │                                               │
   ┌────┼────────────────────┐                          │
   ▼    ▼                    ▼                          │
generate  validate         write_files                  │
(AI Doer) (AI Doer JSON)   (_write_file_impl → disk)    │
   │    │                    │                          │
   └────┴─────────┬──────────┘                          │
                  ▼                                     │
             ┌─────────┐                                │
             │ observe │────────────────────────────────┘
             └────┬────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       END    escalate     fail
```

### 6.3 Component view

```
┌────────────┐   stream    ┌──────────── agent_ai_doer.py ────────────┐
│  main.py   │ ──────────▶ │ set_output_dir                           │
│            │ ◀────────── │ ReAct loop (same as agent2)              │
└────────────┘             │ generate/validate ──HTTPS──▶ AI Doer     │
                           │ write_files ──▶ output_dir or sandbox/   │
                           │ ToolCall schemas + retries + escalate    │
                           └──────────────────────────────────────────┘
```

### 6.4 Strengths / limits (snapshot)

| Strengths | Limits |
|-----------|--------|
| Stronger model (gpt-4o) via AI Doer | Needs API key + credits (not offline) |
| Same ReAct maturity as agent2 | Network / rate-limit sensitivity |
| Smart write location | Slightly more config (`.env`) |
| Automatic write on approval | Cloud billing vs free local Ollama |
| Best “ship a real CLI” option in this repo | |

---

## 7. Step-by-step: advantages of one over another

Read this as an evolution story: each step adds capability, and you choose how far to go.

### Step 1 — Start with `agent.py`

**Advantage over nothing**

- You get a working autonomous loop: generate → validate → write.  
- Tiny graph; perfect for learning LangGraph edges and LLM validation JSON.

**When to use:** teaching, prototypes, offline-only demos with Ollama.

---

### Step 2 — Prefer `agent2.py` over `agent.py`

| Capability | `agent.py` | Advantage of `agent2.py` |
|------------|------------|---------------------------|
| Control flow | Fixed edges | **ReAct** — Reason decides Act from observations |
| Planning | None | **Reactive or plan-and-execute** |
| Tools | Direct `write_file.invoke` | **Pydantic schemas**, parse/validate, sequential vs parallel |
| Stopping | Max revisions only | **Steps + revisions + repeated-action detection** |
| Failure UX | `fail_max_revisions` | **Escalate (human) vs fail (hard error)** |
| Resilience | Validator JSON retry | **LLM/tool retries, rate-limit & timeout backoff** |

**Net:** `agent2.py` is what you show when asked “how do real agents decide, stop, and call tools safely?”

**Tradeoff:** more code and concepts; still limited by local model quality.

---

### Step 3 — Prefer `agent_ai_doer.py` over `agent2.py`

| Capability | `agent2.py` | Advantage of `agent_ai_doer.py` |
|------------|-------------|----------------------------------|
| LLM | Local Ollama | **AI Doer gpt-4o** (OpenAI-compatible cloud) |
| Output path | Mostly fixed conventions | **`set_output_dir`**: mentioned folder, else `sandbox/` |
| Product wiring | Reference module | **Hooked to `main.py`** |
| Write policy | After approve | Same, plus explicit **no human accept** product behavior |
| Docs | `agent2.md` | `agent_ai_doer.md` focused on interview topics |

**Net:** same ReAct architecture as agent2, better model + better file placement + production CLI entrypoint.

**Tradeoff:** requires `AI_DOER_API_KEY`, network, and BDT credits; not a drop-in offline lab.

---

### Step 4 — When `agent2.py` still wins over `agent_ai_doer.py`

| Situation | Prefer |
|-----------|--------|
| No internet / no API budget | `agent2.py` (or `agent.py`) |
| Teaching Ollama-only setups | `agent2.py` |
| Comparing local vs cloud quality | Keep both; swap `main.py` import |

### Step 5 — When `agent.py` still wins over both

| Situation | Prefer |
|-----------|--------|
| First day learning LangGraph | `agent.py` |
| Minimal diff for a blog/tutorial | `agent.py` |
| Debugging “is validation JSON the problem?” without ReAct noise | `agent.py` |

---

## 8. Side-by-side comparison matrix

| Dimension | `agent.py` | `agent2.py` | `agent_ai_doer.py` |
|-----------|------------|-------------|---------------------|
| Pattern | Fixed generate→validate | ReAct | ReAct |
| LLM | Ollama local | Ollama local | AI Doer gpt-4o |
| Planning | — | reactive / plan_and_execute | same |
| Tool schemas | Minimal | Full | Full |
| Parallel tools | — | Supported | Supported |
| Loop detection | — | Yes | Yes |
| Escalate vs fail | Fail only | Both | Both |
| Retries / backoff | Thin | Full | Full |
| Output folder inference | — | — | **Yes** (`set_output_dir`) |
| Human accept | No | No | No |
| `main.py` | Not default | Not default | **Default** |
| Best for | Learn basics | Learn agents | Ship the CLI |

---

## 9. Shared design decisions (all three)

1. **Validate before write** — disk only after LLM approval (or fail/escalate).  
2. **No interactive accept** in the autonomous designs — quality gate is the validator.  
3. **LangGraph + MemorySaver** — thread-scoped state.  
4. **Fresh `thread_id` per task** (in CLI) — runs do not pollute each other.  
5. **Protected agent filenames** — do not overwrite `agent*.py` / `main.py` at repo root.

---

## 10. Configuration

### Local Ollama agents (`agent.py`, `agent2.py`)

| Variable | Default | Notes |
|----------|---------|-------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `MODEL_NAME` | `qwen2.5-coder:7b` | |
| `MAX_REVISIONS` | `6` | |

### AI Doer agent (`agent_ai_doer.py`)

| Variable | Default | Notes |
|----------|---------|-------|
| `AI_DOER_API_KEY` | required | `aiob_…` |
| `AI_DOER_BASE_URL` | `https://ai-doer.com/v1/api` | |
| `MODEL_NAME` | `gpt-4o` | |
| `PLAN_MODE` | `reactive` | or `plan_and_execute` |
| `MAX_STEPS` / `MAX_REVISIONS` / `MAX_REPEATED_ACTIONS` | `12` / `6` / `3` | Stopping |
| `MAX_TOOL_RETRIES` / `MAX_VALIDATOR_ATTEMPTS` | `3` / `2` | Retries |
| `SANDBOX_DIR` | `sandbox` | Default output folder |

---

## 11. Execution

```bash
source .venv/bin/activate
# .env: AI_DOER_API_KEY=...
python main.py          # uses agent_ai_doer
# or: ./run.sh
```

Offline ReAct (agent2):

```bash
ollama serve
ollama pull qwen2.5-coder:7b
# temporarily change main.py to: from agent2 import ...
```

---

## 12. Further reading

| Doc | Covers |
|-----|--------|
| [`agent.md`](agent.md) | Fixed-loop agent walkthrough |
| [`agent2.md`](agent2.md) | ReAct / tools / planning / stopping on Ollama |
| [`agent_ai_doer.md`](agent_ai_doer.md) | Same topics + AI Doer calling & output dirs |

---

## 13. One-line recommendation

**Learn on `agent.py` → understand agents on `agent2.py` → run the product on `agent_ai_doer.py`.**
