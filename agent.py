"""
Autonomous LangGraph coding agent.

Flow:
  generate_code → validate_code ⟲ (revise until approved) → write_files → END

A local Ollama model (default qwen2.5-coder:7b) both writes and validates.
Files are written only after the validator fully approves.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "6"))

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

_PROTECTED_NAMES = {
    "agent.py",
    "main.py",
    "architecture.md",
    "agent.md",
    ".gitignore",
}


# ---------------------------------------------------------------------------
# Tools / file IO
# ---------------------------------------------------------------------------
@tool
def write_file(relative_path: str, content: str) -> str:
    """Write content to a file inside the project folder.

    Args:
        relative_path: Path relative to the project root, e.g. "hello_world.py".
        content: Full source-file contents (never JSON / tool-call text).
    """
    target = (PROJECT_ROOT / relative_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return (
            f"Error: path '{relative_path}' escapes the project folder. "
            "Use a path relative to the project root."
        )

    name = target.name.lower()
    if name in _PROTECTED_NAMES:
        return f"Error: refusing to overwrite protected file '{relative_path}'."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} bytes to {relative_path}"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _looks_like_json_blob(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return (
            '"fixed_code"' in stripped
            or '"relative_path"' in stripped
            or '"approved"' in stripped
        )


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _code_from_fences(text: str) -> str | None:
    blocks = re.findall(r"```(?!json)(\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return None
    bodies = [body.strip() for _, body in blocks if body.strip()]
    if not bodies:
        return None
    candidate = max(bodies, key=len)
    return None if _looks_like_json_blob(candidate) else candidate


def _sanitize_path(path: str) -> str:
    path = path.strip().lstrip("./")
    if Path(path).name.lower() in _PROTECTED_NAMES:
        stem = Path(path).stem + "_app"
        path = str(Path(path).with_name(stem + (Path(path).suffix or ".py")))
    return path


def _normalize_files(raw_files: Any) -> list[dict[str, str]]:
    """Turn validator `files` into a clean list of {relative_path, content}."""
    if not isinstance(raw_files, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        path = item.get("relative_path") or item.get("path") or item.get("filename")
        content = item.get("content") or item.get("code")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        path = _sanitize_path(path)
        content = content.strip()
        fenced = _code_from_fences(content)
        if fenced:
            content = fenced
        if not path or not content or _looks_like_json_blob(content):
            continue
        cleaned.append({"relative_path": path, "content": content})
    return cleaned


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CODE_GEN_SYSTEM_PROMPT = """You are an expert software engineer working inside an autonomous agent loop.

Your job:
1. Implement the user's request as complete, runnable source code.
2. If you see a message starting with [VALIDATOR], treat it as mandatory review feedback and revise the code to fix every issue.
3. Do NOT write files to disk yourself — another step does that after approval.
4. Prefer markdown fenced code blocks (```python ... ```) for each file.
5. Clearly state the intended filename(s) (e.g. hello_world.py). Never use agent.py or main.py.
6. Follow language best practices; include brief comments for non-obvious logic.
7. Output ONLY the code (and short notes if needed) — no fake tool-call JSON."""

VALIDATE_SYSTEM_PROMPT = """You are a strict senior code reviewer validating an autonomous coding agent.

Judge whether the latest generated code fully and correctly satisfies the ORIGINAL user request.

Return ONLY a JSON object (no markdown commentary outside JSON) with this shape:

If NOT approved:
{
  "approved": false,
  "feedback": "Clear, actionable list of what to fix. Be specific.",
  "files": []
}

If fully approved and ready to write to disk:
{
  "approved": true,
  "feedback": "",
  "files": [
    {
      "relative_path": "meaningful_name.ext",
      "content": "<complete source code only — no fences, no JSON wrapper>"
    }
  ]
}

Rules:
- approved=true ONLY if the code is correct, complete, and meets the request.
- When approved, files[].content MUST be pure source code for that language.
- Choose intuitive filenames (hello_world.py, show_time.py). Correct extensions.
- Never use agent.py, main.py, architecture.md, or agent.md as paths.
- Prefer a single file unless multiple files are clearly required.
- If code is missing, broken, or incomplete → approved=false with precise feedback."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(MessagesState, total=False):
    """Graph state: chat history + validation outcome."""

    status: str  # generating | validating | revising | writing | done | failed
    approved: bool
    feedback: str
    files: list  # [{relative_path, content}, ...]
    iteration: int
    write_results: list  # status strings from write_file


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def generate_code(state: AgentState) -> AgentState:
    """Produce or revise code. Does not write files."""
    messages = [
        {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
        *state["messages"],
    ]
    response = llm.invoke(messages)
    iteration = int(state.get("iteration") or 0)
    return {
        "messages": [response],
        "status": "generating" if iteration == 0 else "revising",
        "approved": False,
    }


def validate_code(state: AgentState) -> AgentState:
    """LLM validates latest code. Approves + provides files, or returns feedback."""
    iteration = int(state.get("iteration") or 0) + 1

    messages = [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        *state["messages"],
        {
            "role": "user",
            "content": (
                "Validate the code against the original user request. "
                "Reply with ONLY the JSON object described in your instructions."
            ),
        },
    ]

    raw = ""
    data: dict | None = None
    for attempt in range(2):
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""
        data = _parse_json_object(raw)
        if data is not None and "approved" in data:
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    'Invalid. Reply with ONLY JSON including "approved" (boolean), '
                    '"feedback" (string), and "files" (array).'
                ),
            }
        )

    if data is None or "approved" not in data:
        feedback = (
            "Validator returned unparseable output. Regenerate the full solution "
            "clearly in markdown fences with an explicit filename."
        )
        note = HumanMessage(
            content=(
                f"[VALIDATOR] approved=false (iteration {iteration}/{MAX_REVISIONS})\n"
                f"Feedback:\n{feedback}"
            )
        )
        return {
            "messages": [AIMessage(content=raw or "(empty validator response)"), note],
            "status": "validating",
            "approved": False,
            "feedback": feedback,
            "files": [],
            "iteration": iteration,
        }

    approved = bool(data.get("approved"))
    feedback = str(data.get("feedback") or "").strip()
    files = _normalize_files(data.get("files"))

    # Approved but missing/invalid files → treat as not approved
    if approved and not files:
        approved = False
        feedback = (
            feedback
            or "Approved flag was set but no valid source files were provided. "
            "Return approved=true with a files array of pure source code."
        )

    if approved:
        summary = (
            f"[VALIDATOR] approved=true (iteration {iteration}/{MAX_REVISIONS})\n"
            f"Ready to write {len(files)} file(s): "
            + ", ".join(f["relative_path"] for f in files)
        )
        return {
            "messages": [AIMessage(content=summary)],
            "status": "writing",
            "approved": True,
            "feedback": "",
            "files": files,
            "iteration": iteration,
        }

    if not feedback:
        feedback = "Code is not yet acceptable. Improve correctness and completeness."

    note = HumanMessage(
        content=(
            f"[VALIDATOR] approved=false (iteration {iteration}/{MAX_REVISIONS})\n"
            f"Feedback:\n{feedback}\n\n"
            "Revise the FULL solution addressing every point. "
            "Do not write files yet — output updated code in fences."
        )
    )
    return {
        "messages": [AIMessage(content=raw), note],
        "status": "validating",
        "approved": False,
        "feedback": feedback,
        "files": [],
        "iteration": iteration,
    }


def write_files(state: AgentState) -> AgentState:
    """Persist approved files to disk via write_file."""
    files = state.get("files") or []
    results: list[str] = []
    for item in files:
        path = item.get("relative_path", "")
        content = item.get("content", "")
        result = write_file.invoke({"relative_path": path, "content": content})
        results.append(result)

    if not results:
        results = ["Error: no files to write."]

    done_msg = AIMessage(
        content="📝 Write results:\n" + "\n".join(f"- {r}" for r in results)
    )
    status = "done" if all(r.startswith("Successfully") for r in results) else "failed"
    return {
        "messages": [done_msg],
        "status": status,
        "write_results": results,
    }


def fail_max_revisions(state: AgentState) -> AgentState:
    """Stop after too many revision rounds without approval."""
    feedback = state.get("feedback") or "unknown issues"
    msg = AIMessage(
        content=(
            f"❌ Stopped after {MAX_REVISIONS} revision(s) without approval.\n"
            f"Last validator feedback:\n{feedback}"
        )
    )
    return {"messages": [msg], "status": "failed", "approved": False}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def after_validate(state: AgentState) -> Literal["write_files", "generate_code", "fail_max_revisions"]:
    if state.get("approved") and state.get("files"):
        return "write_files"
    if int(state.get("iteration") or 0) >= MAX_REVISIONS:
        return "fail_max_revisions"
    return "generate_code"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("generate_code", generate_code)
workflow.add_node("validate_code", validate_code)
workflow.add_node("write_files", write_files)
workflow.add_node("fail_max_revisions", fail_max_revisions)

workflow.set_entry_point("generate_code")

workflow.add_edge("generate_code", "validate_code")
workflow.add_conditional_edges(
    "validate_code",
    after_validate,
    {
        "write_files": "write_files",
        "generate_code": "generate_code",
        "fail_max_revisions": "fail_max_revisions",
    },
)
workflow.add_edge("write_files", END)
workflow.add_edge("fail_max_revisions", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def new_thread_id() -> str:
    """Fresh thread per project request so runs don't pollute each other."""
    return f"run-{uuid.uuid4().hex[:12]}"
