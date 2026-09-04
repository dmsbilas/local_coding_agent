"""
langgraph-based AI coding agent that uses a local Ollama instance
with the qwen2.5-coder:7b model to generate, review, and fix code.
"""

import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# Project root — all file writes are confined here
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def write_file(relative_path: str, content: str) -> str:
    """Write content to a file inside the project folder.

    Use this tool to create or overwrite source files so the generated
    code lives on disk under the project root (not just in chat).

    Args:
        relative_path: Path relative to the project root, e.g. "src/app.py"
            or "hello.py". Do not use absolute paths.
        content: Full file contents to write.
    """
    # Normalize and resolve; reject path traversal outside PROJECT_ROOT
    target = (PROJECT_ROOT / relative_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return (
            f"Error: path '{relative_path}' escapes the project folder. "
            "Use a path relative to the project root."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} bytes to {relative_path}"


TOOLS = [write_file]
llm_with_tools = llm.bind_tools(TOOLS)

# Reserved filenames that must never be overwritten by generated code
_PROTECTED_NAMES = {
    "agent.py",
    "main.py",
    "architecture.md",
    ".gitignore",
}


def _looks_like_json_blob(text: str) -> bool:
    """True if text is (or is dominated by) a JSON object/array."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        # Common: JSON wrapped in prose or slightly malformed — still treat as JSON-ish
        return '"fixed_code"' in stripped or '"relative_path"' in stripped or '"name"' in stripped


def _code_from_fences(text: str) -> str | None:
    """Extract the largest fenced code block from markdown text."""
    blocks = re.findall(r"```(?!json)(\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        # Fallback: any fence except pure json label handled below
        blocks_plain = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
        if not blocks_plain:
            return None
        candidate = max(blocks_plain, key=len).strip()
        return None if _looks_like_json_blob(candidate) else candidate
    # blocks are (lang, body) tuples
    bodies = [body.strip() for _, body in blocks if body.strip()]
    if not bodies:
        return None
    candidate = max(bodies, key=len)
    return None if _looks_like_json_blob(candidate) else candidate


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a JSON object from raw or fenced text."""
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


def _code_from_tool_json(text: str) -> tuple[str | None, str | None]:
    """Unwrap source code (+ path) from a write_file-style JSON blob in chat text."""
    data = _parse_json_object(text)
    if not data:
        return None, None

    # {"name": "write_file", "arguments": {"relative_path": ..., "content": ...}}
    if data.get("name") == "write_file" or "arguments" in data:
        args = data.get("arguments") or data.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict) and args.get("content"):
            content = str(args["content"]).strip()
            if content and not _looks_like_json_blob(content):
                return content, args.get("relative_path")

    # {"relative_path": "...", "content": "..."}
    if data.get("content") and "relative_path" in data:
        content = str(data["content"]).strip()
        if content and not _looks_like_json_blob(content):
            return content, data.get("relative_path")

    return None, None


def _code_from_review_json(text: str) -> str | None:
    """Pull fixed_code from a reviewer JSON blob if present."""
    data = _parse_json_object(text)
    if not data:
        return None
    fixed = data.get("fixed_code")
    if not isinstance(fixed, str) or not fixed.strip():
        return None
    fixed = fixed.strip()
    return None if _looks_like_json_blob(fixed) else fixed


def extract_accepted_code(messages: list) -> tuple[str | None, str | None]:
    """Find the best final code (+ optional path) from the conversation.

    Preference order:
      1. Last write_file tool-call args (path + content)
      2. write_file JSON embedded in AI text
      3. Reviewer JSON ``fixed_code``
      4. Largest markdown code fence in the latest AI messages
    Never returns raw JSON / tool-call blobs as code.
    """
    for msg in reversed(messages):
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
            if name == "write_file" and args.get("content"):
                content = str(args["content"]).strip()
                if content and not _looks_like_json_blob(content):
                    return content, args.get("relative_path")

    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        text = (getattr(msg, "content", None) or "").strip()
        if not text:
            continue

        tool_code, tool_path = _code_from_tool_json(text)
        if tool_code:
            return tool_code, tool_path

        from_json = _code_from_review_json(text)
        if from_json:
            return from_json, None

        from_fence = _code_from_fences(text)
        if from_fence:
            return from_fence, None

    return None, None


def _conversation_digest(messages: list, limit: int = 12) -> str:
    """Compact recent human/AI text for a finalize LLM call."""
    parts: list[str] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            continue
        if msg_type == "human":
            parts.append(f"USER: {content}")
        elif msg_type == "ai":
            # Keep AI text but cap length so the finalize prompt stays small
            clipped = content if len(content) <= 4000 else content[:4000] + "\n...[truncated]"
            parts.append(f"ASSISTANT: {clipped}")
    return "\n\n".join(parts[-limit:])


_FINALIZE_SYSTEM = """You prepare accepted code for saving to disk.

Return ONLY a JSON object (no markdown fences, no commentary) with this shape:
{
  "relative_path": "meaningful_name.ext",
  "content": "<complete source code only>"
}

Rules for relative_path:
- Intuitive name from the user's request (e.g. hello_world.py, fizzbuzz.py, todo_cli.py)
- Correct extension for the language (.py, .js, .ts, .go, .rs, .java, etc.)
- snake_case for Python; never use generated.py, output.py, or code.py
- Do NOT use agent.py, main.py, or other project infrastructure names
- Single file at project root unless a subdirectory clearly helps

Rules for content:
- Must be runnable source code in the target language — nothing else
- Never JSON, never write_file tool calls, never review metadata, never markdown fences
- No surrounding explanation"""


def _finalize_with_llm(messages: list) -> tuple[str | None, str | None]:
    """Ask the LLM for clean source code + a meaningful filename."""
    digest = _conversation_digest(messages)
    # Prefer any already-extracted pure source as a strong hint
    hint_code, hint_path = extract_accepted_code(messages)
    hint = ""
    if hint_code:
        hint = (
            f"\n\nCandidate source code already extracted (may need cleanup; "
            f"suggested path: {hint_path or 'unknown'}):\n{hint_code}"
        )

    prompt = [
        {"role": "system", "content": _FINALIZE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Conversation to finalize:\n\n"
                f"{digest}"
                f"{hint}\n\n"
                "Produce the final JSON with relative_path and content now."
            ),
        },
    ]

    for _ in range(2):
        response = llm.invoke(prompt)
        text = getattr(response, "content", "") or ""
        data = _parse_json_object(text)
        if not data:
            prompt.append({"role": "assistant", "content": text})
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        "Invalid response. Reply with ONLY JSON: "
                        '{"relative_path": "...", "content": "..."} '
                        "where content is pure source code."
                    ),
                }
            )
            continue

        path = data.get("relative_path") or data.get("filename")
        content = data.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        path = path.strip().lstrip("./")
        content = content.strip()
        # Strip accidental fences around content
        fenced = _code_from_fences(content)
        if fenced:
            content = fenced
        if not path or not content or _looks_like_json_blob(content):
            prompt.append({"role": "assistant", "content": text})
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        "content must be pure source code, not JSON. "
                        "Pick a meaningful filename. Try again — JSON only."
                    ),
                }
            )
            continue
        if Path(path).name.lower() in _PROTECTED_NAMES:
            stem = Path(path).stem + "_app"
            path = str(Path(path).with_name(stem + Path(path).suffix))
        return content, path

    return None, None


def save_accepted_code(
    messages: list,
    relative_path: str | None = None,
) -> str:
    """Persist accepted code as real source (not JSON), with an intuitive name.

    Calls the LLM to finalize clean code + filename when needed.
    """
    code, inferred_path = _finalize_with_llm(messages)

    # Fallback: local extraction if the LLM finalize fails
    if not code:
        code, inferred_path = extract_accepted_code(messages)
    if not code:
        return "Error: could not produce clean source code to save."

    path = relative_path or inferred_path or "generated.py"
    path = path.strip().lstrip("./")
    if Path(path).name.lower() in _PROTECTED_NAMES:
        path = f"{Path(path).stem}_app{Path(path).suffix or '.py'}"

    if _looks_like_json_blob(code):
        return (
            "Error: refused to save JSON/tool-call blob as source code. "
            "Try accepting again after refining."
        )

    return write_file.invoke({"relative_path": path, "content": code})

# System prompt for the coding agent
CODE_GEN_SYSTEM_PROMPT = """You are an expert software engineer. Your task is to write clean, correct, and well-commented code based on the user's description.

Rules:
1. Ask clarifying questions if the request is ambiguous.
2. Write complete, runnable code — do not omit key parts unless they are trivial boilerplate.
3. Include inline comments explaining non-obvious logic.
4. Follow community best practices for the language.
5. Prefer showing the code in a markdown fenced code block (```python ... ```) so it is easy to read.
6. Choose an intuitive filename when discussing the file (e.g. hello_world.py). Do NOT overwrite agent.py or main.py.
7. If you use the write_file tool, put ONLY source code in the content argument — never JSON, never tool-call text.

If asked to fix or improve existing code, show the full corrected version."""


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------
class AgentState(MessagesState, total=False):
    """LangGraph state extending MessagesState with status tracking."""

    status: str  # "generating" | "reviewing" | "fixed" | "done"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def generate_code(state: AgentState) -> AgentState:
    """First pass: generate code from the user's request (may call tools)."""
    messages = [
        {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
        *state["messages"],
    ]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "status": "generating",
    }


def review_code(state: AgentState) -> AgentState:
    """Second pass: review generated code and suggest improvements."""
    system_review = """You are a senior code reviewer. Review the code carefully for:
- Bugs, logical errors, or edge-case failures
- Missing error handling
- Performance issues
- Style / readability problems

Respond with a short review, then the corrected code in a markdown fenced block
(e.g. ```python ... ```). The fenced block must contain ONLY source code —
no JSON, no tool-call objects, no explanations inside the fence.

If the code is already fine, still repeat the final source code in a fence."""

    messages = [
        {"role": "system", "content": system_review},
        *state["messages"],
    ]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "status": "reviewing",
    }


def ask_user(state: AgentState) -> AgentState:
    """Pause — wait for user feedback after code generation + review."""
    # Append a system message reminding the user to provide feedback
    return {
        "messages": [],  # don't add; we handle this in main.py
        "status": "pending_feedback",
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def after_generate(state: AgentState) -> Literal["tools", "review_code"]:
    """After generation: run tools if the model requested them, else review."""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and getattr(last_msg, "tool_calls", None):
        return "tools"
    return "review_code"


def after_review(state: AgentState) -> Literal["tools", "ask_user"]:
    """After review: run tools if requested (e.g. write fixed files), else ask user."""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and getattr(last_msg, "tool_calls", None):
        return "tools"
    return "ask_user"


def after_tools(state: AgentState) -> Literal["generate_code", "ask_user"]:
    """After tools run: continue generating, or finish review → ask user."""
    if state.get("status") == "reviewing":
        return "ask_user"
    return "generate_code"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_code", generate_code)
workflow.add_node("review_code", review_code)
workflow.add_node("ask_user", ask_user)
workflow.add_node("tools", ToolNode(TOOLS))

# Set entry point
workflow.set_entry_point("generate_code")

# Edges: generate ↔ tools → review → (tools?) → ask_user
workflow.add_conditional_edges(
    "generate_code",
    after_generate,
    {"tools": "tools", "review_code": "review_code"},
)
workflow.add_conditional_edges(
    "review_code",
    after_review,
    {"tools": "tools", "ask_user": "ask_user"},
)
workflow.add_conditional_edges(
    "tools",
    after_tools,
    {"generate_code": "generate_code", "ask_user": "ask_user"},
)

# compile with memory checkpointing
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
