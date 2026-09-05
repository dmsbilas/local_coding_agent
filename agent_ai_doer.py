"""
Autonomous LangGraph coding agent powered by AI Doer (gpt-4o).

Same ReAct workflow as agent2.py:
  set_output_dir → create_plan → reason → act → observe → … → write_files → END

Writes happen automatically after LLM validation approves — no human accept step.
If the user names a folder, files go there; otherwise under sandbox/.

Uses the AI Doer OpenAI-compatible API instead of local Ollama:
  Base URL: https://ai-doer.com/v1/api
  Auth:     Authorization: Bearer <AI_DOER_API_KEY>

Set AI_DOER_API_KEY in your environment or .env file (keys start with aiob_).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox").strip() or "sandbox"
SANDBOX_ROOT = (PROJECT_ROOT / DEFAULT_SANDBOX_DIR).resolve()
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)

AI_DOER_BASE_URL = os.getenv("AI_DOER_BASE_URL", "https://ai-doer.com/v1/api")
AI_DOER_API_KEY = os.getenv("AI_DOER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Planning: "reactive" or "plan_and_execute"
PLAN_MODE = os.getenv("PLAN_MODE", "reactive").strip().lower()

# Hard safety bounds: these are also stopping conditions.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "6"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
MAX_TOOL_RETRIES = int(os.getenv("MAX_TOOL_RETRIES", "3"))
MAX_VALIDATOR_ATTEMPTS = int(os.getenv("MAX_VALIDATOR_ATTEMPTS", "2"))
MAX_REPEATED_ACTIONS = int(os.getenv("MAX_REPEATED_ACTIONS", "3"))

# Tool/LLM timeout and retry configuration.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "1.0"))
RETRY_MAX_SECONDS = float(os.getenv("RETRY_MAX_SECONDS", "8.0"))

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=AI_DOER_API_KEY or "missing-key",
    base_url=AI_DOER_BASE_URL,
    temperature=0.2,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=0,  # retries handled by invoke_with_retry()
    model_kwargs={"stream": False},  # AI Doer streams by default; we need JSON responses
)

_PROTECTED_NAMES = {
    "agent.py",
    "agent2.py",
    "agent_ai_doer.py",
    "main.py",
    "architecture.md",
    "agent.md",
    "agent2.md",
    "agent_ai_doer.md",
    ".gitignore",
    ".env",
    ".env.example",
}

# Directories the agent must never treat as an output project root.
_BLOCKED_OUTPUT_DIRS = {
    ".git",
    ".venv",
    ".env",
    "__pycache__",
    "node_modules",
}


def _require_api_key() -> None:
    if not AI_DOER_API_KEY:
        raise ValueError(
            "AI_DOER_API_KEY is not set. Add it to your .env file "
            "(create a key at https://ai-doer.com → Integrations)."
        )


# ---------------------------------------------------------------------------
# Explicit tool schemas
# ---------------------------------------------------------------------------
class WriteFileInput(BaseModel):
    """Schema presented to a function/tool caller."""

    relative_path: str = Field(
        ...,
        description=(
            "File path relative to the chosen output folder, e.g. "
            "'hello_world.py' or 'src/main.py'."
        ),
        min_length=1,
    )
    content: str = Field(
        ...,
        description="Complete source-file contents; never JSON or tool-call text",
    )


class ReadFileInput(BaseModel):
    relative_path: str = Field(
        ...,
        description="File path relative to the chosen output folder.",
        min_length=1,
    )


class ToolCall(BaseModel):
    """Normalized representation of a model-produced tool call."""

    name: Literal["write_file", "read_file"]
    arguments: dict[str, Any]


def _strip_dot_slash(path: str) -> str:
    """Remove leading ./ segments without treating '../' as characters to strip."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _normalize_output_dir(raw: str | None) -> str:
    """Return a safe project-relative output dir, or the default sandbox name."""
    if not raw:
        return DEFAULT_SANDBOX_DIR

    cleaned = _strip_dot_slash(str(raw))
    if cleaned in {"", ".", ".."} or cleaned.startswith("../") or "/../" in f"/{cleaned}/":
        return DEFAULT_SANDBOX_DIR

    if Path(cleaned).is_absolute():
        return DEFAULT_SANDBOX_DIR

    first = cleaned.split("/", 1)[0].lower()
    if first in _BLOCKED_OUTPUT_DIRS or cleaned.lower() in _PROTECTED_NAMES:
        return DEFAULT_SANDBOX_DIR

    target = (PROJECT_ROOT / cleaned).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return DEFAULT_SANDBOX_DIR

    if target == PROJECT_ROOT:
        return DEFAULT_SANDBOX_DIR

    return cleaned


def infer_output_dir_from_text(text: str) -> str:
    """Detect a user-mentioned target folder; otherwise default to sandbox/.

    Examples:
      - "create a todo app in projects/todo"
      - "put the files under ./my_app"
      - "write to folder demo_cli"
    If nothing matches → DEFAULT_SANDBOX_DIR.
    """
    if not text:
        return DEFAULT_SANDBOX_DIR

    patterns = [
        r"(?i)\b(?:in|into|under|to|at)\s+(?:the\s+)?(?:folder|directory|dir|path)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\b(?:folder|directory|dir|path)\s*[:=]\s*[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\b(?:in|into|under|to|at)\s+\./([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*)",
        r"(?i)\b(?:in|into|under|to|at)\s+[\"'`]?(\.?/?[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)+)[\"'`]?",
        r"(?i)\b(?:in|into|under|to|at)\s+[\"'`](\.?/?[A-Za-z0-9_./-]+)[\"'`]",
        r"(?i)\bsave\s+(?:it|them|files?|code)?\s*(?:in|into|under|to)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\bwrite\s+(?:it|them|files?|code)?\s*(?:in|into|under|to)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
        r"(?i)\bcreate\s+(?:(?:a|the)\s+)?(?:project|app|files?)?\s*(?:in|into|under|at)\s+[\"'`]?([A-Za-z0-9_./-]+)[\"'`]?",
    ]

    stopwords = {
        "python", "javascript", "typescript", "java", "rust", "go",
        "code", "file", "files", "script", "project", "app",
        "the", "a", "an", "this", "that", "here", "there",
        "disk", "repo", "repository",
    }

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = _strip_dot_slash(match.group(1))
            if not candidate:
                continue
            if "/" not in candidate and candidate.lower() in stopwords:
                continue
            normalized = _normalize_output_dir(candidate)
            if normalized != DEFAULT_SANDBOX_DIR or candidate.lower() in {
                DEFAULT_SANDBOX_DIR,
                "sandbox",
            }:
                return normalized
            if any(k in pattern for k in ("folder", "directory", "dir", "path")):
                if candidate.lower() not in stopwords:
                    return _normalize_output_dir(candidate)

    return DEFAULT_SANDBOX_DIR


def _first_user_text(messages: list[Any]) -> str:
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type in {"human", "user"}:
            return str(getattr(msg, "content", "") or "")
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            return str(msg.get("content") or "")
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] in {"user", "human"}:
            return str(msg[1])
    return ""


def _output_root(output_dir: str | None) -> Path:
    """Absolute root directory where generated files are written."""
    rel = _normalize_output_dir(output_dir)
    root = (PROJECT_ROOT / rel).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_under_output(relative_path: str, output_dir: str | None) -> Path | str:
    """Map a relative file path into the chosen output root."""
    root = _output_root(output_dir)
    root_name = root.name
    cleaned = _strip_dot_slash(relative_path)

    while True:
        lower = cleaned.lower()
        stripped = False
        for prefix in (root_name, DEFAULT_SANDBOX_DIR, "sandbox"):
            if lower == prefix:
                cleaned = ""
                stripped = True
                break
            if lower.startswith(prefix + "/"):
                cleaned = _strip_dot_slash(cleaned[len(prefix) + 1 :])
                stripped = True
                break
        if not stripped:
            break

    if (
        not cleaned
        or cleaned in {".", ".."}
        or cleaned.startswith("../")
        or "/../" in f"/{cleaned}/"
    ):
        return (
            f"Error: path '{relative_path}' is empty or escapes the output folder. "
            "Use a path relative to the output directory."
        )

    target = (root / cleaned).resolve()
    try:
        target.relative_to(root)
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return (
            f"Error: path '{relative_path}' escapes the allowed output area. "
            "Use a path relative to the output directory."
        )
    return target


def _write_file_impl(relative_path: str, content: str, output_dir: str | None) -> str:
    target = _resolve_under_output(relative_path, output_dir)
    if isinstance(target, str):
        return target

    # Only block overwriting agent infrastructure at the repo root.
    if target.parent == PROJECT_ROOT and target.name.lower() in _PROTECTED_NAMES:
        return f"Error: refusing to overwrite protected file '{relative_path}'."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    try:
        shown = str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(target)
    return f"Successfully wrote {len(content)} bytes to {shown}"


def _read_file_impl(relative_path: str, output_dir: str | None) -> str:
    target = _resolve_under_output(relative_path, output_dir)
    if isinstance(target, str):
        return target

    if not target.exists():
        return f"Error: file '{relative_path}' does not exist in the output folder."
    if not target.is_file():
        return f"Error: '{relative_path}' is not a file."

    return target.read_text(encoding="utf-8")


@tool(args_schema=WriteFileInput)
def write_file(relative_path: str, content: str) -> str:
    """Write source content inside the chosen output folder (sandbox if none)."""
    return _write_file_impl(relative_path, content, DEFAULT_SANDBOX_DIR)


@tool(args_schema=ReadFileInput)
def read_file(relative_path: str) -> str:
    """Read a file safely from the chosen output folder (sandbox if none)."""
    return _read_file_impl(relative_path, DEFAULT_SANDBOX_DIR)


TOOLS: dict[str, BaseTool] = {
    "write_file": write_file,
    "read_file": read_file,
}

# ---------------------------------------------------------------------------
# Retry / error handling helpers
# ---------------------------------------------------------------------------
def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort classification for provider/backend rate-limit errors."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate-limit",
            "too many requests",
            "quota exceeded",
        )
    )


def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, TimeoutError) or any(
        marker in text for marker in ("timeout", "timed out", "deadline exceeded")
    )


def _backoff(attempt: int) -> float:
    """Exponential backoff with a small jitter, bounded by configuration."""
    delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**attempt))
    return delay + random.uniform(0, delay * 0.1)


def invoke_with_retry(callable_fn, *, attempts: int, operation: str):
    """Retry transient failures, while keeping a strict retry bound."""
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return callable_fn()
        except Exception as exc:  # noqa: BLE001 - boundary around external systems
            last_error = exc
            transient = _is_rate_limit_error(exc) or _is_timeout_error(exc)

            if not transient or attempt >= attempts - 1:
                break

            # Rate limits and timeouts should not cause a tight retry loop.
            time.sleep(_backoff(attempt))

    raise RuntimeError(
        f"{operation} failed after {attempts} attempt(s): {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Tool/function-calling mechanics
# ---------------------------------------------------------------------------
def _parse_tool_call(raw: Any) -> ToolCall | None:
    """Parse and validate a model-produced tool call.

    Handles:
      - a real dict/object;
      - JSON text;
      - JSON inside markdown fences;
      - surrounding text containing one JSON object.

    Invalid output returns None instead of being executed blindly.
    """
    if isinstance(raw, ToolCall):
        return raw

    if isinstance(raw, dict):
        candidate = raw
    else:
        text = str(raw or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                candidate = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    try:
        return ToolCall.model_validate(candidate)
    except ValidationError:
        return None


def _validate_tool_arguments(call: ToolCall) -> dict[str, Any]:
    """Validate arguments against the exact tool schema before execution."""
    tool_obj = TOOLS.get(call.name)
    if tool_obj is None:
        raise ValueError(f"Unknown tool '{call.name}'.")

    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return call.arguments

    validated = args_schema.model_validate(call.arguments)
    return validated.model_dump()


def execute_tool_call(call: ToolCall) -> str:
    """Execute one validated tool call with bounded retries."""
    tool_obj = TOOLS.get(call.name)
    if tool_obj is None:
        return f"Tool error: unknown tool '{call.name}'."

    try:
        arguments = _validate_tool_arguments(call)
    except ValidationError as exc:
        return f"Tool error: malformed arguments for {call.name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Tool error: {exc}"

    try:
        return invoke_with_retry(
            lambda: tool_obj.invoke(arguments),
            attempts=MAX_TOOL_RETRIES,
            operation=f"tool {call.name}",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Tool error after retries: {exc}"


def execute_tool_calls(calls: list[ToolCall], *, parallel: bool = False) -> list[str]:
    """Execute tool calls sequentially or in parallel.

    Parallel execution is only appropriate when calls are independent. File writes
    can have ordering/conflict concerns, so the default for this agent is sequential.
    """
    if not calls:
        return []

    if not parallel or len(calls) == 1:
        return [execute_tool_call(call) for call in calls]

    # Independent read/check operations can be parallelized.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = [executor.submit(execute_tool_call, call) for call in calls]
        return [future.result() for future in futures]


# ---------------------------------------------------------------------------
# Parsing helpers for generated/validated source files
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
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
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


def _sanitize_path(path: str, output_dir: str | None = None) -> str:
    """Normalize a file path relative to the output folder (no output-dir prefix)."""
    path = _strip_dot_slash(path)
    root_name = _normalize_output_dir(output_dir)
    while True:
        lower = path.lower()
        stripped = False
        for prefix in (root_name, DEFAULT_SANDBOX_DIR, "sandbox"):
            if lower == prefix:
                return ""
            if lower.startswith(prefix + "/"):
                path = _strip_dot_slash(path[len(prefix) + 1 :])
                stripped = True
                break
        if not stripped:
            break
    if Path(path).name.lower() in _PROTECTED_NAMES:
        stem = Path(path).stem + "_app"
        path = str(Path(path).with_name(stem + (Path(path).suffix or ".py")))
    return path


def _normalize_files(raw_files: Any, output_dir: str | None = None) -> list[dict[str, str]]:
    """Normalize validator files into paths relative to the chosen output folder."""
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

        path = _sanitize_path(path, output_dir)
        content = content.strip()
        fenced = _code_from_fences(content)
        if fenced:
            content = fenced

        if not path or not content or _looks_like_json_blob(content):
            continue

        resolved = _resolve_under_output(path, output_dir)
        if isinstance(resolved, str):
            continue

        cleaned.append({"relative_path": path, "content": content})

    return cleaned


# ---------------------------------------------------------------------------
# Prompts: explicit ReAct + planning behavior
# ---------------------------------------------------------------------------
CODE_GEN_SYSTEM_PROMPT = """You are an expert software engineer inside an autonomous agent loop.

REACT behavior:
- REASON: understand the current task and validator feedback.
- ACT: generate or revise complete source code.
- OBSERVE: the validator will evaluate the result.

Rules:
1. Implement the user's request as complete, runnable source code.
2. If you see [VALIDATOR] feedback, treat it as mandatory review feedback.
3. Do NOT write files to disk yourself. When validation approves, the agent writes
   automatically — no human confirmation is required.
4. Prefer markdown fenced code blocks for each file.
5. File paths are relative to the chosen OUTPUT DIRECTORY for this run (shown in
   a system note). If the user named a folder, use that; otherwise files go under
   sandbox/. Do not repeat the output-directory name as a path prefix.
6. Never use protected agent filenames (agent.py, main.py, agent_ai_doer.py, etc.).
7. Output source code, not fake tool-call JSON.
"""

VALIDATE_SYSTEM_PROMPT = """You are a strict senior code reviewer validating an autonomous coding agent.

Judge whether the latest generated code fully and correctly satisfies the ORIGINAL user request.

Return ONLY JSON with exactly this conceptual shape:
{
  "approved": true | false,
  "feedback": "string",
  "files": [
    {"relative_path": "file.ext", "content": "pure source code"}
  ]
}

Rules:
- approved=true ONLY if code is correct, complete, and meets the request.
- approved=true requires at least one valid source file — the agent will then write
  those files automatically with NO human permission step.
- approved=false must include actionable feedback.
- files[].content must be pure source code, with no JSON wrapper.
- relative_path is relative to the run's OUTPUT DIRECTORY (see system note). Do not
  prefix with the output directory name or "sandbox/" unless that is the filename.
- Never use agent.py, agent2.py, agent_ai_doer.py, main.py, architecture.md,
  agent.md, or .gitignore.
"""

PLAN_SYSTEM_PROMPT = """You are planning an autonomous coding task.
Return ONLY JSON:
{
  "steps": [
    "short, concrete step 1",
    "short, concrete step 2"
  ]
}

Make the plan short and bounded. Files will be written automatically after approval
to the user-mentioned folder, or to sandbox/ if none was mentioned.
Do not invent tools or claim that files were written.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(MessagesState, total=False):
    """Graph state used by the ReAct/planning workflow."""

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
    output_dir: str  # project-relative folder; default sandbox when none mentioned


# ---------------------------------------------------------------------------
# LLM calls with retry + timeout/error handling
# ---------------------------------------------------------------------------
def invoke_llm(messages: list[Any], operation: str):
    _require_api_key()
    return invoke_with_retry(
        lambda: llm.invoke(messages),
        attempts=MAX_TOOL_RETRIES,
        operation=operation,
    )


# ---------------------------------------------------------------------------
# Reason: planning + ReAct decision
# ---------------------------------------------------------------------------
def set_output_dir(state: AgentState) -> AgentState:
    """Pick write target: user-mentioned folder, else sandbox/. No human approval."""
    if state.get("output_dir"):
        out = _normalize_output_dir(state["output_dir"])
    else:
        out = infer_output_dir_from_text(_first_user_text(state.get("messages") or []))
    root = _output_root(out)
    note = (
        f"OUTPUT DIRECTORY for this run: {out}/ "
        f"(absolute: {root}). "
        "When validation approves, files are written here automatically "
        "with no human confirmation."
    )
    return {
        "output_dir": out,
        "status": "ready",
        "last_observation": note,
        "messages": [AIMessage(content=note)],
    }


def create_plan(state: AgentState) -> AgentState:
    """Optional plan-and-execute strategy: create a bounded plan once."""
    if PLAN_MODE != "plan_and_execute" or state.get("plan"):
        return {}

    try:
        response = invoke_llm(
            [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                *state["messages"],
                {
                    "role": "user",
                    "content": "Create a short implementation plan for the original request.",
                },
            ],
            "planning",
        )
        data = _parse_json_object(getattr(response, "content", "") or "")
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Planner returned no valid steps array.")
        plan = [str(step).strip() for step in steps if str(step).strip()]
        if not plan:
            raise ValueError("Planner returned an empty plan.")
        return {"plan": plan[:MAX_STEPS], "plan_index": 0}
    except Exception as exc:  # noqa: BLE001
        # Planning failure should not create an infinite retry loop. Fall back to
        # reactive planning, which is safer because it chooses one next action at a time.
        return {
            "plan": [],
            "plan_index": 0,
            "last_observation": f"Planning failed; falling back to reactive mode: {exc}",
        }


def reason(state: AgentState) -> AgentState:
    """ReAct REASON step: choose the next action from current observations."""
    step_count = int(state.get("step_count") or 0) + 1
    revisions = int(state.get("revision_count") or 0)
    approved = bool(state.get("approved"))
    files = state.get("files") or []

    if approved and files:
        action = "write_files"
    elif revisions >= MAX_REVISIONS:
        action = "escalate"
    elif step_count >= MAX_STEPS:
        action = "escalate"
    elif PLAN_MODE == "plan_and_execute" and state.get("plan"):
        # The plan is guidance; execution still checks the actual state each round.
        index = int(state.get("plan_index") or 0)
        plan = state["plan"]
        if index >= len(plan):
            action = "generate_code" if not state.get("files") else "validate_code"
        elif "validate" in plan[index].lower() or "review" in plan[index].lower():
            action = "validate_code"
        else:
            action = "generate_code"
    else:
        # Reactive strategy: choose the next action based on what just happened.
        # On the first turn there is no generated code yet, so generation is the
        # first action; later turns react to validator/tool observations.
        if not state.get("iteration") and not state.get("feedback") and state.get("status") in {None, "reasoning"}:
            action = "generate_code"
        elif state.get("feedback") and not approved:
            action = "generate_code"
        elif state.get("status") in {"generating", "revising"} and not approved:
            action = "validate_code"
        elif state.get("status") == "writing":
            action = "write_files"
        else:
            action = "validate_code"

    history = list(state.get("action_history") or [])
    history.append(action)

    return {
        "status": "reasoning",
        "step_count": step_count,
        "current_action": action,
        "action_history": history,
    }


# ---------------------------------------------------------------------------
# Act: generate / validate / write / escalate
# ---------------------------------------------------------------------------
def generate_code(state: AgentState) -> AgentState:
    """ACT step: produce or revise code."""
    output_dir = _normalize_output_dir(state.get("output_dir"))
    messages = [
        {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                f"OUTPUT DIRECTORY for this run: {output_dir}/. "
                "Emit filenames relative to that directory. "
                "After approval, files are written there automatically — no human OK needed."
            ),
        },
        *state["messages"],
    ]

    if state.get("plan"):
        index = int(state.get("plan_index") or 0)
        plan = state.get("plan") or []
        if index < len(plan):
            messages.append(
                {
                    "role": "user",
                    "content": f"Execute this plan step: {plan[index]}",
                }
            )

    try:
        response = invoke_llm(messages, "code generation")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "last_observation": f"Code generation failed: {exc}",
            "escalation_reason": f"Code generation failed after retries: {exc}",
        }

    iteration = int(state.get("iteration") or 0)
    return {
        "messages": [response],
        "status": "generating" if iteration == 0 else "revising",
        "approved": False,
        "current_action": "validate_code",
    }


def validate_code(state: AgentState) -> AgentState:
    """ACT step: validate code; malformed validator output is retried safely."""
    iteration = int(state.get("iteration") or 0) + 1
    output_dir = _normalize_output_dir(state.get("output_dir"))
    messages = [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                f"OUTPUT DIRECTORY for this run: {output_dir}/. "
                "files[].relative_path must be relative to that directory. "
                "If approved=true, the agent writes immediately with no human permission."
            ),
        },
        *state["messages"],
        {
            "role": "user",
            "content": "Validate the code. Reply with ONLY the required JSON object.",
        },
    ]

    raw = ""
    data: dict | None = None

    # This is a separate bounded retry for malformed *structured output*.
    # Transport/rate-limit/timeout failures are handled inside invoke_llm().
    for attempt in range(MAX_VALIDATOR_ATTEMPTS):
        try:
            response = invoke_llm(messages, "code validation")
        except Exception as exc:  # noqa: BLE001
            feedback = f"Validator failed after retries: {exc}"
            return {
                "status": "escalating",
                "approved": False,
                "feedback": feedback,
                "files": [],
                "iteration": iteration,
                "last_observation": feedback,
                "escalation_reason": feedback,
            }

        raw = getattr(response, "content", "") or ""
        data = _parse_json_object(raw)
        if data is not None and "approved" in data:
            break

        # Observe malformed output, then ask for correction rather than executing it.
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    'Malformed structured output. Reply with ONLY JSON containing '
                    '"approved" (boolean), "feedback" (string), and "files" (array).'
                ),
            }
        )

    if data is None or "approved" not in data:
        feedback = (
            "Validator returned unparseable output after bounded retries. "
            "Regenerate the full solution clearly in markdown fences with explicit filenames."
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
            "revision_count": int(state.get("revision_count") or 0) + 1,
            "last_observation": feedback,
        }

    approved = bool(data.get("approved"))
    feedback = str(data.get("feedback") or "").strip()
    files = _normalize_files(data.get("files"), output_dir=output_dir)

    # Never trust an approved flag without valid payload data.
    if approved and not files:
        approved = False
        feedback = (
            feedback
            or "Approved flag was set but no valid source files were provided."
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
            "last_observation": summary,
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
        "revision_count": int(state.get("revision_count") or 0) + 1,
        "last_observation": feedback,
    }


def write_files(state: AgentState) -> AgentState:
    """ACT step: persist approved files automatically (no human confirmation)."""
    files = state.get("files") or []
    output_dir = _normalize_output_dir(state.get("output_dir"))
    results: list[str] = []
    for item in files:
        results.append(
            _write_file_impl(
                item.get("relative_path", ""),
                item.get("content", ""),
                output_dir,
            )
        )

    if not results:
        results = ["Error: no files to write."]

    status = "done" if all(r.startswith("Successfully") for r in results) else "failed"
    observation = (
        f"📝 Wrote to {output_dir}/ (automatic — no human permission):\n"
        + "\n".join(f"- {r}" for r in results)
    )
    return {
        "messages": [AIMessage(content=observation)],
        "status": status,
        "write_results": results,
        "last_observation": observation,
        "output_dir": output_dir,
    }


def escalate(state: AgentState) -> AgentState:
    """Stop safely and surface the unresolved problem to a human."""
    reason = state.get("escalation_reason") or state.get("feedback") or "Agent could not complete the task safely."
    message = AIMessage(
        content=(
            "🧑‍💻 Human escalation required.\n"
            f"Reason: {reason}\n"
            f"Steps: {state.get('step_count', 0)}; "
            f"revisions: {state.get('revision_count', 0)}"
        )
    )
    return {
        "messages": [message],
        "status": "escalated",
        "escalation_reason": reason,
        "last_observation": reason,
    }


def fail(state: AgentState) -> AgentState:
    """Terminal failure when a non-recoverable operation fails."""
    reason = state.get("escalation_reason") or state.get("last_observation") or "Unknown failure."
    return {
        "messages": [AIMessage(content=f"❌ Agent failed: {reason}")],
        "status": "failed",
    }


# ---------------------------------------------------------------------------
# Observe: collect the result of the previous action
# ---------------------------------------------------------------------------
def observe(state: AgentState) -> AgentState:
    """REACT OBSERVE step.

    A compact observation is stored so the next REASON step can react to actual
    results rather than blindly following a fixed sequence.
    """
    observation = state.get("last_observation") or "No observation available."
    observations = list(state.get("observations") or [])
    observations.append(observation)

    # Advance a plan only after an action produced an observation.
    plan_index = int(state.get("plan_index") or 0)
    if PLAN_MODE == "plan_and_execute" and state.get("plan"):
        plan_index += 1

    return {
        "status": state.get("status") or "observing",
        "observations": observations[-10:],
        "plan_index": plan_index,
    }


# ---------------------------------------------------------------------------
# Loop detection / stopping conditions
# ---------------------------------------------------------------------------
def _repeated_action_count(history: list[str]) -> int:
    if not history:
        return 0
    latest = history[-1]
    count = 0
    for action in reversed(history):
        if action != latest:
            break
        count += 1
    return count


def _state_signature(state: AgentState) -> str:
    """Cheap signature for detecting repeated identical agent state."""
    material = {
        "action": state.get("current_action"),
        "feedback": state.get("feedback", ""),
        "iteration": state.get("iteration", 0),
        "revision_count": state.get("revision_count", 0),
        "plan_index": state.get("plan_index", 0),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def after_reason(state: AgentState) -> Literal[
    "generate_code", "validate_code", "write_files", "escalate", "fail"
]:
    action = state.get("current_action") or "generate_code"

    if state.get("status") in {"failed", "escalated", "done"}:
        return "fail" if state.get("status") == "failed" else "escalate"

    if _repeated_action_count(list(state.get("action_history") or [])) >= MAX_REPEATED_ACTIONS:
        return "escalate"

    if int(state.get("step_count") or 0) > MAX_STEPS:
        return "escalate"

    if action not in {"generate_code", "validate_code", "write_files", "escalate"}:
        return "fail"

    return action


def after_act(state: AgentState) -> Literal["observe", "escalate", "fail"]:
    if state.get("status") == "done":
        return "observe"
    if state.get("status") == "escalating":
        return "escalate"
    if state.get("status") == "failed":
        return "fail"
    return "observe"


def after_observe(state: AgentState) -> Literal["reason", "escalate", "fail", "end"]:
    if state.get("status") == "done":
        return "end"
    if state.get("status") in {"escalated", "failed"}:
        return "escalate" if state.get("status") == "escalated" else "fail"
    if int(state.get("step_count") or 0) >= MAX_STEPS:
        return "escalate"
    return "reason"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("set_output_dir", set_output_dir)
workflow.add_node("create_plan", create_plan)
workflow.add_node("reason", reason)
workflow.add_node("generate_code", generate_code)
workflow.add_node("validate_code", validate_code)
workflow.add_node("write_files", write_files)
workflow.add_node("observe", observe)
workflow.add_node("escalate", escalate)
workflow.add_node("fail", fail)

workflow.set_entry_point("set_output_dir")
workflow.add_edge("set_output_dir", "create_plan")
workflow.add_edge("create_plan", "reason")

# ReAct: REASON -> ACT
workflow.add_conditional_edges(
    "reason",
    after_reason,
    {
        "generate_code": "generate_code",
        "validate_code": "validate_code",
        "write_files": "write_files",
        "escalate": "escalate",
        "fail": "fail",
    },
)

# ACT -> OBSERVE
for node in ("generate_code", "validate_code", "write_files"):
    workflow.add_conditional_edges(
        node,
        after_act,
        {"observe": "observe", "escalate": "escalate", "fail": "fail"},
    )

# OBSERVE -> REASON, creating the ReAct loop.
workflow.add_conditional_edges(
    "observe",
    after_observe,
    {"reason": "reason", "escalate": "escalate", "fail": "fail", "end": END},
)

workflow.add_edge("escalate", END)
workflow.add_edge("fail", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def new_thread_id() -> str:
    """Fresh thread per project request so runs do not pollute each other."""
    return f"run-{uuid.uuid4().hex[:12]}"
