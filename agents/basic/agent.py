"""
Minimal autonomous LangGraph coding agent (learning / baseline).

Flow:
  generate_code → validate_code ⟲ (revise until approved) → write_files → END

Uses a local Ollama model. Requires: pip install langchain-ollama && ollama serve
"""

from __future__ import annotations

import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from config.settings import (
    DEFAULT_SANDBOX_DIR,
    MAX_REVISIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME as MODEL_NAME,
)
from prompts.basic import CODE_GEN_SYSTEM_PROMPT, VALIDATE_SYSTEM_PROMPT
from tools.file_tools import normalize_files, write_file_impl
from utils.parsing import parse_json_object

try:
    from langchain_ollama import ChatOllama
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "agents.basic requires langchain-ollama. "
        "Install with: pip install langchain-ollama"
    ) from exc

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


class AgentState(MessagesState, total=False):
    """Graph state: chat history + validation outcome."""

    status: str
    approved: bool
    feedback: str
    files: list
    iteration: int
    write_results: list


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
    for _attempt in range(2):
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""
        data = parse_json_object(raw)
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
    files = normalize_files(data.get("files"), output_dir=DEFAULT_SANDBOX_DIR)

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
    """Persist approved files under sandbox/."""
    files = state.get("files") or []
    results: list[str] = []
    for item in files:
        results.append(
            write_file_impl(
                item.get("relative_path", ""),
                item.get("content", ""),
                DEFAULT_SANDBOX_DIR,
            )
        )

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


def after_validate(
    state: AgentState,
) -> Literal["write_files", "generate_code", "fail_max_revisions"]:
    if state.get("approved") and state.get("files"):
        return "write_files"
    if int(state.get("iteration") or 0) >= MAX_REVISIONS:
        return "fail_max_revisions"
    return "generate_code"


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
    return f"run-{uuid.uuid4().hex[:12]}"
