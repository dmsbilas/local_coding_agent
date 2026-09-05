"""
Autonomous LangGraph coding agent powered by AI Doer (gpt-4o).

Flow:
  set_output_dir → create_plan → reason → act → observe → … → write_files → END

Writes happen automatically after LLM validation approves — no human accept step.
If the user names a folder, files go there; otherwise under sandbox/.

Uses the AI Doer OpenAI-compatible API:
  Base URL: https://ai-doer.com/v1/api
  Auth:     Authorization: Bearer <AI_DOER_API_KEY>
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from config.settings import (
    AI_DOER_API_KEY,
    AI_DOER_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    MAX_REPEATED_ACTIONS,
    MAX_REVISIONS,
    MAX_STEPS,
    MAX_TOOL_RETRIES,
    MAX_VALIDATOR_ATTEMPTS,
    MODEL_NAME,
    PLAN_MODE,
    require_ai_doer_api_key,
)
from prompts.react import (
    CODE_GEN_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    VALIDATE_SYSTEM_PROMPT,
)
from tools.file_tools import normalize_files, write_file_impl
from tools.paths import (
    first_user_text,
    infer_output_dir_from_text,
    normalize_output_dir,
    output_root,
)
from utils.parsing import parse_json_object
from utils.retry import invoke_with_retry

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=AI_DOER_API_KEY or "missing-key",
    base_url=AI_DOER_BASE_URL,
    temperature=0.2,
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=0,
    model_kwargs={"stream": False},
)


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
    output_dir: str


def invoke_llm(messages: list[Any], operation: str):
    require_ai_doer_api_key()
    return invoke_with_retry(
        lambda: llm.invoke(messages),
        attempts=MAX_TOOL_RETRIES,
        operation=operation,
    )


def set_output_dir(state: AgentState) -> AgentState:
    """Pick write target: user-mentioned folder, else sandbox/."""
    if state.get("output_dir"):
        out = normalize_output_dir(state["output_dir"])
    else:
        out = infer_output_dir_from_text(first_user_text(state.get("messages") or []))
    root = output_root(out)
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
        data = parse_json_object(getattr(response, "content", "") or "")
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Planner returned no valid steps array.")
        plan = [str(step).strip() for step in steps if str(step).strip()]
        if not plan:
            raise ValueError("Planner returned an empty plan.")
        return {"plan": plan[:MAX_STEPS], "plan_index": 0}
    except Exception as exc:  # noqa: BLE001
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
        index = int(state.get("plan_index") or 0)
        plan = state["plan"]
        if index >= len(plan):
            action = "generate_code" if not state.get("files") else "validate_code"
        elif "validate" in plan[index].lower() or "review" in plan[index].lower():
            action = "validate_code"
        else:
            action = "generate_code"
    else:
        if not state.get("iteration") and not state.get("feedback") and state.get("status") in {
            None,
            "reasoning",
            "ready",
        }:
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


def generate_code(state: AgentState) -> AgentState:
    """ACT step: produce or revise code."""
    output_dir = normalize_output_dir(state.get("output_dir"))
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
    output_dir = normalize_output_dir(state.get("output_dir"))
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
        data = parse_json_object(raw)
        if data is not None and "approved" in data:
            break

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
    files = normalize_files(data.get("files"), output_dir=output_dir)

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
    output_dir = normalize_output_dir(state.get("output_dir"))
    results: list[str] = []
    for item in files:
        results.append(
            write_file_impl(
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
    reason = (
        state.get("escalation_reason")
        or state.get("feedback")
        or "Agent could not complete the task safely."
    )
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


def observe(state: AgentState) -> AgentState:
    """ReAct OBSERVE step — feed results into the next REASON turn."""
    observation = state.get("last_observation") or "No observation available."
    observations = list(state.get("observations") or [])
    observations.append(observation)

    plan_index = int(state.get("plan_index") or 0)
    if PLAN_MODE == "plan_and_execute" and state.get("plan"):
        plan_index += 1

    return {
        "status": state.get("status") or "observing",
        "observations": observations[-10:],
        "plan_index": plan_index,
    }


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

for node in ("generate_code", "validate_code", "write_files"):
    workflow.add_conditional_edges(
        node,
        after_act,
        {"observe": "observe", "escalate": "escalate", "fail": "fail"},
    )

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
