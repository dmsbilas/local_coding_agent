#!/usr/bin/env python3
"""
CLI entry point for the autonomous ReAct coding agent (agent2).

Usage:
    .env/bin/python main.py

Describe what you want once. The agent plans (optionally), reasons, generates,
validates, revises until approved, writes files, then finishes — no manual accept.
"""

import sys

from agent2 import MAX_REVISIONS, MAX_STEPS, PLAN_MODE, app, new_thread_id


def read_user_input(prompt: str = "What would you like to build? ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)


def _print_partial_state(partial: dict) -> None:
    """Print messages and key state fields from a streamed node update."""
    for msg in partial.get("messages") or []:
        msg_type = getattr(msg, "type", "")
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            continue
        if msg_type == "ai":
            print(content)
            print()
        elif msg_type == "human" and content.startswith("[VALIDATOR]"):
            print(f"🔎 {content}\n")

    plan = partial.get("plan")
    if plan:
        print("📋 Plan:")
        for i, step in enumerate(plan, start=1):
            print(f"   {i}. {step}")
        print()

    action = partial.get("current_action")
    observation = partial.get("last_observation")
    status = partial.get("status")
    iteration = partial.get("iteration")
    step_count = partial.get("step_count")
    revision_count = partial.get("revision_count")
    approved = partial.get("approved")
    escalation_reason = partial.get("escalation_reason")

    parts: list[str] = []
    if status is not None:
        parts.append(f"status={status}")
    if action:
        parts.append(f"action={action}")
    if step_count is not None:
        parts.append(f"steps={step_count}/{MAX_STEPS}")
    if iteration is not None:
        parts.append(f"validate_round={iteration}")
    if revision_count is not None:
        parts.append(f"revisions={revision_count}/{MAX_REVISIONS}")
    if approved is not None:
        parts.append(f"approved={approved}")
    if escalation_reason:
        parts.append(f"escalation={escalation_reason}")
    if observation and status in {"observing", "validating", "writing", "done", "failed"}:
        preview = observation if len(observation) <= 200 else observation[:200] + "..."
        parts.append(f"obs={preview}")

    if parts:
        print(f"📋 {', '.join(parts)}\n")


def run_agent() -> None:
    """Ask for a task, run the ReAct loop until files are written or escalation."""
    while True:
        user_input = read_user_input(
            "Describe the code you want (or type 'quit' to exit): "
        )

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nBye!")
            break

        if not user_input:
            continue

        config = {
            "configurable": {"thread_id": new_thread_id()},
            # ReAct runs reason → act → observe per step; allow headroom.
            "recursion_limit": max(100, MAX_STEPS * 8 + 20),
        }

        print("\n⏳ Autonomous ReAct run (agent2)\n")
        print(f"   plan_mode={PLAN_MODE}")
        print(f"   max_steps={MAX_STEPS}, max_revisions={MAX_REVISIONS}\n")
        print("   Flow: plan → reason → act → observe → … → write → end\n")

        for update in app.stream(
            {
                "messages": [("user", user_input)],
                "iteration": 0,
                "approved": False,
                "files": [],
                "feedback": "",
                "status": "starting",
                "step_count": 0,
                "revision_count": 0,
                "action_history": [],
                "observations": [],
                "plan": [],
                "plan_index": 0,
            },
            config=config,
            stream_mode="updates",
        ):
            for node_name, partial in update.items():
                print(f"—— {node_name} ——")
                _print_partial_state(partial)

        final_state = app.get_state(config).values
        status = final_state.get("status", "unknown")
        write_results = final_state.get("write_results") or []
        escalation_reason = final_state.get("escalation_reason")

        if status == "done":
            print("✅ Project finished. Files written:")
            for line in write_results:
                print(f"   • {line}")
            print()
        elif status == "escalated":
            print("🧑‍💻 Agent escalated — human review recommended.")
            if escalation_reason:
                print(f"   Reason: {escalation_reason}")
            print()
        elif status == "failed":
            print("❌ Agent failed without completing the task.")
            if escalation_reason:
                print(f"   Reason: {escalation_reason}")
            print()
        else:
            print(f"📋 Finished with status: {status}\n")


if __name__ == "__main__":
    run_agent()
