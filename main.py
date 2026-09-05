#!/usr/bin/env python3
"""
CLI entry point for the autonomous ReAct coding agent (AI Doer / gpt-4o).

Usage (use the project venv — NOT plain `python` unless .venv is activated):
    source .venv/bin/activate
    python main.py

    # or without activating:
    .venv/bin/python main.py

Requires AI_DOER_API_KEY in your environment or .env file.
"""

import sys

try:
    from agent_ai_doer import MAX_REVISIONS, MAX_STEPS, MODEL_NAME, PLAN_MODE, app, new_thread_id
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    print("Missing Python package:", missing, file=sys.stderr)
    print(file=sys.stderr)
    print("You are probably not using the project virtualenv (.venv).", file=sys.stderr)
    print("Fix:", file=sys.stderr)
    print("  deactivate          # exit any old (.env) venv", file=sys.stderr)
    print("  source .venv/bin/activate", file=sys.stderr)
    print("  pip install -r requirements.txt", file=sys.stderr)
    print("  python main.py", file=sys.stderr)
    print(file=sys.stderr)
    print("Or run directly:", file=sys.stderr)
    print("  .venv/bin/python main.py", file=sys.stderr)
    sys.exit(1)


def read_user_input(prompt: str = "What would you like to build? ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)


def _print_partial_state(partial: dict | None) -> None:
    """Print messages and key state fields from a streamed node update."""
    if not partial:
        print("(no state change)\n")
        return

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
    output_dir = partial.get("output_dir")

    parts: list[str] = []
    if status is not None:
        parts.append(f"status={status}")
    if output_dir:
        parts.append(f"output_dir={output_dir}")
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

        print("\n⏳ Autonomous ReAct run (AI Doer / agent_ai_doer)\n")
        print(f"   model={MODEL_NAME}, plan_mode={PLAN_MODE}")
        print(f"   max_steps={MAX_STEPS}, max_revisions={MAX_REVISIONS}")
        print("   Writes automatically after approval (no human accept).")
        print("   Output: user-mentioned folder, else sandbox/\n")
        print("   Flow: set_output_dir → plan → reason → act → observe → write → end\n")

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
                "output_dir": "",
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
        output_dir = final_state.get("output_dir") or "sandbox"

        if status == "done":
            print(f"✅ Project finished. Files written under {output_dir}/:")
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
