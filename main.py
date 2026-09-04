#!/usr/bin/env python3
"""
CLI entry point for the autonomous AI coding agent.

Usage:
    .env/bin/python main.py

Describe what you want once. The agent generates code, validates it with the
LLM, revises until approved, writes files, then finishes — no manual accept.
"""

import sys

from agent import MAX_REVISIONS, app, new_thread_id


def read_user_input(prompt: str = "What would you like to build? ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)


def run_agent() -> None:
    """Ask for a task, run the autonomous loop until files are written (or fail)."""
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
            "recursion_limit": max(50, MAX_REVISIONS * 4 + 10),
        }

        print("\n⏳ Autonomous run: generate → validate → revise → write files\n")
        print(f"(max {MAX_REVISIONS} revision rounds)\n")

        for update in app.stream(
            {
                "messages": [("user", user_input)],
                "iteration": 0,
                "approved": False,
                "files": [],
                "feedback": "",
                "status": "starting",
            },
            config=config,
            stream_mode="updates",
        ):
            for node_name, partial in update.items():
                print(f"—— {node_name} ——")
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

                status = partial.get("status")
                iteration = partial.get("iteration")
                approved = partial.get("approved")
                if status is not None:
                    extra = f"status={status}"
                    if iteration is not None:
                        extra += f", iteration={iteration}"
                    if approved is not None:
                        extra += f", approved={approved}"
                    print(f"📋 {extra}\n")

        final_state = app.get_state(config).values
        status = final_state.get("status", "unknown")
        write_results = final_state.get("write_results") or []

        if status == "done":
            print("✅ Project finished. Files written:")
            for line in write_results:
                print(f"   • {line}")
            print()
        elif status == "failed":
            print("❌ Agent stopped without writing files.\n")
        else:
            print(f"📋 Finished with status: {status}\n")


if __name__ == "__main__":
    run_agent()
