#!/usr/bin/env python3
"""
CLI entry point for the AI code-writing agent.

Usage:
    .env/bin/python main.py

Interactively receives user input describing the code they want,
then runs the LangGraph agent (agent.py) to generate / review / iterate on it.
"""

import sys

from agent import app, save_accepted_code


def read_user_input(prompt: str = "What would you like to build? ", label: str = "") -> str:
    """Prompt the user for a text input on stdin/stdout."""
    if label:
        prompt = f"{label}: {prompt}"
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)


def print_code(code: str) -> None:
    """Print generated/reviewed code with markdown code-fence styling."""
    print("\n```python\n" + code.strip() + "\n```\n")


def display_new_messages(messages: list, start_index: int) -> None:
    """Print assistant/tool messages produced since start_index."""
    for msg in messages[start_index:]:
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", None)
        if not content or not str(content).strip():
            continue
        text = str(content).strip()
        if msg_type == "ai":
            print(text)
            print()
        elif msg_type == "tool":
            print(f"🔧 {text}\n")


def run_agent() -> None:
    """Main loop — drive the LangGraph agent from user input."""
    config = {"configurable": {"thread_id": "default-thread"}}
    displayed_upto = 0

    while True:
        # Read initial request or feedback
        user_input = read_user_input("Describe the code you want (or type 'quit' to exit): ")

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        # --- First interaction: start fresh graph run ---
        print("\n⏳ Generating code...\n")

        result = app.invoke(
            {"messages": [("user", user_input)]},
            config=config,
        )

        display_new_messages(result["messages"], start_index=displayed_upto)
        displayed_upto = len(result["messages"])
        print(f"📋 Status: {result.get('status', 'unknown')}\n")

        # After generate + review, ask user for feedback or acceptance
        while True:
            feedback = read_user_input(
                "Review the output above. Type feedback to refine, or 'accept' to finish:",
                label="[USER]",
            ).strip()

            if feedback.lower() == "accept":
                print("\n💾 Finalizing clean source and saving...\n")
                outcome = save_accepted_code(result["messages"])
                print(f"🔧 {outcome}\n")
                if outcome.startswith("Error"):
                    print("❌ Save failed. You can type feedback to refine, or 'accept' to retry.\n")
                    continue
                print("✅ Code accepted and saved.\n")
                break

            if feedback.lower() in ("quit", "exit", "q"):
                print("\nBye!")
                sys.exit(0)

            # Feed feedback back into the graph
            print("\n🔄 Refining code based on your feedback...\n")
            result = app.invoke(
                {"messages": [("user", f"[USER] {feedback}")]},
                config=config,
            )

            display_new_messages(result["messages"], start_index=displayed_upto)
            displayed_upto = len(result["messages"])
            print(f"📋 Status: {result.get('status', 'unknown')}\n")


if __name__ == "__main__":
    run_agent()
