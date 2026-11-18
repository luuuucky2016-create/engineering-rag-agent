"""Interactive terminal chat:  python -m app.cli  (type 'exit' to quit)."""

from __future__ import annotations

import sys
import uuid

from app.factory import build_assistant


def main() -> None:
    thread_id = sys.argv[1] if len(sys.argv) > 1 else f"cli-{uuid.uuid4().hex[:8]}"
    print("Loading models and indexing documents...")
    assistant = build_assistant()
    print(f"Ready. thread_id={thread_id}  (type 'exit' to quit)\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        r = assistant.ask(msg, thread_id=thread_id)
        print(f"\nAgent: {r['answer']}")
        if r["tools_used"]:
            print(f"  tools: {', '.join(r['tools_used'])}")
        if r["sources"]:
            print(f"  sources: {', '.join(r['sources'])}")
        print()


if __name__ == "__main__":
    main()
