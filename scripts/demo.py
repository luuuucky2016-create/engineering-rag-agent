"""Scripted end-to-end demo against a REAL model. Prints the conversation and saves demo_transcript.md.

    python scripts/demo.py
"""

from __future__ import annotations

import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.factory import build_assistant  # noqa: E402

QUESTIONS = [
    # 1. RAG: grounded answer with citations
    "What does IP65 mean, and would it suit a rooftop isolator in dusty Riyadh weather?",
    # 2. RAG + calculator in one turn
    "I have a three-phase load drawing 32 A over a 60 m run of 10 mm² copper cable on a 400 V supply. "
    "What is the voltage drop in volts and as a percentage, and is it within the usual limit?",
    # 3. Memory: relies on the details from turn 2
    "Same circuit, but with 6 mm² cable instead. What would the drop be?",
    # 4. Honesty: not in the knowledge base
    "What is the boiling point of mercury?",
]

CYAN, GREEN, DIM, RESET = "\033[96m", "\033[92m", "\033[2m", "\033[0m"


def main() -> None:
    s = get_settings()
    thread_id = f"demo-{uuid.uuid4().hex[:6]}"
    print(f"{DIM}provider={s.llm_provider} chat_model={s.chat_model} embed_model={s.embed_model} thread={thread_id}{RESET}")
    print(f"{DIM}Indexing {s.docs_dir} ...{RESET}\n")
    assistant = build_assistant(s)

    lines = [
        "# Demo transcript",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.machine()}",
        f"- Model: `{s.chat_model}` via {s.llm_provider} | Embeddings: `{s.embed_model}` | Thread: `{thread_id}`",
        "",
    ]
    for i, q in enumerate(QUESTIONS, 1):
        print(f"{CYAN}[{i}] You:{RESET} {q}")
        t0 = time.perf_counter()
        r = assistant.ask(q, thread_id=thread_id)
        dt = time.perf_counter() - t0
        print(f"{GREEN}Agent:{RESET} {r['answer']}")
        rel = ", ".join(f"{k} {v:.2f}" for k, v in r.get("relevance", {}).items())
        print(f"{DIM}  tools: {', '.join(r['tools_used']) or '-'} | sources: {', '.join(r['sources']) or '-'} | {dt:.1f}s{RESET}")
        removed = ", ".join(r.get("citations_removed", []))
        print(f"{DIM}  retrieval relevance: {rel or '-'}{' | removed fake citations: ' + removed if removed else ''}{RESET}\n")
        lines += [
            f"## {i}. {q}",
            "",
            r["answer"],
            "",
            f"*Tools used:* {', '.join(r['tools_used']) or 'none'}  ",
            f"*Sources:* {', '.join(r['sources']) or 'none'}  ",
            f"*Latency:* {dt:.1f}s",
            "",
        ]
    Path("demo_transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{DIM}Saved demo_transcript.md{RESET}")


if __name__ == "__main__":
    main()
