"""Interactive terminal chat for manual end-to-end checks.

Usage: ``python -m scripts.chat_cli``  (type 'exit' to quit)

Requires a seeded database, an ingested vector store and a configured LLM.
This is a script, so printing is intentional.
"""

from __future__ import annotations

import sys

from app.chatbot.llm import LLMConfigurationError
from app.chatbot.orchestrator import ConversationSession, build_orchestrator
from app.config import configure_logging


def main() -> int:
    """Run a read-eval-print loop over the orchestrator."""
    configure_logging("WARNING")
    try:
        orchestrator = build_orchestrator()
    except LLMConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    session = ConversationSession()
    print("CityPark Central assistant. Type 'exit' to quit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        reply = orchestrator.handle(message, session)
        print(f"bot> {reply.text}")
        if reply.sources:
            cited = ", ".join(f"{s.source} ({s.score:.2f})" for s in reply.sources)
            print(f"     sources: {cited}")
        flow_label = f", flow={reply.flow}]" if reply.flow else "]"
        print(f"     [intent={reply.intent.value}{flow_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
