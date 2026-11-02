"""Check the configured live LLM with a synthetic prompt, not parking or user data."""

from __future__ import annotations

import logging

from app.chatbot.llm import LLMConfigurationError, LLMError, build_llm_client
from app.config import configure_logging

logger = logging.getLogger(__name__)
EXPECTED_REPLY = "LLM_OK"


def main() -> int:
    """Return zero only when the configured model answers the smoke-test prompt."""
    configure_logging()
    try:
        client = build_llm_client()
        reply = client.complete(
            "This is a connectivity check. Reply with exactly LLM_OK and nothing else.",
            "Run the connectivity check.",
        )
    except LLMConfigurationError:
        logger.error(
            "LLM check failed: configure a valid LLM_API_KEY for OpenAI, or set "
            "LLM_PROVIDER=ollama with a running server and an installed LLM_MODEL."
        )
        return 1
    except LLMError:
        logger.error("LLM check failed: the provider did not return a usable reply.")
        return 1

    if reply != EXPECTED_REPLY:
        logger.error("LLM check failed: a reply arrived but did not match the requested token.")
        return 1

    logger.info("LLM check passed: the configured model returned the requested token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
