"""Output safety filtering and PII protection for responses."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.guardrails.pii import GuardrailProcessingError, PIIRedactor

_INTERNAL_PROMPT_MARKERS = (
    re.compile(r"\byou are the virtual assistant for citypark central\b", re.IGNORECASE),
    re.compile(
        r"\banswer the user's question using only the numbered context passages\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bif the context does not contain the answer, reply exactly\b",
        re.IGNORECASE,
    ),
)


@dataclass
class OutputFilter:
    """Redact model-produced PII and credentials before returning a response."""

    redactor: PIIRedactor = field(default_factory=PIIRedactor)

    def filter(self, text: str, *, sensitive_values: Iterable[str] = ()) -> str:
        """Return a response with detected sensitive values removed."""
        try:
            if any(pattern.search(text) for pattern in _INTERNAL_PROMPT_MARKERS):
                raise GuardrailProcessingError("Unsafe model output.")
            for value in sorted(set(sensitive_values), key=len, reverse=True):
                if value:
                    text = re.sub(re.escape(value), "[REDACTED]", text, flags=re.IGNORECASE)
            return self.redactor.redact(text)
        except GuardrailProcessingError:
            raise
        except Exception as exc:
            raise GuardrailProcessingError("Output filtering failed.") from exc


__all__ = ["OutputFilter"]
