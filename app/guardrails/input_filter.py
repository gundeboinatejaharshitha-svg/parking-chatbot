"""Input sanitization and prompt-injection guardrails."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.guardrails.pii import GuardrailProcessingError, PIIRedactor, RedactionResult

PROMPT_INJECTION_MESSAGE = (
    "I can't help with requests to bypass safety instructions. "
    "Please ask a CityPark Central question instead."
)

_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,50}\b"
        r"(?:all|any|the|previous|prior|above|system|developer|safety|security)?\s*"
        r"(?:instructions?|rules?|prompts?|guardrails?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat|output|expose|leak|dump)\b.{0,60}\b"
        r"(?:system|developer|hidden|internal)\b.{0,30}\b"
        r"(?:prompt|instructions?|message|policy|rules?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:act as|pretend to be|roleplay as)\b.{0,40}\b"
        r"(?:system|developer|unrestricted|administrator|DAN)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:you are now|from now on)\b.{0,40}\b"
        r"(?:unrestricted|ignore|bypass|developer|system)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(?:jailbreak|do anything now)\b", re.IGNORECASE),
)


class PromptInjectionError(ValueError):
    """Raised when input contains a direct attempt to override model instructions."""


@dataclass
class InputFilter:
    """Block direct prompt injection and redact PII before model use."""

    redactor: PIIRedactor = field(default_factory=PIIRedactor)

    def check_injection(self, text: str) -> None:
        """Reject instruction-override attempts without logging their content."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", normalized)
        if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
            raise PromptInjectionError("Input blocked by safety policy.")

    def filter(self, text: str) -> str:
        """Validate input and return a PII-redacted version."""
        return self.filter_with_values(text).text

    def filter_with_values(self, text: str) -> RedactionResult:
        """Return redacted input and values that must not be echoed by the model."""
        self.check_injection(text)
        try:
            return self.redactor.redact_with_values(text)
        except GuardrailProcessingError:
            raise
        except Exception as exc:
            raise GuardrailProcessingError("Input filtering failed.") from exc


__all__ = [
    "GuardrailProcessingError",
    "InputFilter",
    "PROMPT_INJECTION_MESSAGE",
    "PromptInjectionError",
]
