"""Shared PII recognition and redaction using Presidio."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

REDACTED = "[REDACTED]"

_BOOKING_REFERENCE = re.compile(r"\bCP\s*-?\s*\d{6}\b", re.IGNORECASE)
_LABELED_PLATE = re.compile(
    r"\b(?:vehicle\s+)?(?:registration(?:\s+number)?|reg(?:istration)?|number\s+plate|plate)"
    r"\s*(?:(?:is|:|=)\s*)"
    r"(?P<plate>(?:[A-Z0-9][ -]?){1,7}[A-Z0-9])",
    re.IGNORECASE,
)
_COMMON_PLATE = re.compile(
    r"\b(?:[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]{1,3}|[A-Z]{2}\s*\d{2}\s*[A-Z]{3})\b",
    re.IGNORECASE,
)
_LABELED_NAME = re.compile(
    r"\b(?:my name is|name)\s*(?:is|:)?\s*"
    r"(?P<name>(?!(?:and|but|with|for|about|what|where|when|who|why|how|can|could|do|does|"
    r"is|are|please)\b)[A-Z][A-Za-z'’-]*(?:\s+(?!(?:and|but|with|for|about|what|where|when|"
    r"who|why|how|can|could|do|does|is|are|please)\b)[A-Z][A-Za-z'’-]*){0,3})",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"\b(?:api[_ -]?key|password|access[_ -]?token)\s*[:=]\s*"
    r"(?P<secret>[^\s,;]+)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+(?P<secret>[A-Za-z0-9._~+/-]+=*)", re.IGNORECASE)
_SECRET_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\."
    r"[A-Za-z0-9_-]+)\b"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PUBLIC_INFORMATION = re.compile(
    r"28 Station Square, Crown District, City Centre, B1 7QR|B1 7QR|"
    r"\+44 20 5555 0101|support@cityparkcentral\.example|"
    r"(?:the )?(?:Station|Retail) Entrance|CityPark Central|\b24 hours?\b",
    re.IGNORECASE,
)


class Analyzer(Protocol):
    """Subset of Presidio's AnalyzerEngine used by this module."""

    def analyze(self, *, text: str, language: str) -> list[Any]:
        """Return detected entity spans."""
        ...


class Anonymizer(Protocol):
    """Subset of Presidio's AnonymizerEngine used by this module."""

    def anonymize(self, *, text: str, analyzer_results: list[Any], operators: dict) -> Any:
        """Return text with detected spans anonymized."""
        ...


class GuardrailProcessingError(RuntimeError):
    """Raised when PII recognition or anonymization cannot be completed."""


@dataclass
class PIIRedactor:
    """Redact model-detected and project-specific sensitive values."""

    analyzer: Analyzer | None = None
    anonymizer: Anonymizer | None = None
    _component_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def redact(self, text: str) -> str:
        """Return text with recognized PII and credentials replaced."""
        return self.redact_with_values(text).text

    def redact_with_values(self, text: str) -> RedactionResult:
        """Return safe text and the values removed so they cannot be echoed later."""
        if not text:
            return RedactionResult(text, ())
        try:
            analyzer, anonymizer = self._components()
            results = list(analyzer.analyze(text=text, language="en"))
            results.extend(self._custom_results(text))
            results = self._remove_overlaps(results, text)
            if not results:
                return RedactionResult(text, ())

            from presidio_anonymizer.entities import OperatorConfig

            result = anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": REDACTED})},
            )
            sensitive_values = tuple(
                dict.fromkeys(text[item.start : item.end] for item in results)
            )
            return RedactionResult(result.text, sensitive_values)
        except GuardrailProcessingError:
            raise
        except Exception as exc:
            raise GuardrailProcessingError("PII processing failed.") from exc

    def _components(self) -> tuple[Analyzer, Anonymizer]:
        if self.analyzer is None or self.anonymizer is None:
            with self._component_lock:
                try:
                    from presidio_analyzer import AnalyzerEngine
                    from presidio_anonymizer import AnonymizerEngine

                    if self.analyzer is None:
                        self.analyzer = AnalyzerEngine()
                    if self.anonymizer is None:
                        self.anonymizer = AnonymizerEngine()
                except Exception as exc:
                    raise GuardrailProcessingError("PII analyzer is unavailable.") from exc
        if self.analyzer is None or self.anonymizer is None:
            raise GuardrailProcessingError("PII components are unavailable.")
        return self.analyzer, self.anonymizer

    @staticmethod
    def _custom_results(text: str) -> list[Any]:
        """Build Presidio-compatible spans for identifiers absent from generic NER."""
        from presidio_analyzer import RecognizerResult

        results: list[Any] = []
        patterns = (
            ("BOOKING_REFERENCE", _BOOKING_REFERENCE, None),
            ("VEHICLE_REGISTRATION", _LABELED_PLATE, "plate"),
            ("VEHICLE_REGISTRATION", _COMMON_PLATE, None),
            ("PERSON", _LABELED_NAME, "name"),
            ("CREDENTIAL", _CREDENTIAL, "secret"),
            ("CREDENTIAL", _BEARER, "secret"),
            ("CREDENTIAL", _SECRET_TOKEN, None),
            ("EMAIL_ADDRESS", _EMAIL, None),
        )
        for entity_type, pattern, group in patterns:
            for match in pattern.finditer(text):
                start, end = match.span(group) if group else match.span()
                if end > start:
                    results.append(
                        RecognizerResult(
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            score=1.0,
                        )
                    )
        return results

    @staticmethod
    def _remove_overlaps(results: list[Any], text: str) -> list[Any]:
        """Keep valid, non-overlapping spans so Presidio can anonymize reliably."""
        public_spans = [match.span() for match in _PUBLIC_INFORMATION.finditer(text)]
        ordered = sorted(
            (
                item
                for item in results
                if item.start < item.end
                and not any(
                    public_start <= item.start and item.end <= public_end
                    for public_start, public_end in public_spans
                )
            ),
            key=lambda item: (item.start, -(item.end - item.start), -item.score),
        )
        accepted: list[Any] = []
        last_end = -1
        for item in ordered:
            if item.start >= last_end:
                accepted.append(item)
                last_end = item.end
        return accepted


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text and original values removed during the operation."""

    text: str
    sensitive_values: tuple[str, ...]


__all__ = ["GuardrailProcessingError", "PIIRedactor", "REDACTED", "RedactionResult"]
