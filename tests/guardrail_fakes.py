"""Model-free guardrail collaborators shared by Stage 3 tests."""

from __future__ import annotations

from typing import Any

from presidio_anonymizer import AnonymizerEngine

from app.guardrails.input_filter import InputFilter
from app.guardrails.output_filter import OutputFilter
from app.guardrails.pii import PIIRedactor


class EmptyAnalyzer:
    """Presidio-compatible analyzer that does not load an NLP model."""

    def analyze(self, *, text: str, language: str) -> list[Any]:
        return []


class FailingAnalyzer:
    """Raise during analysis to verify the LLM boundary fails closed."""

    def analyze(self, *, text: str, language: str) -> list[Any]:
        raise RuntimeError("analyzer unavailable")


def build_test_filters() -> tuple[InputFilter, OutputFilter]:
    """Build deterministic filters using only project regexes and the Presidio anonymizer."""
    input_filter = InputFilter(
        redactor=PIIRedactor(analyzer=EmptyAnalyzer(), anonymizer=AnonymizerEngine())
    )
    output_filter = OutputFilter(
        redactor=PIIRedactor(analyzer=EmptyAnalyzer(), anonymizer=AnonymizerEngine())
    )
    return input_filter, output_filter
