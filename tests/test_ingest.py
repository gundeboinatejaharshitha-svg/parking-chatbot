"""Tests for app.data.ingest (pure chunking; no vector store, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.ingest import (
    EXPECTED_CHUNKS,
    EXPECTED_FILES,
    STATIC_DIR,
    IngestionError,
    chunk_markdown,
    clean_title,
    load_static_chunks,
    slugify,
    split_words,
)

SAMPLE = """# Parking rules

## What are the standard parking rules at CityPark Central?
Drivers must park fully within a marked bay.

## Where is stopping or parking prohibited at CityPark Central?
Stopping is prohibited on ramps and in lanes.

## What happens if I park outside a marked bay at CityPark Central?
A warning first, then a penalty notice.
"""


# ---- title cleaning --------------------------------------------------------
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Can I get a refund at CityPark Central?", "Can I get a refund?"),
        (
            "How do I contact CityPark Central support?",
            "How do I contact CityPark Central support?",
        ),
        ("Where is CityPark Central located?", "Where is CityPark Central located?"),
        ("Can motorcycles use CityPark Central?", "Can motorcycles use?"),
        ("Plain title", "Plain title"),
    ],
)
def test_clean_title_strips_only_trailing_suffix(title: str, expected: str) -> None:
    assert clean_title(title) == expected


def test_slugify_truncates_and_strips_hyphens() -> None:
    assert slugify("Where is stopping or parking prohibited?") == (
        "where-is-stopping-or-parking-prohibited"
    )
    long = slugify("Where is stopping or parking prohibited at CityPark Central?")
    assert len(long) <= 40 and not long.endswith("-")


# ---- chunk IDs and structure -----------------------------------------------
def test_chunk_ids_follow_documented_format() -> None:
    chunks = chunk_markdown(SAMPLE, "03_parking_rules.md")
    assert [c.id for c in chunks] == [
        "03_parking_rules::01-what-are-the-standard-parking-rules::0",
        "03_parking_rules::02-where-is-stopping-or-parking-prohibited::0",
        "03_parking_rules::03-what-happens-if-i-park-outside-a-marked::0",
    ]


def test_chunk_text_and_metadata() -> None:
    chunk = chunk_markdown(SAMPLE, "03_parking_rules.md")[1]
    assert chunk.text == (
        "Where is stopping or parking prohibited?\nStopping is prohibited on ramps and in lanes."
    )
    assert chunk.section == "Where is stopping or parking prohibited at CityPark Central?"
    assert chunk.source == "03_parking_rules.md"
    assert chunk.chunk_index == 0


def test_preamble_and_empty_sections_are_ignored() -> None:
    text = "# Title\nintro text\n\n## Empty section at CityPark Central?\n\n## Real one?\nBody.\n"
    chunks = chunk_markdown(text, "x.md")
    assert [c.id for c in chunks] == ["x::02-real-one::0"]


def test_document_without_headings_yields_no_chunks() -> None:
    assert chunk_markdown("# Only a title\nSome text.", "y.md") == []
    assert chunk_markdown("", "z.md") == []


# ---- long-section splitting ------------------------------------------------
def test_split_words_overlap() -> None:
    words = [f"w{i}" for i in range(10)]
    pieces = split_words(" ".join(words), max_words=4, overlap=1)
    assert pieces == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_split_words_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        split_words("a b c", max_words=2, overlap=2)


def test_long_section_is_split_with_overlap_and_shared_title() -> None:
    body = " ".join(f"word{i}" for i in range(900))
    text = f"## Long section at CityPark Central?\n{body}\n"
    chunks = chunk_markdown(text, "long.md")  # defaults: 400 words, 50 overlap

    assert [c.id for c in chunks] == [f"long::01-long-section::{i}" for i in range(3)]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.text.startswith("Long section?\n") for c in chunks)

    bodies = [c.text.split("\n", 1)[1].split() for c in chunks]
    assert len(bodies[0]) == 400
    assert bodies[0][-50:] == bodies[1][:50]  # overlap preserved
    assert bodies[1][-50:] == bodies[2][:50]
    assert bodies[2][-1] == "word899"  # nothing dropped at the tail


# ---- real static corpus ----------------------------------------------------
def test_static_corpus_produces_expected_counts_and_unique_ids() -> None:
    file_count, chunks = load_static_chunks(STATIC_DIR)
    assert file_count == EXPECTED_FILES == 10
    assert len(chunks) == EXPECTED_CHUNKS == 38
    assert len({c.id for c in chunks}) == len(chunks)
    assert "03_parking_rules::02-where-is-stopping-or-parking-prohibited::0" in {
        c.id for c in chunks
    }


def test_load_static_chunks_fails_loudly_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_static_chunks(tmp_path / "nope")
    with pytest.raises(IngestionError):
        load_static_chunks(tmp_path)  # exists but has no .md files
