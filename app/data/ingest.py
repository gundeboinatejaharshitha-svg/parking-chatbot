"""Static knowledge ingestion: Markdown -> chunks -> vector store.

Usage: ``python -m app.data.ingest [--reset]``

Each ``## `` section of the Markdown files in ``app/data/static`` becomes one
chunk (long sections are split into overlapping word windows). Chunk IDs are
deterministic so re-running the ingestion upserts in place and never
duplicates data.

Chunk ID format::

    {file_stem}::{section_number:02d}-{slug[:40]}::{chunk_index}
    e.g. 03_parking_rules::02-where-is-stopping-or-parking-prohibited::0

``section_number`` is 1-based within the file; ``slug`` is derived from the
section title after the facility-name suffix has been removed.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import configure_logging
from app.data.vector_store import VectorStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXPECTED_CHUNKS = 38
EXPECTED_FILES = 10

MAX_CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
SLUG_MAX_LEN = 40

FACILITY_NAME = "CityPark Central"
_HEADING = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
_SUFFIXES = (f" at {FACILITY_NAME}", f" {FACILITY_NAME}")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


class IngestionError(Exception):
    """Raised when the knowledge base does not look like what we expect."""


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of static knowledge.

    ``text`` is what gets embedded and stored: the cleaned title followed by the
    section body. ``section`` keeps the full original heading for citations.
    """

    id: str
    text: str
    source: str
    section: str
    chunk_index: int


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def clean_title(title: str) -> str:
    """Strip the trailing facility name so embeddings focus on the question.

    ``"Where is CityPark Central located?"`` is left unchanged because the name
    is in the middle, not a trailing suffix.
    """
    cleaned = title.strip()
    punctuation = ""
    if cleaned and cleaned[-1] in "?.!":
        cleaned, punctuation = cleaned[:-1], cleaned[-1]
    for suffix in _SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return f"{cleaned.rstrip()}{punctuation}"


def slugify(title: str, max_len: int = SLUG_MAX_LEN) -> str:
    """Lowercase, hyphen-separated slug truncated to ``max_len`` characters."""
    slug = _NON_SLUG.sub("-", title.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def split_words(
    text: str, max_words: int = MAX_CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    """Split ``text`` into windows of at most ``max_words`` with ``overlap`` shared words."""
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("Need max_words >= 1 and 0 <= overlap < max_words.")
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max_words - overlap
    pieces: list[str] = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    return pieces


def chunk_markdown(
    text: str,
    source: str,
    *,
    max_words: int = MAX_CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split one Markdown document into chunks, one per ``## `` section.

    Text before the first ``## `` heading (the ``# `` title) is ignored. Empty
    sections are skipped. Sections longer than ``max_words`` are split into
    overlapping windows that share the same title and section number.
    """
    file_stem = Path(source).stem
    headings = list(_HEADING.finditer(text))
    chunks: list[Chunk] = []

    for number, match in enumerate(headings, start=1):
        title = match.group(1).strip()
        body_end = headings[number].start() if number < len(headings) else len(text)
        body = text[match.end() : body_end].strip()
        if not body:
            logger.warning("Skipping empty section %r in %s", title, source)
            continue

        short_title = clean_title(title)
        slug = slugify(short_title)
        for index, piece in enumerate(split_words(body, max_words, overlap)):
            chunks.append(
                Chunk(
                    id=f"{file_stem}::{number:02d}-{slug}::{index}",
                    text=f"{short_title}\n{piece}",
                    source=source,
                    section=title,
                    chunk_index=index,
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Loading and running
# --------------------------------------------------------------------------
def load_static_chunks(static_dir: Path = STATIC_DIR) -> tuple[int, list[Chunk]]:
    """Read every ``*.md`` file in ``static_dir`` (sorted) and chunk it.

    Returns ``(file_count, chunks)``.
    """
    if not static_dir.is_dir():
        raise IngestionError(f"Static knowledge directory not found: {static_dir}")
    files = sorted(static_dir.glob("*.md"))
    if not files:
        raise IngestionError(f"No Markdown files found in {static_dir}")

    chunks: list[Chunk] = []
    for path in files:
        file_chunks = chunk_markdown(path.read_text(encoding="utf-8"), path.name)
        logger.info("%s -> %d chunks", path.name, len(file_chunks))
        chunks.extend(file_chunks)

    ids = [chunk.id for chunk in chunks]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise IngestionError(f"Duplicate chunk IDs produced: {duplicates}")
    return len(files), chunks


def run_ingestion(
    store: VectorStore,
    *,
    static_dir: Path = STATIC_DIR,
    reset: bool = False,
    expected_chunks: int | None = EXPECTED_CHUNKS,
    expected_files: int | None = EXPECTED_FILES,
) -> int:
    """Chunk the static docs and upsert them into ``store``; returns the chunk count.

    Pass ``expected_*=None`` to skip the sanity checks (used for custom corpora).
    """
    file_count, chunks = load_static_chunks(static_dir)
    if expected_files is not None and file_count != expected_files:
        raise IngestionError(f"Expected {expected_files} Markdown files, found {file_count}.")
    if expected_chunks is not None and len(chunks) != expected_chunks:
        raise IngestionError(
            f"Expected {expected_chunks} chunks from the static docs, got {len(chunks)}. "
            "The static files or the chunker changed; review before ingesting."
        )

    if reset:
        logger.warning("Resetting vector collection '%s'", store.collection_name)
        store.reset()

    written = store.add_chunks(chunks)
    logger.info(
        "Ingestion complete: %d files, %d chunks written, %d stored",
        file_count,
        written,
        store.count(),
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Ingest the static knowledge base.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args(argv)

    configure_logging()
    try:
        run_ingestion(VectorStore(), reset=args.reset)
    except IngestionError as exc:
        logger.error("Ingestion aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
