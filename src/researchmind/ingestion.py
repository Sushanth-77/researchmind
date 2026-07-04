"""
PDF ingestion and chunking.

Extracts raw text from a PDF and splits it into fixed-size overlapping
character chunks. Overlap preserves context across chunk boundaries so
sentences split mid-chunk aren't orphaned from their surrounding meaning.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Defaults chosen for typical research-paper prose: large enough to hold
# a full paragraph or table row, small enough to keep embeddings focused.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    """A single chunk of extracted text with tracking metadata."""

    text: str
    chunk_index: int
    source_file: str


def extract_text(pdf_path: Path) -> str:
    """
    Extract raw text from a PDF file, concatenating all pages.

    Raises:
        FileNotFoundError: if pdf_path does not exist.
        ValueError: if the PDF has no extractable text (e.g. scanned
            image-only PDF with no OCR layer).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            f"No extractable text found in {pdf_path}. "
            f"It may be a scanned/image-only PDF with no OCR layer."
        )

    return full_text


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split text into fixed-size overlapping character chunks.

    Args:
        text: full extracted text to split.
        source_file: filename for provenance tracking on each chunk.
        chunk_size: max characters per chunk.
        chunk_overlap: characters repeated between consecutive chunks.

    Raises:
        ValueError: if chunk_overlap >= chunk_size (would loop forever
            or produce meaningless chunks).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk_str = text[start:end].strip()

        if chunk_str:
            chunks.append(
                Chunk(text=chunk_str, chunk_index=index, source_file=source_file)
            )
            index += 1

        # Advance by (chunk_size - overlap) so the next chunk re-includes
        # the tail of this one.
        start += chunk_size - chunk_overlap

    return chunks


def ingest_pdf(
    pdf_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Full ingestion pipeline: PDF file -> list of text chunks."""
    text = extract_text(pdf_path)
    return chunk_text(text, source_file=pdf_path.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)