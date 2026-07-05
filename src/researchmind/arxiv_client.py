"""
Thin client for arXiv's free, public Atom API (export.arxiv.org).

No API key or paid tier involved — arXiv's search and PDF endpoints are
open and unauthenticated by design. This lets ResearchMind fetch new
papers directly by arXiv ID or search query, rather than requiring the
user to manually find and download PDFs before ingestion.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class ArxivSearchResult:
    """One paper returned from an arXiv search."""

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str


def search_arxiv(query: str, max_results: int = 5) -> list[ArxivSearchResult]:
    """
    Search arXiv by keyword and return basic metadata for each result.

    Raises:
        ValueError: if query is empty.
        RuntimeError: if the arXiv API request fails.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"arXiv API request failed: {exc}") from exc

    root = ET.fromstring(response.content)
    results = []

    for entry in root.findall("atom:entry", ATOM_NS):
        id_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url

        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS).strip()
        authors = [
            author.findtext("atom:name", default="", namespaces=ATOM_NS)
            for author in entry.findall("atom:author", ATOM_NS)
        ]

        results.append(
            ArxivSearchResult(arxiv_id=arxiv_id, title=title, authors=authors, summary=summary)
        )

    return results


def fetch_arxiv_pdf(arxiv_id: str, dest_path: Path) -> None:
    """
    Download a paper's PDF by arXiv ID (e.g. "2510.04950v1") to dest_path.

    Raises:
        RuntimeError: if the download fails or the ID doesn't exist.
    """
    url = ARXIV_PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to download PDF for arXiv ID '{arxiv_id}': {exc}") from exc

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(response.content)