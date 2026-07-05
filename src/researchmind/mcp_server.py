"""
MCP server exposing ResearchMind's capabilities to any MCP-compatible
client (e.g. Claude Desktop), over stdio.

Tools vs. resource: tools are actions an MCP client's model decides to
invoke; the resource is context a host can attach directly without a
tool call, which fits "list of currently ingested papers" well — the
filesystem-browsing capability, scoped to this project's data/papers
directory rather than the whole OS filesystem.

Every function here is a thin wrapper around existing pipeline code
(ingestion, vectorstore, graph, knowledge_graph) — no logic is
duplicated, only exposed through a new transport.
"""

from mcp.server.fastmcp import FastMCP

from researchmind.agents.graph_agent import query_graph as _query_graph
from researchmind.arxiv_client import fetch_arxiv_pdf, search_arxiv
from researchmind.config import DATA_DIR
from researchmind.graph import run_query
from researchmind.ingestion import ingest_pdf
from researchmind.knowledge_graph import populate_graph_for_all_papers
from researchmind.vectorstore import list_source_files, store_chunks

mcp = FastMCP("ResearchMind")


@mcp.tool()
def list_ingested_papers() -> list[str]:
    """List the filenames of papers currently stored in the vector store."""
    return list_source_files()


@mcp.tool()
def search_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv for papers matching a keyword query.

    Returns a list of {arxiv_id, title, authors, summary} for each match,
    so a caller can pick one to ingest via ingest_arxiv_paper.
    """
    results = search_arxiv(query, max_results=max_results)
    return [
        {"arxiv_id": r.arxiv_id, "title": r.title, "authors": r.authors, "summary": r.summary}
        for r in results
    ]


@mcp.tool()
def ingest_arxiv_paper(arxiv_id: str) -> dict:
    """
    Download a paper from arXiv by ID (e.g. "2510.04950v1") and ingest
    it into the vector store, ready for querying.
    """
    pdf_path = DATA_DIR / "papers" / f"{arxiv_id}.pdf"
    fetch_arxiv_pdf(arxiv_id, pdf_path)

    chunks = ingest_pdf(pdf_path)
    store_chunks(chunks)

    return {"arxiv_id": arxiv_id, "filename": pdf_path.name, "chunks_created": len(chunks)}


@mcp.tool()
def ask_researchmind(query: str) -> dict:
    """
    Ask a question about the ingested papers. Routes automatically to
    single-paper Q&A, metadata extraction, comparison, analysis, survey
    generation, or knowledge-graph lookup based on the question.
    """
    result = run_query(query)
    decision = result["planner"]["decision"]

    return {
        "intent": decision.intent,
        "source_files": decision.source_files,
        "answer": result["final_answer"],
    }


@mcp.tool()
def populate_knowledge_graph() -> list[dict]:
    """Populate the Neo4j knowledge graph from all currently ingested papers."""
    return populate_graph_for_all_papers()


@mcp.tool()
def query_knowledge_graph(query: str) -> str:
    """Query the knowledge graph for structural facts (shared authors, papers by metric, summary)."""
    msg = _query_graph(query)
    return msg.context["answer"]


@mcp.resource("papers://ingested")
def ingested_papers_resource() -> str:
    """Browsable resource listing papers currently ingested (the 'filesystem' view)."""
    papers = list_source_files()
    if not papers:
        return "No papers ingested yet."
    return "\n".join(f"- {p}" for p in papers)


if __name__ == "__main__":
    mcp.run()