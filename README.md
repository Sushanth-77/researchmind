# ResearchMind

A multi-agent research-paper analysis platform built entirely on free-tier
tools: Groq (LLM inference), local sentence-transformers (embeddings),
Chroma (vector store), Neo4j (knowledge graph), and LangGraph (agent
orchestration).

Ingest PDFs, ask grounded questions, extract structured metadata, compare
papers, get cross-paper analysis and literature-review summaries, query a
knowledge graph of papers/authors/methods, and drive it all from a
Streamlit UI, a FastAPI backend, or an MCP server usable directly from
Claude Desktop.

## Features

- **PDF ingestion & chunking** — overlapping character chunks via `pypdf`
- **Grounded RAG Q&A** — hybrid BM25 + embedding retrieval, explicit
  refusal when context is insufficient, chunk-level citation
- **Structured metadata extraction** — Pydantic-validated title, authors,
  methodology, dataset, evaluation metrics, with retry/repair on
  malformed LLM output
- **Multi-paper comparison** — 2 or more papers, per-paper filtered
  retrieval, cross-paper attribution
- **Multi-agent orchestration** — Planner, Extractor, QA, Analysis,
  Survey, and Knowledge Graph agents coordinated via LangGraph, with
  isolated per-agent state namespaces
- **Multi-turn conversation memory** — follow-up questions resolved
  against chat history before routing
- **Cross-paper analysis & survey generation** — trend/gap detection and
  literature-review synthesis, with explicit guardrails against
  manufacturing false connections between unrelated papers
- **Neo4j knowledge graph** — Paper/Author/Methodology/Dataset/Metric
  schema; shared-author and metric-based structural queries
- **Resilience** — Groq call retry/backoff with structured JSONL logging
  (tokens, latency, retry counts) for every LLM call
- **FastAPI backend** — background ingestion, uniform error responses,
  full REST API
- **Streamlit frontend** — paper upload, chat, and multi-paper comparison
  views
- **MCP server** — exposes arXiv search/ingest and paper Q&A as tools
  usable directly from Claude Desktop
- **Docker Compose** — full stack (backend, frontend, Neo4j) containerized
- **Testing** — pytest unit suite (corpus-independent) + regression eval
  harness (corpus-aware, built from real verified query/answer pairs)

## Architecture
PDF → ingestion → Chroma (vectors) ─┐
├─→ LangGraph orchestration ─→ Groq
Neo4j (knowledge graph) ─────────────┘         │
▼
FastAPI backend ←→ Streamlit frontend
│
MCP server (Claude Desktop)

## Stack

| Layer | Tool |
|---|---|
| LLM inference | Groq — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers` (local, CPU) |
| Vector store | Chroma (local, persistent, cosine + BM25 hybrid) |
| Knowledge graph | Neo4j Community Edition |
| Orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Protocol integration | MCP (Model Context Protocol) |
| Observability | Structured JSONL logging + optional LangSmith tracing |
| Containerization | Docker Compose (CPU-only torch) |
| Testing | pytest + corpus-aware regression eval harness |

## Setup

### Prerequisites
- Python 3.10+
- A free Groq API key: https://console.groq.com/keys
- (Optional, for knowledge graph) Neo4j Desktop: https://neo4j.com/download/
- (Optional, for containerized deployment) Docker Desktop

### Local development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env
# edit .env with your real GROQ_API_KEY (and NEO4J_PASSWORD if using the graph)
```

Verify the base setup:
```powershell
python scripts/test_groq.py
python scripts/test_embeddings.py
```

Ingest a paper and try the pipeline end to end:
```powershell
python scripts/test_vectorstore.py your_paper.pdf
python scripts/run_graph.py "What does this paper find?"
```

Run the backend and frontend (two terminals):
```powershell
uvicorn researchmind.api.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

### Docker (full stack)

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Backend: http://localhost:8000/health · Frontend: http://localhost:8501

See [DEPLOYMENT.md](DEPLOYMENT.md) for full Docker instructions, known
limitations, and public-hosting notes.

### Testing

```powershell
# Unit tests — no external services required
pytest tests/ -v

# Regression eval harness — requires your ingested corpus
python scripts/run_eval.py
```

### MCP server (Claude Desktop integration)

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "researchmind": {
      "command": "C:\\path\\to\\researchmind\\.venv\\Scripts\\python.exe",
      "args": ["-m", "researchmind.mcp_server"]
    }
  }
}
```

## Project Structure
researchmind/
├── src/researchmind/
│   ├── agents/              # Planner, Extractor, QA, Analysis, Survey, Graph agents
│   ├── api/                 # FastAPI backend
│   ├── frontend/            # Streamlit API client
│   ├── ingestion.py         # PDF parsing & chunking
│   ├── embeddings.py        # Local embedding model
│   ├── vectorstore.py       # Chroma wrapper
│   ├── retrieval.py         # Hybrid BM25 + embedding retrieval
│   ├── qa.py                # Grounded generation
│   ├── comparison.py        # Multi-paper comparison
│   ├── metadata_extraction.py
│   ├── analysis.py          # Cross-paper trend/gap detection
│   ├── survey.py            # Literature review synthesis
│   ├── knowledge_graph.py   # Neo4j integration
│   ├── conversation.py      # Multi-turn query resolution
│   ├── observability.py     # Groq call logging + retry/backoff
│   ├── graph.py             # LangGraph state machine
│   ├── graph_nodes.py / graph_state.py
│   ├── mcp_server.py        # MCP tool/resource exposure
│   └── arxiv_client.py      # arXiv search/download
├── frontend/streamlit_app.py
├── tests/                   # pytest unit suite
├── eval/test_cases.json     # regression eval cases
├── scripts/                 # CLI entry points for every capability
├── Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
└── DEPLOYMENT.md