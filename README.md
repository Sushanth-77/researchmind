# ResearchMind

A multi-agent research-paper analysis platform built entirely on free-tier
tools: Groq (LLM inference), local sentence-transformers (embeddings),
ChromaDB (vector store), Neo4j (knowledge graph), and LangGraph (agent
orchestration).

Ingest PDFs, ask grounded questions, extract structured metadata, compare
papers, generate literature-review summaries, query a knowledge graph of
papers/authors/methods, and drive it all from a Streamlit UI, a FastAPI
backend, or an MCP server.

[![Tests](https://github.com/Sushanth-77/researchmind/actions/workflows/tests.yml/badge.svg)](https://github.com/Sushanth-77/researchmind/actions/workflows/tests.yml)

---

## Resume

> **ResearchMind — Multi-Agent Research Paper Analysis Platform**
>
> – Architected a multi-agent orchestration system using **LangGraph** with six specialised agents (Planner, Extractor, QA, Analysis, Survey, Knowledge Graph), implementing isolated per-agent state namespaces, retry-and-repair loops for structured LLM output, and multi-turn conversation memory for follow-up resolution.
>
> – Built a **hybrid BM25 + embedding retrieval** pipeline (ChromaDB + sentence-transformers, local CPU) with grounded RAG Q&A, explicit refusal on insufficient context, Pydantic-validated metadata extraction, and multi-paper comparison with per-paper filtered retrieval and cross-paper attribution.
>
> – Engineered a production-grade **FastAPI backend** with background PDF ingestion, persistent SQLite-backed task/cache store, duplicate upload detection (HTTP 409), 50 MB size guard (HTTP 413), and a `DELETE /papers/{filename}` endpoint; decoupled from a Streamlit frontend and exposed as an **MCP server** compatible with Claude Desktop.
>
> – Implemented an **observability and resilience layer** — structured JSONL logging of every Groq call (tokens, latency, retry count), exponential backoff with per-exception retry classification, process-level client singletons for Groq and ChromaDB, and a corpus-aware regression eval harness built from hand-verified query/answer pairs.

---

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
  literature-review synthesis
- **Neo4j knowledge graph** — Paper/Author/Methodology/Dataset/Metric
  schema; shared-author and metric-based structural queries
- **Resilience** — Groq call retry/backoff with structured JSONL logging
  (tokens, latency, retry counts) for every LLM call
- **FastAPI backend** — background ingestion, duplicate/size guards,
  paper deletion, persistent SQLite task store, uniform REST API
- **Streamlit frontend** — paper upload with ingestion status polling,
  chat with history, multi-paper comparison, per-paper delete button
- **MCP server** — exposes arXiv search/ingest and paper Q&A as MCP tools
- **Docker Compose** — full stack (backend, frontend, Neo4j) containerized
- **Testing** — pytest unit suite (corpus-independent) + corpus-aware
  regression eval harness

## Architecture

```
PDF → ingestion → ChromaDB (vectors) ─┐
                                       ├─→ LangGraph orchestration ─→ Groq
Neo4j (knowledge graph) ───────────────┘         │
                                                  ▼
                    FastAPI backend ←→ Streamlit frontend
                          │
              MCP server (stdio; usable from Claude Desktop
              or any MCP-compatible host)
```

## Stack

| Layer | Tool |
|---|---|
| LLM inference | Groq — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers` (local, CPU) |
| Vector store | ChromaDB (local, persistent, cosine + BM25 hybrid) |
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

Note: don't run a local `streamlit run` / `uvicorn` at the same time as
the Docker stack below — both default to the same ports and will conflict.

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

### MCP server

Tested against a standalone MCP client (`scripts/test_mcp_client.py`) —
lists tools, searches arXiv, and reads the ingested-papers resource over
stdio. To try it with Claude Desktop, add to
`%APPDATA%\Claude\claude_desktop_config.json`:
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

```
researchmind/
├── src/researchmind/
│   ├── agents/               # Planner, Extractor, QA, Analysis, Survey, Graph agents
│   ├── api/                  # FastAPI backend (main, schemas, task_store, ingestion_service)
│   ├── frontend/             # Streamlit HTTP client
│   ├── ingestion.py          # PDF parsing & chunking
│   ├── embeddings.py         # Local embedding model (singleton)
│   ├── vectorstore.py        # ChromaDB wrapper (singleton client + collection)
│   ├── retrieval.py          # Hybrid BM25 + embedding retrieval
│   ├── qa.py                 # Grounded generation
│   ├── comparison.py         # Multi-paper comparison
│   ├── metadata_extraction.py
│   ├── analysis.py           # Cross-paper trend/gap detection
│   ├── survey.py             # Literature review synthesis
│   ├── knowledge_graph.py    # Neo4j integration
│   ├── conversation.py       # Multi-turn query resolution
│   ├── observability.py      # Groq call logging + retry/backoff (singleton client)
│   ├── kv_store.py           # Persistent SQLite-backed key-value store
│   ├── utils.py              # Shared utilities (strip_code_fences)
│   ├── schemas.py            # Core Pydantic schemas (PaperMetadata, PlannerDecision)
│   ├── graph.py              # LangGraph state machine
│   ├── graph_nodes.py        # Agent node functions + intent routing
│   ├── graph_state.py        # TypedDict state definition
│   ├── mcp_server.py         # MCP tool/resource exposure
│   └── arxiv_client.py       # arXiv search/download
├── frontend/streamlit_app.py
├── tests/                    # pytest unit suite
├── eval/test_cases.json      # Regression eval cases
├── scripts/                  # CLI entry points for every capability
├── Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
└── DEPLOYMENT.md
```