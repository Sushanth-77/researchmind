# ResearchMind

A multi-agent research-paper analysis platform built entirely on free-tier
tools: Groq (LLM inference), local sentence-transformers (embeddings), and
Chroma (vector store).

## Setup

1. Clone the repo and enter the directory.

2. Create and activate a virtual environment:
```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```

3. Install the package in editable mode (this also installs dependencies
   from `pyproject.toml`):
```powershell
   pip install -e .
```

4. Copy `.env.example` to `.env` and fill in your real Groq API key
   (free key: https://console.groq.com/keys):
```powershell
   copy .env.example .env
```

5. Verify the setup:
```powershell
   python scripts/test_groq.py
   python scripts/test_embeddings.py
```

## Project Structure

- `src/researchmind/` — main package (importable as `researchmind`)
- `tests/` — pytest test suite
- `scripts/` — one-off manual smoke tests, not part of the package
- `.env` — local secrets (never committed)
- `.env.example` — template for required environment variables

## Stack

| Layer      | Tool                                  |
|------------|----------------------------------------|
| LLM        | Groq — `llama-3.3-70b-versatile`      |
| Embeddings | `sentence-transformers` (local)       |
| Vector DB  | Chroma (local, persistent, cosine)    |
| PDF parse  | `pypdf`                                |
| Orchestration | Plain Python → LangGraph (Phase 5) |