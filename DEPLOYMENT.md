# Deployment Notes

## Running the full stack locally via Docker

```powershell
docker compose build
docker compose up -d
docker compose ps      # wait until neo4j shows "healthy"
```

Backend: http://localhost:8000/health · Frontend: http://localhost:8501

The backend and frontend containers reuse your existing local `chroma_db/`
and `data/` folders (bind-mounted, not named volumes), so previously
ingested papers are available immediately. **Neo4j is the exception**: the
containerized Neo4j starts with an empty graph, separate from any local
Neo4j Desktop instance. Re-populate it once the stack is up:

```powershell
docker compose exec backend python scripts/populate_graph.py
```

Stop everything with `docker compose down` (add `-v` to also delete the
Neo4j data volume and start fully fresh next time).

## Known limitations before any real/public deployment

These are honest gaps, not oversights — each was an explicit scope choice
earlier in the build, made for good reasons at the time, but worth revisiting
before this runs anywhere beyond your own machine:

- **In-memory ingestion task store** (Phase 8): task status lives in a
  Python dict, lost on restart and invisible across multiple backend
  instances. Fine for one person, one process. A real multi-user or
  multi-instance deployment needs this backed by Redis or a database table.
- **No authentication on the FastAPI backend**: anyone who can reach port
  8000 can ingest papers and run queries. Add an API key or proper auth
  before exposing this beyond your own machine/network.
- **Groq free-tier rate limits**: the retry/backoff logic added this session
  smooths over transient limits but doesn't remove the underlying ceiling.
  Real multi-user traffic would need a paid tier or a fallback provider.
- **Chroma is an embedded, single-process file store**: great for one
  instance, not built for concurrent multi-instance writes. A scaled
  deployment would want a hosted vector database instead.
- **Neo4j Community Edition has no clustering/HA**: fine for personal use;
  a real deployment might use Neo4j Aura's free managed tier instead of
  self-hosting.

## If you want to actually host this somewhere cheaply

- **Backend + frontend containers**: Render, Railway, or Fly.io all have
  free/hobby tiers that run Docker images directly from this repo's
  Dockerfiles.
- **Neo4j**: Neo4j Aura Free (managed, hosted) avoids running your own
  Neo4j container in production.
- **Secrets**: never bake `.env` into a Docker image (it's already in
  `.dockerignore` for this reason) — set `GROQ_API_KEY`, `NEO4J_PASSWORD`,
  etc. via the hosting platform's environment variable/secrets manager.
- **HTTPS**: put a reverse proxy (Caddy, or the hosting platform's built-in
  TLS) in front of both services if exposed to the public internet.

## MCP server

The MCP server (Phase 10) runs over stdio and is spawned locally by an MCP
host (e.g. Claude Desktop) — it isn't part of the Docker Compose stack and
doesn't need to be, since it's inherently a local, single-user integration
point rather than a network service.