"""
Neo4j knowledge graph integration.

Builds a Paper/Author/Methodology/Dataset/Metric graph from the same
PaperMetadata Phase 2 already produces — populating the graph requires
no new LLM calls, only a new way of storing and querying existing facts.

Requires Neo4j 5.x (constraint syntax below is version-specific) running
locally via Neo4j Desktop. Connection is opt-in: if NEO4J_PASSWORD is
unset, functions here raise a clear error only when called, so the rest
of the app is unaffected if Neo4j isn't running.
"""

import os

from neo4j import Driver, GraphDatabase

from researchmind.metadata_extraction import extract_metadata
from researchmind.vectorstore import list_source_files

_driver: Driver | None = None


def _get_driver() -> Driver:
    """Lazily create and cache the Neo4j driver for this process."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password or password.startswith("your_"):
            raise EnvironmentError(
                "NEO4J_PASSWORD is not set. Add it to .env to use the knowledge "
                "graph (see .env.example). Requires Neo4j Desktop running locally."
            )

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
        except Exception as exc:
            raise RuntimeError(f"Could not connect to Neo4j at {uri}: {exc}") from exc

        _driver = driver

    return _driver


def close_driver() -> None:
    """Close the cached driver, if one exists. Call on app shutdown."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def ensure_constraints() -> None:
    """
    Create uniqueness constraints so repeated population upserts (via
    MERGE) instead of duplicating nodes for the same paper/author/etc.
    """
    driver = _get_driver()
    constraints = [
        "CREATE CONSTRAINT paper_source IF NOT EXISTS FOR (p:Paper) REQUIRE p.source_file IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT methodology_desc IF NOT EXISTS FOR (m:Methodology) REQUIRE m.description IS UNIQUE",
        "CREATE CONSTRAINT dataset_desc IF NOT EXISTS FOR (d:Dataset) REQUIRE d.description IS UNIQUE",
        "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (me:Metric) REQUIRE me.name IS UNIQUE",
    ]
    with driver.session() as session:
        for stmt in constraints:
            session.run(stmt)


def populate_graph_for_paper(source_file: str) -> dict:
    """
    Extract metadata (reusing Phase 2's extract_metadata) and upsert it
    into the graph as a Paper node connected to Author/Methodology/
    Dataset/Metric nodes.

    Raises:
        ValueError: propagated if source_file has no chunks in the vector store.
        RuntimeError: propagated if Groq or Neo4j calls fail.
        EnvironmentError: propagated if Neo4j credentials are unset.
    """
    metadata = extract_metadata(source_file)
    driver = _get_driver()

    query = """
    MERGE (p:Paper {source_file: $source_file})
    SET p.title = $title
    WITH p
    UNWIND $authors AS author_name
    MERGE (a:Author {name: author_name})
    MERGE (a)-[:AUTHORED]->(p)
    WITH p
    FOREACH (_ IN CASE WHEN $methodology <> 'Not specified' THEN [1] ELSE [] END |
        MERGE (m:Methodology {description: $methodology})
        MERGE (p)-[:USES_METHODOLOGY]->(m)
    )
    FOREACH (_ IN CASE WHEN $dataset <> 'Not specified' THEN [1] ELSE [] END |
        MERGE (d:Dataset {description: $dataset})
        MERGE (p)-[:USES_DATASET]->(d)
    )
    WITH p
    UNWIND $metrics AS metric_name
    MERGE (me:Metric {name: metric_name})
    MERGE (p)-[:EVALUATED_WITH]->(me)
    """

    with driver.session() as session:
        session.run(
            query,
            source_file=source_file,
            title=metadata.title,
            authors=metadata.authors or ["Unknown"],
            methodology=metadata.methodology,
            dataset=metadata.dataset,
            metrics=metadata.evaluation_metrics or [],
        )

    return {
        "source_file": source_file,
        "title": metadata.title,
        "authors": metadata.authors,
        "methodology": metadata.methodology,
        "dataset": metadata.dataset,
        "evaluation_metrics": metadata.evaluation_metrics,
    }


def populate_graph_for_all_papers() -> list[dict]:
    """Populate the graph for every paper currently in the vector store."""
    ensure_constraints()
    return [populate_graph_for_paper(sf) for sf in list_source_files()]


def find_shared_authors() -> list[dict]:
    """Find authors who appear on more than one ingested paper."""
    driver = _get_driver()
    query = """
    MATCH (a:Author)-[:AUTHORED]->(p:Paper)
    WITH a, collect(p.source_file) AS papers
    WHERE size(papers) > 1
    RETURN a.name AS author, papers
    """
    with driver.session() as session:
        result = session.run(query)
        return [{"author": r["author"], "papers": r["papers"]} for r in result]


def find_papers_by_metric(metric_keyword: str) -> list[str]:
    """Find papers evaluated using a metric matching metric_keyword (case-insensitive)."""
    driver = _get_driver()
    query = """
    MATCH (p:Paper)-[:EVALUATED_WITH]->(m:Metric)
    WHERE toLower(m.name) CONTAINS toLower($metric_keyword)
    RETURN DISTINCT p.source_file AS source_file
    """
    with driver.session() as session:
        result = session.run(query, metric_keyword=metric_keyword)
        return [r["source_file"] for r in result]


def get_graph_summary() -> dict:
    """Return a count of each node type currently in the graph."""
    driver = _get_driver()
    label_to_key = {
        "Paper": "papers",
        "Author": "authors",
        "Methodology": "methodologies",
        "Dataset": "datasets",
        "Metric": "metrics",
    }
    counts = {}
    with driver.session() as session:
        for label, key in label_to_key.items():
            record = session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()
            counts[key] = record["count"]
    return counts