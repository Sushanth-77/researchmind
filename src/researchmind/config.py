"""
Central configuration module.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise EnvironmentError(
            f"Missing or placeholder value for required env var: {name}. "
            f"Check your .env file against .env.example."
        )
    return value


def _optional_env(name: str) -> Optional[str]:
    """Fetch an optional env var, treating unset or placeholder values as None."""
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        return None
    return value


GROQ_API_KEY: str = _require_env("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Optional: API key required on the FastAPI backend (sent as the
# X-API-Key header). Leave unset to run the API unauthenticated — fine
# for local-only use, NOT recommended if the backend is ever reachable
# beyond your own machine.
API_KEY: Optional[str] = _optional_env("API_KEY")

if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGCHAIN_API_KEY").startswith("your_"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "researchmind"))

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"