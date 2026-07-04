"""
Central configuration module.

Loads environment variables once and exposes them as typed constants
so no other module has to touch os.environ or dotenv directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    """Fetch a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise EnvironmentError(
            f"Missing or placeholder value for required env var: {name}. "
            f"Check your .env file against .env.example."
        )
    return value


GROQ_API_KEY: str = _require_env("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Optional: LangSmith tracing (free tier). If LANGCHAIN_API_KEY is unset,
# LangGraph simply runs untraced — this is opt-in, not required.
if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGCHAIN_API_KEY").startswith("your_"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "researchmind"))

# Standard paths used across phases
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"