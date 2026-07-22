"""
env.py — load secrets from .env once, at program entry.

Entry-point SCRIPTS call load_env() on their first line. Libraries (adapters, schema)
never load env themselves; they assume the process environment is already populated.
This keeps the .env dependency in one place and out of the engine-specific code.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root


def load_env() -> None:
    """Load KEY=VALUE pairs from the project-root .env into os.environ if present.
    No-op if python-dotenv or the file is missing (env may already be set externally)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:
        pass
