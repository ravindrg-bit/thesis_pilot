"""
base.py — shared, engine-AGNOSTIC adapter plumbing.

Engine-specific code (the API call and the citation parsing) lives in each engine's
own adapter. The bits identical for every engine -- writing a BronzeRecord to disk
under the standard naming, and parsing a URL's domain -- live here so they are
defined once (V3 sec.7.3: adapters hold only what is truly engine-specific).
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import thesis_config as cfg
from src.schema import BronzeRecord


def write_bronze(rec: BronzeRecord) -> Path:
    """Write one verbatim BronzeRecord as JSON. One file per (engine, query, run);
    the stable name makes re-writes idempotent."""
    cfg.BRONZE.mkdir(parents=True, exist_ok=True)
    path = cfg.BRONZE / f"{rec.engine}__{rec.query_id}__r{rec.run_index}.json"
    path.write_text(json.dumps(rec.__dict__, indent=2, default=str))
    return path


def domain_of(url):
    """Best-effort domain (netloc) for grouping/counting later."""
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None
