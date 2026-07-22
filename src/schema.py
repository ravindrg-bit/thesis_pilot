"""
schema.py — the shape of a captured response, defined once.

Two record types matching the medallion layers:
  - BronzeRecord:    raw, verbatim capture (one per query x engine x repeat).
  - CanonicalRecord: engine-agnostic record produced after a per-engine adapter
                     normalises that engine's native citation format (V3 sec.7.3).
                     ALL metric code reads only CanonicalRecords, so it runs
                     identically across engines.

Dataclasses (not a database) keep this test-grade and inspectable.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


def _current_run_profile() -> str:
    import thesis_config as cfg          # lazy import avoids any circular dependency
    return cfg.RUN_PROFILE


@dataclass(frozen=True)
class CitedSource:
    """One source the engine cited, in the order it appeared.
    `provenance` records how the citation was obtained: 'provider_certified' (read from a
    structured tool output) or 'self_reported_prompt_elicited' (parsed from answer text;
    Kimi only). Default keeps the five structured engines unchanged."""
    position: int                 # 1-based order in the answer
    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    provenance: str = "provider_certified"


@dataclass(frozen=True)
class ClaimCitationPair:
    """A claim paired with the source position(s) cited for it.
    Basis for the AIS verifiability rate (V3 RQ4; Rashkin et al. 2023)."""
    claim_text: str
    cited_positions: tuple = ()


@dataclass
class BronzeRecord:
    """Raw, immutable capture. Nothing here is cleaned or interpreted."""
    query_id: str
    engine: str                   # short key from thesis_config.ENGINES
    run_index: int                # which of the k repeats (1-based)
    model_requested: str          # the model id we asked for
    model_served: str             # EXACT id the API returned (V3 sec.7.4) - may differ
    timestamp_utc: str            # ISO 8601, UTC
    query_text: str
    request_params: dict          # temperature, seed, etc. actually sent
    raw_response: dict            # full provider payload, verbatim
    answer_text: str = ""         # best-effort top-level text (full parse is silver)
    usage: dict = field(default_factory=dict)  # token counts if provided
    run_profile: str = field(default_factory=_current_run_profile)


@dataclass
class CanonicalRecord:
    """Engine-agnostic record from a per-engine adapter (V3 sec.7.3).
    The single shape every metric is computed from."""
    query_id: str
    engine: str
    run_index: int
    model_served: str
    timestamp_utc: str
    answer_text: str
    cited_sources: list = field(default_factory=list)        # list[CitedSource]
    claim_citation_pairs: list = field(default_factory=list) # list[ClaimCitationPair]
    run_profile: str = field(default_factory=_current_run_profile)
