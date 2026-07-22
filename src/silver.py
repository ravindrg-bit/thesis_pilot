"""
silver.py - build the SILVER layer from BRONZE (V3 sec.7.3).

Pure transform: every BronzeRecord is run through its engine's adapter.normalise()
to produce the canonical shape, then flattened into two analysis tables:
  - responses : one row per (engine, query, run) cell  -> feeds SIS, CV
  - citations : one row per cited source (long)         -> feeds PAWC, AIS

NO API keys or network needed -- normalise() reads only the captured bronze payload,
so silver can be rebuilt from the immutable bronze at any time (medallion reproducibility).
"""

import json
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd

import thesis_config as cfg
from src.schema import BronzeRecord
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.claude_adapter import ClaudeAdapter
from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.perplexity_adapter import PerplexityAdapter
from src.adapters.kimi_adapter import KimiAdapter
from src.adapters.mistral_adapter import MistralAdapter

# engine key (as stored in bronze) -> adapter class
_ADAPTER_CLASSES = {
    "chatgpt": OpenAIAdapter,
    "claude": ClaudeAdapter,
    "gemini": GeminiAdapter,
    "perplexity": PerplexityAdapter,
    "kimi": KimiAdapter,
    "mistral": MistralAdapter,
}

# tracking params to strip during URL canonicalisation
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = {
    "srsltid", "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid",
    "mc_eid", "igshid", "ref", "ref_src", "yclid", "_hsenc", "_hsmi", "spm",
}


def canonicalise_url(url):
    """Lowercase host, drop fragment + tracking params, trim trailing slash.
    Best-effort; returns the original string on any parse failure."""
    if not url:
        return url
    try:
        p = urlsplit(url.strip())
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not (k.lower().startswith(_TRACKING_PREFIXES) or k.lower() in _TRACKING_KEYS)]
        path = p.path.rstrip("/") if p.path not in ("", "/") else p.path
        return urlunsplit((scheme, netloc, path, urlencode(q), ""))
    except Exception:
        return url


def reconstruct_bronze(d: dict) -> BronzeRecord:
    """Rebuild a BronzeRecord from a loaded bronze JSON dict."""
    return BronzeRecord(
        query_id=d["query_id"],
        engine=d["engine"],
        run_index=d["run_index"],
        model_requested=d.get("model_requested", ""),
        model_served=d.get("model_served", ""),
        timestamp_utc=d.get("timestamp_utc", ""),
        query_text=d.get("query_text", ""),
        request_params=d.get("request_params", {}) or {},
        raw_response=d.get("raw_response", {}) or {},
        answer_text=d.get("answer_text", "") or "",
        usage=d.get("usage", {}) or {},
        run_profile=d.get("run_profile", cfg.RUN_PROFILE),
    )


def normalise_record(bronze: BronzeRecord):
    """Run a bronze record through its engine's adapter.normalise().
    Instantiates the adapter WITHOUT __init__ (no API client/key needed), since
    normalise() is a pure read of bronze. carries run_profile through from bronze."""
    cls = _ADAPTER_CLASSES.get(bronze.engine)
    if cls is None:
        raise KeyError(f"no adapter for engine {bronze.engine!r}")
    inst = cls.__new__(cls)                          # skip __init__ -> no key required
    inst.spec = cfg.ENGINES.get(bronze.engine, {})   # safety belt if normalise reads self.spec
    canonical = inst.normalise(bronze)
    canonical.run_profile = bronze.run_profile       # stamp matches the source bronze
    return canonical


def load_all_bronze() -> list:
    """Read every bronze JSON in cfg.BRONZE into BronzeRecords."""
    out = []
    for f in sorted(cfg.BRONZE.glob("*.json")):
        out.append(reconstruct_bronze(json.loads(f.read_text())))
    return out


def assert_single_profile(records) -> set:
    """Refuse to build silver across mixed run_profiles (hygiene guard)."""
    profiles = {r.run_profile for r in records}
    if len(profiles) > 1:
        raise ValueError(
            f"Refusing to build silver across mixed run_profiles: {profiles}. "
            f"Bronze must come from a single profile."
        )
    if profiles and cfg.RUN_PROFILE not in profiles:
        print(f"[WARN] bronze profile {profiles} != current RUN_PROFILE '{cfg.RUN_PROFILE}'")
    return profiles


def build_responses_table(canonicals) -> pd.DataFrame:
    rows = []
    for c in canonicals:
        domains = {s.domain for s in c.cited_sources if s.domain}
        provs = {s.provenance for s in c.cited_sources}
        txt = c.answer_text or ""
        rows.append({
            "query_id": c.query_id,
            "engine": c.engine,
            "run_index": c.run_index,
            "model_served": c.model_served,
            "timestamp_utc": c.timestamp_utc,
            "run_profile": c.run_profile,
            "answer_text": txt,
            "answer_char_len": len(txt),
            "answer_word_count": len(txt.split()),
            "n_cited_sources": len(c.cited_sources),
            "n_unique_domains": len(domains),
            "has_citations": len(c.cited_sources) > 0,
            "citation_provenance": ";".join(sorted(provs)) if provs else "",
        })
    return pd.DataFrame(rows)


def build_citations_table(canonicals) -> pd.DataFrame:
    rows = []
    for c in canonicals:
        for s in c.cited_sources:
            rows.append({
                "query_id": c.query_id,
                "engine": c.engine,
                "run_index": c.run_index,
                "position": s.position,
                "url": s.url,
                "url_canonical": canonicalise_url(s.url),
                "domain": s.domain,
                "title": s.title,
                "provenance": s.provenance,
                # populated in the AIS batch: resolved_status, support_label
            })
    return pd.DataFrame(rows)


def build_silver(write: bool = True):
    """Full bronze -> silver build. Returns (responses_df, citations_df, report)."""
    records = load_all_bronze()
    profiles = assert_single_profile(records)
    canonicals, failures = [], []
    for r in records:
        try:
            canonicals.append(normalise_record(r))
        except Exception as e:
            failures.append({"engine": r.engine, "query_id": r.query_id,
                             "run_index": r.run_index, "error": f"{type(e).__name__}: {e}"})
    responses = build_responses_table(canonicals)
    citations = build_citations_table(canonicals)
    if write:
        cfg.SILVER.mkdir(parents=True, exist_ok=True)
        responses.to_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet", index=False)
        citations.to_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet", index=False)
    report = {
        "bronze_records": len(records),
        "normalised": len(canonicals),
        "failures": failures,
        "profiles": sorted(profiles),
        "n_responses_rows": len(responses),
        "n_citations_rows": len(citations),
    }
    return responses, citations, report
