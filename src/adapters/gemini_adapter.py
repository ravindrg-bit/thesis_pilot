"""
gemini_adapter.py — the ONLY Gemini-specific code (V3 sec.7.3).

Gemini specifics (verified 30 May 2026):
  - gemini-3.5-flash via the google-genai SDK, generate_content surface.
  - Citations require the google_search grounding tool. The model DECIDES whether to
    search (like OpenAI/Claude; unlike Perplexity, which always grounds).
  - Sources: candidates[0].grounding_metadata.grounding_chunks, each {web: {uri, title}}:
        uri   = a vertexaisearch.cloud.google.com REDIRECT (not the publisher URL)
        title = the publisher DOMAIN (e.g. "uefa.com"), not a page title
    We put title into the canonical 'domain' field; resolving the redirect to the true
    URL is a documented scale-up step (like utm-stripping), not done in the pilot.
  - SDK dumps may use snake_case OR camelCase keys; normalise() tolerates both.
"""

from datetime import datetime, timezone

from google import genai
from google.genai import types

import os

import thesis_config as cfg
from src.adapters.base import write_bronze
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "gemini"


def _first(d, *keys):
    """First present key's value (tolerates snake_case/camelCase serialisation)."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


class GeminiAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        if not (os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()):
            raise RuntimeError(
                "GEMINI_API_KEY missing -- call src.env.load_env() in your script."
            )
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def fetch(self, query_id, query_text, run_index) -> BronzeRecord:
        # google_search grounding ON; the model decides whether to actually search.
        # thinking_level deferred (SDK field still settling for Gemini 3) -> model default.
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        request_params = {"model": self.spec["model"], "tools": ["google_search"]}

        response = self.client.models.generate_content(
            model=self.spec["model"],
            contents=query_text,
            config=config,
        )

        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = response.model_dump()

        rec = BronzeRecord(
            query_id=query_id,
            engine=ENGINE_KEY,
            run_index=run_index,
            model_requested=self.spec["model"],
            model_served=getattr(response, "model_version", "") or "",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            query_text=query_text,
            request_params=request_params,
            raw_response=raw,
            answer_text=getattr(response, "text", "") or "",
            usage=(_first(raw, "usage_metadata", "usageMetadata") or {}),
        )
        write_bronze(rec)
        return rec

    def normalise(self, rec: BronzeRecord) -> CanonicalRecord:
        raw = rec.raw_response
        cited = []
        seen = set()

        candidates = raw.get("candidates") or []
        cand = candidates[0] if candidates else {}
        gm = _first(cand, "grounding_metadata", "groundingMetadata") or {}
        chunks = _first(gm, "grounding_chunks", "groundingChunks") or []
        for ch in chunks:
            web = (ch or {}).get("web") or {}
            uri = web.get("uri")
            title = web.get("title")  # Gemini puts the source DOMAIN here
            if uri and uri not in seen:
                seen.add(uri)
                cited.append(CitedSource(
                    position=len(cited) + 1,
                    url=uri,                              # vertexaisearch redirect (verbatim)
                    title=title,
                    domain=web.get("domain") or title,    # real publisher domain
                ))

        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=rec.answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],  # grounding_supports -> pairs in the RQ4 batch
        )
