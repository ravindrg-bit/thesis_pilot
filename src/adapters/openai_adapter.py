"""
openai_adapter.py — the ONLY OpenAI-specific code (V3 sec.7.3).

Two responsibilities, matching the medallion layers:
  fetch()     -> make the grounded call; write a verbatim BronzeRecord to data/bronze/
  normalise() -> parse that raw payload into a CanonicalRecord (answer + cited sources)

OpenAI specifics (verified 30 May 2026):
  - gpt-5.5 is a REASONING model on the RESPONSES API (not Chat Completions).
  - Citations require the web_search tool. The response is a typed array of items
    (reasoning, web_search_call(s), message). Parse by item .type, NEVER by position.
  - Inline url_citation annotations = what was CITED; the separate sources list =
    everything CONSULTED. We keep both; the distinction matters for the metrics.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import thesis_config as cfg
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "chatgpt"


class OpenAIAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment

    # --- fetch: live call -> verbatim bronze --------------------------------
    def fetch(self, query_id: str, query_text: str, run_index: int) -> BronzeRecord:
        request_params = {
            "model": self.spec["model"],
            "tools": [{"type": self.spec["grounding"]["tool"]}],  # web_search ON
            "reasoning": {"effort": cfg.OPENAI_REASONING_EFFORT},
            # include: ["web_search_call.action.sources"] intentionally removed.
            # Decision (2026-06-06): 299/300 pilot captures were made without it
            # (one mid-collection artefact). Removing keeps scale-up directly
            # comparable to pilot. The CONSULTED-sources list is not used in any
            # metric computation. See conflict #2 in pilot_llm_configurations.md.
        }
        response = self.client.responses.create(input=query_text, **request_params)

        # Convert the SDK object to a plain dict so bronze is pure JSON (verbatim).
        raw = response.model_dump()

        rec = BronzeRecord(
            query_id=query_id,
            engine=ENGINE_KEY,
            run_index=run_index,
            model_requested=self.spec["model"],
            model_served=raw.get("model", ""),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            query_text=query_text,
            request_params=request_params,
            raw_response=raw,
            answer_text=getattr(response, "output_text", "") or "",
            usage=(raw.get("usage") or {}),
        )
        self._write_bronze(rec)
        return rec

    def _write_bronze(self, rec: BronzeRecord) -> Path:
        cfg.BRONZE.mkdir(parents=True, exist_ok=True)
        # One file per (engine, query, run). Stable name = idempotent re-writes.
        path = cfg.BRONZE / f"{rec.engine}__{rec.query_id}__r{rec.run_index}.json"
        path.write_text(json.dumps(rec.__dict__, indent=2, default=str))
        return path

    # --- normalise: verbatim bronze -> canonical (silver shape) -------------
    def normalise(self, rec: BronzeRecord) -> CanonicalRecord:
        raw = rec.raw_response
        answer_text = rec.answer_text
        cited = []

        # Citations live ONLY as url_citation annotations on the message item's
        # content (verified against real bronze). NOT inline [1][2] markers, NOT a
        # top-level "citations" array. Walk the typed output array and dispatch on
        # .type (never by position: order is reasoning/web_search_call/message and
        # not guaranteed). A URL cited across several text spans appears as multiple
        # annotations -> de-dupe so each source counts once, at its first position
        # (1-based, in annotation order). If no message carries a url_citation, the
        # list stays empty: that's the model choosing not to search/cite, not a parse
        # failure (~72% of pilot chatgpt rows; the structural zero-citation floor).
        seen_urls = set()
        position = 0
        for item in (raw.get("output") or []):
            if item.get("type") != "message":
                continue  # skip reasoning / web_search_call items
            for block in (item.get("content") or []):
                if not answer_text and block.get("type") == "output_text":
                    answer_text = block.get("text", "")
                for ann in (block.get("annotations") or []):
                    if ann.get("type") == "url_citation":
                        url = ann.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            position += 1
                            cited.append(CitedSource(
                                position=position,
                                url=url,
                                title=ann.get("title"),
                                domain=_domain_of(url),
                            ))

        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],  # built in the metric-extraction batch (RQ4)
        )


def _domain_of(url: str):
    """Host of a URL with any leading 'www.' stripped, so 'www.vogue.com' and
    'vogue.com' collapse to one domain for source-diversity analysis. A non-www
    subdomain (e.g. 'us.fashionnetwork.com') is preserved."""
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or None
    except Exception:
        return None
