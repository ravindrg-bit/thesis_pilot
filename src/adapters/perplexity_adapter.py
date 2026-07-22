"""
perplexity_adapter.py — the ONLY Perplexity-specific code (V3 sec.7.3).

Perplexity specifics (verified 30 May 2026):
  - Sonar is OpenAI-COMPATIBLE: call it with the openai client pointed at the
    Perplexity base_url; answer text is choices[0].message.content.
  - Sonar is INHERENTLY grounded -- every answer carries sources, no tool to enable.
  - Citations come back in the PAYLOAD: a top-level `search_results` array of
    {title, url, date, last_updated, snippet}. Older responses may carry a flat
    `citations` list (URL strings); we read search_results first, then fall back.
"""

import os
from datetime import datetime, timezone

from openai import OpenAI

import thesis_config as cfg
from src.adapters.base import write_bronze, domain_of
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "perplexity"


class PerplexityAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "PERPLEXITY_API_KEY missing -- call src.env.load_env() in your script."
            )
        self.client = OpenAI(api_key=key, base_url=self.spec["base_url"])

    def fetch(self, query_id, query_text, run_index) -> BronzeRecord:
        # Sonar grounds every query natively; no tool to enable. Sampling left at
        # API defaults -- determinism rests on k repeats + CV (thesis_config sec.3/4).
        request_params = {
            "model": self.spec["model"],
            "messages": [{"role": "user", "content": query_text}],
        }
        resp = self.client.chat.completions.create(**request_params)

        raw = resp.model_dump()
        # Safety net: ensure Perplexity's non-standard top-level fields
        # (search_results, citations) survive the SDK's typed parsing.
        extra = getattr(resp, "model_extra", None) or {}
        for k, v in extra.items():
            raw.setdefault(k, v)

        try:
            answer = resp.choices[0].message.content or ""
        except Exception:
            answer = ""

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
            answer_text=answer,
            usage=(raw.get("usage") or {}),
        )
        write_bronze(rec)
        return rec

    def normalise(self, rec: BronzeRecord) -> CanonicalRecord:
        raw = rec.raw_response
        cited = []
        seen = set()

        results = raw.get("search_results")
        if isinstance(results, list) and results:
            for r in results:
                url = (r or {}).get("url")
                if url and url not in seen:
                    seen.add(url)
                    cited.append(CitedSource(
                        position=len(cited) + 1,
                        url=url,
                        title=(r or {}).get("title"),
                        domain=domain_of(url),
                    ))
        else:
            for url in (raw.get("citations") or []):
                if url and url not in seen:
                    seen.add(url)
                    cited.append(CitedSource(
                        position=len(cited) + 1,
                        url=url,
                        title=None,
                        domain=domain_of(url),
                    ))

        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=rec.answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],  # built in the metric-extraction batch (RQ4)
        )
