"""
claude_adapter.py — the ONLY Claude-specific code (V3 sec.7.3).

Claude specifics (verified 30 May 2026):
  - claude-sonnet-4-6 via the Anthropic Messages API (anthropic SDK).
  - Citations require the web_search server tool (type web_search_20250305). The model
    DECIDES whether to search (like OpenAI/Gemini; unlike Perplexity).
  - max_tokens is REQUIRED by the Messages API.
  - The response 'content' is a list of typed blocks. Parse by block .type:
        text                   -> answer text; carries a 'citations' list
        server_tool_use        -> the search query Claude issued (skip)
        web_search_tool_result -> the raw results consulted (skip for cited_sources)
    Inline citations on text blocks have type 'web_search_result_location' with
    url / title / cited_text -> these are what was CITED (analogous to OpenAI's
    url_citation annotations). Dedup on URL.
"""

import os
from datetime import datetime, timezone

import anthropic

import thesis_config as cfg
from src.adapters.base import write_bronze, domain_of
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "claude"
MAX_TOKENS = 2048


class ClaudeAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing -- call src.env.load_env() in your script."
            )
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def fetch(self, query_id, query_text, run_index) -> BronzeRecord:
        # web_search server tool ON; the model decides whether to actually search.
        tools = [{
            "type": self.spec["grounding"]["tool_version"],  # web_search_20250305
            "name": self.spec["grounding"]["tool"],           # web_search
            "max_uses": 5,
        }]
        request_params = {
            "model": self.spec["model"],
            "max_tokens": MAX_TOKENS,
            "tools": tools,
        }
        response = self.client.messages.create(
            messages=[{"role": "user", "content": query_text}],
            **request_params,
        )

        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = response.model_dump()

        answer_text = "".join(
            getattr(b, "text", "") for b in response.content
            if getattr(b, "type", "") == "text"
        )

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
            answer_text=answer_text,
            usage=(raw.get("usage") or {}),
        )
        write_bronze(rec)
        return rec

    def normalise(self, rec: BronzeRecord) -> CanonicalRecord:
        raw = rec.raw_response
        answer_parts = []
        cited = []
        seen = set()

        for block in (raw.get("content") or []):
            if block.get("type") != "text":
                continue  # skip server_tool_use / web_search_tool_result blocks
            answer_parts.append(block.get("text") or "")
            for cit in (block.get("citations") or []):
                if cit.get("type") == "web_search_result_location":
                    url = cit.get("url")
                    if url and url not in seen:
                        seen.add(url)
                        cited.append(CitedSource(
                            position=len(cited) + 1,
                            url=url,
                            title=cit.get("title"),
                            domain=domain_of(url),
                        ))

        answer_text = rec.answer_text or "".join(answer_parts)
        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],  # cited_text -> claim pairing in the RQ4 batch
        )
