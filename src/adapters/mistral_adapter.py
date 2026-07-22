"""
mistral_adapter.py — the ONLY Mistral-specific code (V3 sec.7.3).

Mistral specifics (verified 31 May 2026):
  - mistral-medium-3-5 via the Agents API (NOT plain chat completions), mistralai SDK.
  - Web search is a built-in CONNECTOR; declared as a tool {"type": "web_search"} and
    invoked directly on conversations.start (no pre-created agent). The model DECIDES
    whether to search (like OpenAI/Claude/Gemini; unlike Perplexity).
  - Response = response.outputs, a list of ENTRIES. Parse by entry .type:
        tool.execution  -> the web_search ran (skip)
        message.output  -> the answer; its .content is a LIST OF CHUNKS
    Each content chunk .type is 'text' (answer) or 'tool_reference' (a CITATION).
    tool_reference sources are provider-certified. Dedup on URL.
  - tool_reference field names are confirmed against the live response in the test
    (bronze-first); normalise() reads url/title defensively.
"""

import os
from datetime import datetime, timezone

from mistralai import Mistral

import thesis_config as cfg
from src.adapters.base import write_bronze, domain_of
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "mistral"


def _extract(raw):
    """Walk outputs -> message.output -> content chunks. Returns (answer_text, cited_list)."""
    answer_parts, cited, seen = [], [], set()
    for entry in (raw.get("outputs") or []):
        if entry.get("type") != "message.output":
            continue
        content = entry.get("content")
        if isinstance(content, str):
            answer_parts.append(content)
            continue
        for chunk in (content or []):
            ctype = chunk.get("type")
            if ctype == "text":
                answer_parts.append(chunk.get("text") or "")
            elif ctype == "tool_reference":
                url = chunk.get("url") or chunk.get("source")
                if url and url not in seen:
                    seen.add(url)
                    cited.append(CitedSource(
                        position=len(cited) + 1,
                        url=url,
                        title=chunk.get("title"),
                        domain=domain_of(url),
                    ))
    return "".join(answer_parts), cited


def _served_model(raw):
    if raw.get("model"):
        return raw["model"]
    for entry in (raw.get("outputs") or []):
        if entry.get("type") == "message.output" and entry.get("model"):
            return entry["model"]
    return ""


class MistralAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not key:
            raise RuntimeError("MISTRAL_API_KEY missing -- call src.env.load_env() in your script.")
        self.client = Mistral(api_key=key)

    def fetch(self, query_id, query_text, run_index) -> BronzeRecord:
        request_params = {"model": self.spec["model"], "tools": [{"type": "web_search"}]}
        response = self.client.beta.conversations.start(
            model=self.spec["model"],
            inputs=[{"role": "user", "content": query_text}],
            tools=[{"type": "web_search"}],
        )
        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = response.model_dump()

        answer_text, _ = _extract(raw)

        rec = BronzeRecord(
            query_id=query_id,
            engine=ENGINE_KEY,
            run_index=run_index,
            model_requested=self.spec["model"],
            model_served=_served_model(raw),
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
        answer_text, cited = _extract(rec.raw_response)
        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=rec.answer_text or answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],
        )
