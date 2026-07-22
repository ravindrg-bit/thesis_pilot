"""
openai_forced_search_adapter.py — FORCED-web_search variant for sensitivity testing.

IDENTICAL to OpenAIAdapter (it subclasses it) except for ONE behavioural change:
the Responses API request carries `tool_choice={"type": "web_search"}`, so the
model MUST invoke web_search on every call instead of deciding for itself. This
probes the chatgpt zero-citation floor (~72% of pilot rows): when the model is
*forced* to search, what does it cite?

Everything else is inherited verbatim:
  - same Responses API request shape (tools, reasoning, include) as the main adapter
  - same BronzeRecord schema
  - normalise() is REUSED directly from the parent (same url_citation parsing,
    same CONSULTED-vs-CITED de-dupe) — no new extraction code.

Sensitivity-analysis-only. NOT wired into the main pipeline (thesis_config routing
is untouched). Bronze is written to a SEPARATE tree (data/bronze/openai_forced/)
under a distinct engine tag ("openai_forced") so it never collides with the main
adapter's data/pilot/bronze/ "chatgpt__*" files.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import thesis_config as cfg
from src.adapters.openai_adapter import OpenAIAdapter
from src.schema import BronzeRecord

ENGINE_KEY = "openai_forced"

# Separate bronze tree so forced-search captures never collide with the auto
# adapter's data/pilot/bronze/ files. Deliberately outside the run-profile data
# tree (this is sensitivity analysis, not part of the medallion pipeline).
BRONZE_FORCED = cfg.DATA / "forced" / "bronze" / ENGINE_KEY


class OpenAIForcedSearchAdapter(OpenAIAdapter):
    """OpenAIAdapter with web_search FORCED via tool_choice.

    __init__ is inherited: self.spec is still cfg.ENGINES["chatgpt"], so the
    model id and grounding tool are identical to the main adapter. normalise()
    is inherited unchanged (the response shape is identical; only the request
    differs). Only fetch() and _write_bronze() are overridden.
    """

    # --- fetch: live call -> verbatim bronze (forced web_search) ------------
    def fetch(self, query_id: str, query_text: str, run_index: int) -> BronzeRecord:
        request_params = {
            "model": self.spec["model"],
            "tools": [{"type": self.spec["grounding"]["tool"]}],  # web_search ON
            # THE single behavioural change vs the auto adapter: force the tool so
            # the model cannot decline to search. Everything else is byte-identical
            # to OpenAIAdapter.fetch (same reasoning effort, same include).
            "tool_choice": {"type": self.spec["grounding"]["tool"]},
            "reasoning": {"effort": cfg.OPENAI_REASONING_EFFORT},
            # Return the FULL set of sources the model fetched, not just the ones it
            # cited, populating the otherwise-null web_search_call.action.sources.
            "include": ["web_search_call.action.sources"],
        }
        response = self.client.responses.create(input=query_text, **request_params)

        # Convert the SDK object to a plain dict so bronze is pure JSON (verbatim).
        raw = response.model_dump()

        rec = BronzeRecord(
            query_id=query_id,
            engine=ENGINE_KEY,            # "openai_forced", NOT "chatgpt" -> no collision
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
        BRONZE_FORCED.mkdir(parents=True, exist_ok=True)
        # One file per (engine, query, run). Stable name = idempotent re-writes.
        path = BRONZE_FORCED / f"{rec.engine}__{rec.query_id}__r{rec.run_index}.json"
        path.write_text(json.dumps(rec.__dict__, indent=2, default=str))
        return path

    # normalise() intentionally NOT overridden — inherited verbatim from
    # OpenAIAdapter (identical response shape, identical url_citation parsing).
