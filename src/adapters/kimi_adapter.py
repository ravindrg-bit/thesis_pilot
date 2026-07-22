"""
kimi_adapter.py — the ONLY Kimi/Moonshot-specific code (V3 sec.7.3).

Kimi specifics (verified 31 May 2026):
  - kimi-k2.6 via Moonshot's OpenAI-COMPATIBLE endpoint (api.moonshot.ai/v1).
  - Web search is a builtin_function "$web_search" run as a TOOL-CALL LOOP (declare in every
    request; on finish_reason=="tool_calls" append the assistant message then echo
    tool_call.function.arguments back as a role="tool" message; repeat). Thinking mode must
    be disabled (extra_body {"thinking": {"type": "disabled"}}) for the tool to work.
  - NO STRUCTURED CITATION CHANNEL on any surface: $web_search returns an opaque search_id;
    the Formula API returns encrypted_output. The model holds the URLs server-side.
  - We therefore ELICIT citations into the answer text via a system prompt (the consumer
    app's approach, through the official API) and parse the URLs out. These are SELF-REPORTED
    (prompt-elicited), not provider-certified -> each CitedSource is flagged accordingly, and
    AIS is computed at cited-source-SET granularity uniformly across engines (methodology_notes).
    The full conversation AND the system prompt are captured to bronze for auditability.
"""

import os
import re
from datetime import datetime, timezone

from openai import OpenAI

import thesis_config as cfg
from src.adapters.base import write_bronze, domain_of
from src.schema import BronzeRecord, CanonicalRecord, CitedSource

ENGINE_KEY = "kimi"
MAX_TOOL_ROUNDS = 5
MAX_TOKENS = 4096

# Citation-instructing system prompt (verified to elicit full URLs, 31 May 2026).
CITE_SYSTEM_PROMPT = (
    "You have a web search tool. Answer the user's question using it, then end your reply "
    "with a section headed exactly 'Sources:' that lists every source you used as a numbered "
    "list. For each source, give its COMPLETE URL beginning with https:// - never just the "
    "site name or a bracketed number. If you used a source, its full URL must appear."
)

_URL_RE = re.compile(r"https?://[^\s\)\]\>\"'<]+")
_TRAIL = ".,;:)]}>\"'"


def _serialize(msg):
    if isinstance(msg, dict):
        return msg
    try:
        return msg.model_dump()
    except Exception:
        return {"role": getattr(msg, "role", "?"), "content": str(getattr(msg, "content", ""))}


class KimiAdapter:
    def __init__(self):
        self.spec = cfg.ENGINES[ENGINE_KEY]
        key = os.environ.get("MOONSHOT_API_KEY", "").strip()
        if not key:
            raise RuntimeError("MOONSHOT_API_KEY missing -- call src.env.load_env() in your script.")
        self.client = OpenAI(api_key=key, base_url=self.spec["base_url"])
        self.search_fn = self.spec["grounding"]["builtin_function"]  # "$web_search"

    def fetch(self, query_id, query_text, run_index) -> BronzeRecord:
        tools = [{"type": "builtin_function", "function": {"name": self.search_fn}}]
        messages = [
            {"role": "system", "content": CITE_SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ]
        search_calls = []
        rounds = 0
        finish_reason = None
        completion = None

        while (finish_reason is None or finish_reason == "tool_calls") and rounds <= MAX_TOOL_ROUNDS:
            completion = self.client.chat.completions.create(
                model=self.spec["model"],
                messages=messages,
                max_tokens=MAX_TOKENS,
                tools=tools,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in (choice.message.tool_calls or []):
                    if tc.function.name == self.search_fn:
                        search_calls.append(tc.function.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": tc.function.arguments,
                        })
                rounds += 1

        final = completion.model_dump() if completion is not None else {}
        try:
            answer = completion.choices[0].message.content or ""
        except Exception:
            answer = ""

        raw = {
            "final_response": final,
            "conversation": [_serialize(m) for m in messages],
            "search_calls": search_calls,
            "n_search_rounds": rounds,
        }
        rec = BronzeRecord(
            query_id=query_id,
            engine=ENGINE_KEY,
            run_index=run_index,
            model_requested=self.spec["model"],
            model_served=(final.get("model", "") or ""),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            query_text=query_text,
            request_params={
                "model": self.spec["model"],
                "tools": [self.search_fn],
                "thinking": "disabled",
                "citation_mode": "prompt_elicited",   # SELF-REPORTED sources (see schema/notes)
                "system_prompt": CITE_SYSTEM_PROMPT,   # logged for provenance auditability
                "max_tool_rounds": MAX_TOOL_ROUNDS,
            },
            raw_response=raw,
            answer_text=answer,
            usage=(final.get("usage") or {}),
        )
        write_bronze(rec)
        return rec

    def normalise(self, rec: BronzeRecord) -> CanonicalRecord:
        # No structured channel -> parse the prompt-elicited URLs from the answer text.
        # SELF-REPORTED (not provider-certified); flagged on every source.
        cited = []
        seen = set()
        for raw_url in _URL_RE.findall(rec.answer_text or ""):
            url = raw_url.rstrip(_TRAIL)
            if url and url not in seen:
                seen.add(url)
                cited.append(CitedSource(
                    position=len(cited) + 1,
                    url=url,
                    title=None,
                    domain=domain_of(url),
                    provenance="self_reported_prompt_elicited",
                ))
        return CanonicalRecord(
            query_id=rec.query_id,
            engine=rec.engine,
            run_index=rec.run_index,
            model_served=rec.model_served,
            timestamp_utc=rec.timestamp_utc,
            answer_text=rec.answer_text,
            cited_sources=cited,
            claim_citation_pairs=[],
        )
