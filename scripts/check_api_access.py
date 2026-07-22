"""
check_api_access.py — confirm each engine's API key + SDK + model string work.
Run AFTER putting your keys in .env. From the project root:
    python scripts/check_api_access.py
Makes ONE tiny call per engine (a few cents total). Never prints keys.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import thesis_config as cfg

from src.env import load_env
load_env()

results = {}

def report(engine, ok, detail):
    results[engine] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {engine:10s} | {detail}")

# --- OpenAI (Responses API) -------------------------------------------------
try:
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    client = OpenAI()
    m = cfg.ENGINES["chatgpt"]["model"]
    r = client.responses.create(
        model=m,
        input="Reply with the single word: ok",
        reasoning={"effort": cfg.OPENAI_REASONING_EFFORT},
    )
    report("openai", True, f"requested={m} served={getattr(r, 'model', None)} -> {r.output_text[:40]!r}")
except Exception as e:
    report("openai", False, f"{type(e).__name__}: {e}")

# --- Anthropic / Claude (Messages API) --------------------------------------
try:
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("ANTHROPIC_API_KEY missing in .env")
    aclient = anthropic.Anthropic()
    m = cfg.ENGINES["claude"]["model"]
    resp = aclient.messages.create(
        model=m,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    report("claude", True, f"requested={m} served={getattr(resp, 'model', None)} -> {text[:40]!r}")
except Exception as e:
    report("claude", False, f"{type(e).__name__}: {e}")

# --- Perplexity (OpenAI-compatible) -----------------------------------------
try:
    from openai import OpenAI
    key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY missing in .env")
    pclient = OpenAI(api_key=key, base_url=cfg.ENGINES["perplexity"]["base_url"])
    m = cfg.ENGINES["perplexity"]["model"]
    resp = pclient.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    report("perplexity", True, f"requested={m} served={getattr(resp, 'model', None)} -> {resp.choices[0].message.content[:40]!r}")
except Exception as e:
    report("perplexity", False, f"{type(e).__name__}: {e}")

# --- Gemini (generate_content) ----------------------------------------------
try:
    from google import genai
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    gclient = genai.Client()
    m = cfg.ENGINES["gemini"]["model"]
    resp = gclient.models.generate_content(model=m, contents="Reply with the single word: ok")
    report("gemini", True, f"requested={m} served={getattr(resp, 'model_version', None)} -> {(resp.text or '')[:40]!r}")
except Exception as e:
    report("gemini", False, f"{type(e).__name__}: {e}")

print("-" * 60)
passed = sum(1 for v in results.values() if v)
print(f"{passed}/{len(results)} engines reachable.")
sys.exit(0 if passed == len(results) else 1)
