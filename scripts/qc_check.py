"""
qc_check.py — comprehensive pre-pilot QC (READ-ONLY; no API calls; free).
Validates config, schema, the frozen v2 registry, all six adapters (import + interface +
dispatch completeness), API-key presence, and re-normalises the existing single-query bronze
files (golden-file regression) to confirm each adapter parses real responses correctly with
the right citation provenance. Run BEFORE the collection loop. Live key/model reachability is
a separate step: scripts/check_api_access.py.
"""

import dataclasses
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd
import thesis_config as cfg

R = {"pass": 0, "fail": 0, "warn": 0}
def ok(m):   print(f"  [PASS] {m}");  R["pass"] += 1
def fail(m): print(f"  [FAIL] {m}");  R["fail"] += 1
def warn(m): print(f"  [WARN] {m}");  R["warn"] += 1
def section(t): print(f"\n=== {t} ===")

EXPECTED = {
    "chatgpt":    {"model": "gpt-5.5-2026-04-23", "env": "OPENAI_API_KEY",     "region": "US"},
    "claude":     {"model": "claude-sonnet-4-6",  "env": "ANTHROPIC_API_KEY",  "region": "US"},
    "gemini":     {"model": "gemini-3.5-flash",   "env": "GEMINI_API_KEY",     "region": "US"},
    "perplexity": {"model": "sonar",              "env": "PERPLEXITY_API_KEY", "region": "US"},
    "kimi":       {"model": "kimi-k2.6",          "env": "MOONSHOT_API_KEY",   "region": "China"},
    "mistral":    {"model": "mistral-medium-3-5", "env": "MISTRAL_API_KEY",    "region": "Europe"},
}
ADAPTERS = [
    ("src.adapters.openai_adapter",     "OpenAIAdapter"),
    ("src.adapters.perplexity_adapter", "PerplexityAdapter"),
    ("src.adapters.gemini_adapter",     "GeminiAdapter"),
    ("src.adapters.claude_adapter",     "ClaudeAdapter"),
    ("src.adapters.kimi_adapter",       "KimiAdapter"),
    ("src.adapters.mistral_adapter",    "MistralAdapter"),
]
GROUNDING = {
    "chatgpt":    lambda g: g.get("tool") == "web_search",
    "gemini":     lambda g: g.get("tool") == "google_search",
    "perplexity": lambda g: g.get("inherent") is True,
    "claude":     lambda g: g.get("tool_version") == "web_search_20250305",
    "kimi":       lambda g: g.get("builtin_function") == "$web_search",
    "mistral":    lambda g: g.get("connector") == "web_search",
}

print("=" * 72)
print("PRE-PILOT QC CHECK (read-only; no API calls)")
print("=" * 72)

# ---- schema import (needed by C and G) ----
SCHEMA_OK = True
try:
    from src.schema import BronzeRecord, CanonicalRecord, CitedSource, ClaimCitationPair
except Exception as e:
    SCHEMA_OK = False
    print(f"  [FATAL] cannot import src.schema: {type(e).__name__}: {e}")

# ---- A. ENGINE CONFIG ----
section("A. Engine config (thesis_config.ENGINES)")
keys = set(cfg.ENGINES.keys())
exp = set(EXPECTED.keys())
ok("exactly the 6 expected engines") if keys == exp else fail(f"engine mismatch: missing={exp-keys} extra={keys-exp}")
fail("deepseek still present") if "deepseek" in keys else ok("deepseek absent (correctly dropped)")
req = ["provider","model","api","grounding","supports_temperature","supports_seed","env_key","region","notes","docs"]
for ek in sorted(keys & exp):
    spec = cfg.ENGINES[ek]
    miss = [f for f in req if f not in spec]
    if miss: fail(f"{ek}: missing fields {miss}")
    ok(f"{ek}: model = {spec.get('model')}") if spec.get("model") == EXPECTED[ek]["model"] \
        else fail(f"{ek}: model = {spec.get('model')} (expected {EXPECTED[ek]['model']})")
    if spec.get("env_key") != EXPECTED[ek]["env"]:
        fail(f"{ek}: env_key = {spec.get('env_key')} (expected {EXPECTED[ek]['env']})")
    if spec.get("region") != EXPECTED[ek]["region"]:
        warn(f"{ek}: region = {spec.get('region')} (expected {EXPECTED[ek]['region']})")
    if ek in GROUNDING:
        g = spec.get("grounding") or {}
        ok(f"{ek}: grounding spec correct") if GROUNDING[ek](g) else fail(f"{ek}: grounding wrong -> {g}")
regions = {}
for ek in keys & exp:
    regions[cfg.ENGINES[ek]["region"]] = regions.get(cfg.ENGINES[ek]["region"], 0) + 1
print(f"  region distribution: {regions}")
ok("cross-region = US:4 China:1 Europe:1") if (regions.get("US")==4 and regions.get("China")==1 and regions.get("Europe")==1) \
    else warn(f"region spread unexpected: {regions}")
# Claude tool-version regression guard (the structured-citation fix)
ctv = (cfg.ENGINES.get("claude", {}).get("grounding", {}) or {}).get("tool_version")
ok("claude tool_version = web_search_20250305 (citation fix held)") if ctv == "web_search_20250305" \
    else fail(f"claude tool_version = {ctv} (MUST be web_search_20250305 or structured citations break)")

# ---- B. PILOT SCOPE / REPRO ----
section("B. Pilot scope & reproducibility constants")
for name, val, want in [("N_QUERIES", getattr(cfg,"N_QUERIES",None), 100),
                        ("REGISTRY_VERSION", getattr(cfg,"REGISTRY_VERSION",None), "v2"),
                        ("PRODUCTION_K", getattr(cfg,"PRODUCTION_K",None), 8),
                        ("QUERY_ID_HASH_LEN", getattr(cfg,"QUERY_ID_HASH_LEN",None), 16)]:
    ok(f"{name} = {val}") if val == want else fail(f"{name} = {val} (expected {want})")
for name in ["PILOT_K","SEED","QUERY_REGISTRY","BRONZE","SILVER","GOLD"]:
    ok(f"{name} present") if getattr(cfg, name, None) is not None else fail(f"{name} missing from config")

# ---- C. SCHEMA ----
section("C. Schema (src/schema.py)")
if not SCHEMA_OK:
    fail("schema import failed (see FATAL above)")
else:
    ok("imports: BronzeRecord, CanonicalRecord, CitedSource, ClaimCitationPair")
    cs = {f.name: f for f in dataclasses.fields(CitedSource)}
    for f in ["position","url","title","domain","provenance"]:
        ok(f"CitedSource.{f} present") if f in cs else fail(f"CitedSource.{f} MISSING")
    pd_def = cs["provenance"].default if "provenance" in cs else None
    ok("CitedSource.provenance default = 'provider_certified'") if pd_def == "provider_certified" \
        else fail(f"CitedSource.provenance default = {pd_def!r} (expected 'provider_certified')")
    for dc in [BronzeRecord, CanonicalRecord, CitedSource]:
        ok(f"{dc.__name__} is frozen") if dc.__dataclass_params__.frozen else warn(f"{dc.__name__} NOT frozen")
    cr = {f.name for f in dataclasses.fields(CanonicalRecord)}
    for f in ["cited_sources","claim_citation_pairs","answer_text","engine","query_id","run_index","model_served"]:
        ok(f"CanonicalRecord.{f} present") if f in cr else fail(f"CanonicalRecord.{f} MISSING")

# ---- D. FROZEN REGISTRY ----
section("D. Frozen query registry (v2)")
reg = cfg.QUERY_REGISTRY
if not reg.exists():
    fail(f"registry not found: {reg}")
else:
    ok(f"active registry = {reg.name}") if "v2" in reg.name else warn(f"registry name lacks 'v2': {reg.name}")
    try:
        df = pd.read_parquet(reg)
        ok(f"registry rows = {len(df)}") if len(df) == cfg.N_QUERIES else fail(f"rows = {len(df)} (expected {cfg.N_QUERIES})")
        for col in ["query_id","query_text"]:
            ok(f"column '{col}' present") if col in df.columns else fail(f"column '{col}' MISSING")
        if "query_id" in df.columns:
            ids = df["query_id"].astype(str)
            ok("query_id unique") if ids.is_unique else fail(f"query_id has {int(ids.duplicated().sum())} duplicates")
            ok("query_id all non-empty") if (ids.str.len() > 0).all() else fail("query_id has empties")
            print(f"  sample query_ids: {ids.head(3).tolist()}")
            hashes = ids.str.replace(r'^[A-Za-z]+[_-]?', '', regex=True)
            ok(f"query_id hashes are {cfg.QUERY_ID_HASH_LEN} hex") if hashes.str.fullmatch(rf'[0-9a-fA-F]{{{cfg.QUERY_ID_HASH_LEN}}}').all() \
                else warn(f"some query_id hashes != {cfg.QUERY_ID_HASH_LEN} hex")
        if "query_text" in df.columns:
            ok("query_text all non-empty") if (df["query_text"].astype(str).str.strip().str.len() > 0).all() else fail("query_text has empties")
    except Exception as e:
        fail(f"registry load error: {type(e).__name__}: {e}")
    man = reg.parent / f"pilot_queries_{cfg.REGISTRY_VERSION}.manifest.json"
    if man.exists():
        try:
            json.loads(man.read_text()); ok(f"manifest present + valid JSON ({man.name})")
        except Exception as e:
            warn(f"manifest unreadable: {e}")
    else:
        warn(f"manifest not found: {man.name}")

# ---- E. ADAPTERS ----
section("E. Adapters (import, interface, dispatch completeness)")
ad_keys, ad_cls = {}, {}
for mod_path, cls_name in ADAPTERS:
    try:
        mod = importlib.import_module(mod_path)
        ek = getattr(mod, "ENGINE_KEY", None)
        cls = getattr(mod, cls_name, None)
        if cls is None: fail(f"{mod_path}: class {cls_name} not found"); continue
        if ek is None: fail(f"{mod_path}: ENGINE_KEY missing"); continue
        hf, hn = callable(getattr(cls,"fetch",None)), callable(getattr(cls,"normalise",None))
        ok(f"{cls_name}: imports; ENGINE_KEY='{ek}'; fetch+normalise present") if (hf and hn) \
            else fail(f"{cls_name}: missing {'fetch ' if not hf else ''}{'normalise' if not hn else ''}")
        ad_keys[ek] = mod_path; ad_cls[ek] = cls
    except Exception as e:
        fail(f"{mod_path}: import error {type(e).__name__}: {e}")
ok(f"every config engine has exactly one adapter") if set(ad_keys) == set(cfg.ENGINES) \
    else fail(f"adapter/config mismatch: engines w/o adapter={set(cfg.ENGINES)-set(ad_keys)}, adapters w/o engine={set(ad_keys)-set(cfg.ENGINES)}")
try:
    from src.adapters.base import write_bronze, domain_of
    ok("base.py exports write_bronze + domain_of")
except Exception as e:
    fail(f"base.py import error: {e}")

# ---- F. API KEYS (presence only; never prints values) ----
section("F. API keys present in environment (presence only)")
for ek in sorted(cfg.ENGINES):
    envk = cfg.ENGINES[ek].get("env_key", "")
    ok(f"{ek}: {envk} set") if os.environ.get(envk, "").strip() else warn(f"{ek}: {envk} NOT set (needed for the run)")

# ---- G. BRONZE RE-NORMALISATION (golden-file regression; offline) ----
section("G. Re-normalise existing single-query bronze (golden-file regression)")
if SCHEMA_OK:
    bfields = {f.name for f in dataclasses.fields(BronzeRecord)}
    for ek in sorted(cfg.ENGINES):
        files = sorted(cfg.BRONZE.glob(f"{ek}__*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            warn(f"{ek}: no bronze found (run its single-query test) — skipping"); continue
        bf = files[0]
        try:
            d = json.loads(bf.read_text())
            rec = BronzeRecord(**{k: v for k, v in d.items() if k in bfields})
        except Exception as e:
            fail(f"{ek}: bronze {bf.name} won't load into BronzeRecord: {type(e).__name__}: {e}"); continue
        served, want = (rec.model_served or "").strip(), cfg.ENGINES[ek]["model"]
        if not served: warn(f"{ek}: empty model_served in bronze")
        elif served == want or served.startswith(want) or want in served: ok(f"{ek}: served model '{served}' matches requested")
        else: warn(f"{ek}: served '{served}' != requested '{want}' (drift or dated variant?)")
        if not os.environ.get(cfg.ENGINES[ek].get("env_key",""), "").strip():
            warn(f"{ek}: key not set, cannot instantiate adapter — skipping normalise"); continue
        if ek not in ad_cls:
            warn(f"{ek}: adapter class unavailable — skipping normalise"); continue
        try:
            canon = ad_cls[ek]().normalise(rec)
        except Exception as e:
            fail(f"{ek}: normalise() failed on {bf.name}: {type(e).__name__}: {e}"); continue
        nc = len(canon.cited_sources)
        ok(f"{ek}: normalise OK — {nc} cited sources, answer {len(canon.answer_text or '')} chars  [{bf.name}]") \
            if (canon.answer_text or "").strip() else warn(f"{ek}: normalise produced EMPTY answer_text  [{bf.name}]")
        if nc == 0:
            warn(f"{ek}: 0 cited sources in this bronze (ok if a non-grounding query — verify expected)")
        else:
            want_prov = "self_reported_prompt_elicited" if ek == "kimi" else "provider_certified"
            provs = {s.provenance for s in canon.cited_sources}
            ok(f"{ek}: citation provenance = {want_prov}") if provs == {want_prov} \
                else fail(f"{ek}: provenance {provs} (expected all '{want_prov}')")
else:
    warn("schema unavailable — skipping bronze re-normalisation")

# ---- H. CLEANLINESS + DOCS ----
section("H. Cleanliness & documentation")
diags = list((ROOT / "scripts").glob("_diag_*.py"))
ok("no leftover _diag_* scripts") if not diags else warn(f"leftover diagnostics: {[p.name for p in diags]}")
for doc in ["docs/methodology_notes.md", "docs/data_provenance.md"]:
    ok(f"{doc} present") if (ROOT / doc).exists() else warn(f"{doc} missing")

# ---- SUMMARY ----
print("\n" + "=" * 72)
print(f"QC SUMMARY:  PASS={R['pass']}  WARN={R['warn']}  FAIL={R['fail']}")
if R["fail"] == 0:
    print("RESULT: GREEN - no hard failures. Review WARNs, then run scripts/check_api_access.py")
    print("        (live 6-engine reachability) as the final gate before the collection loop.")
else:
    print("RESULT: RED - hard failures above must be fixed before the pilot.")
print("=" * 72)
sys.exit(1 if R["fail"] else 0)
