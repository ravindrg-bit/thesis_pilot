"""
verify_scaleup_config.py — pre-launch integration check for the scale-up.

Imports the config and adapters under THESIS_RUN_PROFILE=scaleup, resolves
the profile, and prints a 7-item pass/fail checklist. Makes NO API calls.

Run from the project root:
    THESIS_RUN_PROFILE=scaleup python scripts/verify_scaleup_config.py
"""

import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before importing thesis_config
if os.environ.get("THESIS_RUN_PROFILE") != "scaleup":
    print("[WARN] THESIS_RUN_PROFILE is not 'scaleup'. Re-run as:")
    print("       THESIS_RUN_PROFILE=scaleup python scripts/verify_scaleup_config.py")
    sys.exit(1)

import thesis_config as cfg
from src.adapters.openai_adapter import OpenAIAdapter

results = []

def chk(label, passed, expected="", got=""):
    tag = "✅ PASS" if passed else "❌ FAIL"
    detail = f" (expected {expected!r}, got {got!r})" if not passed and expected else ""
    results.append((passed, f"  {tag}  {label}{detail}"))


# 1. n_queries == 250
chk("n_queries = 250",
    cfg.N_QUERIES == 250, expected=250, got=cfg.N_QUERIES)

# 2. k == 8
chk("k = 8",
    cfg.K == 8, expected=8, got=cfg.K)

# 3. engines == 5, correct set, no mistral
expected_engines = {"chatgpt", "claude", "gemini", "kimi", "perplexity"}
actual_engines   = set(cfg.ENGINES.keys())
chk("engines = {chatgpt, claude, gemini, kimi, perplexity} (no mistral)",
    actual_engines == expected_engines,
    expected=sorted(expected_engines), got=sorted(actual_engines))

# 4. registry file exists and is the right one
reg = cfg.QUERY_REGISTRY
chk("registry file = scaleup_queries_v2.parquet (and exists)",
    reg.exists() and reg.name == "scaleup_queries_v2.parquet",
    expected="scaleup_queries_v2.parquet (exists)", got=f"{reg.name} (exists={reg.exists()})")

# 5. total planned calls = 250 × 8 × 5 = 10,000
total = cfg.N_QUERIES * cfg.K * len(cfg.ENGINES)
chk("total planned calls = 10,000 (250 × 8 × 5)",
    total == 10_000, expected=10_000, got=total)

# 6. mistral is absent from both ENGINES and ADAPTERS
# Import ADAPTERS via AST so we don't execute main()
import ast
adapter_src = (ROOT / "scripts" / "run_collection.py").read_text()
tree = ast.parse(adapter_src)
adapter_keys = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "ADAPTERS":
                adapter_keys = [k.value for k in node.value.keys
                                if isinstance(k, ast.Constant)]
mistral_in_engines  = "mistral" in cfg.ENGINES
mistral_in_adapters = "mistral" in adapter_keys
chk("mistral absent from ENGINES and ADAPTERS",
    not mistral_in_engines and not mistral_in_adapters,
    expected="absent", got=f"ENGINES={mistral_in_engines}, ADAPTERS={mistral_in_adapters}")

# 7. No 'include' key in ChatGPT adapter request_params
fetch_src = inspect.getsource(OpenAIAdapter.fetch)
has_include = '"include"' in fetch_src and "intentionally removed" not in fetch_src
chk('no "include" key in openai_adapter request_params',
    not has_include,
    expected="absent", got=f"present={has_include}")

# ── Print report ──────────────────────────────────────────────────────────────
print("=" * 64)
print("SCALE-UP PRE-LAUNCH INTEGRATION CHECK")
print(f"Profile: {cfg.RUN_PROFILE}  |  root: {cfg.DATA}")
print("=" * 64)
for _, line in results:
    print(line)
passed = sum(1 for p, _ in results if p)
failed = len(results) - passed
print("-" * 64)
print(f"  {passed} PASS  /  {failed} FAIL")
print("=" * 64)
if failed:
    print("\n[ACTION REQUIRED] Fix the failing items before launching the scale-up.")
    sys.exit(1)
else:
    print("\nAll checks pass. Safe to proceed with scale-up collection.")
