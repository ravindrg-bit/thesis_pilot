"""
verify_setup.py — confirm Batch 1 stood up correctly. Run from the project root:
    python scripts/verify_setup.py
Prints a clear PASS/FAIL. Never prints secrets. Missing API keys are a WARN, not a
FAIL (they are not needed until the API batch).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # make project root importable from anywhere

import os

ok = True

def check(label, condition, fix=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    line = f"[{mark}] {label}"
    if not condition and fix:
        line += f"  -> {fix}"
    print(line)

def warn(label, condition, note=""):
    mark = "OK" if condition else "WARN"
    line = f"[{mark}] {label}"
    if not condition and note:
        line += f"  -> {note}"
    print(line)

print("=" * 60)
print("BATCH 1 SETUP CHECK")
print("=" * 60)

try:
    import thesis_config as cfg
    check("thesis_config imports", True)
except Exception as e:
    check("thesis_config imports", False, f"import error: {e}")
    print("\nCannot continue without config. Stopping.")
    sys.exit(1)

for label, p in [
    ("data/bronze", cfg.BRONZE), ("data/silver", cfg.SILVER), ("data/gold", cfg.GOLD),
    ("docs", cfg.DOCS), ("src", ROOT / "src"),
    ("src/adapters", ROOT / "src" / "adapters"), ("scripts", ROOT / "scripts"),
]:
    check(f"directory {label}", Path(p).is_dir(), "re-run the skeleton prompt")

try:
    from src import schema
    rec = schema.CanonicalRecord(
        query_id="gb_test", engine="chatgpt", run_index=1,
        model_served="verify-me", timestamp_utc="1970-01-01T00:00:00Z",
        answer_text="hello",
    )
    check("src.schema CanonicalRecord instantiates", rec.query_id == "gb_test")
except Exception as e:
    check("src.schema CanonicalRecord instantiates", False, f"{e}")

for lib in ["pandas", "pyarrow", "dotenv"]:
    try:
        m = __import__(lib)
        ver = getattr(m, "__version__", "n/a")
        check(f"library {lib} ({ver})", True)
    except Exception:
        check(f"library {lib}", False, "pip install -r requirements.txt")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass
for var in ["OPENAI_API_KEY", "PERPLEXITY_API_KEY", "GEMINI_API_KEY"]:
    present = bool(os.environ.get(var, "").strip())
    warn(f"env {var}", present, "add to .env when you reach the API batch")

print("-" * 60)
print("Resolved config:")
for k, v in cfg.resolved_summary().items():
    print(f"  {k}: {v}")

print("=" * 60)
print("RESULT:", "PASS - Batch 1 is ready." if ok else "FAIL - fix the items above.")
print("=" * 60)
sys.exit(0 if ok else 1)
