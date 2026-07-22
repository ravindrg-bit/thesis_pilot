"""
cost_ledger.py - consolidated pilot cost ledger (5 engines, 100 queries, k=3).

(a) ENGINE COLLECTION: totals REAL token usage recorded in every bronze record's `usage`
    block -> actual incurred cost (not an estimate), priced via cfg.ENGINE_PRICES.
(b) PAWC/AIS JUDGE: projects from the smoke-test measured per-cell token rate
    (per-sentence = the validated method; batched shown for context), priced via JUDGE_PRICES.
(c) Per-query and grand totals.
No API calls. All prices are VERIFY-me constants in thesis_config.
"""
import json, sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import thesis_config as cfg

ENGINES = ["chatgpt", "claude", "gemini", "perplexity", "kimi"]
N_QUERIES, K = 100, 3
EXPECTED_CELLS = N_QUERIES * K * len(ENGINES)   # 1500

def usd(tok_in, tok_out, price):
    return (tok_in/1e6)*price["in"] + (tok_out/1e6)*price["out"]

# ---------- (a) ENGINE COLLECTION COST from real bronze usage ----------
rows = []
for f in glob.glob(str(cfg.BRONZE / "*.json")):
    d = json.loads(Path(f).read_text())
    eng = d.get("engine")
    if eng not in ENGINES:
        continue
    u = d.get("usage", {}) or {}
    # token field names vary by engine; cover the common ones
    tin = u.get("prompt_tokens") or u.get("input_tokens") or u.get("prompt_token_count") or 0
    tout = u.get("completion_tokens") or u.get("output_tokens") or u.get("candidates_token_count") or 0
    total = u.get("total_tokens") or (tin + tout)
    rows.append({"engine": eng, "in": tin, "out": tout, "total": total})
bdf = pd.DataFrame(rows)

print("="*72)
print("PILOT COST LEDGER  |  5 engines x 100 queries x k=3")
print("="*72)
print(f"bronze records found: {len(bdf)} (expected {EXPECTED_CELLS} for full 5-engine pilot)")
if len(bdf) < EXPECTED_CELLS:
    print(f"  NOTE: {EXPECTED_CELLS - len(bdf)} fewer than expected (e.g. ChatGPT cells that")
    print("        returned 0 tokens, or any not yet collected). Cost reflects what's on disk.")

print("\n(a) ENGINE COLLECTION COST  [REAL tokens from bronze; prices = VERIFY]")
coll = bdf.groupby("engine").agg(cells=("engine","count"),
        tok_in=("in","sum"), tok_out=("out","sum")).reset_index()
coll_total = 0.0
print(f"  {'engine':11s} {'cells':>6s} {'tok_in':>12s} {'tok_out':>10s} {'$ cost':>9s} {'$/query':>9s}")
for _, r in coll.iterrows():
    p = cfg.ENGINE_PRICES.get(r['engine'], {"in":0,"out":0,"search_per_call":0})
    c = usd(r['tok_in'], r['tok_out'], p) + r['cells']*p.get("search_per_call",0)
    coll_total += c
    per_q = c / N_QUERIES
    print(f"  {r['engine']:11s} {r['cells']:6d} {r['tok_in']:12,d} {r['tok_out']:10,d} {c:9.2f} {per_q:9.4f}")
print(f"  {'TOTAL':11s} {coll['cells'].sum():6d} {coll['tok_in'].sum():12,d} {coll['tok_out'].sum():10,d} {coll_total:9.2f}")
print(f"  collection cost PER QUERY (all 5 engines, k=3): ${coll_total/N_QUERIES:.4f}")

# ---------- (b) PAWC/AIS JUDGE COST projected from the smoke ----------
print("\n(b) PAWC/AIS JUDGE COST  [projected from 50-cell smoke; prices = VERIFY]")
smoke_path = cfg.GOLD / f"smoke_pawc_ais_{cfg.RUN_PROFILE}.parquet"        # per-sentence (validated)
jp = cfg.JUDGE_PRICES
if smoke_path.exists():
    s = pd.read_parquet(smoke_path)
    s_cells = len(s)
    s_in, s_out = int(s["input_tokens"].sum()), int(s["output_tokens"].sum())
    s_cost = usd(s_in, s_out, jp)
    in_per_cell, out_per_cell = s_in/s_cells, s_out/s_cells
    cost_per_cell = s_cost/s_cells
    proj_cells = EXPECTED_CELLS
    proj_cost_sync = cost_per_cell * proj_cells
    print(f"  smoke basis: {s_cells} cells | {s_in:,} in / {s_out:,} out | ${cost_per_cell:.5f}/cell (per-sentence)")
    print(f"  projected over {proj_cells} cells:")
    print(f"     judge cost SYNC : ${proj_cost_sync:,.2f}   (${proj_cost_sync/N_QUERIES:.4f}/query)")
    print(f"     judge cost BATCH: ${proj_cost_sync*0.5:,.2f}   (50% off, laptop-free)")
    judge_for_total = proj_cost_sync * 0.5   # assume batch for the grand total
else:
    print("  [smoke_pawc_ais_pilot.parquet not found -- run the per-sentence smoke first]")
    judge_for_total = 0.0

# ---------- (c) GRAND TOTAL ----------
print("\n(c) PILOT GRAND TOTAL (collection incurred + judge projected@batch)")
print(f"  engine collection (incurred) : ${coll_total:,.2f}")
print(f"  PAWC/AIS judge (batch proj.)  : ${judge_for_total:,.2f}")
print(f"  ----------------------------------------")
print(f"  PILOT TOTAL                   : ${coll_total + judge_for_total:,.2f}")
print(f"  PILOT TOTAL per query         : ${(coll_total + judge_for_total)/N_QUERIES:.4f}")
print("="*72)
print("Caveats: token counts are REAL (from bronze) and the judge token rate is REAL (smoke);")
print("only the per-MTok PRICES are assumptions -- verify each at the provider's pricing page.")
print("Web-search/grounding fees are included only where set in ENGINE_PRICES.search_per_call.")
