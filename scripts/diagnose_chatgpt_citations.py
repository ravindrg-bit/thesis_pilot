"""
Diagnose ChatGPT's low citation count: does the RAW bronze (Responses API payload) genuinely
contain no url_citation annotations, or is the adapter/silver dropping them?
Reads bronze JSON + silver; no network.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import thesis_config as cfg

citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")
responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")

cg = responses[responses.engine == "chatgpt"].copy()
zero_cells = cg[cg.n_cited_sources == 0]
print(f"chatgpt cells: {len(cg)} | with ZERO cited sources in silver: {len(zero_cells)}")
print(f"chatgpt mean cited sources/cell: {cg.n_cited_sources.mean():.2f}")

# Inspect up to 5 zero-citation chatgpt bronze files: does the RAW payload contain
# any 'url_citation' annotations the adapter might have missed?
import glob
files = sorted(glob.glob(str(cfg.BRONZE / "chatgpt__*.json")))
checked = 0
for f in files:
    d = json.loads(Path(f).read_text())
    qid = d["query_id"]; run = d["run_index"]
    sv = responses[(responses.engine=="chatgpt") & (responses.query_id==qid) & (responses.run_index==run)]
    if sv.empty or sv.iloc[0]["n_cited_sources"] != 0:
        continue
    raw = d.get("raw_response", {})
    blob = json.dumps(raw).lower()
    n_url_citation = blob.count('"url_citation"') + blob.count("url_citation")
    n_annotations = blob.count('"annotations"')
    n_http = blob.count("http")
    print("-"*64)
    print(f"{Path(f).name} | q={qid[:16]} run={run}")
    print(f"  raw contains 'url_citation' tokens : {n_url_citation}")
    print(f"  raw contains 'annotations' keys    : {n_annotations}")
    print(f"  raw contains 'http' occurrences    : {n_http}")
    print(f"  answer length (chars)              : {len(d.get('answer_text',''))}")
    checked += 1
    if checked >= 5:
        break

print("="*64)
print("READ: if url_citation tokens are ~0 AND http occurrences are low across these cells,")
print("ChatGPT genuinely didn't cite -> 0.22 AIS is a TRUE finding. If url_citation/http")
print("appear in raw but n_cited_sources is 0 -> the adapter is DROPPING citations (a bug to fix).")
