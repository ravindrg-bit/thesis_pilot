#!/usr/bin/env python3
"""compute_source_token_lengths.py — precompute DeBERTa token lengths per source.

READ-ONLY on data/. Streams data/scaleup/silver/sources_scaleup.parquet in bounded
row batches (the file is ~3.5 GB; the full cleaned_text column will not fit in memory)
and tokenises each ok-fetched source with the *actual* NLI SentencePiece model
(spm.model from MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). Token count
is (SentencePiece pieces + 2) for the [CLS]/[SEP] pair, matching the DeBERTa encoder.

Only the >512-token decision matters for the truncation-exposure check, so each page is
sliced to its first CHAR_CAP characters before encoding — 512 tokens is ~2,000-3,500
chars, so any page above the limit still reads as >512 and any page below it is captured
in full. This makes the decision exact while bounding tokeniser work per document.

Writes analysis/source_token_length.parquet (url_canonical, n_tokens, n_chars,
fetch_status). This is the input file validation.ipynb V4 loads; regenerate with:
    .venv/bin/python scripts/compute_source_token_lengths.py
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data/scaleup/silver/sources_scaleup.parquet")
OUT = os.path.join(ROOT, "analysis/source_token_length.parquet")
CHAR_CAP = 6000          # >512 tokens is ~2-3.5k chars; 6k is a safe exact margin
BATCH = 300              # rows per streamed batch (bounds peak memory)

# The NLI SentencePiece model, resolved from the local HF cache (offline).
_hits = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    "/snapshots/*/spm.model"))
if not _hits:
    sys.exit("spm.model not found in HF cache — cannot tokenise offline.")
sp = spm.SentencePieceProcessor(model_file=_hits[0])

pf = pq.ParquetFile(SRC)
print(f"streaming {SRC}  ({pf.metadata.num_rows:,} rows) …", flush=True)

urls, ntok, nchr, stat = [], [], [], []
t0 = time.time()
seen = 0
for batch in pf.iter_batches(batch_size=BATCH,
                             columns=["url_canonical", "fetch_status", "cleaned_text"]):
    u = batch.column("url_canonical").to_pylist()
    fs = batch.column("fetch_status").to_pylist()
    ct = batch.column("cleaned_text").to_pylist()
    capped, chars = [], []
    for t in ct:
        t = t or ""
        chars.append(len(t))
        capped.append(t[:CHAR_CAP])
    pieces = sp.encode(capped)
    for i in range(len(u)):
        urls.append(u[i]); stat.append(fs[i])
        nchr.append(chars[i]); ntok.append(len(pieces[i]) + 2)
    seen += len(u)
    if seen % 3000 < BATCH:
        print(f"  {seen:,}/{pf.metadata.num_rows:,}  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame({"url_canonical": urls, "n_tokens": ntok,
                   "n_chars": nchr, "fetch_status": stat})
# sources_scaleup is unique per url_canonical, but guard anyway.
df = df.drop_duplicates("url_canonical").reset_index(drop=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_parquet(OUT, index=False)

ok = df[df.fetch_status == "ok"]
print(f"\nwrote {OUT}  ({len(df):,} urls; {len(ok):,} ok)  in {time.time()-t0:.0f}s", flush=True)
print("ok sources >512 tokens: %.1f%%  | median tokens %.0f"
      % (100 * np.mean(ok.n_tokens > 512), np.median(ok.n_tokens)), flush=True)
