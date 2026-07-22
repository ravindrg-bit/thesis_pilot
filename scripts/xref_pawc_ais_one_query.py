"""
SECONDARY cross-reference of raw PAWC / AIS for a single query, regenerating every
primitive from raw inputs rather than trusting the stored sentence_detail.

Independent steps:
  1. re-segment the raw answer_text with the pinned regex segmenter      -> sentences
  2. recompute |s_i| = len(sentence.split()) from those sentences        -> word counts
  3. recompute w_i = (N - i + 1)/N                                        -> position weights
  4. RE-RUN the DeBERTa NLI model on the actual fetched source pages      -> e(s_i, c)
  5. apply 1[e >= 0.5], then compute PAWC (Eq.1) and AIS (Eq.3) by hand
  6. compare against the values the VM stored in gold

Only the segmenter and the NLI inference use the project's canonical instruments
(reimplementing them would not reproduce the pinned behaviour); the formula arithmetic
in steps 2,3,5 is done here from scratch. Read-only.
"""
import sys, os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow as pa

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import thesis_config as cfg
from src.attribution import segment_sentences          # pinned regex segmenter
from src import nli_attribution as nli                  # _load_nli, _max_entailment

ENG, QID, RUN = 'chatgpt', 'gb_4ee9cc1110279ca5', 2
TAU = 0.5

def _L(x): return x.tolist() if isinstance(x, np.ndarray) else x

# ---- raw inputs -----------------------------------------------------------
gold = pd.read_parquet('data/scaleup/silver/nli_scaleup_cleaned.parquet')
g = gold[(gold.engine == ENG) & (gold.query_id == QID) & (gold.run_index == RUN)].iloc[0]
stored_sd = _L(g.sentence_detail)

resp = pd.read_parquet('data/scaleup/silver/responses_scaleup.parquet')
answer = resp[(resp.engine == ENG) & (resp.query_id == QID) & (resp.run_index == RUN)]['answer_text'].iloc[0]

cit = pd.read_parquet('data/scaleup/silver/citations_scaleup.parquet')
cm = cit[(cit.engine == ENG) & (cit.query_id == QID) & (cit.run_index == RUN)].sort_values('position')
pos2url = dict(zip(cm['position'], cm['url_canonical']))

dset = ds.dataset('data/scaleup/silver/sources_scaleup.parquet')
stab = dset.to_table(filter=pc.is_in(pc.field('url_canonical'), pa.array(list(pos2url.values()))),
                     columns=['url_canonical', 'fetch_status', 'cleaned_text']).to_pandas()
url2text = dict(zip(stab['url_canonical'], stab['cleaned_text']))
url2status = dict(zip(stab['url_canonical'], stab['fetch_status']))

# positions that were fetched ok = the source set C actually scored
positions = [int(p) for p in cm['position'] if url2status.get(pos2url[int(p)]) == 'ok']

print('=' * 80)
print(f'SECONDARY CROSS-REFERENCE  |  {ENG}__{QID}__r{RUN}')
print(f'stored: pawc_total={g.pawc_total}  ais_rate={g.ais_rate}  ais_supported={int(g.ais_supported_sentences)}')
print('=' * 80)

# ---- step 1: re-segment raw answer_text -----------------------------------
sentences = segment_sentences(answer)
N = len(sentences)
print(f'\n[1] re-segmented answer_text -> N = {N} sentences')
seg_ok = (N == len(stored_sd)) and all(sentences[i] == stored_sd[i]['text'] for i in range(N))
print(f'    sentences identical to stored sentence_detail texts: {seg_ok}')
assert seg_ok, 'segmentation does not reproduce stored sentences'

# ---- step 4 prep: load model once -----------------------------------------
print('\n[4] loading NLI model to regenerate e(s_i, c) ...')
nli._load_nli()
src_window = cfg.JUDGE_SOURCE_CHAR_LIMIT
src_text = {p: (url2text[pos2url[p]] or '')[:src_window] for p in positions}

# ---- steps 2,3,4,5: per-sentence recomputation ----------------------------
print(f'\n{"i":>2} {"|s_i|":>5} {"w_i":>7} ' + ' '.join(f'e@{p:<6}' for p in positions) +
      f' {"#c>=t":>5} {"term":>8}')
pawc = 0.0
supported = 0
max_score_dev = 0.0
for i, sent in enumerate(sentences, start=1):
    wc = len(sent.split())                              # [2] independent word count
    w = (N - i + 1) / N                                 # [3] independent position weight
    # [4] regenerate entailment per source from the page text
    regen = {p: nli._max_entailment(sent, src_text[p]) for p in positions}
    # cross-check against stored scores
    stored_scores = {int(s['pos']): float(s['score']) for s in _L(stored_sd[i - 1]['source_scores'])}
    for p in positions:
        max_score_dev = max(max_score_dev, abs(regen[p] - stored_scores.get(p, np.nan)))
    nc = sum(1 for p in positions if regen[p] >= TAU)   # [5] indicator + count
    term = w * wc * nc
    pawc += term
    if nc >= 1:
        supported += 1
    # integrity checks vs stored primitives
    assert wc == int(stored_sd[i - 1]['wc']), f'word count mismatch at i={i}'
    assert abs(w - float(stored_sd[i - 1]['w'])) < 5e-4, f'weight mismatch at i={i}'
    print(f'{i:>2} {wc:>5} {w:>7.4f} ' + ' '.join(f'{regen[p]:>7.4f}' for p in positions) +
          f' {nc:>5} {term:>8.3f}')

ais = supported / N
print('-' * 80)
print(f'[4] max |regenerated score - stored score| over all (s_i,c) pairs : {max_score_dev:.2e}')
print(f'[5] PAWC (recomputed by hand, Eq.1) = {pawc:.4f}   vs stored pawc_total = {g.pawc_total}')
print(f'[5] AIS  (recomputed by hand, Eq.3) = {supported}/{N} = {ais:.6f}   vs stored ais_rate = {g.ais_rate}')

# ---- final assertions -----------------------------------------------------
assert abs(pawc - g.pawc_total) <= 5.1e-4, 'PAWC mismatch beyond 3-dp rounding'
assert supported == int(g.ais_supported_sentences), 'AIS supported-count mismatch'
assert abs(ais - g.ais_rate) < 1e-9, 'AIS rate mismatch'
print('\nPASS - every primitive regenerated from raw inputs reproduces the VM result.')
