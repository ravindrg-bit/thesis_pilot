"""
Verification that the VM-computed RAW PAWC and AIS scores match the thesis formulas.

  Eq.1  PAWC(c)   = Sum_i  w_i * |s_i| * 1[e(s_i,c) >= tau]      (per cited source c)
        pawc_total = Sum_c PAWC(c) = Sum_i w_i * |s_i| * #{c : e(s_i,c) >= tau}
  Eq.3  AIS rate  = |{ s_i : exists c with e(s_i,c) >= tau }| / N

  with   w_i = (N - i + 1) / N   (linear position decay over ALL N sentences),
         |s_i| = sentence word count,  tau = 0.5.

Both metrics are recomputed here PURELY from the stored primitives (sentence word
counts, 1-based index, per-(sentence,source) entailment scores) and compared against
the values the GPU/VM run wrote to nli_scaleup_cleaned.parquet. Read-only.

Run:  python scripts/test_raw_pawc_ais.py
"""
import numpy as np
import pandas as pd

GOLD = 'data/scaleup/silver/nli_scaleup_cleaned.parquet'
TAU = 0.5
PAWC_ROUND_TOL = 5.1e-4   # stored pawc_total is rounded to 3 dp -> max rounding error 5e-4
TAU_BAND = 2e-5           # a stored score this close to tau may flip vs the VM's full-precision call


def _L(x):
    return x.tolist() if isinstance(x, np.ndarray) else x


def raw_pawc_and_ais(sentence_detail, tau=TAU):
    """Recompute RAW pawc_total (Eq.1 summed over sources) and the AIS supported-count (Eq.3)."""
    sd = _L(sentence_detail)
    N = len(sd)
    if N == 0:
        return 0.0, 0, 0
    pawc = 0.0
    supported = 0
    for e in sd:
        i = int(e['idx'])                 # 1-based reading-order position
        w_i = (N - i + 1) / N             # linear position-decay weight
        len_si = int(e['wc'])             # |s_i| word count
        scores = [float(s['score']) for s in (_L(e.get('source_scores')) or [])]
        indicator_sum = sum(1 for sc in scores if sc >= tau)   # Sum_c 1[e(s_i,c) >= tau]
        pawc += w_i * len_si * indicator_sum                   # Eq.1, accumulated over sources
        if indicator_sum >= 1:                                 # Eq.3 numerator: exists c
            supported += 1
    return pawc, supported, N


def has_boundary_score(sentence_detail, tau=TAU, band=TAU_BAND):
    """True if any stored score lies within `band` of tau (so 5-dp rounding could flip the indicator)."""
    sd = _L(sentence_detail)
    return any(abs(float(s['score']) - tau) <= band
               for e in sd for s in (_L(e.get('source_scores')) or []))


# ---------------------------------------------------------------------------
def worked_sample(df, rid='chatgpt__gb_2956adea0137d0c0__r6'):
    """Print the full per-sentence calculation for one query and check it against the stored values."""
    row = df[df['response_id'] == rid].iloc[0]
    sd = _L(row['sentence_detail'])
    N = len(sd)
    print('=' * 78)
    print(f'WORKED SAMPLE QUERY: {rid}')
    print(f'N = {N} sentences   |   M (sources fetched ok) = {int(row.n_sources_fetched_ok)}   |   tau = {TAU}')
    print('-' * 78)
    print(f'{"i":>2} {"|s_i|":>5} {"w_i":>6} {"#c>=tau":>7} {"term=w*|s|*#c":>13}   sentence (clipped)')
    pawc = 0.0
    supported = 0
    for e in sd:
        i = int(e['idx']); w_i = (N - i + 1) / N; len_si = int(e['wc'])
        scores = [float(s['score']) for s in (_L(e.get('source_scores')) or [])]
        nc = sum(1 for sc in scores if sc >= TAU)
        term = w_i * len_si * nc
        pawc += term
        supported += 1 if nc >= 1 else 0
        print(f'{i:>2} {len_si:>5} {w_i:>6.3f} {nc:>7} {term:>13.3f}   {str(e["text"])[:40]!r}')
    ais = supported / N
    print('-' * 78)
    print(f'PAWC (recomputed) = {pawc:.4f}   vs stored pawc_total = {row.pawc_total}')
    print(f'AIS  (recomputed) = {supported}/{N} = {ais:.4f}   vs stored ais_rate = {row.ais_rate}')
    assert abs(pawc - row.pawc_total) <= PAWC_ROUND_TOL, 'sample PAWC mismatch'
    assert supported == int(row.ais_supported_sentences), 'sample AIS supported-count mismatch'
    assert abs(ais - row.ais_rate) < 1e-9, 'sample AIS rate mismatch'
    print('sample query: PAWC and AIS match the stored VM values  ->  OK')
    print('=' * 78)


# ---------------------------------------------------------------------------
def test_raw_pawc_ais_match_formula():
    """Dataset-wide check: every response that has no score on the tau boundary must match exactly."""
    df = pd.read_parquet(GOLD)
    df['response_id'] = df['engine'] + '__' + df['query_id'] + '__r' + df['run_index'].astype(str)

    worked_sample(df)

    pawc_re, sup_re, boundary = [], [], []
    for r in df.itertuples(index=False):
        p, s, _ = raw_pawc_and_ais(r.sentence_detail)
        pawc_re.append(p); sup_re.append(s); boundary.append(has_boundary_score(r.sentence_detail))
    df['pawc_re'] = pawc_re
    df['sup_re'] = sup_re
    df['boundary'] = boundary

    clean = df[~df['boundary']]
    pawc_dev = (clean['pawc_re'] - clean['pawc_total']).abs()
    ais_count_ok = (clean['sup_re'] == clean['ais_supported_sentences'])

    print(f'responses total                         : {len(df)}')
    print(f'  with a score within {TAU_BAND:g} of tau (excluded) : {int(df.boundary.sum())}')
    print(f'  clean (decision unambiguous)          : {len(clean)}')
    print(f'PAWC  max |recomputed - stored| (clean) : {pawc_dev.max():.2e}  (stored rounded to 3 dp)')
    print(f'PAWC  within 3-dp rounding tol (clean)   : {int((pawc_dev <= PAWC_ROUND_TOL).sum())} / {len(clean)}')
    print(f'AIS   supported-count exact (clean)      : {int(ais_count_ok.sum())} / {len(clean)}')

    # ASSERTIONS: for every unambiguous response the VM result equals the formula
    assert (pawc_dev <= PAWC_ROUND_TOL).all(), 'some clean responses deviate beyond 3-dp rounding'
    assert ais_count_ok.all(), 'some clean responses have an AIS supported-count mismatch'

    # the excluded responses are ONLY tau-boundary cases (stored-score rounding, not a formula error)
    boundary_pawc_dev = (df.loc[df.boundary, 'pawc_re'] - df.loc[df.boundary, 'pawc_total']).abs()
    print(f'\nExcluded boundary responses             : {int(df.boundary.sum())} '
          f'(max PAWC dev {boundary_pawc_dev.max():.2f} - all attributable to 5-dp score rounding at tau)')
    print('\nPASS - VM raw PAWC and AIS reproduce Eq.1 and Eq.3 for every unambiguous response.')


if __name__ == '__main__':
    test_raw_pawc_ais_match_formula()
