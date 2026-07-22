"""
Unit test for the PAWC_norm arithmetic (screenshot Eq. 2):

    PAWC_norm = ( sum_{c in C} PAWC(c) ) / ( M * Omega ),
    Omega = sum_{i=1}^{N} w_i * |s_i|   (every sentence, supported or not),
    M = |C| = number of sources fetched ok.

Two candidate definitions are computed from primitives and BOTH compared to the
stored `pawc_norm`, to determine -- without assuming -- which one the VM actually wrote:

  (A) LITERAL Eq. 2  : Omega and numerator over ALL N sentences, raw weights w_i=(N-i+1)/N.
  (B) CLEANED (impl) : artefacts removed; survivors re-indexed k=1..N*, w*_k=(N*-k+1)/N*;
                       Omega and numerator over survivors only.  (pp_step3_4_cleaned_pawc.py)

is_artifact is copied verbatim from pp_step3_4_cleaned_pawc.py. Read-only.
"""
import re
import numpy as np
import pandas as pd

GOLD = 'data/scaleup/silver/nli_scaleup_cleaned.parquet'
TAU = 0.5
TAU_BAND = 2e-5          # exclude responses with a score this close to tau (5-dp rounding ambiguity)


def is_artifact(text: str) -> bool:                      # verbatim from pp_step3_4_cleaned_pawc.py
    t = text.strip()
    if not t:
        return True
    if re.match(r'^https?://\S+$', t):
        return True
    if t.startswith('#'):
        return True
    if re.match(r'^[-*=_]{3,}\s*$', t):
        return True
    if re.match(r'^(sources?|references?|citations?)\s*:?\s*$', t, re.IGNORECASE):
        return True
    if len(t.split()) <= 3:
        return True
    if re.match(r'^[\#\-\*\d\.\s]{1,10}$', t):
        return True
    return False


def _L(x): return x.tolist() if isinstance(x, np.ndarray) else x


def both_norms(sentence_detail, M):
    """Return (literal_eq2_norm, cleaned_norm, components dict)."""
    sd = _L(sentence_detail)
    N = len(sd)
    # ---------- (A) LITERAL Eq. 2 : all sentences, raw weights ----------
    omega_raw = 0.0
    num_raw = 0.0
    for e in sd:
        i = int(e['idx']); w_i = (N - i + 1) / N; wc = int(e['wc'])
        nsup = sum(1 for s in (_L(e.get('source_scores')) or []) if float(s['score']) >= TAU)
        omega_raw += w_i * wc
        num_raw += w_i * wc * nsup
    norm_raw = (num_raw / (M * omega_raw)) if (M > 0 and omega_raw > 0) else np.nan
    # ---------- (B) CLEANED implementation : survivors, re-indexed ----------
    survivors = [e for e in sd if not is_artifact(e.get('text', ''))]
    n_star = len(survivors)
    omega_cl = 0.0
    num_cl = 0.0
    for k, e in enumerate(survivors, start=1):
        w_star = (n_star - k + 1) / n_star
        wc = int(e['wc'])
        nsup = sum(1 for s in (_L(e.get('source_scores')) or []) if float(s['score']) >= TAU)
        omega_cl += wc * w_star
        num_cl += wc * w_star * nsup
    norm_cl = (num_cl / (M * omega_cl)) if (M > 0 and n_star > 0 and omega_cl > 0) else np.nan
    return norm_raw, norm_cl, dict(N=N, n_star=n_star, omega_raw=omega_raw, omega_cl=omega_cl,
                                   num_raw=num_raw, num_cl=num_cl)


def near_tau(sd):
    return any(abs(float(s['score']) - TAU) <= TAU_BAND
              for e in _L(sd) for s in (_L(e.get('source_scores')) or []))


def worked_sample(df):
    # pick a response with artefacts (so cleaned != literal), M>0, no boundary score
    cand = df[(df.n_artifacts >= 1) & (df.n_sources_fetched_ok > 0) & (df.cleaned_n > 0)
              & (~df.sentence_detail.map(near_tau))].copy()
    cand['rid'] = cand.engine + '__' + cand.query_id + '__r' + cand.run_index.astype(str)
    row = cand.sort_values('rid').iloc[0]
    M = int(row.n_sources_fetched_ok)
    norm_raw, norm_cl, comp = both_norms(row.sentence_detail, M)
    print('=' * 84)
    print(f'WORKED SAMPLE: {row.rid}')
    print(f'  N(all sentences)={comp["N"]}   N*(survivors)={comp["n_star"]}   '
          f'n_artifacts={int(row.n_artifacts)}   M={M}')
    print('-' * 84)
    print('  (A) LITERAL Eq.2  (all sentences, raw weights):')
    print(f'        Omega_raw = sum_i w_i*|s_i|              = {comp["omega_raw"]:.4f}')
    print(f'        numerator = sum PAWC(c)                  = {comp["num_raw"]:.4f}')
    print(f'        PAWC_norm = {comp["num_raw"]:.4f} / ({M} * {comp["omega_raw"]:.4f}) = {norm_raw:.6f}')
    print('  (B) CLEANED impl  (survivors, re-indexed weights):')
    print(f'        Omega*    = sum_k w*_k*|s_k|             = {comp["omega_cl"]:.4f}')
    print(f'        numerator = cleaned_pawc_total           = {comp["num_cl"]:.4f}')
    print(f'        PAWC_norm = {comp["num_cl"]:.4f} / ({M} * {comp["omega_cl"]:.4f}) = {norm_cl:.6f}')
    print('-' * 84)
    print(f'  STORED pawc_norm                                = {row.pawc_norm:.6f}')
    print(f'  STORED cleaned_pawc_total                       = {row.cleaned_pawc_total}')
    print(f'  matches (A) literal Eq.2 : {np.isclose(norm_raw, row.pawc_norm, atol=1e-6)}')
    print(f'  matches (B) cleaned impl : {np.isclose(norm_cl,  row.pawc_norm, atol=1e-6)}')
    print('=' * 84)
    return row.rid


def test_pawc_norm():
    df = pd.read_parquet(GOLD)
    rid = worked_sample(df)

    sub = df[(df.n_sources_fetched_ok > 0) & (df.cleaned_n > 0) & (~df.sentence_detail.map(near_tau))].copy()
    raw_match = cl_match = 0
    cl_dev = []
    for r in sub.itertuples(index=False):
        nr, nc, _ = both_norms(r.sentence_detail, int(r.n_sources_fetched_ok))
        if np.isclose(nr, r.pawc_norm, atol=1e-6): raw_match += 1
        if np.isclose(nc, r.pawc_norm, atol=1e-9): cl_match += 1
        cl_dev.append(abs(nc - r.pawc_norm))
    n = len(sub)
    print(f'\nresponses tested (M>0, N*>0, no boundary score): {n}')
    print(f'  stored pawc_norm == LITERAL Eq.2 (atol 1e-6)  : {raw_match} / {n}')
    print(f'  stored pawc_norm == CLEANED impl (atol 1e-9)  : {cl_match} / {n}')
    print(f'  max |cleaned recompute - stored| (clean set)  : {max(cl_dev):.2e}')

    # the arithmetic the VM stored is the CLEANED definition; assert it reproduces exactly
    assert cl_match == n, 'stored pawc_norm does not match the cleaned definition for some response'
    # and it is demonstrably NOT the literal Eq.2 for most responses with artefacts
    print('\nPASS - stored pawc_norm = cleaned_pawc_total / (M * Omega*) exactly (cleaned definition).')
    print('NOTE - it does NOT equal the literal screenshot Eq.2 (which sums all N sentences with')
    print('       raw weights); the implementation removes artefacts and re-indexes weights over N*.')


if __name__ == '__main__':
    test_pawc_norm()
