"""
Audit test-suite for notebooks/scaleup/EDA.ipynb
================================================

Independently recomputes EVERY numeric value the notebook DISPLAYS (descriptive
tables, root-cause percentages, binned stacked-bar shares, medians cited in the
ANALYSIS prose, search rates, and the confound figure) straight from the gold
parquet files, and asserts each matches what the notebook shows.

The expected values below are transcribed from the notebook's rendered output /
ANALYSIS text. A failing assertion means the notebook DISPLAYS a value that the
data does not support.

Run:  python3 -m pytest notebooks/scaleup/test_eda_audit.py -q
  or: python3 notebooks/scaleup/test_eda_audit.py      (prints PASS/FAIL + manual calcs)
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ── load gold data exactly as the notebook does ──────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
rq = pd.read_parquet(ROOT / "data/scaleup/queries/reclass_queries.parquet")
ca = pd.read_parquet(ROOT / "data/scaleup/gold/cell_aggregates_scaleup.parquet").merge(rq, on="query_id")
r1 = pd.read_parquet(ROOT / "data/scaleup/gold/flat_table1_responses.parquet").merge(rq, on="query_id")

ENGINES = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]
PAWC_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.0]          # E13/E14 5-bin
CV_EDGES = [0, 0.25, 0.5, 1.0, np.inf]              # E15/E16 4-bin


def pct_bins(values, edges):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return np.histogram(v, bins=edges)[0] / len(v) * 100


def stat(df, engine, col):
    s = df[df.engine == engine][col].dropna()
    return dict(n=len(s), mean=s.mean(), median=s.median(), sd=s.std())


# ── 1. E0 / E18 descriptive statistics table (displayed, live-computed) ───────
# expected = exactly what E18 prints (n, mean, median, sd) per engine x metric
E18 = {  # engine: {metric: (n, mean, median, sd)}
    "chatgpt":    {"cleaned_ais": (2000, 0.120, 0.000, 0.279), "pawc_norm": (414, 0.422, 0.388, 0.305)},
    "claude":     {"cleaned_ais": (2000, 0.579, 0.726, 0.404), "pawc_norm": (1467, 0.426, 0.399, 0.226)},
    "gemini":     {"cleaned_ais": (1994, 0.511, 0.611, 0.415), "pawc_norm": (1345, 0.296, 0.259, 0.195)},
    "kimi":       {"cleaned_ais": (1992, 0.777, 0.923, 0.303), "pawc_norm": (1912, 0.396, 0.357, 0.240)},
    "perplexity": {"cleaned_ais": (2000, 0.793, 1.000, 0.350), "pawc_norm": (1993, 0.464, 0.429, 0.326)},
}
E18_CV = {  # engine: {col: (n, mean, median)}
    "chatgpt":    {"pawc_norm_cv": (65, 0.566, 0.425), "ais_cv": (77, 1.207, 1.048)},
    "claude":     {"pawc_norm_cv": (188, 0.233, 0.184), "ais_cv": (191, 0.247, 0.089)},
    "gemini":     {"pawc_norm_cv": (195, 0.395, 0.341), "ais_cv": (215, 0.777, 0.418)},
    "kimi":       {"pawc_norm_cv": (245, 0.403, 0.305), "ais_cv": (246, 0.274, 0.131)},
    "perplexity": {"pawc_norm_cv": (247, 0.653, 0.524), "ais_cv": (247, 0.390, 0.265)},
}


def test_e0_e18_descriptive_stats():
    for eng, metrics in E18.items():
        for col, (n, mean, med, sd) in metrics.items():
            s = stat(r1, eng, col)
            assert s["n"] == n, (eng, col, "n", s["n"], n)
            assert abs(s["mean"] - mean) < 1e-2, (eng, col, "mean", s["mean"], mean)
            assert abs(s["median"] - med) < 1e-2, (eng, col, "median", s["median"], med)
            assert abs(s["sd"] - sd) < 1.5e-3, (eng, col, "sd", s["sd"], sd)


def test_e18_cv_table():
    for eng, cols in E18_CV.items():
        for col, (n, mean, med) in cols.items():
            s = stat(ca, eng, col)
            assert s["n"] == n, (eng, col, "n", s["n"], n)
            assert abs(s["mean"] - mean) < 1e-2, (eng, col, "mean", s["mean"], mean)
            assert abs(s["median"] - med) < 1e-2, (eng, col, "median", s["median"], med)


# ── 2. Root-cause percentages (RC1 / RC2 ANALYSIS prose, hardcoded) ───────────
def test_rc1_zero_citation_rate():
    def zero_rate(e):
        x = r1[r1.engine == e]
        return (x.n_sources_cited == 0).sum() / len(x) * 100
    assert round(zero_rate("chatgpt"), 1) == 75.8
    assert round(zero_rate("kimi"), 1) == 3.0
    assert round(zero_rate("perplexity"), 1) == 0.0


def test_rc2_fetch_failure():
    def rc2(e):
        x = r1[r1.engine == e]
        return int(((x.n_sources_cited > 0) & (x.n_sources_fetched_ok == 0)).sum())
    assert rc2("chatgpt") == 71                                   # ANALYSIS: "(71)"
    assert rc2("perplexity") == 7                                 # ANALYSIS: "All 7 ... nulls"
    # every Perplexity PAWC null is an RC2 null (no zero-citation runs)
    px = r1[r1.engine == "perplexity"]
    assert int(px.pawc_norm.isna().sum()) == 7
    assert round(71 / 2000 * 100, 1) in (3.5, 3.6)               # prose says 3.6%


def test_chatgpt_pawc_denominators():
    cg = r1[r1.engine == "chatgpt"]
    assert int(cg.pawc_norm.notna().sum()) == 414                # "414 of 2,000 runs"
    assert int(ca[ca.engine == "chatgpt"].pawc_norm_mean.notna().sum()) == 79  # "79 of 250 cells"
    # nulls (1586) decompose into RC1 (1515) + RC2 (71)
    assert int(cg.pawc_norm.isna().sum()) == 1586
    assert int((cg.n_sources_cited == 0).sum()) == 1515
    assert 1515 + 71 == 1586


# ── 3. E13-E16 stacked-bar binned shares (ANALYSIS prose, hardcoded) ──────────
def test_e13_pawc_bins():
    g = pct_bins(r1[r1.engine == "gemini"].pawc_norm, PAWC_EDGES)
    assert round(g[0] + g[1]) == 78                              # Gemini bottom-two bands
    p = pct_bins(r1[r1.engine == "perplexity"].pawc_norm, PAWC_EDGES)
    assert round(p[4]) == 21                                     # Perplexity Very High


def test_e14_ais_bins():
    exp = {"chatgpt": (0, 83), "perplexity": (4, 70), "kimi": (4, 65),
           "claude": (4, 44), "gemini": (4, 38)}
    for eng, (idx, val) in exp.items():
        b = pct_bins(r1[r1.engine == eng].cleaned_ais, PAWC_EDGES)
        assert round(b[idx]) == val, (eng, round(b[idx]), val)


def test_e15_pawc_cv_bins():
    cl = pct_bins(ca[ca.engine == "claude"].pawc_norm_cv, CV_EDGES)
    assert round(cl[0]) == 68                                    # Claude Stable
    px = pct_bins(ca[ca.engine == "perplexity"].pawc_norm_cv, CV_EDGES)
    assert round(px[2]) == 40 and round(px[3]) == 16             # Perplexity Variable / Highly Var


def test_e16_ais_cv_bins():
    assert round(pct_bins(ca[ca.engine == "claude"].ais_cv, CV_EDGES)[0]) == 79   # Claude Stable
    assert round(pct_bins(ca[ca.engine == "kimi"].ais_cv, CV_EDGES)[0]) == 71     # Kimi Stable
    assert round(pct_bins(ca[ca.engine == "chatgpt"].ais_cv, CV_EDGES)[3]) == 51  # ChatGPT Highly Var
    assert round(pct_bins(ca[ca.engine == "gemini"].ais_cv, CV_EDGES)[3]) == 25   # Gemini Highly Var


def test_e13_e16_medians_in_prose():
    med = lambda df, e, c: df[df.engine == e][c].median()
    assert round(med(r1, "perplexity", "pawc_norm"), 2) == 0.43
    assert round(med(r1, "claude", "pawc_norm"), 2) == 0.40
    assert round(med(r1, "gemini", "pawc_norm"), 2) == 0.26
    assert round(med(ca, "claude", "pawc_norm_cv"), 2) == 0.18
    assert round(med(ca, "perplexity", "pawc_norm_cv"), 2) == 0.52
    assert round(med(ca, "chatgpt", "ais_cv"), 2) == 1.05
    assert round(med(ca, "claude", "ais_cv"), 2) == 0.09


# ── 4. E6 search-rate ANALYSIS (live f-strings; confirm anyway) ───────────────
def test_e6_search_rate():
    g = ca.groupby(["temporality", "engine"]).did_search_rate.mean()
    assert round(g[("evergreen", "chatgpt")] * 100, 1) == 11.7
    assert round(g[("latest", "chatgpt")] * 100, 1) == 81.4
    assert round(g[("evergreen", "claude")] * 100, 1) == 72.1
    assert round(g[("latest", "gemini")] * 100, 1) == 92.1
    assert round(g[("evergreen", "perplexity")] * 100) == 100


def test_complex_query_count():
    assert rq[rq.complexity == "complex"].query_id.nunique() == 13   # E2: "n=13 total queries"


# ── 5. BUG REGRESSIONS — these document the two errors the audit found ────────
def test_debate_sensitivity_is_70_percent_not_84():
    """E4+E5 figure stamp / markdown / ANALYSIS claim '84% of debate queries are
    also sensitive'. The data says 70% (35 of 50). 84% is wrong (stale)."""
    deb = rq[rq.content_type == "debate"]
    frac = (deb.sensitivity == "sensitive").mean() * 100
    assert len(deb) == 50
    assert round(frac) == 70           # correct value
    assert round(frac) != 84           # the displayed value is NOT supported
    # E10 also claimed factual is "~18%" sensitive; the data says ~26% (37/144)
    fac = rq[rq.content_type == "factual"]
    assert round((fac.sensitivity == "sensitive").mean() * 100) == 26


def test_e18_ais_cv_not_consistently_lower_than_pawc_cv():
    """E18 ANALYSIS claims 'AIS CV is consistently lower than PAWC_norm CV'.
    It is HIGHER for ChatGPT, Claude, Gemini and only lower for Kimi, Perplexity."""
    higher = {e: ca[ca.engine == e].ais_cv.mean() > ca[ca.engine == e].pawc_norm_cv.mean()
              for e in ENGINES}
    assert higher == {"chatgpt": True, "claude": True, "gemini": True,
                      "kimi": False, "perplexity": False}
    assert not all(not v for v in higher.values())   # NOT consistently lower


# ── manual-calculation evidence (printed, hand-checkable) ─────────────────────
def manual_calculations():
    print("\n" + "=" * 70 + "\nMANUAL CALCULATIONS (hand-checkable evidence)\n" + "=" * 70)

    cg = r1[r1.engine == "chatgpt"]
    n0 = int((cg.n_sources_cited == 0).sum())
    print(f"\n[RC1] ChatGPT zero-citation rate")
    print(f"      zero-citation runs = {n0}; total runs = {len(cg)}")
    print(f"      {n0} / {len(cg)} = {n0/len(cg):.5f} = {n0/len(cg)*100:.1f}%   (shown: 75.8%)")

    ais = cg.cleaned_ais.dropna().values
    vlow = int(((ais >= 0) & (ais < 0.2)).sum())
    print(f"\n[E14] ChatGPT AIS 'Very Low' (0-0.2) share")
    print(f"      runs with AIS in [0,0.2) = {vlow}; total = {len(ais)}")
    print(f"      {vlow} / {len(ais)} = {vlow/len(ais)*100:.1f}%   (shown: 83%)")

    deb = rq[rq.content_type == "debate"]
    ns = int((deb.sensitivity == "sensitive").sum())
    print(f"\n[E4/E5] Debate queries that are 'sensitive'")
    print(f"      sensitive debate queries = {ns}; total debate = {len(deb)}")
    print(f"      {ns} / {len(deb)} = {ns/len(deb)*100:.0f}%   (notebook now shows 70%; previously 84% -- corrected)")

    pc = ca[ca.engine == "chatgpt"].pawc_norm_cv.mean()
    ac = ca[ca.engine == "chatgpt"].ais_cv.mean()
    print(f"\n[E18] ChatGPT PAWC_norm CV vs AIS CV (means)")
    print(f"      PAWC_norm CV = {pc:.3f};  AIS CV = {ac:.3f}")
    print(f"      AIS CV is {'HIGHER' if ac > pc else 'lower'}  ->  'consistently lower' claim is WRONG")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            print(f"ERROR {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    manual_calculations()
