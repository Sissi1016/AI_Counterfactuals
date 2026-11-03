"""Extra tests for the appendix's sections on the design (A3) and on memorization (A4).

1. C1's index in the permutation test, by center. 35 (Figure III) runs the permutation test of Kleinberg et al.
   (2018) with C1's index, the algorithm's predicted citations (ml_cites), in its contraction cells (test and
   imputation sets; art unit × year × USPC class with at least five examiners; 35's equal-caseload quintiles; 1,000
   within-cell permutations) and writes only the pooled share of cells with p < 0.05 (c1_kllm_all_permutation.csv).
   This recomputes it from 35's cached predictions (c1_preds_all_all.parquet) with 35's random stream and cell
   order, checks that it reproduces 35's pooled numbers, and keeps the cell-level p-values, so that 41's
   permutation table can show TC 3600 next to the eight centers.
2. The test of Frandsen, Lefgren, and Leslie (2023) as in 39 (39's fll and fll_stat: within each art unit, examiner
   means of granted × citations, net of cell means, against a cubic in the examiner's grant rate; permutation
   calibration with 49 within-cell shuffles of citations among granted applications, 39's seed). 39 uses an
   examiner's applications in an art unit only when there are at least 20 of them (MIN_FLL) and an art unit only
   when at least eight examiners qualify. Women are 9% of first inventors, so few examiners qualify for women's
   applications. Here (a) the distribution of a group's applications per examiner and per examiner × art unit, and
   how many examiners and applications each minimum keeps; (b) 39's test with the minimum lowered to 10 and 5, for
   all, women's, and men's applications (and at 20, which must reproduce 39). Below 20, some examiners' applications
   in an art unit all have the same residualized grant and outcome (for example, all rejected within one cell);
   their sampling variance is zero and 39's statistic becomes infinite. They are left out (the only change to 39's
   fll_stat; at a minimum of 20 no examiner is in that case, so the statistic is 39's). (c) For women only, the
   same test with 499 shuffles (own random stream), since a p-value from 49 shuffles moves in steps of 0.02.
3. Memorization (23): in TC 3600 and TC 1600, the AUC for the grant decision of the zero-shot LLM and of the
   grant-trained text model in the more and the less familiar half of 23's 30,000 scored applications (Min-20% Prob,
   split at the median within each center, as in 23), their difference (LLM − text model) in each half, and the
   difference of that difference across halves; 95% intervals from 1,000 bootstrap draws of applications within each
   center × half (the same draw for both models). If the LLM answered from memory, it would gain on the text model in
   the more familiar half; a text model trained on our data has no memory, so its own change across halves measures
   how much easier that half is.
4. Claims removed in examination (the number of claims granted mixes what was filed with what examination does).
   Claims at filing are counted in the HUPD claim text of
   the published application (all_text.claims; count_claims: claims numbered 1, 2, 3, ... in order, ranges such as
   '1-20.' included, minus those marked canceled or deleted, i.e. the claims pending at publication) and validated
   against the Office Action Research Dataset (the highest claim number named in the first non-final rejection).
   Claims at grant are PatentsView's num_claims (as 42). On leniency × 0.1 within art unit × year × class cells:
   claims at filing for all applications (a balance check, no controls, as 42's balance), and claims at filing, at
   grant, and removed (filing − grant) for granted patents with 42's exclusion controls and clustering; the grant
   coefficient is the filing coefficient minus the removed one, which separates selection (what lenient examiners'
   grants were filed with) from examination. The same on applications published before the examiner's first action
   (PatEx), whose published claims cannot reflect examination (37 CFR 1.215 lets a publication include claim
   amendments filed before its preparation).

Runs on CPUs only: 4 single-threaded processes for 2 (about 50 minutes, most of it the 499-shuffle runs), about 5
minutes for 4, the rest under a minute. Run under taskset -c 0-15. Parts: --c1perm, --fll, --mem, --claims (default:
all four); --support only 2a; --report only rewrites the outputs from the csvs.
Input: data/processed/{all_text,all_sample,c1_preds_all_all,memorization,citation_scores,patex_transactions}.parquet,
       the record of cfe.check (pilot), data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip,
       output/tables/{c1_kllm_all_permutation,c3_checks_fll,memorization_auc_by_half}.csv;
       code/18 (f_stat), 35 (quintiles), 39 (load, fll, fll_stat), 42 (load: sample, claims at grant, controls)
Output: output/tables/tx_{c1_permutation,c1_permutation_cells,fll,fll_support,fll_art_units,fll_female_b499,
        memorization,claims,claims_validation,claims_validation_dist}.csv,
        output/latex/tx_tab_{fll,memorization}.tex, output/latex/design_tab_claims.tex,
        output/latex/tx_numbers.tex (macros \\xTx...), output/tx.md
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_v] = "1"

import functools  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
T, TEX = OUT / "tables", OUT / "latex"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
# CFE_LOSS=poisson: part 1 only, on 35's Poisson variant (c1_preds_all_poisson_all.parquet, checked against
# c1_kllm_all_poisson_permutation.csv), written to tx_c1_permutation_poisson{,_cells}.csv; nothing else is rewritten
PSFX = "_poisson" if os.environ.get("CFE_LOSS", "squared_error") == "poisson" else ""
JOBS = 4
SEED = 20261101
THRESHOLDS = [20, 10, 5]   # 39's MIN_FLL and the two lower minimums
B_EXTRA = 499              # women's test with more shuffles (2c)
B_MEM = 1000
GROUPS = [("All", None), ("Female", 1), ("Male", 0)]
TH_WORD = {20: "Twenty", 10: "Ten", 5: "Five"}
TC_WORD = {"1600": "OneSix", "3600": "ThreeSix"}
G_WORD = {"All": "All", "Female": "Fem", "Male": "Male"}
OA_ZIP = ROOT / "data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. C1's index in the permutation test

def c1_permutation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """35's Figure III permutation test, cell by cell (35's run_tc: ev, quintiles, prng = SEED + 7, sorted cells)."""
    m35 = load_module("m35", "35_c1_kllm.py")
    f_stat = load_module("m18", "18_assignment_checks.py").f_stat
    # 35's load_all() row order (all_text keys, sorted by application number) with the columns the test uses
    keys = pd.read_parquet(P / "all_text.parquet", columns=["application_number"])
    d = keys.merge(pd.read_parquet(P / "all_sample.parquet",
                                   columns=["application_number", "tech_center", "au_class", "examiner_id",
                                            "lenient_grant", "granted"]), on="application_number")
    d = d.sort_values("application_number").reset_index(drop=True)
    pr = pd.read_parquet(P / f"c1_preds_all{PSFX}_all.parquet", columns=["application_number", "part", "ml_cites"])
    x = d.merge(pr, on="application_number", how="inner")
    assert len(x) == len(d) == len(pr)
    ev_all = x[x.part.isin(["test", "impute"])].copy()
    e = ev_all.copy()
    e["q"] = m35.quintiles(e)
    e = e[e.q.notna()]
    prng = np.random.default_rng(m35.SEED + 7)
    rows = []
    for c, z in e.groupby(m35.CELL, sort=True):
        assert z.tech_center.nunique() == 1
        gq = z.q.to_numpy(int) - 1
        yy = z.ml_cites.to_numpy(float)
        f0 = f_stat(yy, gq, m35.Q)[0]
        fp = f_stat(yy, gq[np.argsort(prng.random((m35.M, len(gq))), axis=1)], m35.Q)
        rows.append(dict(tech_center=z.tech_center.iloc[0], cell=c, n=len(z), examiners=z.examiner_id.nunique(),
                         p=(1 + (fp >= f0 - 1e-12).sum()) / (1 + m35.M)))
    cells = pd.DataFrame(rows)
    summ = []
    for smp, units in [("All centers", TCS)] + [(tc, [tc]) for tc in TCS]:
        s = cells[cells.tech_center.isin(units)]
        summ.append(dict(sample=smp, cells_tested=len(s), apps_tested=int(s.n.sum()),
                         apps_evaluated=int(ev_all.tech_center.isin(units).sum()),
                         apps_covered=s.n.sum() / ev_all.tech_center.isin(units).sum(),
                         mean_examiners=s.examiners.mean(), p_below_05=(s.p < 0.05).mean(),
                         p_below_10=(s.p < 0.10).mean(), mean_p=s.p.mean()))
    summ = pd.DataFrame(summ)
    o = pd.read_csv(T / f"c1_kllm_all{PSFX}_permutation.csv").iloc[0]
    a = summ.iloc[0]
    ok = (int(o.cells) == a.cells_tested) and np.isclose(o.share_p_below_05, a.p_below_05) and np.isclose(
        o.share_p_below_10, a.p_below_10)
    check = (f"35's c1_kllm_all{PSFX}_permutation.csv: {int(o.cells):,} cells, {100 * o.share_p_below_05:.2f}% p<0.05, "
             f"{100 * o.share_p_below_10:.2f}% p<0.10; recomputed: {a.cells_tested:,} cells, "
             f"{100 * a.p_below_05:.2f}%, {100 * a.p_below_10:.2f}% -> {'reproduced' if ok else 'NOT reproduced'}")
    print(check, flush=True)
    if not ok:
        sys.exit("C1 permutation test not reproduced")
    return summ, cells


# ---------------------------------------------------------------- 2. Frandsen, Lefgren, and Leslie

def fll_stat(x: pd.DataFrame, min_n: int, info: dict | None = None) -> tuple[float, int, float, int]:
    """39's fll_stat with the minimum as an argument; examiners whose applications in the art unit all have the same
    residualized grant and outcome (zero sampling variance) are left out before the fit."""
    x = x.assign(dd=x.granted.astype(float))
    for v in ["y", "dd"]:
        x[v + "_r"] = x[v] - x.groupby("au_class")[v].transform("mean")
    stat, df, slope_min, n_ex, n_const = 0.0, 0, np.inf, 0, 0
    for _, z in x.groupby("art_unit"):
        e = z.groupby("examiner_id").agg(n=("y_r", "size"), p=("dd_r", "mean"), y=("y_r", "mean"),
                                         ry=("y_r", lambda v: v.max() - v.min()),
                                         rd=("dd_r", lambda v: v.max() - v.min()))
        e = e[e.n >= min_n]
        const = (e.ry < 1e-9) & (e.rd < 1e-9)
        n_const += int(const.sum())
        e = e[~const]
        if len(e) < 8:
            continue
        V = np.vander(e.p.to_numpy(), 4, increasing=True)
        w = e.n.to_numpy(float)
        beta = np.linalg.lstsq(V * np.sqrt(w)[:, None], e.y.to_numpy() * np.sqrt(w), rcond=None)[0]
        slope = beta[1] + 2 * beta[2] * e.p.to_numpy() + 3 * beta[3] * e.p.to_numpy() ** 2
        zz = z[z.examiner_id.isin(e.index)]
        u = zz.y_r - zz.examiner_id.map(pd.Series(slope, index=e.index)) * zz.dd_r
        var = u.groupby(zz.examiner_id).var(ddof=1).reindex(e.index).to_numpy() / e.n.to_numpy()
        stat += float(((e.y.to_numpy() - V @ beta) ** 2 / var).sum())
        df += len(e) - 4
        n_ex += len(e)
        grid = np.linspace(np.percentile(e.p, 5), np.percentile(e.p, 95), 25)
        slope_min = min(slope_min, float((beta[1] + 2 * beta[2] * grid + 3 * beta[3] * grid ** 2).min()))
    if info is not None:
        info["const"] = n_const
    return stat, df, slope_min, n_ex


def fll_run(min_n: int) -> pd.DataFrame:
    """39's fll() (its seed, its 49 shuffles, its groups) with fll_stat replaced by the version above."""
    warnings.filterwarnings("ignore")
    m39 = load_module("m39", "39_c3_checks.py")
    d = m39.load()
    m39.fll_stat = functools.partial(fll_stat, min_n=min_n)
    t0 = time.time()
    r = m39.fll(d)
    info = {}
    for gname, sel in GROUPS:   # examiners left out for zero variance in the observed data
        x = d[d.lenient_grant.notna()] if sel is None else d[d.lenient_grant.notna() & (d.female == sel)]
        g = (x.granted == 1).to_numpy()
        fll_stat(x.assign(y=np.where(g, x.cites_ns.fillna(0).to_numpy(float), 0.0)), min_n, info)
        r.loc[r.group == gname, "left_out_zero_variance"] = info["const"]
    print(f"FLL, minimum {min_n}: {time.time() - t0:.0f}s", flush=True)
    return r.assign(min_apps=min_n, shuffles=m39.B_FLL)


def fll_female_extra(min_n: int, B: int) -> dict:
    """Women's applications only, as 39's fll() but with B shuffles from an own random stream."""
    warnings.filterwarnings("ignore")
    m39 = load_module("m39", "39_c3_checks.py")
    d = m39.load()
    rng = np.random.default_rng(SEED + min_n)
    x = d[d.lenient_grant.notna() & (d.female == 1)].reset_index(drop=True)
    g = (x.granted == 1).to_numpy()
    cites = x.cites_ns.fillna(0).to_numpy(float)
    stat, df, slope_min, n_ex = fll_stat(x.assign(y=np.where(g, cites, 0.0)), min_n)
    cell = x.au_class.to_numpy()
    idx = np.where(g)[0]
    order = idx[np.argsort(cell[idx], kind="mergesort")]
    bounds = np.flatnonzero(np.r_[True, cell[order][1:] != cell[order][:-1], True])
    null = []
    for _ in range(B):
        y = np.zeros(len(x))
        perm = order.copy()
        for a_, b_ in zip(bounds[:-1], bounds[1:]):
            perm[a_:b_] = rng.permutation(order[a_:b_])
        y[order] = cites[perm]
        null.append(fll_stat(x.assign(y=y), min_n)[0])
    null = np.array(null)
    return dict(group="Female", min_apps=min_n, shuffles=B, examiners=n_ex, chi2=stat, df=df, ratio=stat / df,
                null_ratio=float(np.mean(null) / df), null_median_ratio=float(np.median(null) / df),
                p=(1 + (null >= stat).sum()) / (1 + B))


def fll_support() -> pd.DataFrame:
    """A group's applications per examiner and per examiner × art unit; what each minimum keeps (39's rules: at least
    min applications of the group in the art unit, at least eight such examiners in the art unit)."""
    m39 = load_module("m39", "39_c3_checks.py")
    d = m39.load()
    x = d[d.lenient_grant.notna()]
    rows = []
    for gname, sel in GROUPS:
        z = x if sel is None else x[x.female == sel]
        pe = z.groupby("examiner_id").size()
        pa = z.groupby(["art_unit", "examiner_id"]).size()
        base = dict(group=gname, applications=len(z), share_of_sample=len(z) / len(x), examiners=len(pe),
                    examiner_au_pairs=len(pa), per_examiner_mean=pe.mean(), per_examiner_median=pe.median(),
                    per_examiner_p90=pe.quantile(0.9), per_examiner_p99=pe.quantile(0.99),
                    per_pair_median=pa.median(), per_pair_p90=pa.quantile(0.9), per_pair_p99=pa.quantile(0.99))
        for t in THRESHOLDS:
            ok = pa[pa >= t]
            nau = ok.groupby(level=0).size()
            au = nau.index[nau >= 8]
            kept = ok[ok.index.get_level_values(0).isin(au)]
            rows.append(dict(base, min_apps=t, pairs_at_min=len(ok), pairs_at_min_share=len(ok) / len(pa),
                             examiners_at_min=ok.index.get_level_values(1).nunique(),
                             examiners_at_min_share=ok.index.get_level_values(1).nunique() / len(pe),
                             art_units_with_8=len(au), pairs_used=len(kept), apps_used=int(kept.sum()),
                             apps_used_share=kept.sum() / len(z)))
            if gname == "Female" and t == THRESHOLDS[0]:   # the art units behind 39's women's test
                units = [dict(art_unit=a, tech_center=x.loc[x.art_unit == a, "tech_center"].iloc[0],
                              pairs_used=int(nau[a]), female_apps_used=int(kept.loc[a].sum()),
                              female_apps_in_art_unit=int((z.art_unit == a).sum()),
                              female_share_in_art_unit=float(x.loc[x.art_unit == a, "female"].mean()))
                         for a in au]
    return pd.DataFrame(rows), pd.DataFrame(units)


# ---------------------------------------------------------------- 3. memorization

def auc_w(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUC (ties count one half): the Mann–Whitney statistic with bootstrap multiplicities as weights."""
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(s)]))
    wp = np.bincount(grp, weights=w * y)
    wn = np.bincount(grp, weights=w * (1 - y))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def memorization() -> pd.DataFrame:
    """23's halves and AUCs, with bootstrap intervals for the LLM − text model difference and its change."""
    from sklearn.metrics import roc_auc_score
    from cfe.check import load_record
    d = load_record(ROOT)
    d = d.merge(pd.read_parquet(P / "memorization.parquet"), on="application_number", how="left")
    sc = pd.read_parquet(P / "citation_scores.parquet", columns=["application_number", "LLM, zero-shot",
                                                                  "Train on grant"])
    d = d.merge(sc, on="application_number", how="left")
    d["familiar"] = np.nan
    for tc in ["3600", "1600"]:
        m = (d.tech_center == tc) & d.min_k.notna()
        d.loc[m, "familiar"] = (d.loc[m, "min_k"] > d.loc[m, "min_k"].median()).astype(float)
    old = pd.read_csv(T / "memorization_auc_by_half.csv", dtype={"tech_center": str})
    rng = np.random.default_rng(SEED + 3)
    rows, draws = [], {}
    for tc in ["3600", "1600"]:
        for half, lab in [(1.0, "More familiar half"), (0.0, "Less familiar half")]:
            x = d[(d.tech_center == tc) & (d.familiar == half)]
            y = x.granted.to_numpy(float)
            a = x["LLM, zero-shot"].to_numpy(float)
            b = x["Train on grant"].to_numpy(float)
            a0, b0 = roc_auc_score(y, a), roc_auc_score(y, b)
            o = old[(old.tech_center == tc) & (old.half == lab)].iloc[0]
            assert len(x) == o.n and np.isclose(a0, o.auc_llm_grant) and np.isclose(b0, o.auc_grant_model), \
                f"23's AUCs not reproduced ({tc}, {lab})"
            assert np.isclose(auc_w(y, a, np.ones(len(y))), a0) and np.isclose(auc_w(y, b, np.ones(len(y))), b0)
            W = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=B_MEM).astype(float)
            da = np.array([auc_w(y, a, w) for w in W])
            db = np.array([auc_w(y, b, w) for w in W])
            draws[(tc, half)] = (da, db)
            rows.append(dict(tech_center=tc, half=lab, n=len(x), granted=y.mean(), auc_llm=a0, auc_text=b0,
                             diff=a0 - b0, auc_llm_lo=np.percentile(da, 2.5), auc_llm_hi=np.percentile(da, 97.5),
                             auc_text_lo=np.percentile(db, 2.5), auc_text_hi=np.percentile(db, 97.5),
                             diff_lo=np.percentile(da - db, 2.5), diff_hi=np.percentile(da - db, 97.5)))
        r1, r0 = rows[-2], rows[-1]
        (a1, b1), (a0_, b0_) = draws[(tc, 1.0)], draws[(tc, 0.0)]
        for k, est, v in [("diff", r1["diff"] - r0["diff"], (a1 - b1) - (a0_ - b0_)),
                          ("auc_llm", r1["auc_llm"] - r0["auc_llm"], a1 - a0_),
                          ("auc_text", r1["auc_text"] - r0["auc_text"], b1 - b0_)]:
            rows.append(dict(tech_center=tc, half="More − less familiar", n=r1["n"] + r0["n"], **{
                k: est, f"{k}_lo": np.percentile(v, 2.5), f"{k}_hi": np.percentile(v, 97.5)}))
    t = pd.DataFrame(rows)
    return t.groupby(["tech_center", "half"], sort=False).first().reset_index()


# ---------------------------------------------------------------- 4. claims removed in examination

# a claim starts with its number (or a range of numbers, '1-20.' or '1.-20.') and a period; 1) and 1: as a fallback
CLAIM_START = [re.compile(r"(?<![\w,.\-/])(\d{1,3})(?:\s*\.?\s*(?:-|–|—|to|through)\s*(\d{1,3}))?\s*\.(?!\d)"),
               re.compile(r"(?<![\w,.\-/(])(\d{1,3})(?:\s*(?:-|–|—|to|through)\s*(\d{1,3}))?\s*[):](?!\d)")]
CANCELED = re.compile(r"\s*[\(\[]?\s*(?:cancel+ed|deleted)", re.I)


def count_claims(t: str) -> tuple[int, int]:
    """Claims of a published application's claim text: the highest number reached by claims numbered 1, 2, 3, ...
    in order (a number that is not the next one, such as 'claim 1.' inside claim 4, is skipped; a range '1-20.'
    covers its numbers), and how many of them are marked canceled or deleted (preliminary amendments). (0, 0) if no
    numbered claim is found."""
    for pat in CLAIM_START:
        exp, canc = 1, 0
        for m in pat.finditer(t):
            a = int(m.group(1))
            if a != exp:
                continue
            b = int(m.group(2)) if m.group(2) else a
            if b < a:
                continue
            if CANCELED.match(t, m.end()):
                canc += b - a + 1
            exp = b + 1
        if exp > 1:
            return exp - 1, canc
    return 0, 0


def claims_at_filing() -> pd.DataFrame:
    """Claims of every main-sample application in its HUPD text (the published application)."""
    t = pd.read_parquet(P / "all_text.parquet", columns=["application_number", "claims"])
    r = np.array([count_claims(x or "") for x in t.claims], dtype=float)
    out = pd.DataFrame({"application_number": t.application_number, "claims_highest": r[:, 0],
                        "claims_canceled": r[:, 1]})
    out.loc[out.claims_highest == 0, ["claims_highest", "claims_canceled"]] = np.nan
    out["claims_filed"] = out.claims_highest - out.claims_canceled   # pending at publication
    out.loc[out.claims_filed <= 0, "claims_filed"] = np.nan
    return out


def oa_first_action(apps: pd.Series) -> pd.DataFrame:
    """Office Action Research Dataset: each application's first non-final rejection (CTNF, earliest mail date) and the
    highest claim number named in its rejections."""
    keep = set(apps.astype(str))
    oa, rej = [], []
    with zipfile.ZipFile(OA_ZIP) as z:
        with z.open("office_actions.csv") as f:
            for c in pd.read_csv(f, usecols=["app_id", "ifw_number", "document_cd", "mail_dt"], dtype=str,
                                 chunksize=1_000_000):
                oa.append(c[(c.document_cd == "CTNF") & c.app_id.isin(keep)])
        oa = pd.concat(oa).sort_values(["app_id", "mail_dt", "ifw_number"]).drop_duplicates("app_id")
        first = set(oa.ifw_number)
        with z.open("rejections.csv") as f:
            for c in pd.read_csv(f, usecols=["app_id", "ifw_number", "claim_numbers"], dtype=str,
                                 chunksize=1_000_000):
                c = c[c.ifw_number.isin(first)]
                c = c.assign(mx=c.claim_numbers.fillna("").str.findall(r"\d+").map(
                    lambda v: max((int(i) for i in v if len(i) <= 3), default=np.nan)))
                rej.append(c[["ifw_number", "mx"]])
    rej = pd.concat(rej).groupby("ifw_number").mx.max()
    oa["oa_max_claim"] = oa.ifw_number.map(rej)
    return oa.rename(columns={"app_id": "application_number"})[["application_number", "mail_dt", "oa_max_claim"]]


def publication_timing(apps: pd.Series) -> pd.DataFrame:
    """PatEx: date of the pre-grant publication (PG-ISSUE) and of the examiner's first action (non-final or final
    rejection, restriction, Quayle action, miscellaneous action, or notice of allowance). A publication can include
    claim amendments filed before its preparation (37 CFR 1.215), including replies to an early action; claims of an
    application published before any action are the claims as filed or preliminarily amended."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads = 4; SET memory_limit = '40GB'")
    con.register("apps", pd.DataFrame({"application_number": apps.astype(str)}))
    t = con.execute(f"""
    SELECT a.application_number,
           min(t.recorded_date) FILTER (WHERE t.event_code = 'PG-ISSUE') AS pub_date,
           min(t.recorded_date) FILTER (WHERE t.event_code IN ('CTNF', 'CTFR', 'CTRS', 'CTEQ', 'CTMS', 'MN/=.', 'N/=.'))
             AS first_action_date
    FROM apps a LEFT JOIN '{P}/patex_transactions.parquet' t USING (application_number) GROUP BY 1""").df()
    pub, fa = pd.to_datetime(t.pub_date), pd.to_datetime(t.first_action_date)
    t["published_before_action"] = (pub.notna() & (fa.isna() | (pub < fa))).astype(float)
    t.loc[pub.isna(), "published_before_action"] = np.nan
    return t[["application_number", "published_before_action"]]


def claims() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Claims at filing (HUPD text), at grant (PatentsView, 42's extras), and removed in examination, on leniency within
    art unit × year × class cells: 42's exclusion regression (granted patents; small entity, female first inventor,
    inventors; errors clustered by examiner; coefficient × 0.1) and 42's balance regression (all applications, no
    controls). Pooled over the eight centers and by center."""
    import pyfixest as pf
    m42 = load_module("m42", "42_design_balance.py")
    d = m42.load()
    d = d.merge(claims_at_filing(), on="application_number", how="left", validate="1:1")
    d = d.merge(publication_timing(d.application_number), on="application_number", how="left", validate="1:1")
    d["claims_removed"] = d.claims_filed - d.num_claims
    d["claims_removed_highest"] = d.claims_highest - d.num_claims
    v = oa_first_action(d.application_number)
    x = d[["application_number", "claims_highest", "claims_filed"]].merge(v, on="application_number")
    x = x[x.oa_max_claim.notna() & x.claims_highest.notna()]
    val = dict(apps=len(d), parsed=float(d.claims_highest.notna().mean()), with_oa=len(x),
               equal=float((x.oa_max_claim == x.claims_highest).mean()),
               below=float((x.oa_max_claim < x.claims_highest).mean()),
               above=float((x.oa_max_claim > x.claims_highest).mean()),
               within2=float(((x.oa_max_claim - x.claims_highest).abs() <= 2).mean()),
               corr=float(np.corrcoef(x.oa_max_claim, x.claims_highest)[0, 1]),
               canceled_any=float((d.claims_canceled > 0).mean()), mean_filed=float(d.claims_filed.mean()),
               median_filed=float(d.claims_filed.median()),
               pub_known=float(d.published_before_action.notna().mean()),
               published_before_action=float(d.published_before_action.mean()))
    ctl = list(m42.CONTROLS)
    rows = []
    specs = [("filed_all", "claims_filed", "Claims at filing (all applications)", False, False),
             ("filed_granted", "claims_filed", "Claims at filing (granted patents)", True, True),
             ("granted", "num_claims", "Claims at grant", True, True),
             ("removed", "claims_removed", "Claims removed in examination (filing - grant)", True, True),
             ("filed_highest_granted", "claims_highest", "Highest claim number at filing (granted patents)", True,
              True),
             ("removed_highest", "claims_removed_highest", "Highest claim number at filing - claims at grant", True,
              True),
             ("granted_42", "num_claims", "Claims at grant (42's exclusion sample)", True, True)]
    pre = " (published before the first action)"
    specs += [(k + "_pre", y, lab + pre, go, c) for k, y, lab, go, c in specs[:4]]
    for sname, units in m42.DESIGN.samples:
        base = d[d.tech_center.isin(units) & d.lenient_grant.notna()]
        g = base[(base.granted == 1) & base.num_claims.notna() & base.claims_filed.notna()]
        g42 = base[(base.granted == 1)].dropna(subset=["num_claims", "pendency_years"])
        for key, y, lab, granted_only, controls in specs:
            s = g42 if key == "granted_42" else g if granted_only else base[base.claims_filed.notna()]
            if key.endswith("_pre"):
                s = s[s.published_before_action == 1]
            f = f"{y} ~ lenient_grant" + (" + " + " + ".join(ctl) if controls else "") + " | au_class"
            fit = pf.feols(f, data=s, vcov={"CRV1": "examiner_id"})
            rows.append(dict(sample=sname, key=key, variable=lab, mean=s[y].mean(),
                             coef_per_10pp=0.1 * fit.coef()["lenient_grant"], se=0.1 * fit.se()["lenient_grant"],
                             n=fit._N))
    return pd.DataFrame(rows), x, val


# ---------------------------------------------------------------- formatting and report

def num(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def br(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{num(lo, nd)}, {num(hi, nd)}]"


def plain(s: str) -> str:
    return s.replace("$-$", "−").replace(r"\%", "%")


def report() -> None:
    N, L = {}, ["# Extra tests for A3 and A4", "", "Generated by code/46_tests_extra.py.", ""]

    # 1. C1 permutation
    f = T / "tx_c1_permutation.csv"
    if f.exists():
        s = pd.read_csv(f, dtype={"sample": str}).set_index("sample")
        a, b = s.loc["All centers"], s.loc["3600"]
        bc = s.drop(index="All centers")
        o = pd.read_csv(T / "c1_kllm_all_permutation.csv").iloc[0]
        same = int(o.cells) == int(a.cells_tested) and np.isclose(o.share_p_below_05, a.p_below_05) and np.isclose(
            o.share_p_below_10, a.p_below_10)
        check = (f"Pooled, this reproduces 35's c1_kllm_all_permutation.csv ({int(o.cells):,} cells, "
                 f"{100 * o.share_p_below_05:.2f}% with p < 0.05, {100 * o.share_p_below_10:.2f}% with p < 0.10): "
                 f"{'yes' if same else 'NO'}.")
        N.update(PermCOne=num(100 * a.p_below_05, 1), PermCOneCells=f"{int(a.cells_tested):,}",
                 PermCOneCover=num(100 * a.apps_covered, 0), PermCOneEvaluated=f"{int(a.apps_evaluated):,}",
                 PermCOneThreeSix=num(100 * b.p_below_05, 1), PermCOneThreeSixCells=f"{int(b.cells_tested):,}",
                 PermCOneMax=num(100 * bc.p_below_05.max(), 1), PermCOneMaxTC=bc.p_below_05.idxmax(),
                 PermCOneMin=num(100 * bc.p_below_05.min(), 1), PermCOneMinTC=bc.p_below_05.idxmin())
        L += ["## 1. C1's index (predicted citations) in the permutation test, by center", "",
              "35's Figure III test (test and imputation sets, cells with at least five examiners, 35's quintiles, "
              "1,000 permutations), recomputed cell by cell from 35's cached predictions. " + check, "",
              "| Sample | Cells tested | Applications in them | Share of the test and imputation sets | "
              "Examiners per cell (mean) | p < 0.05 | p < 0.10 | Mean p |", "|---|---|---|---|---|---|---|---|"]
        for r in s.reset_index().itertuples():
            L.append(f"| {r.sample} | {r.cells_tested:,} | {r.apps_tested:,} | {100 * r.apps_covered:.1f}% | "
                     f"{r.mean_examiners:.1f} | {100 * r.p_below_05:.1f}% | {100 * r.p_below_10:.1f}% | "
                     f"{r.mean_p:.2f} |")
        L.append("")

    # 2. FLL
    f, fs, fx = T / "tx_fll.csv", T / "tx_fll_support.csv", T / "tx_fll_female_b499.csv"
    if fs.exists():
        sp = pd.read_csv(fs)
        L += ["## 2. Frandsen, Lefgren, and Leslie (2023): why 36 examiners for women's applications", "",
              "39's test uses an examiner's applications of the group in an art unit only if there are at least the "
              "minimum (39: 20), and an art unit only if at least eight examiners qualify. 'Examiners' in 39's table "
              "are examiner × art unit pairs.", "",
              "| Group | Applications | Share | Examiners | Per examiner: mean / median / p90 / p99 | "
              "Per examiner × art unit: median / p90 / p99 |", "|---|---|---|---|---|---|"]
        for r in sp.drop_duplicates("group").itertuples():
            L.append(f"| {r.group} | {r.applications:,} | {100 * r.share_of_sample:.1f}% | {r.examiners:,} | "
                     f"{r.per_examiner_mean:.1f} / {r.per_examiner_median:.0f} / {r.per_examiner_p90:.0f} / "
                     f"{r.per_examiner_p99:.0f} | {r.per_pair_median:.0f} / {r.per_pair_p90:.0f} / "
                     f"{r.per_pair_p99:.0f} |")
        L += ["", "| Group | Minimum | Examiner × art unit pairs at the minimum | Examiners at the minimum | "
              "Art units with at least 8 | Pairs used | Group's applications used |", "|---|---|---|---|---|---|---|"]
        for r in sp.itertuples():
            L.append(f"| {r.group} | {r.min_apps} | {r.pairs_at_min:,} ({100 * r.pairs_at_min_share:.1f}%) | "
                     f"{r.examiners_at_min:,} ({100 * r.examiners_at_min_share:.1f}%) | {r.art_units_with_8} | "
                     f"{r.pairs_used:,} | {r.apps_used:,} ({100 * r.apps_used_share:.1f}%) |")
        L.append("")
        fe = sp[sp.group == "Female"].set_index("min_apps")
        al = sp[sp.group == "All"].set_index("min_apps")
        N.update(FemShare=num(100 * fe.share_of_sample.iloc[0], 1), FemApps=f"{int(fe.applications.iloc[0]):,}",
                 FemPerEx=num(fe.per_examiner_mean.iloc[0], 1), FemPerExMedian=f"{fe.per_examiner_median.iloc[0]:.0f}",
                 FemPerPairMedian=f"{fe.per_pair_median.iloc[0]:.0f}", FemPerPairNinety=f"{fe.per_pair_p90.iloc[0]:.0f}",
                 AllPerPairMedian=f"{al.per_pair_median.iloc[0]:.0f}", AllPerEx=num(al.per_examiner_mean.iloc[0], 1))
        for t in THRESHOLDS:
            w = TH_WORD[t]
            N[f"FemPairs{w}"] = f"{int(fe.loc[t, 'pairs_at_min']):,}"
            N[f"FemPairs{w}Share"] = num(100 * fe.loc[t, "pairs_at_min_share"], 1)
            N[f"FemAU{w}"] = str(int(fe.loc[t, "art_units_with_8"]))
            N[f"FemUsed{w}"] = f"{int(fe.loc[t, 'pairs_used']):,}"
            N[f"FemAppsUsed{w}"] = f"{int(fe.loc[t, 'apps_used']):,}"
            N[f"FemAppsUsed{w}Share"] = num(100 * fe.loc[t, "apps_used_share"], 0)
            N[f"AllAppsUsed{w}Share"] = num(100 * al.loc[t, "apps_used_share"], 0)
            N[f"AllAU{w}"] = str(int(al.loc[t, "art_units_with_8"]))
    if f.exists():
        fl = pd.read_csv(f)
        old = pd.read_csv(T / "c3_checks_fll.csv").set_index("group")
        r20 = fl[fl.min_apps == 20].set_index("group")
        same = all(np.isclose(r20.loc[g, c], old.loc[g, c]) for g in old.index for c in ["chi2", "p", "null_ratio"]) \
            and all(int(r20.loc[g, "examiners"]) == int(old.loc[g, "examiners"]) for g in old.index)
        L += ["### 39's test with lower minimums (49 shuffles, 39's seed)", "",
              f"At a minimum of 20 this reproduces 39's c3_checks_fll.csv: {'yes' if same else 'NO'}.", "",
              "| Minimum | Group | Examiner × art unit pairs | Left out (zero variance) | Misfit / df | "
              "Same, null (mean) | p |", "|---|---|---|---|---|---|---|"]
        for r in fl.itertuples():
            L.append(f"| {r.min_apps} | {r.group} | {r.examiners:,} | {int(r.left_out_zero_variance):,} | "
                     f"{r.ratio:.2f} | {r.null_ratio:.2f} | {r.p:.2f} |")
        L.append("")
        for r in fl.itertuples():
            k = f"Fll{G_WORD[r.group]}{TH_WORD[r.min_apps]}"
            N[k + "N"], N[k + "P"], N[k + "Ratio"] = f"{r.examiners:,}", num(r.p, 2), num(r.ratio, 2)
            N[k + "Null"] = num(r.null_ratio, 2)
        rows = []
        for t in THRESHOLDS:
            z = fl[fl.min_apps == t].set_index("group")
            fe_used = (pd.read_csv(fs).set_index(["group", "min_apps"]).loc[("Female", t), "apps_used_share"]
                       if fs.exists() else np.nan)
            rows.append([str(t)] + [f"{int(z.loc[g, 'examiners']):,}" for g in ["All"]] + [num(z.loc["All", "p"])] +
                        [f"{int(z.loc['Female', 'examiners']):,}", num(100 * fe_used, 0), num(z.loc["Female", "p"])] +
                        [f"{int(z.loc['Male', 'examiners']):,}", num(z.loc["Male", "p"])])
        hdr = [r" & \multicolumn{2}{c}{All} & \multicolumn{3}{c}{Women} & \multicolumn{2}{c}{Men} \\",
               r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
               r"Minimum applications & Examiners & $p$ & Examiners & Applications used (\%) & $p$ & "
               r"Examiners & $p$ \\"]
        Lt = [r"\begin{tabular}{lrrrrrrr}", r"\toprule"] + hdr + [r"\midrule"] + [" & ".join(r) + r" \\" for r in rows]
        if fx.exists():
            xx = pd.read_csv(fx).set_index("min_apps")
            Lt.append(r"\addlinespace")
            Lt.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{Women, {B_EXTRA} shuffles}}}} \\")
            for t in THRESHOLDS:
                Lt.append(" & ".join([str(t), "", "", f"{int(xx.loc[t, 'examiners']):,}", "",
                                      num(xx.loc[t, "p"], 3), "", ""]) + r" \\")
        Lt += [r"\bottomrule", r"\end{tabular}"]
        (TEX / "tx_tab_fll.tex").write_text("\n".join(Lt) + "\n")
    if fx.exists():
        xx = pd.read_csv(fx)
        L += [f"### Women's applications with {B_EXTRA} shuffles (own random stream)", "",
              "| Minimum | Examiner × art unit pairs | Misfit / df | Same, null (mean) | Same, null (median) | p |",
              "|---|---|---|---|---|---|"]
        for r in xx.itertuples():
            L.append(f"| {r.min_apps} | {r.examiners:,} | {r.ratio:.2f} | {r.null_ratio:.2f} | "
                     f"{r.null_median_ratio:.2f} | {r.p:.3f} |")
            N[f"FllFem{TH_WORD[r.min_apps]}PExtra"] = num(r.p, 3)
        N["FllShufflesExtra"] = str(B_EXTRA)
        L.append("")
    fa = T / "tx_fll_art_units.csv"
    if fa.exists() and f.exists() and fs.exists() and fx.exists():
        au = pd.read_csv(fa, dtype={"art_unit": str, "tech_center": str})
        fl = pd.read_csv(f).set_index(["group", "min_apps"])
        sp = pd.read_csv(fs).set_index(["group", "min_apps"])
        xx = pd.read_csv(fx).set_index("min_apps")
        tcs = sorted(au.tech_center.unique())
        N["FemAUList"] = " and ".join(au.art_unit)
        N["FemAUTC"] = " and ".join(tcs)
        L += ["Art units behind the women's test at a minimum of 20: " + "; ".join(
            f"{r.art_unit} (TC {r.tech_center}): {r.pairs_used} examiners, {r.female_apps_used:,} of its "
            f"{r.female_apps_in_art_unit:,} women's applications, female share {100 * r.female_share_in_art_unit:.0f}%"
            for r in au.itertuples()) + ".", ""]

        def p_(g, t):
            return f"{fl.loc[(g, t), 'p']:.2f}"
        fe = {t: sp.loc[("Female", t)] for t in THRESHOLDS}
        L += ["Reading. Under exclusion (the examiner affects citations only through the grant) and monotonicity "
              "(what a stricter examiner grants, a more lenient one would grant too), examiners with the same grant "
              "rate grant the same applications, so an examiner's mean of granted × citations depends on the examiner "
              "only through the grant rate, along a smooth curve whose slope is the mean citations of the applications "
              "at the margin. A rejection for women's applications says that, within art units, examiners with "
              "similar grant rates on women's applications differ in the citations of what they grant by more than "
              "sampling noise: either they grant different women's applications (monotonicity fails: examiners order "
              "women's applications differently, not only draw the line in different places) or they change the "
              "citations of the patents they grant (exclusion fails). For C3 it would mean that the most lenient "
              "quintile's extra grants of women's applications are not one set of applications that every stricter "
              "examiner rejects. "
              f"At 39's minimum of 20 the test rests on {int(fe[20].pairs_used)} examiner × art unit pairs in "
              f"{int(fe[20].art_units_with_8)} art units (TC {' and '.join(tcs)}; {int(fe[20].apps_used):,} women's "
              f"applications, {100 * fe[20].apps_used_share:.0f}% of {int(fe[20].applications):,}): women are "
              f"{100 * fe[20].share_of_sample:.0f}% of first inventors, so the median examiner has "
              f"{fe[20].per_examiner_median:.0f} women's applications in the sample and the median examiner × art unit "
              f"pair {fe[20].per_pair_median:.0f}; only {int(fe[20].pairs_at_min):,} pairs "
              f"({100 * fe[20].pairs_at_min_share:.1f}%) reach 20. There p = {p_('Female', 20)}, and "
              f"{xx.loc[20, 'p']:.3f} with {B_EXTRA} shuffles, so the rejection in those art units is not a "
              f"simulation accident. With a minimum of 10 ({int(fl.loc[('Female', 10), 'examiners']):,} pairs in "
              f"{int(fe[10].art_units_with_8)} art units, {100 * fe[10].apps_used_share:.0f}% of women's "
              f"applications) p = {p_('Female', 10)} ({xx.loc[10, 'p']:.3f} with {B_EXTRA} shuffles); with 5 "
              f"({int(fl.loc[('Female', 5), 'examiners']):,} pairs, {int(fe[5].art_units_with_8)} art units, "
              f"{100 * fe[5].apps_used_share:.0f}%) p = {p_('Female', 5)} ({xx.loc[5, 'p']:.3f}). All applications "
              f"(p = {p_('All', 20)}, {p_('All', 10)}, {p_('All', 5)}) and men's ({p_('Male', 20)}, {p_('Male', 10)}, "
              f"{p_('Male', 5)}) do not reject at any minimum. The women's rejection is therefore confined to the two "
              "art units that the minimum of 20 leaves and does not survive when more of women's applications enter. "
              "Caveat: with a minimum of 5 each examiner's sampling variance rests on five applications and the null "
              f"distribution is very heavy-tailed (misfit / df under the null: mean {xx.loc[5, 'null_ratio']:.0f}, "
              f"median {xx.loc[5, 'null_median_ratio']:.1f}), so that version has little power; at 10 the null is "
              f"well behaved (median {xx.loc[10, 'null_median_ratio']:.2f}, observed {xx.loc[10, 'ratio']:.2f}).", ""]

    # 3. memorization
    f = T / "tx_memorization.csv"
    if f.exists():
        m = pd.read_csv(f, dtype={"tech_center": str})
        rows = []
        L += ["## 3. Memorization: the LLM against the text model in the more and the less familiar half", "",
              "AUC for the grant decision; 95% intervals from 1,000 bootstrap draws of applications within each "
              "center × half (the same draw for both models).", "",
              "| TC | Half | Applications | AUC, LLM | AUC, text model | LLM − text model |", "|---|---|---|---|---|---|"]
        for tc in ["3600", "1600"]:
            z = m[m.tech_center == tc].set_index("half")
            for h in ["More familiar half", "Less familiar half", "More − less familiar"]:
                r = z.loc[h]
                L.append(f"| {tc} | {h} | {int(r.n):,} | {r.auc_llm:.3f} [{r.auc_llm_lo:.3f}, {r.auc_llm_hi:.3f}] | "
                         f"{r.auc_text:.3f} [{r.auc_text_lo:.3f}, {r.auc_text_hi:.3f}] | "
                         f"{r['diff']:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}] |")
            f1, f0, dd = z.loc["More familiar half"], z.loc["Less familiar half"], z.loc["More − less familiar"]
            rows.append([f"TC {tc}", num(f1["diff"], 3), num(f0["diff"], 3), num(dd["diff"], 3),
                         br(dd.diff_lo, dd.diff_hi)])
            w = TC_WORD[tc]
            N[f"Mem{w}Fam"], N[f"Mem{w}FamCI"] = num(f1["diff"], 2), br(f1.diff_lo, f1.diff_hi, 2)
            N[f"Mem{w}Less"], N[f"Mem{w}LessCI"] = num(f0["diff"], 2), br(f0.diff_lo, f0.diff_hi, 2)
            N[f"Mem{w}Diff"], N[f"Mem{w}DiffCI"] = num(dd["diff"], 3), br(dd.diff_lo, dd.diff_hi, 3)
            N[f"Mem{w}LLMGain"], N[f"Mem{w}LLMGainCI"] = num(dd.auc_llm, 3), br(dd.auc_llm_lo, dd.auc_llm_hi, 3)
            N[f"Mem{w}TextGain"], N[f"Mem{w}TextGainCI"] = num(dd.auc_text, 3), br(dd.auc_text_lo, dd.auc_text_hi, 3)
            for h, k in [("More familiar half", "Fam"), ("Less familiar half", "Less")]:
                N[f"Mem{w}{k}LLM"], N[f"Mem{w}{k}Text"] = num(z.loc[h, "auc_llm"], 3), num(z.loc[h, "auc_text"], 3)
        L.append("")
        z3, z1 = (m[m.tech_center == tc].set_index("half") for tc in ["3600", "1600"])
        fam, less, dd = "More familiar half", "Less familiar half", "More − less familiar"
        L += ["Reading. Familiarity (Min-20% Prob) measures whether the model has seen an application's text, not "
              "its outcome; a text model trained on our data cannot remember outcomes, so its change across halves "
              "measures how much easier the more familiar half is to rank. Memory of outcomes would make the LLM "
              "gain on the text model in the more familiar half (a positive difference of differences). "
              f"TC 3600: LLM − text model {z3.loc[fam, 'diff']:.3f} in the more familiar half and "
              f"{z3.loc[less, 'diff']:.3f} in the less familiar half; difference {z3.loc[dd, 'diff']:.3f} "
              f"[{z3.loc[dd, 'diff_lo']:.3f}, {z3.loc[dd, 'diff_hi']:.3f}]. TC 1600: {z1.loc[fam, 'diff']:.3f} and "
              f"{z1.loc[less, 'diff']:.3f}; difference {z1.loc[dd, 'diff']:.3f} [{z1.loc[dd, 'diff_lo']:.3f}, "
              f"{z1.loc[dd, 'diff_hi']:.3f}]. In TC 1600 both models rank the more familiar half better (LLM "
              f"{z1.loc[dd, 'auc_llm']:+.3f} [{z1.loc[dd, 'auc_llm_lo']:.3f}, {z1.loc[dd, 'auc_llm_hi']:.3f}], text "
              f"model {z1.loc[dd, 'auc_text']:+.3f} [{z1.loc[dd, 'auc_text_lo']:.3f}, {z1.loc[dd, 'auc_text_hi']:.3f}]"
              "), so the LLM's higher AUC there is the half being easier, not memory. The upper ends of the "
              "intervals of the difference of differences bound any gain from memory. Intervals resample applications "
              "and ignore clustering by examiner or art unit.", ""]
        hdr = [r" & \multicolumn{2}{c}{AUC of the LLM $-$ AUC of the text model} & & \\",
               r"\cmidrule(lr){2-3}",
               r" & More familiar half & Less familiar half & Difference & 95\% CI \\"]
        Lt = [r"\begin{tabular}{lrrrr}", r"\toprule"] + hdr + [r"\midrule"] + [" & ".join(r) + r" \\" for r in rows]
        Lt += [r"\bottomrule", r"\end{tabular}"]
        (TEX / "tx_tab_memorization.tex").write_text("\n".join(Lt) + "\n")

    # 4. claims
    f, fv = T / "tx_claims.csv", T / "tx_claims_validation.csv"
    if f.exists() and fv.exists():
        cl = pd.read_csv(f, dtype={"sample": str})
        v = pd.read_csv(fv).iloc[0]
        TCS8 = TCS

        def est(key, smp):
            return cl[(cl.key == key) & (cl["sample"] == smp)].iloc[0]

        def sig(key):
            c = cl[(cl.key == key) & cl["sample"].isin(TCS8)]
            lo, hi = c.coef_per_10pp - 1.96 * c.se, c.coef_per_10pp + 1.96 * c.se
            assert len(c) == len(TCS8)
            return int((hi < 0).sum()), int((lo > 0).sum())

        def cell(r):
            return (f"{num(r.coef_per_10pp)} [{num(r.coef_per_10pp - 1.96 * r.se)}, "
                    f"{num(r.coef_per_10pp + 1.96 * r.se)}] ({num(r['mean'], 1)})")
        main_rows = [("filed_all", "At filing, all applications"), ("filed_granted", "At filing, granted patents"),
                     ("granted", "At grant"), ("removed", "Removed in examination")]
        rows = []
        for key, lab in main_rows:
            ng, ps = sig(key)
            rows.append([lab, cell(est(key, "All centers")), cell(est(key, "3600")), f"{ng} / {ps}"])
        Lt = [r"\begin{tabular}{lrrc}", r"\toprule",
              r"Claims & All centers & TC 3600 & Centers, CI $<$0 / $>$0 (of 8) \\", r"\midrule"]
        Lt += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
        (TEX / "design_tab_claims.tex").write_text("\n".join(Lt) + "\n")
        g42 = est("granted_42", "All centers")
        L += ["## 4. Claims removed in examination (comment 11)", "",
              "Claims at filing: the claims of the published application in HUPD (all_text.claims), counted as the "
              "claims numbered 1, 2, 3, ... in order (ranges such as '1-20.' included), minus those marked canceled "
              "or deleted by a preliminary amendment, i.e. the claims pending when the application was published "
              "(usually before the first action). Claims at grant: PatentsView num_claims (as 42). Removed = filing − "
              "grant, for granted patents. Coefficient on leniency × 0.1 within art unit × year × class cells; "
              "granted patents with 42's controls (small entity, female first inventor, inventors), all "
              "applications without controls (42's balance); 95% CI clustered by examiner; mean in parentheses.", "",
              f"Parsing: numbered claims found for {100 * v.parsed:.2f}% of {int(v.apps):,} applications; "
              f"{100 * v.canceled_any:.1f}% have claims canceled before publication; claims pending at publication: "
              f"mean {v.mean_filed:.1f}, median {v.median_filed:.0f}. Validation against the Office Action Research "
              f"Dataset (first non-final rejection, highest claim number named in its rejections; {int(v.with_oa):,} "
              f"applications): equal to the highest claim number at publication in {100 * v.equal:.1f}%, within two "
              f"in {100 * v.within2:.1f}%, lower in {100 * v.below:.1f}% (not every claim rejected), higher in "
              f"{100 * v.above:.1f}% (claims added after publication, or a count too low); correlation "
              f"{v['corr']:.3f}. 42's claims-at-grant estimate reproduced on 42's sample: {g42.coef_per_10pp:.3f}. "
              f"Published before the examiner's first action (PatEx PG-ISSUE before the first rejection, restriction, "
              f"or allowance): {100 * v.published_before_action:.0f}% of applications with a publication date; rows "
              "marked so repeat the estimates on them, whose published claims cannot reflect examination.", "",
              "| Variable | Sample | Applications | Mean | Coefficient × 0.1 [95% CI] |", "|---|---|---|---|---|"]
        for r in cl.itertuples():
            L.append(f"| {r.variable} | {r.sample} | {r.n:,} | {r.mean:.2f} | {r.coef_per_10pp:.3f} "
                     f"[{r.coef_per_10pp - 1.96 * r.se:.3f}, {r.coef_per_10pp + 1.96 * r.se:.3f}] |")
        L.append("")
        keys = [("filed_all", "Filed"), ("filed_granted", "FiledGranted"), ("granted", "Grant"), ("removed", "Removed")]
        for key, k in keys + [(a + "_pre", b + "Pre") for a, b in keys]:
            for smp, w in [("All centers", ""), ("3600", "ThreeSix")]:
                r = est(key, smp)
                N[f"Claims{k}{w}"] = num(r.coef_per_10pp)
                N[f"Claims{k}{w}CI"] = br(r.coef_per_10pp - 1.96 * r.se, r.coef_per_10pp + 1.96 * r.se, 2)
                N[f"Claims{k}{w}Mean"] = num(r["mean"], 1)
            N[f"Claims{k}Neg"], N[f"Claims{k}Pos"] = (str(i) for i in sig(key))
        N["ClaimsGrantedN"] = f"{int(est('removed', 'All centers').n):,}"
        N["ClaimsPubBefore"] = num(100 * v.published_before_action, 0)
        N["ClaimsPreN"] = f"{int(est('filed_all_pre', 'All centers').n):,}"
        N["ClaimsParsed"] = num(100 * v.parsed, 1)
        N["ClaimsCanceledAny"] = num(100 * v.canceled_any, 1)
        N["ClaimsValN"] = f"{int(v.with_oa):,}"
        N["ClaimsValEqual"] = num(100 * v.equal, 0)
        N["ClaimsValWithinTwo"] = num(100 * v.within2, 0)
        N["ClaimsValAbove"] = num(100 * v.above, 0)
        N["ClaimsValBelow"] = num(100 * v.below, 0)
        N["ClaimsValCorr"] = num(v["corr"], 2)

    for k in N:
        assert k.isalpha(), k
    (TEX / "tx_numbers.tex").write_text("% Generated by code/46_tests_extra.py; do not edit by hand.\n" + "".join(
        rf"\newcommand{{\xTx{k}}}{{{v}}}" + "\n" for k, v in N.items()))
    L += ["## Macros (output/latex/tx_numbers.tex)", "", "| Macro | Value |", "|---|---|"]
    L += [f"| `\\xTx{k}` | {plain(v)} |" for k, v in N.items()]
    (OUT / "tx.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    warnings.filterwarnings("ignore")
    args = set(sys.argv[1:])
    allp = {"--c1perm", "--fll", "--mem", "--claims"}
    parts = allp & args or (allp if not {"--report", "--support"} & args else set())
    t0 = time.time()
    if PSFX:   # the Poisson variant: part 1 only, own files, no report
        summ, cells = c1_permutation()
        summ.to_csv(T / f"tx_c1_permutation{PSFX}.csv", index=False)
        cells.to_csv(T / f"tx_c1_permutation{PSFX}_cells.csv", index=False)
        print(summ.to_string(), flush=True)
        return
    if "--c1perm" in parts:
        summ, cells = c1_permutation()
        summ.to_csv(T / "tx_c1_permutation.csv", index=False)
        cells.to_csv(T / "tx_c1_permutation_cells.csv", index=False)
        print(f"C1 permutation: {time.time() - t0:.0f}s", flush=True)
    if "--fll" in parts or "--support" in args:
        sp, au = fll_support()
        sp.to_csv(T / "tx_fll_support.csv", index=False)
        au.to_csv(T / "tx_fll_art_units.csv", index=False)
    if "--fll" in parts:
        jobs = [delayed(fll_run)(t) for t in THRESHOLDS] + [delayed(fll_female_extra)(t, B_EXTRA) for t in THRESHOLDS]
        res = Parallel(n_jobs=JOBS, backend="loky")(jobs)
        pd.concat(res[:len(THRESHOLDS)]).to_csv(T / "tx_fll.csv", index=False)
        pd.DataFrame(res[len(THRESHOLDS):]).to_csv(T / "tx_fll_female_b499.csv", index=False)
        print(f"FLL: {time.time() - t0:.0f}s", flush=True)
    if "--mem" in parts:
        memorization().to_csv(T / "tx_memorization.csv", index=False)
        print(f"memorization: {time.time() - t0:.0f}s", flush=True)
    if "--claims" in parts:
        cl, x, val = claims()
        cl.to_csv(T / "tx_claims.csv", index=False)
        pd.DataFrame([val]).to_csv(T / "tx_claims_validation.csv", index=False)
        dd = (x.oa_max_claim - x.claims_highest).clip(-10, 10)
        dd.value_counts().sort_index().rename_axis("oa_max_minus_highest").rename("applications").to_csv(
            T / "tx_claims_validation_dist.csv")
        print(f"claims: {time.time() - t0:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
