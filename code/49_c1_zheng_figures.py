"""C1 appendix figures after Zheng (2025, JFQA, "How Can Innovation Screening Be Improved? A Machine Learning Analysis
with Economic Consequences for Firm Performance"), built from 35's main run (Poisson loss; cached predictions
data/processed/c1_preds_all_poisson_all.parquet) without refitting.

Zheng's Algorithm MLQuality is our algorithm (trained on citations, 35's ml_cites); his Algorithm MLDecision is our
predicted examiner (trained on grant decisions, 35's ml_judge). Sample, cells, quintiles and weights are 35's own,
through its functions (load_all, cached, quintiles): Figures 1 and 4 use the test set, as 35's Figure II; Figures 2
and 3 use the test and imputation sets with the lock box excluded, as 35's Table III.

1. fig_c1z_grant_rate (Zheng's Figure 5): (a) the examiners' grant rate in 1,000 equal-size bins of the algorithm's
   predicted citations (x: the rank of the bin's mean prediction), dots and a smoothed line; the final grant mixes
   the examiner's reading with the applicant's persistence after a rejection, so the same is drawn by stage:
   (b) the first-action allowance (the examiner's first reading, before the applicant replies) and (c) granted in the
   end among applications whose first action was not an allowance (the later stage). Each panel adds the means of
   20 equal-size bins with 95% intervals from 1,000 bootstrap draws of art units. Reported with it (c1z_stages.csv):
   the highest 5% of predictions against the lowest 5% (same bootstrap), the slope of each outcome on the
   prediction's percentile (pooled, within center × filing year, within art unit × filing year; errors clustered by
   examiner), and how well the algorithm's prediction and the predicted examiner's separate each outcome (AUC).
2. fig_c1z_reject_cites (Zheng's Figure 6): reject rates in 20 equal-size bins of predicted citations. Bottom bar: the
   most lenient quintile's reject rate in the bin. On top: (a) the strictest quintile's additional reject rate in the
   bin (strictest rate − most lenient rate, not clipped at zero); (b) the algorithm's additional rejections; (c) the
   predicted examiner's, on the same bins of predicted citations so the two AI rules share the axis. An AI rule works
   within each cell on the most lenient quintile's caseload: it rejects that quintile's grants with the lowest score
   until the quintile's reject rate equals the strictest quintile's in the cell, so in every cell it makes exactly the
   strictest quintile's additional rejections. Fractional cases as in 35's contraction() (k = n_lenient × (strictest
   rate − lenient rate) cases; the last one counts in part, as 35's linear interpolation). In the cells where the
   strictest quintile rejects less than the most lenient one, the rule matches the rate the other way: it grants the
   lenient quintile's rejections with the highest score (negative additional rejections), so that per cell and in
   total its additional rejections equal the strictest quintile's. The rule's additional reject rate in a bin is its
   weighted extra rejections over the lenient quintile's weighted cases in the bin.
3. fig_c1z_reject_grant (Zheng's Figure 7): the same with 20 equal-size bins of the predicted examiner's grant
   probability: (a) the strictest quintile, (b) the predicted examiner.
4. fig_c1z_calibration (Zheng's Internet Appendix Figure IA.1): observed against predicted citations of granted test
   applications in 200 equal-size bins, dots, a smoothed line and the 45-degree line.
Weights: each case n_c / n_cq (cell size over the cell's cases in its quintile), 35's reweighting, so all quintiles
have the same mix of cells; a quintile's overall reject rate is then the cell-size-weighted mean of its cell rates, as
in 35's Table III. Shares of additional rejections by bin follow 35's Figure VI (misranking()): the strictest
quintile's are the lenient quintile's weighted grants in the bin minus the strictest quintile's.

Differences from Zheng (2025), and why:
- Lenient and strict: Zheng splits examiners within an art unit at the median reject rate (above/below); here they are
  the most lenient and the strictest of five quintiles of equal caseload within art unit × filing year × USPC class
  cells with at least five examiners, ordered by leave-one-out leniency (35's quintiles(): assignment is as good as
  random only within these cells, and an examiner's rate on the same cases would sort on the decisions themselves).
  Cases are reweighted so the two quintiles have the same cell composition.
- Models: Zheng uses XGBoost; here 35's models, scikit-learn histogram gradient boosting tuned by cross-validation as
  in Kleinberg et al. (2018): the algorithm with a Poisson loss on citations within five years of grant excluding
  self-citations (Zheng: citations of granted patents), the predicted examiner a classifier of the grant decision;
  inputs are 64 principal components of the text embedding (title, abstract, claims), entity size, number of
  inventors, technology center and filing year.
- AI rules within cells: Zheng's algorithm rejects within an art unit until it matches the strict examiners' grant
  rate; here within 35's cells, with fractional cases and, where the strictest quintile grants more than the most
  lenient one, the symmetric step described in 2 (Zheng's median split cannot produce such cells).
- Sample: Figures 2 and 3 use the test and imputation sets (35's Table III; Zheng: the test set), so the quintiles
  hold more cases; the lock box is left out.
- Smoothed lines: local linear regression with tricube weights on the nearest 30% of the bins (LOWESS, no robustness
  iterations); Zheng's ggplot2 geom_smooth uses loess or a GAM. In Figure 4 the smoother runs on log predicted
  citations (the predictions have a long right tail) and the axes are linear as in Zheng's; the top bin lies off the
  scale and is labelled.
- 1,000 bins of a 70,679-application test set hold about 71 applications each, so Figure 1's dots are noisy.

Runs on CPUs in about a minute. --figure redraws the figures from output/tables.
Input: 35's cached predictions and data (through 35's load_all and cached); output/tables/c1_kllm_all_poisson_
       {quintiles,fig6,fit}.csv for the checks; 44's FIG_SERIES colours.
Output: output/tables/c1z_{grant_rate,reject_cites,reject_grant,calibration,checks,stages,stage_bins,
        stage_bins20}.csv,
        output/latex/c1z_numbers.tex (macros \\xCz...), output/c1z_figures.md,
        output/figures/journal/fig_c1z_{grant_rate,reject_cites,reject_grant,calibration}.{pdf,png}
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")
os.environ["CFE_SAMPLE"], os.environ["CFE_LOSS"] = "all", "poisson"   # 35's main specification (outputs *_all_poisson)
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
T = OUT / "tables"
TEX = OUT / "latex"
S35 = "_all_poisson"
SEED = 20261101 + 49        # tie-breaks among equal scores (own stream)
N_FINE, N_BINS, N_CAL, LOW = 1000, 20, 200, 5   # bins of Figures 1, 2–3, 4; 'lowest bins' of 20 for the shares
SPAN = 0.3                  # LOWESS: share of the bins in each local fit
RULES = {"algorithm": "ml_cites", "judge": "ml_judge"}   # AI rule → its score (rejected first = lowest)
STAGES = [("granted", "Final grant"), ("fa_allow", "First-action allowance"),
          ("later", "Granted in the end, first action not an allowance")]
B_STAGE = 1000              # bootstrap draws of art units for Figure 1's panels and the top − bottom 5% differences


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def lowess(x: np.ndarray, y: np.ndarray, frac: float = SPAN) -> np.ndarray:
    """Local linear fit at each x with tricube weights on the nearest ceil(frac · n) points (loess, degree 1, no
    robustness iterations)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    k = int(np.ceil(frac * n))
    out = np.empty(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        idx = np.argpartition(dist, k - 1)[:k]
        h = max(dist[idx].max(), 1e-12) * 1.0000001
        w = (1 - (dist[idx] / h) ** 3) ** 3
        X = np.c_[np.ones(k), x[idx] - x[i]]
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y[idx] * np.sqrt(w), rcond=None)[0]
        out[i] = beta[0]
    return out


def equal_bins(score: pd.Series, n: int) -> np.ndarray:
    """Equal-size bins of a score in the frame's row order (ties by position), 0 = lowest, as 35's misranking()."""
    return pd.qcut(pd.Series(np.asarray(score)).rank(method="first"), n, labels=False).to_numpy()


# ---------------------------------------------------------------- data (35's sample, cells, quintiles, weights)

def data(M) -> tuple:
    x = M.cached(M.load_all(), "all")
    te = x[x.part == "test"].copy()                          # 35's Figure II
    ev = x[x.part.isin(["test", "impute"])].copy()           # 35's Table III (contraction), lock box excluded
    ev["q"] = M.quintiles(ev)
    ev = ev[ev.q.notna()].copy()
    ev["q"] = ev.q.astype(int)                               # 5 = most lenient, 1 = strictest
    n_c = ev.groupby(M.CELL)[M.CELL].transform("size").to_numpy(float)
    n_cq = ev.groupby([M.CELL, "q"])[M.CELL].transform("size").to_numpy(float)
    ev["w"] = n_c / n_cq
    return x, te, ev


def matched_rule(ev: pd.DataFrame, score: np.ndarray, cell: str, tb: np.ndarray, contract_only: bool = False) -> tuple:
    """Per case of the most lenient quintile, the share of it that the rule rejects beyond that quintile's decisions
    (negative: grants). In each cell, k = n_lenient × (strictest rate − lenient rate) cases: the lenient quintile's
    grants with the lowest score (k > 0), or its rejections with the highest score (k < 0; skipped with
    contract_only); fractional as 35's linear interpolation. Returns (f, per-cell table)."""
    q, g = ev.q.to_numpy(int), ev.granted.to_numpy(float)
    f = np.zeros(len(ev))
    rows = []
    for c, idx in ev.groupby(cell, sort=True).indices.items():
        five, one = idx[q[idx] == 5], idx[q[idx] == 1]
        d = g[five].mean() - g[one].mean()                  # strictest − lenient reject rate in the cell
        k = d * len(five)
        if k > 0:
            cand = five[g[five] == 1]
            cand = cand[np.lexsort((tb[cand], score[cand]))]       # lowest score first
        elif k < 0 and not contract_only:
            cand = five[g[five] == 0]
            cand = cand[np.lexsort((tb[cand], -score[cand]))]      # highest score first
        else:
            cand = five[:0]
        take = np.clip(abs(k) - np.arange(len(cand)), 0, 1)
        f[cand] = np.sign(k) * take
        rows.append(dict(cell=c, n=len(idx), n_lenient=len(five), strict_extra=d,
                         rule_extra=f[five].sum() / len(five)))
    return f, pd.DataFrame(rows)


def bin_table(ev: pd.DataFrame, b: np.ndarray, score_col: str, fs: dict) -> pd.DataFrame:
    """Reject rates by bin: the most lenient (q = 5) and strictest (q = 1) quintiles' weighted reject rates, the
    strictest quintile's additional reject rate, each AI rule's additional reject rate (on the lenient quintile's
    cases in the bin), and each series' share of the additional rejections by bin."""
    q, g, w = ev.q.to_numpy(int), ev.granted.to_numpy(float), ev.w.to_numpy(float)
    s = ev[score_col].to_numpy(float)
    rows = []
    for i in range(b.max() + 1):
        inb = b == i
        L, S = inb & (q == 5), inb & (q == 1)
        r = dict(bin=i + 1, n=int(inb.sum()), score_mean=s[inb].mean(), score_min=s[inb].min(),
                 score_max=s[inb].max(), lenient_reject=(w * (1 - g))[L].sum() / w[L].sum(),
                 strict_reject=(w * (1 - g))[S].sum() / w[S].sum(),
                 strict_mass=(w * g)[L].sum() - (w * g)[S].sum())
        r["strict_extra"] = r["strict_reject"] - r["lenient_reject"]
        for k, f in fs.items():
            r[f"{k}_extra"] = (w * f)[L].sum() / w[L].sum()
            r[f"{k}_mass"] = (w * f)[L].sum()
        rows.append(r)
    t = pd.DataFrame(rows)
    for k in ["strict"] + list(fs):
        t[f"{k}_share"] = t[f"{k}_mass"] / t[f"{k}_mass"].sum()
    return t.drop(columns=[c for c in t.columns if c.endswith("_mass")])


# ---------------------------------------------------------------- computation and checks

def stages(te: pd.DataFrame, bf: np.ndarray) -> dict:
    """Figure 1 by stage of examination (test set, the same 1,000 bins): final grant, first-action allowance, and
    granted in the end among applications whose first action was not an allowance."""
    import pyfixest as pf
    from sklearn.metrics import roc_auc_score
    ex = pd.read_parquet(P / "all_sample.parquet", columns=["application_number", "examiner_id", "art_unit",
                                                           "au_year", "tech_center", "filing_year", "fa_allow"])
    t = pd.DataFrame(dict(application_number=te.application_number.to_numpy(), granted=te.granted.to_numpy(float),
                          ml_cites=te.ml_cites.to_numpy(float), ml_judge=te.ml_judge.to_numpy(float), bin=bf))
    t = t.merge(ex, on="application_number", how="left", validate="one_to_one")
    assert len(t) == len(te) and t.fa_allow.notna().all()
    t["later"] = np.where(t.fa_allow == 0, t.granted, np.nan)
    t["pct"] = t.ml_cites.rank(pct=True)
    t["tcy"] = t.tech_center.astype(str) + "_" + t.filing_year.astype(str)
    lo, hi = (t.bin < N_FINE * 0.05).to_numpy(), (t.bin >= N_FINE * 0.95).to_numpy()
    aus = np.sort(t.art_unit.unique())
    rng = np.random.default_rng(SEED + 1)
    W = np.stack([np.bincount(rng.integers(0, len(aus), len(aus)), minlength=len(aus)) for _ in range(B_STAGE)])
    idx = np.searchsorted(aus, t.art_unit.to_numpy())
    b20 = t.bin.to_numpy() * N_BINS // N_FINE
    rows, bins20 = [], []
    for y, lab in STAGES:
        ok = t[y].notna().to_numpy()
        v = np.nan_to_num(t[y].to_numpy(float))

        def bmean(m):
            m = m & ok
            sm = np.bincount(idx[m], weights=v[m], minlength=len(aus))
            ct = np.bincount(idx[m], minlength=len(aus)).astype(float)
            return (W @ sm) / np.maximum(W @ ct, 1e-9)
        db = bmean(hi) - bmean(lo)
        tt = t[ok]
        row = dict(outcome=y, label=lab, n=int(ok.sum()), mean=float(v[ok].mean()), low5=float(v[ok & lo].mean()),
                   high5=float(v[ok & hi].mean()))
        row["diff"] = row["high5"] - row["low5"]
        row["diff_lo"], row["diff_hi"] = np.percentile(db, [2.5, 97.5])
        for name, fe in [("pooled", None), ("center_year", "tcy"), ("au_year", "au_year")]:
            m = pf.feols(f"{y} ~ pct" + (f" | {fe}" if fe else ""), data=tt, vcov={"CRV1": "examiner_id"})
            row[f"slope_{name}"], row[f"slope_{name}_se"] = float(m.coef()["pct"]), float(m.se()["pct"])
        row["auc_algorithm"] = roc_auc_score(tt[y], tt.ml_cites)
        row["auc_predicted_examiner"] = roc_auc_score(tt[y], tt.ml_judge)
        rows.append(row)
        for b in range(N_BINS):
            m = ok & (b20 == b)
            bs = bmean(b20 == b)
            bins20.append(dict(outcome=y, bin=b + 1, mean=float(v[m].mean()), lo=np.percentile(bs, 2.5),
                               hi=np.percentile(bs, 97.5)))
    fine = t.groupby("bin").agg(granted=("granted", "mean"), fa_allow=("fa_allow", "mean"),
                                later=("later", "mean")).reset_index()
    fine["bin"] += 1
    for y, _ in STAGES:
        fine[f"{y}_smoothed"] = lowess(fine.bin.to_numpy(float), fine[y].to_numpy())
    corr = float(t[["ml_cites", "ml_judge"]].corr(method="spearman").iloc[0, 1])
    return dict(stages=pd.DataFrame(rows), stage_bins=fine, stage_bins20=pd.DataFrame(bins20), rank_corr=corr)


def compute(M) -> dict:
    x, te, ev = data(M)
    tb = np.random.default_rng(SEED).random(len(ev))
    q, g, w = ev.q.to_numpy(int), ev.granted.to_numpy(float), ev.w.to_numpy(float)
    chk = []

    # check 1: quintile reject rates against 35's Table III (extra rejections of quintile labels 2–5 = q 4..1)
    rej = {k: (w * (1 - g))[q == k].sum() / w[q == k].sum() for k in range(1, 6)}
    q35 = pd.read_csv(T / f"c1_kllm{S35}_quintiles.csv")
    q35 = q35[(q35["sample"] == "test + imputation") & (q35.outcome == "cites")].set_index("quintile").extra_rejections
    for k in [4, 3, 2, 1]:
        chk.append(dict(check=f"Extra rejection rate, quintile {6 - k} vs most lenient (35's Table III)",
                        ours=rej[k] - rej[5], c35=float(q35[str(6 - k)])))

    # Figure 2 and 3: the AI rules' matched extra rejections, cell by cell
    fs, fs_c, cells = {}, {}, {}
    for k, col in RULES.items():
        fs[k], cells[k] = matched_rule(ev, ev[col].to_numpy(float), M.CELL, tb)
        fs_c[k], _ = matched_rule(ev, ev[col].to_numpy(float), M.CELL, tb, contract_only=True)
        cl = cells[k]
        chk.append(dict(check=f"Max |rule − strictest| extra rejections per cell ({k})", ours=float(
            (cl.rule_extra - cl.strict_extra).abs().max()), c35=0.0))
        tot = (cl.n * cl.rule_extra).sum() / cl.n.sum()
        chk.append(dict(check=f"Extra rejection rate, cell-weighted ({k}) vs strictest quintile (35)", ours=tot,
                        c35=float(q35["5"])))
    cl = cells["algorithm"]
    neg = cl.strict_extra < 0
    b20 = equal_bins(ev.ml_cites, N_BINS)
    rc = bin_table(ev, b20, "ml_cites", fs)
    rc_c = bin_table(ev, b20, "ml_cites", fs_c)
    rg = bin_table(ev, equal_bins(ev.ml_judge, N_BINS), "ml_judge", fs)
    rg_c = bin_table(ev, equal_bins(ev.ml_judge, N_BINS), "ml_judge", fs_c)

    # check 2: bins of predicted citations against 35's Figure VI (misranking, strictest quintile = label 5)
    f6 = pd.read_csv(T / f"c1_kllm{S35}_fig6.csv")
    f6 = f6[pd.to_numeric(f6.quintile, errors="coerce") == 5].sort_values("bin").reset_index(drop=True)
    chk.append(dict(check="Max |lenient reject rate by bin − 35's Figure VI|",
                    ours=float(np.abs((1 - rc.lenient_reject) - f6.grant_rate_lenient).max()), c35=0.0))
    chk.append(dict(check="Max |strictest reject rate by bin − 35's Figure VI|",
                    ours=float(np.abs((1 - rc.strict_reject) - f6.grant_rate_quintile).max()), c35=0.0))
    chk.append(dict(check="Max |strictest share of extra rejections by bin − 35's Figure VI|",
                    ours=float(np.abs(rc.strict_share - f6.examiners).max()), c35=0.0))

    # Figure 1: grant rate by 1,000 bins of predicted citations (test set)
    bf = equal_bins(te.ml_cites, N_FINE)
    gr = pd.DataFrame(dict(bin=bf + 1, p=te.ml_cites.to_numpy(), g=te.granted.to_numpy(float))).groupby("bin").agg(
        n=("g", "size"), predicted_cites=("p", "mean"), grant_rate=("g", "mean")).reset_index()
    gr["smoothed"] = lowess(gr.bin.to_numpy(float), gr.grant_rate.to_numpy())
    st = stages(te, bf)
    fit = pd.read_csv(T / f"c1_kllm{S35}_fit.csv").iloc[0]
    tg_ = te.granted.to_numpy(float)
    chk.append(dict(check="Test-set grant rate (35's fit table)", ours=float(tg_.mean()), c35=float(fit.grant_rate)))
    low1 = (te.ml_cites <= te.ml_cites.quantile(0.01)).to_numpy()
    chk.append(dict(check="Grant rate, lowest 1% of predicted citations (35's fit table)", ours=float(tg_[low1].mean()),
                    c35=float(fit.lowest1_grant_rate)))

    # Figure 4: calibration among granted test applications, 200 bins
    tg = te[te.granted == 1]
    bc = equal_bins(tg.ml_cites, N_CAL)
    cal = pd.DataFrame(dict(bin=bc + 1, p=tg.ml_cites.to_numpy(), y=tg.cites.to_numpy(float))).groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    cal["smoothed"] = lowess(np.log(cal.predicted.to_numpy()), cal.observed.to_numpy())

    ck = pd.DataFrame(chk)
    ck["abs_diff"] = (ck.ours - ck.c35).abs()
    return dict(te=te, ev=ev, rej=rej, cells=cl, neg=neg, rc=rc, rc_c=rc_c, rg=rg, rg_c=rg_c, gr=gr, cal=cal,
                checks=ck, f6=f6, tgm=(float(tg.ml_cites.mean()), float(tg.cites.mean())),
                grant_low5=float(tg_[bf < N_FINE * 0.05].mean()), grant_top5=float(tg_[bf >= N_FINE * 0.95].mean()),
                grant_low1=float(tg_[bf < N_FINE * 0.01].mean()), grant_all=float(tg_.mean()), **st)


# ---------------------------------------------------------------- outputs

def low(t: pd.DataFrame, k: str) -> float:
    return float(t.loc[t.bin <= LOW, f"{k}_share"].sum())


def write(r: dict) -> None:
    rename = {"score_mean": "predicted_cites_mean", "score_min": "predicted_cites_min",
              "score_max": "predicted_cites_max"}
    rc = r["rc"].rename(columns=rename)
    rg = r["rg"].rename(columns={k: v.replace("cites", "grant_prob") for k, v in rename.items()})
    for t, tc in [(rc, r["rc_c"]), (rg, r["rg_c"])]:   # the contract-only variant (no extra grants in any cell)
        for k in RULES:
            t[f"{k}_extra_contract_only"] = tc[f"{k}_extra"].to_numpy()
            t[f"{k}_share_contract_only"] = tc[f"{k}_share"].to_numpy()
    rc.to_csv(T / "c1z_reject_cites.csv", index=False)
    rg.to_csv(T / "c1z_reject_grant.csv", index=False)
    r["gr"].to_csv(T / "c1z_grant_rate.csv", index=False)
    r["cal"].to_csv(T / "c1z_calibration.csv", index=False)
    r["checks"].to_csv(T / "c1z_checks.csv", index=False)
    for k in ["stages", "stage_bins", "stage_bins20"]:
        r[k].to_csv(T / f"c1z_{k}.csv", index=False)

    cl, neg, rej = r["cells"], r["neg"], r["rej"]
    nb_c, nb_g = int((rc.strict_extra < 0).sum()), int((rg.strict_extra < 0).sum())
    f6_alg = float(r["f6"].loc[r["f6"].bin <= LOW, "algorithm"].sum())
    pm, om = r["tgm"]
    N = {"xCzGrantAll": num(100 * r["grant_all"]), "xCzGrantLowOne": num(100 * r["grant_low1"]),
         "xCzGrantLowFive": num(100 * r["grant_low5"]), "xCzGrantTopFive": num(100 * r["grant_top5"]),
         "xCzRejLenient": num(100 * rej[5]), "xCzRejStrict": num(100 * rej[1]),
         "xCzRejExtra": num(100 * (rej[1] - rej[5])),
         "xCzShareStrictLowFive": num(100 * low(rc, "strict")),
         "xCzShareAlgoLowFive": num(100 * low(rc, "algorithm")),
         "xCzShareJudgeLowFive": num(100 * low(rc, "judge")),
         "xCzShareGlobalAlgoLowFive": num(100 * f6_alg),
         "xCzShareStrictLowFiveGrant": num(100 * low(rg, "strict")),
         "xCzShareJudgeLowFiveGrant": num(100 * low(rg, "judge")),
         "xCzShareAlgoLowFiveGrant": num(100 * low(rg, "algorithm")),
         "xCzNegCells": num(100 * neg.mean()), "xCzNegCellsApps": num(100 * cl.n[neg].sum() / cl.n.sum()),
         "xCzNegBinsCites": str(nb_c), "xCzNegBinsGrant": str(nb_g),
         "xCzCalPred": num(pm, 2), "xCzCalObs": num(om, 2),
         "xCzCells": f"{len(cl):,}", "xCzApps": f"{len(r['ev']):,}", "xCzTestApps": f"{len(r['te']):,}",
         "xCzRankCorrModels": num(r["rank_corr"], 2)}
    sg = r["stages"].set_index("outcome")
    for y, tag in [("granted", "Grant"), ("fa_allow", "Fa"), ("later", "Later")]:
        g_ = sg.loc[y]
        N.update({f"xCz{tag}Mean": num(100 * g_["mean"]), f"xCz{tag}Low": num(100 * g_.low5),
                  f"xCz{tag}High": num(100 * g_.high5), f"xCz{tag}Diff": num(100 * g_["diff"]),
                  f"xCz{tag}DiffCI": f"[{num(100 * g_.diff_lo)}, {num(100 * g_.diff_hi)}]",
                  f"xCz{tag}AucAlgo": num(g_.auc_algorithm, 2), f"xCz{tag}AucJudge": num(g_.auc_predicted_examiner, 2),
                  f"xCz{tag}N": f"{int(g_.n):,}"})
        for nm, tg2 in [("pooled", "Pool"), ("center_year", "Tc"), ("au_year", "Au")]:
            N[f"xCz{tag}Slope{tg2}"] = num(100 * g_[f"slope_{nm}"])
            N[f"xCz{tag}Slope{tg2}Se"] = num(100 * g_[f"slope_{nm}_se"])
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "c1z_numbers.tex").write_text("% C1 figures after Zheng (2025) (49_c1_zheng_figures.py)\n" + "\n".join(
        rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items()) + "\n")

    def pct(v, nd=1):
        return f"{100 * v:.{nd}f}".replace("-", "−")
    ck = r["checks"]
    L = ["# C1 figures after Zheng (2025, JFQA)", "",
         "From 35's main run (Poisson loss, cached predictions; 49_c1_zheng_figures.py). Zheng's MLQuality = the "
         "algorithm (trained on citations); MLDecision = the predicted examiner (trained on grant decisions). "
         "Figures 1 and 4: test set; Figures 2 and 3: test and imputation sets in 35's cells (art unit × filing year "
         "× USPC class, at least five examiners), most lenient vs strictest leniency quintile, cases weighted "
         "n_c / n_cq. Details and differences from Zheng in the script's docstring.", "",
         "## Checks against 35", "", "| Check | This script | 35 | Absolute difference |", "|---|---|---|---|"]
    for c in ck.itertuples():
        L.append(f"| {c.check} | {c.ours:.6f} | {c.c35:.6f} | {c.abs_diff:.1e} |")
    L += ["", "## Figure 1 (fig_c1z_grant_rate; Zheng's Figure 5)", "",
          f"- Test set: {len(r['te']):,} applications in {N_FINE:,} bins of predicted citations "
          f"(about {len(r['te']) / N_FINE:.0f} per bin).",
          f"- Examiners' grant rate: {pct(r['grant_all'])}% overall; {pct(r['grant_low1'])}% in the lowest 1% of "
          f"predicted citations, {pct(r['grant_low5'])}% in the lowest 5%, {pct(r['grant_top5'])}% in the highest "
          f"5%.",
          f"- Smoothed grant rate (LOWESS): {pct(r['gr'].smoothed.min())}% to {pct(r['gr'].smoothed.max())}% across "
          "the bins.", "",
          "## Figure 2 (fig_c1z_reject_cites; Zheng's Figure 6) and Figure 3 (fig_c1z_reject_grant; Zheng's Figure 7)",
          "", f"- {len(r['ev']):,} applications in {len(cl):,} cells.",
          f"- Reject rate: most lenient quintile {pct(rej[5])}%, strictest {pct(rej[1])}% "
          f"(additional {pct(rej[1] - rej[5])} points, = 35's Table III).",
          f"- Cells where the strictest quintile rejects less than the most lenient one: {pct(neg.mean())}% of "
          f"cells, {pct(cl.n[neg].sum() / cl.n.sum())}% of applications; their (negative) extra rejections add "
          f"{pct((cl.n * cl.strict_extra.clip(upper=0)).sum() / cl.n.sum())} points to the "
          f"{pct((cl.n * cl.strict_extra).sum() / cl.n.sum())}-point total "
          f"({pct((cl.n * cl.strict_extra.clip(lower=0)).sum() / cl.n.sum())} from the other cells). There the AI "
          "rules grant the lenient quintile's rejections with the highest score (see the docstring).",
          f"- Bins of predicted citations with a negative strictest − lenient reject rate: {nb_c} of {N_BINS}; "
          f"bins of predicted grant probability: {nb_g} of {N_BINS}.", "",
          f"Share of the additional rejections in the lowest {LOW} of {N_BINS} bins (%):", "",
          "| Series | Bins of predicted citations | Bins of predicted grant probability |", "|---|---|---|",
          f"| Strictest quintile of examiners | {pct(low(rc, 'strict'))} | {pct(low(rg, 'strict'))} |",
          f"| Algorithm (learns citations), within cells | {pct(low(rc, 'algorithm'))} | {pct(low(rg, 'algorithm'))} |",
          f"| Predicted examiner (learns decisions), within cells | {pct(low(rc, 'judge'))} | {pct(low(rg, 'judge'))} |",
          f"| Algorithm, within cells, contract only | {pct(low(r['rc_c'], 'algorithm'))} | "
          f"{pct(low(r['rg_c'], 'algorithm'))} |",
          f"| Predicted examiner, within cells, contract only | {pct(low(r['rc_c'], 'judge'))} | "
          f"{pct(low(r['rg_c'], 'judge'))} |",
          f"| Algorithm, pooled across cells (35's Figure VI) | {pct(f6_alg)} | |", "",
          "Contract only: no extra grants in the cells where the strictest quintile rejects less (shares of the "
          "rule's own total, which then exceeds the strictest quintile's).", "",
          "By bin of predicted citations (reject rates and additional reject rates, %):", "",
          "| Bin | Predicted citations (mean) | Lenient | Strictest | Strictest − lenient | Algorithm | "
          "Predicted examiner |", "|---|---|---|---|---|---|---|"]
    for t in r["rc"].itertuples():
        L.append(f"| {t.bin} | {t.score_mean:.2f} | {pct(t.lenient_reject)} | {pct(t.strict_reject)} | "
                 f"{pct(t.strict_extra)} | {pct(t.algorithm_extra)} | {pct(t.judge_extra)} |")
    L += ["", "By bin of predicted grant probability (%):", "",
          "| Bin | Predicted grant probability (mean) | Lenient | Strictest | Strictest − lenient | "
          "Predicted examiner | Algorithm |", "|---|---|---|---|---|---|---|"]
    for t in r["rg"].itertuples():
        L.append(f"| {t.bin} | {t.score_mean:.3f} | {pct(t.lenient_reject)} | {pct(t.strict_reject)} | "
                 f"{pct(t.strict_extra)} | {pct(t.judge_extra)} | {pct(t.algorithm_extra)} |")
    cal = r["cal"]
    L += ["", "## Figure 4 (fig_c1z_calibration; Zheng's Figure IA.1)", "",
          f"- Granted test applications: {int(cal.n.sum()):,} in {N_CAL} bins; mean predicted citations {pm:.2f}, "
          f"observed {om:.2f} (the Poisson model's level is low, as noted for 35's Figure IV).",
          f"- Bins with observed above predicted: {int((cal.observed > cal.predicted).sum())} of {N_CAL}; "
          f"correlation of bin means {np.corrcoef(cal.predicted, cal.observed)[0, 1]:.3f} "
          f"(on logs {np.corrcoef(np.log(cal.predicted), np.log(cal.observed))[0, 1]:.3f}).",
          f"- Top bin (off the plotted scale): predicted {cal.predicted.iloc[-1]:.1f}, observed "
          f"{cal.observed.iloc[-1]:.1f}.", "", "## Macros (output/latex/c1z_numbers.tex)", ""]
    L += [f"- `\\{k}` = {v.replace('$-$', '−')}" for k, v in N.items()]
    (OUT / "c1z_figures.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- figures

def colours() -> dict:
    C3 = load_module("m44", "44_c3_extra.py")   # the paper's colours (Figure 2 of the main text)
    col = {key: c for key, _, c, _ in C3.FIG_SERIES}
    return dict(examiners=col["Examiners"], judge=col[C3.G], algorithm=col[C3.C], lenient="#C4C4C4")


def stacked(ax, J, t: pd.DataFrame, key: str, col: str, lab: str, cl: dict, ymax: float, xlabel: bool) -> None:
    """Zheng's stacked bars: the most lenient quintile's reject rate, and on top the series' additional reject rate;
    a negative addition is drawn hatched below the top of the lenient bar."""
    from matplotlib.patches import Patch
    x, base, add = t.bin.to_numpy(), 100 * t.lenient_reject.to_numpy(), 100 * t[f"{key}_extra"].to_numpy()
    wd = 0.78
    ax.bar(x, base, width=wd, color=cl["lenient"], lw=0, zorder=2)
    pos, neg = np.clip(add, 0, None), np.clip(add, None, 0)
    ax.bar(x, pos, bottom=base, width=wd, color=col, lw=0, zorder=2)
    if (neg < 0).any():
        ax.bar(x[neg < 0], -neg[neg < 0], bottom=(base + neg)[neg < 0], width=wd, facecolor="white", edgecolor=col,
               hatch="//////", lw=0.4, zorder=3)
    handles = [Patch(color=cl["lenient"], label="Most lenient quintile's rejections"),
               Patch(color=col, label=lab)]
    if (neg < 0).any():
        handles.append(Patch(facecolor="white", edgecolor=col, hatch="//////", lw=0.4,
                             label="Negative: fewer rejections than the most lenient quintile"))
    ax.legend(handles=handles, loc="upper right", fontsize=6, handlelength=1.1, borderaxespad=0.1)
    ax.set_xlim(0.3, N_BINS + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Reject rate (%)")
    if not xlabel:
        ax.set_xticklabels([])
    J.hgrid(ax, "y")


def figures() -> None:
    J = load_module("jf20", "20_journal_figures.py")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5})
    cl = colours()
    gr, cal = pd.read_csv(T / "c1z_grant_rate.csv"), pd.read_csv(T / "c1z_calibration.csv")
    rc, rg = pd.read_csv(T / "c1z_reject_cites.csv"), pd.read_csv(T / "c1z_reject_grant.csv")

    # 1. grant rate by 1,000 bins of predicted citations: (a) final grant, (b) first action, (c) later stage
    from matplotlib.lines import Line2D
    sb, s20 = pd.read_csv(T / "c1z_stage_bins.csv"), pd.read_csv(T / "c1z_stage_bins20.csv")
    sg = pd.read_csv(T / "c1z_stages.csv").set_index("outcome")
    assert np.allclose(sb.granted, gr.grant_rate)               # panel a = Zheng's Figure 5, as before
    fig = J.figure(180, 84)
    spec = [("granted", "a", "Final grant (Zheng's Figure 5)", (30, 100)),
            ("fa_allow", "b", "First action: allowed", (0, 25)),
            ("later", "c", "Granted in the end, first action not allowed", (30, 100))]
    for k, (y, let, title, ylim) in enumerate(spec):
        ax = J.axes(fig, 11 + 58.5 * k, 11, 47, 46)
        ax.scatter(sb.bin, 100 * sb[y], s=1.2, color=cl["examiners"], alpha=0.35, lw=0, zorder=2)
        ax.plot(sb.bin, 100 * sb[f"{y}_smoothed"], color=cl["examiners"], lw=1.3, zorder=3)
        t20 = s20[s20.outcome == y]
        xc = (t20.bin.to_numpy() - 0.5) * N_FINE / N_BINS
        ax.errorbar(xc, 100 * t20["mean"], yerr=[100 * (t20["mean"] - t20.lo), 100 * (t20.hi - t20["mean"])],
                    fmt="o", ms=2.0, color=J.OUTC, elinewidth=0.7, capsize=0, zorder=4)
        r = sg.loc[y]
        ax.axhline(100 * r["mean"], color=J.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-10, 1011)
        ax.set_xticks([1, 500, 1000])
        ax.set_xticklabels(["1", "500", "1,000"])
        ax.set_ylim(*ylim)
        ax.set_xlabel("Bin of predicted citations (1 = lowest)", labelpad=1.5)
        if k == 0:
            ax.set_ylabel("Share of applications (%)")
        d_ = f"{100 * r['diff']:+.1f}".replace("-", "\u2212")
        ci = f"[{100 * r.diff_lo:.1f}, {100 * r.diff_hi:.1f}]".replace("-", "\u2212")
        ax.text(0.0, 1.03, f"Highest 5% \u2212 lowest 5%: {d_} pp {ci}\nAll: {100 * r['mean']:.1f}%",
                transform=ax.transAxes, fontsize=5.8, color=J.INK2, va="bottom")
        J.hgrid(ax, "y")
        J.letter(fig, 1 + 58.5 * k, 72, let, title)
    hs = [Line2D([], [], marker="o", ms=2.0, color=cl["examiners"], alpha=0.6, ls="none"),
          Line2D([], [], color=cl["examiners"], lw=1.3),
          Line2D([], [], marker="o", ms=2.0, color=J.OUTC, ls="-", lw=0.7),
          Line2D([], [], color=J.MUTED, lw=0.7, ls=(0, (3, 2)))]
    fig.legend(hs, [f"1,000 bins of about {gr.n.mean():.0f} test applications", "Smoothed (LOWESS)",
                    "20 bins, 95% CI (art-unit bootstrap)", "All test applications"],
               loc="upper left", bbox_to_anchor=(11 / 180, 1.0), ncol=4, frameon=False, fontsize=6.2,
               handlelength=1.6, columnspacing=1.6)
    J.save(fig, "fig_c1z_grant_rate")

    # 2. reject rates by 20 bins of predicted citations: strictest quintile, algorithm, predicted examiner
    h, gap, bot, top = 37, 9.0, 12, 7.5
    fig = J.figure(120, bot + 3 * h + 2 * gap + top)
    panels = [("strict", cl["examiners"], "Strictest quintile's additional rejections",
               "Strictest quintile of examiners"),
              ("algorithm", cl["algorithm"], "Algorithm's additional rejections",
               "Algorithm (learns citations), within cells"),
              ("judge", cl["judge"], "Predicted examiner's additional rejections",
               "Predicted examiner (learns grant decisions), within cells")]
    for i, (key, col, lab, title) in enumerate(panels):
        y0 = bot + (2 - i) * (h + gap)
        ax = J.axes(fig, 13, y0, 104, h)
        stacked(ax, J, rc, key, col, lab, cl, 100, xlabel=(i == 2))
        sh = rc.loc[rc.bin <= LOW, f"{key}_share"].sum()
        ax.text(0.01, 0.97, f"Bins 1–{LOW}: {100 * sh:.0f}% of the additional rejections".replace("-", "−"),
                transform=ax.transAxes, ha="left", va="top", fontsize=6, color=J.INK2)
        J.letter(fig, 1.5, y0 + h + 5.5, "abc"[i], title)
    ax.set_xlabel("Predicted citations, bin of 5% of applications (1 = lowest)")
    J.save(fig, "fig_c1z_reject_cites")

    # 3. reject rates by 20 bins of the predicted examiner's grant probability
    fig = J.figure(120, bot + 2 * h + gap + top)
    for i, (key, col, lab, title) in enumerate([panels[0], panels[2]]):
        y0 = bot + (1 - i) * (h + gap)
        ax = J.axes(fig, 13, y0, 104, h)
        stacked(ax, J, rg, key, col, lab, cl, 100, xlabel=(i == 1))
        sh = rg.loc[rg.bin <= LOW, f"{key}_share"].sum()
        ax.text(0.99, 0.62, f"Bins 1–{LOW}: {100 * sh:.0f}% of the additional rejections".replace("-", "−"),
                transform=ax.transAxes, ha="right", va="top", fontsize=6, color=J.INK2)
        J.letter(fig, 1.5, y0 + h + 5.5, "ab"[i], title)
    ax.set_xlabel("Predicted examiner's grant probability, bin of 5% of applications (1 = lowest)")
    J.save(fig, "fig_c1z_reject_grant")

    # 4. calibration: observed against predicted citations of granted test applications, 200 bins
    fig = J.figure(90, 86)
    ax = J.axes(fig, 12, 11, 72, 68)
    on = cal.iloc[:-1]
    lim = float(np.ceil(max(on.predicted.max(), on.observed.max()) * 1.1))
    ax.plot([0, lim], [0, lim], color=J.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1, label="45-degree line")
    ax.scatter(on.predicted, on.observed, s=4, color=cl["algorithm"], alpha=0.7, lw=0, zorder=3,
               label=f"Bin of about {cal.n.mean():.0f} granted applications")
    s = cal.smoothed.to_numpy()
    keep = (cal.predicted <= lim).to_numpy()
    ax.plot(cal.predicted[keep], s[keep], color=cl["algorithm"], lw=1.2, zorder=4, label="Smoothed (LOWESS)")
    top = cal.iloc[-1]
    ax.text(0.98 * lim, 0.04 * lim, f"Not shown (off scale): the top bin,\npredicted {top.predicted:.1f}, observed "
            f"{top.observed:.1f}", ha="right", va="bottom", fontsize=6, color=J.INK2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(np.arange(0, lim + 0.1, 5))
    ax.set_yticks(np.arange(0, lim + 0.1, 5))
    ax.set_xlabel("Predicted citations (algorithm)")
    ax.set_ylabel("Observed citations within five years")
    ax.legend(loc="upper left")
    J.hgrid(ax, "both")
    J.letter(fig, 1.5, 84.5, "", "Observed against predicted citations, granted test applications")
    J.save(fig, "fig_c1z_calibration")
    print("saved", ", ".join(f"fig_c1z_{n}" for n in ["grant_rate", "reject_cites", "reject_grant", "calibration"]))


def main() -> None:
    warnings.filterwarnings("ignore")
    if "--figure" not in os.sys.argv:
        r = compute(load_module("m35", "35_c1_kllm.py"))
        write(r)
    figures()


if __name__ == "__main__":
    main()
