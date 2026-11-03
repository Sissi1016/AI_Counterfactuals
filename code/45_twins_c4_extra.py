"""AI examiner twins (C2), for final appendix Tables A30–A31 and Figures A8–A9.

Rebuilds the 20 twins, verifies their predictions against script 17, and computes
decision changes, gender gaps, and threshold-zone shares. Numerical fitting is
single-threaded within each worker. The file number is retained for traceability.
Run with the main Python environment and CFE_THREADS set explicitly.
"""
import os

THREADS = int(os.environ.get("CFE_THREADS", "4"))
# joblib workers inherit these, so the twins' cross-validation runs THREADS single-threaded processes
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS",
           "RAYON_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
os.environ["NUMBA_NUM_THREADS"] = str(THREADS)

import functools  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
from joblib import Parallel  # noqa: E402
from scipy.stats import norm  # noqa: E402

pa.set_cpu_count(THREADS)
pa.set_io_thread_count(THREADS)
duckdb.connect = functools.partial(duckdb.connect, config={"threads": THREADS})  # also for cfe.check.load_record

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
T = OUT / "tables"
TEX = OUT / "latex"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
GROUPS = ["0–2", "3–5", "6 or more"]
GROUP_LABELS = ["Strictest five", "Second five", "Third five", "Most lenient five"]
CIT = "Train on citations (granted only)"
GRANT = "Train on grant"
N_STAGE2 = 15  # Stage 2: the 15 examiners with the most applications in each center


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


J = load_module("jf20", "20_journal_figures.py")
from cfe.check import PATENTS, assign_folds, recalibrate  # noqa: E402
from cfe.estimation import cluster_se, tsls  # noqa: E402


def mean_ci(y: np.ndarray, cl) -> tuple[float, float]:
    """Mean and its standard error clustered by cl."""
    psi = ((y - y.mean()) / len(y))[:, None]
    return y.mean(), cluster_se(psi, cl)[0]


def diff_ci(y: np.ndarray, g: np.ndarray, cl) -> tuple[float, float]:
    """Difference in means (g = 1 minus g = 0) and its standard error clustered by cl."""
    b, psi = tsls(y[:, None], g.astype(float), g.astype(float), np.zeros(len(y)))
    return b[0], cluster_se(psi, cl)[0]


# ---------------------------------------------------------------- C2: rebuild 17's twins

def build_twins() -> dict:
    """17's twins, single-threaded whatever the caller's thread settings: with one feature the twins' penalties are
    chosen among near-ties, and multithreaded linear algebra (e.g. numpy imported before this module set
    OMP_NUM_THREADS=1) can flip a choice and change a twin's grant rate."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _build_twins()


def _build_twins() -> dict:
    """17's main() up to the probabilities, unchanged (features s0, as 17 picks when the q8 inputs are absent)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from cfe.check import load_record
    m17 = load_module("m17", "17_ai_examiners.py")
    # 17 runs 12 cross-validation fits at once; cap the processes at THREADS
    m17.Parallel = lambda n_jobs=None, **kw: Parallel(n_jobs=min(THREADS, n_jobs or THREADS), **kw)
    assert not (P / "emb/ai_ex_q8.npy").exists(), "17 would use q8 features; rebuild with them"
    d = load_record(ROOT)
    emb = np.load(P / "emb/s0.npy", mmap_mode="r")
    xa = pd.read_parquet(P / "ai_examiner_apps.parquet", columns=["application_number", "examiner_id", "granted",
                                                                  "au_year"])
    xemb = np.load(P / "emb/ai_ex_s0.npy").astype(np.float32)
    tc_rows = np.where(d.tech_center == m17.TC)[0]
    counts = d.iloc[tc_rows].groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort")
    top = counts.head(m17.N_EXAMINERS).index.tolist()
    others = tc_rows[~d.examiner_id.iloc[tc_rows].isin(top).to_numpy()]
    E_tc = np.asarray(emb[tc_rows], dtype=np.float32)
    shared, _, _ = m17.fit_shared(np.asarray(emb[others], dtype=np.float32), d.granted.to_numpy()[others].astype(int),
                                  pd.factorize(d.examiner_id.iloc[others])[0])
    pool = d[d.examiner_id.isin(top)].copy()
    pool_rows = pool.index.to_numpy()
    pool["shared"] = shared(np.asarray(emb[pool_rows], dtype=np.float32))
    xa["shared"] = shared(xemb)
    sc_all = StandardScaler().fit(E_tc)
    Z = PCA(m17.N_PC, random_state=m17.SEED).fit(sc_all.transform(E_tc)) if m17.N_PC > 0 else None

    def feats(shared_col, E):
        return np.column_stack([shared_col] + ([Z.transform(sc_all.transform(E))] if Z is not None else []))
    X_full = feats(pool.shared.to_numpy(), np.asarray(emb[pool_rows], dtype=np.float32))
    X_len = pool[["shared"]].to_numpy()
    Xx_full = feats(xa.shared.to_numpy(), xemb)
    Xx_len = xa[["shared"]].to_numpy()
    probs = {"full": {}, "leniency": {}, "leniency_c1": {}}
    grid = list(m17.C_GRID)
    rows = []
    for i, e in enumerate(top):
        own = (pool.examiner_id == e).to_numpy()
        ext = (xa.examiner_id == e).to_numpy()
        y, yx = pool.granted.to_numpy()[own].astype(int), xa.granted.to_numpy()[ext].astype(int)
        res = {}
        # "leniency_c1": the leniency-only twin with the weakest penalty of 17's grid (C = 1), i.e. the examiner's own
        # logistic intercept and slope on the common score. With one feature the held-out AUC hardly depends on the
        # penalty, so 17's choice by AUC can shrink the slope to almost zero: thresholded decisions are unaffected
        # (any positive slope gives the same ranking), probabilities are not
        for kind, X, Xx in [("full", X_full, Xx_full), ("leniency", X_len, Xx_len), ("leniency_c1", X_len, Xx_len)]:
            m17.C_GRID = grid[:1] if kind == "leniency_c1" else grid
            oof, m, sc, auc = m17.fit_cv(Xx[ext], yx, X[own], y, m17.SEED + i)
            p = m.predict_proba(sc.transform(X))[:, 1]
            p[own] = oof
            probs[kind][e] = p
            res[kind] = auc
            res[f"C_{kind}"] = m.C
        m17.C_GRID = grid
        g = pool[own]
        rows.append(dict(examiner=e, n=int(own.sum()), human_leniency_loo=g.lenient_grant.iloc[0],
                         auc_full=res["full"], auc_leniency_only=res["leniency"], auc_c1=res["leniency_c1"],
                         C_leniency=res["C_leniency"], C_full=res["C_full"]))
    ex = pd.DataFrame(rows).sort_values("human_leniency_loo").reset_index(drop=True)
    order = ex.examiner.tolist()
    return dict(pool=pool.reset_index(drop=True), xa=xa, ex=ex, order=order,
                Pl=np.column_stack([probs["leniency"][e] for e in order]),
                Pf=np.column_stack([probs["full"][e] for e in order]),
                P1=np.column_stack([probs["leniency_c1"][e] for e in order]))


def decide(p: np.ndarray) -> np.ndarray:
    """17's decisions: grant as many as the probabilities imply, those rated highest."""
    return (p >= np.quantile(p, 1 - p.mean())).astype(int)


def check_rebuild(tw: dict) -> float:
    """Largest difference from 17's published twins (grant rates and AUCs, by twin number)."""
    pub = pd.read_csv(T / "ai_examiners.csv").sort_values("ai_examiner")
    D = np.column_stack([decide(tw["Pl"][:, j]) for j in range(tw["Pl"].shape[1])])
    diffs = [np.abs(tw["Pl"].mean(0) - pub.ai_grant_rate_leniency.to_numpy()).max(),
             np.abs(tw["Pf"].mean(0) - pub.ai_grant_rate_full.to_numpy()).max(),
             np.abs(tw["ex"].auc_leniency_only.to_numpy() - pub.auc_leniency_only.to_numpy()).max(),
             np.abs(tw["ex"].auc_full.to_numpy() - pub.auc_full.to_numpy()).max(),
             np.abs(tw["ex"].human_leniency_loo.to_numpy() - pub.human_leniency_loo.to_numpy()).max()]
    k = D.sum(1)
    lot = pd.read_csv(T / "ai_examiners_lottery.csv").set_index("group")
    diffs.append(abs(np.mean((k > 0) & (k < D.shape[1])) - lot.loc["All", "depends_on_examiner"]))
    return float(max(diffs))


# ---------------------------------------------------------------- C2: 1. groups of five

def own_gaps(tw: dict) -> pd.DataFrame:
    """Each examiner's own female − male grant gap within the examiner's art unit × year cells: on the examiner's 2010–2014
    applications (the common applications), and on all decisions the twin learned from (2004–2018)."""
    pool, xa = tw["pool"], tw["xa"]
    con = duckdb.connect()
    con.register("x", xa[["application_number"]])
    gx = con.execute(f"""SELECT x.application_number, CASE g.first_inventor_gender WHEN 'F' THEN 1.0
                                WHEN 'M' THEN 0.0 END AS female
                         FROM x LEFT JOIN '{P}/app_gender.parquet' g USING (application_number)""").df()
    xa = xa.merge(gx, on="application_number", how="left", validate="1:1")
    both = pd.concat([pool[["examiner_id", "au_year", "granted", "female"]],
                      xa[["examiner_id", "au_year", "granted", "female"]]], ignore_index=True)
    rows = []
    for e in tw["order"]:
        out = dict(examiner=e)
        for lab, s in [("pilot", pool[pool.examiner_id == e]), ("all", both[both.examiner_id == e])]:
            s = s[s.female.notna()]
            nf = int(s.female.sum())
            out[f"n_female_{lab}"] = nf
            if nf == 0 or nf == len(s):
                out[f"gap_{lab}"] = np.nan
                continue
            b, _ = tsls(s[["granted"]].to_numpy(float), s.female.to_numpy(float), s.female.to_numpy(float), s.au_year)
            out[f"gap_{lab}"] = b[0]
        rows.append(out)
    return pd.DataFrame(rows), both


def group_table(tw: dict) -> tuple[pd.DataFrame, dict]:
    pool, ex, order = tw["pool"], tw["ex"].copy(), tw["order"]
    f = pool.female.to_numpy(float)
    Dl = np.column_stack([decide(tw["Pl"][:, j]) for j in range(len(order))]).astype(float)
    Df = np.column_stack([decide(tw["Pf"][:, j]) for j in range(len(order))]).astype(float)
    out = {}
    D1 = np.column_stack([decide(tw["P1"][:, j]) for j in range(len(order))]).astype(float)
    for lab, Y in [("gap_dec", Dl), ("gap_prob", tw["Pl"]), ("gap_dec_text", Df), ("gap_prob_text", tw["Pf"]),
                   ("gap_dec_c1", D1), ("gap_prob_c1", tw["P1"])]:
        b, psi = tsls(Y, f, f, pool.au_year)
        ex[lab], ex[lab + "_se"] = b, cluster_se(psi, pool.examiner_id)
    ex["grant_rate"], ex["grant_rate_text"], ex["grant_rate_c1"] = tw["Pl"].mean(0), tw["Pf"].mean(0), tw["P1"].mean(0)
    ex["p_sd"], ex["p_sd_c1"] = tw["Pl"].std(0), tw["P1"].std(0)
    og, both = own_gaps(tw)
    ex = ex.merge(og, on="examiner", how="left", validate="1:1")
    ex["grp"] = np.arange(len(ex)) * 4 // len(ex)
    # human gap on the common applications (17's definition) and pooled within examiner × art unit × year
    hb, hpsi = tsls(pool[["granted"]].to_numpy(float), f, f, pool.au_year)
    out["human_gap"], out["human_gap_se"] = hb[0], cluster_se(hpsi, pool.examiner_id)[0]
    rows = []

    def pooled_gap(s: pd.DataFrame) -> tuple[float, float]:
        s = s[s.female.notna()]
        b, psi = tsls(s[["granted"]].to_numpy(float), s.female.to_numpy(float), s.female.to_numpy(float),
                      s.examiner_id + "|" + s.au_year)
        return b[0], cluster_se(psi, s.examiner_id + "|" + s.au_year)[0]
    for k, g in list(ex.groupby("grp")) + [(4, ex)]:
        ids = set(g.examiner)
        pg, pse = pooled_gap(pool[pool.examiner_id.isin(ids)])
        ag, ase = pooled_gap(both[both.examiner_id.isin(ids)])
        rows.append(dict(group=GROUP_LABELS[k] if k < 4 else "All 20", twins=len(g),
                         leniency_median=g.human_leniency_loo.median(), grant_rate_median=g.grant_rate.median(),
                         grant_rate_text_median=g.grant_rate_text.median(),
                         auc_median=g.auc_leniency_only.median(), auc_text_median=g.auc_full.median(),
                         own_gap_median=g.gap_pilot.median(), own_gap_n=int(g.gap_pilot.notna().sum()),
                         own_gap_pooled=pg, own_gap_pooled_se=pse,
                         own_gap_all_years_median=g.gap_all.median(), own_gap_all_years_pooled=ag,
                         own_gap_all_years_pooled_se=ase,
                         female_apps_median=g.n_female_pilot.median(),
                         twin_gap_dec_median=g.gap_dec.median(), twin_gap_prob_median=g.gap_prob.median(),
                         twin_gap_dec_text_median=g.gap_dec_text.median(),
                         twin_gap_prob_text_median=g.gap_prob_text.median(),
                         twin_gap_dec_c1_median=g.gap_dec_c1.median(), twin_gap_prob_c1_median=g.gap_prob_c1.median(),
                         grant_rate_c1_median=g.grant_rate_c1.median(), auc_c1_median=g.auc_c1.median(),
                         # group means for the appendix table: a median of five is one examiner's value
                         leniency_mean=g.human_leniency_loo.mean(), grant_rate_mean=g.grant_rate.mean(),
                         grant_rate_text_mean=g.grant_rate_text.mean(),
                         auc_mean=g.auc_leniency_only.mean(), twin_gap_dec_mean=g.gap_dec.mean(),
                         twin_gap_prob_mean=g.gap_prob.mean()))
    out["n_female_examiner_median"] = ex.n_female_pilot.median()
    out["n_examiners_lt10_female"] = int((ex.n_female_pilot < 10).sum())
    out["n_examiners_no_female"] = int((ex.n_female_pilot == 0).sum())
    # penalties chosen by 17 (counts only) and how flat the leniency-only twins' probabilities are
    out["n_strong_penalty"] = int((ex.C_leniency <= 1e-3).sum())
    out["penalty_counts"] = ", ".join(f"C = {c:g}: {k}" for c, k in ex.C_leniency.value_counts().sort_index(
        ascending=False).items())
    out["n_weakest_penalty"] = int((ex.C_leniency == 1.0).sum())
    out["n_flat"] = int((ex.p_sd < 0.025).sum())
    out["n_flat_c1"] = int((ex.p_sd_c1 < 0.025).sum())
    out["p_sd_median"], out["p_sd_c1_median"] = float(ex.p_sd.median()), float(ex.p_sd_c1.median())
    out["corr_own_twin_gap"] = float(ex.gap_pilot.corr(ex.gap_dec))
    tw["ex_full"] = ex  # stays in memory only
    return pd.DataFrame(rows), out


# ---------------------------------------------------------------- C2: 2. decision-change probability

def pair_dec(D: np.ndarray) -> np.ndarray:
    """Per application: probability that two distinct twins drawn at random decide differently."""
    n, k = D.shape[1], D.sum(1)
    return 2 * k * (n - k) / (n * (n - 1))


def pair_prob(Pm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per application, decisions drawn from the probabilities: two distinct twins, and the same twin twice."""
    n, m, v = Pm.shape[1], Pm.mean(1), Pm.var(1)
    return 2 * m * (1 - m) + 2 * v / (n - 1), 2 * m * (1 - m) - 2 * v


def thresholds(s: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Decisions of twins that rank by one common score s and grant the top r_j."""
    return np.column_stack([s >= np.quantile(s, 1 - r) for r in rates]).astype(int)


def change_table(tw: dict) -> tuple[pd.DataFrame, dict, dict]:
    pool, ex = tw["pool"], tw["ex_full"]
    s = pool.shared.to_numpy()
    fem = pool.female.to_numpy() == 1
    cl = pool.examiner_id.to_numpy()
    Dl = np.column_stack([decide(tw["Pl"][:, j]) for j in range(tw["Pl"].shape[1])])
    Df = np.column_stack([decide(tw["Pf"][:, j]) for j in range(tw["Pf"].shape[1])])
    D1 = np.column_stack([decide(tw["P1"][:, j]) for j in range(tw["P1"].shape[1])])
    prob, same = pair_prob(tw["Pl"])
    prob_f, same_f = pair_prob(tw["Pf"])
    prob_1, same_1 = pair_prob(tw["P1"])
    r_tw, r_ex = ex.grant_rate.to_numpy(), ex.human_leniency_loo.to_numpy()
    k = Dl.sum(1)
    n = Dl.shape[1]
    # twins that know only their grant rate: every application granted with probability r_j
    prob_r, same_r = pair_prob(np.tile(r_tw, (len(pool), 1)))
    A, B = "Thresholded decisions", "Decisions drawn independently from each twin's probability"
    measures = [
        (A, "Leniency alone: one common ranking, thresholds at the examiners' leave-one-out grant rates", "len_exam",
         pair_dec(thresholds(s, r_ex))),
        (A, "Leniency alone: one common ranking, thresholds at the twins' grant rates", "len_twin",
         pair_dec(thresholds(s, r_tw))),
        (A, "Leniency-only twins (each grants its top share, as 17)", "dec", pair_dec(Dl)),
        (A, "Leniency-only twins, weakest penalty (C = 1)", "dec_c1", pair_dec(D1)),
        (A, "Twins with 32 text components", "dec_text", pair_dec(Df)),
        (B, "Grant rates alone (each twin grants every application with its grant rate)", "rate", prob_r),
        (B, "  of which difference across twins", "rate_between", prob_r - same_r),
        (B, "Leniency-only twins (17's penalties)", "prob", prob),
        (B, "  of which same twin drawing twice", "prob_same", same),
        (B, "  of which difference across twins", "prob_between", prob - same),
        (B, "Leniency-only twins, weakest penalty (C = 1)", "prob_c1", prob_1),
        (B, "  of which same twin drawing twice", "prob_c1_same", same_1),
        (B, "  of which difference across twins", "prob_c1_between", prob_1 - same_1),
        (B, "Twins with 32 text components", "prob_text", prob_f),
        (B, "  of which difference across twins", "prob_text_between", prob_f - same_f),
        ("Earlier layout (17)", "Granted by all 20 twins", "all", (k == n).astype(float)),
        ("Earlier layout (17)", "Granted by none", "none", (k == 0).astype(float)),
        ("Earlier layout (17)", "Depends on the twin (1–19 of 20 grant)", "depends", ((k > 0) & (k < n)).astype(float)),
        ("Earlier layout (17)", "Depends on the twin, twins with text", "depends_text",
         ((Df.sum(1) > 0) & (Df.sum(1) < n)).astype(float))]
    rows, per_app = [], {}
    for panel, lab, key, y in measures:
        y = np.asarray(y, float)
        per_app[key] = y
        r = dict(panel=panel, measure=lab, key=key)
        for g, m in [("all", np.ones(len(y), bool)), ("female", fem), ("male", ~fem)]:
            r[g], r[g + "_se"] = mean_ci(y[m], cl[m])
        r["f_minus_m"], r["f_minus_m_se"] = diff_ci(y, fem, cl)
        rows.append(r)
    # closed forms from the grant rates alone
    iu = np.triu_indices(n, 1)
    S = dict(n_common=len(pool), n_female=int(fem.sum()), n_male=int((~fem).sum()),
             rates_twins_pairs=float(np.abs(r_tw[:, None] - r_tw[None, :])[iu].mean()),
             rates_exam_pairs=float(np.abs(r_ex[:, None] - r_ex[None, :])[iu].mean()),
             range_twins=float(r_tw.max() - r_tw.min()), range_exam=float(r_ex.max() - r_ex.min()),
             twin_rank_corr_min=float(min(pd.Series(tw["Pl"][:, j]).corr(pd.Series(s), method="spearman")
                                        for j in range(n))))
    # where the thresholds are: share of applications between the strictest and the most lenient twin's threshold,
    # and mean common-score percentile, by gender
    lo, hi = np.quantile(s, 1 - r_tw.max()), np.quantile(s, 1 - r_tw.min())
    pct = pd.Series(s).rank(pct=True).to_numpy()
    for g, m in [("female", fem), ("male", ~fem)]:
        S[f"between_{g}"] = float(np.mean((s[m] >= lo) & (s[m] < hi)))
        S[f"below_{g}"] = float(np.mean(s[m] < lo))
        S[f"above_{g}"] = float(np.mean(s[m] >= hi))
        S[f"score_pct_{g}"] = float(np.median(pct[m]))
    # pairs: share decided differently against the gap in the two twins' grant rates (for figure panel b)
    pr = []
    for a, b in zip(*iu):
        pr.append(dict(gap=abs(r_tw[a] - r_tw[b]), gap_text=abs(Df[:, a].mean() - Df[:, b].mean()),
                       dec=np.mean(Dl[:, a] != Dl[:, b]), dec_text=np.mean(Df[:, a] != Df[:, b]),
                       prob=np.mean(tw["Pl"][:, a] * (1 - tw["Pl"][:, b]) + tw["Pl"][:, b] * (1 - tw["Pl"][:, a])),
                       prob_text=np.mean(tw["Pf"][:, a] * (1 - tw["Pf"][:, b])
                                         + tw["Pf"][:, b] * (1 - tw["Pf"][:, a]))))
    pr = pd.DataFrame(pr)
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.51]
    labs = ["0–10", "10–20", "20–30", "30–40", "40–51"]
    binned = []
    for x, cols in [("gap", ["dec", "prob"]), ("gap_text", ["dec_text", "prob_text"])]:
        b = pd.cut(pr[x], edges, right=False, labels=labs)
        for lab, g in pr.groupby(b, observed=True):
            binned.append(dict(twins="leniency only" if x == "gap" else "with text", bin_pp=lab, pairs=len(g),
                               grant_rate_gap_mean=g[x].mean(), decisions_differ=g[cols[0]].mean(),
                               decisions_differ_drawn=g[cols[1]].mean()))
    return pd.DataFrame(rows), S, dict(per_app=per_app, binned=pd.DataFrame(binned))


# ---------------------------------------------------------------- C2: 4. women among the common applications

def women(tw: dict, ch: pd.DataFrame, S: dict, G: dict) -> pd.DataFrame:
    a = pd.read_parquet(P / "all_sample.parquet", columns=["examiner_id", "tech_center", "female"])
    rows = [dict(sample="Common applications (20 TC 3600 examiners)", applications=S["n_common"],
                 female=S["n_female"])]
    for tc in ["3600"] + [t for t in TCS if t != "3600"]:
        x = a[a.tech_center == tc]
        rows.append(dict(sample=f"TC {tc}, 2010–2014", applications=len(x), female=int(x.female.sum())))
    rows.append(dict(sample="Eight centers, 2010–2014", applications=len(a), female=int(a.female.sum())))
    # Stage 2: the 15 examiners with the most applications in each center
    tops = []
    for tc in TCS:
        x = a[a.tech_center == tc]
        c = x.groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort").head(N_STAGE2)
        tops.append(x[x.examiner_id.isin(c.index)])
    tops = pd.concat(tops)
    rows.append(dict(sample=f"Stage 2: {N_STAGE2} busiest examiners in each of the eight centers",
                     applications=len(tops), female=int(tops.female.sum())))
    w = pd.DataFrame(rows)
    w["female_share"] = w.female / w.applications
    # implied precision: the human gap's 95% half-width on the common applications, scaled by 1 / sqrt(women)
    hw = 1.96 * G["human_gap_se"]
    w["gap_ci_halfwidth_scaled_pp"] = 100 * hw * np.sqrt(S["n_female"] / w.female)
    return w


# ---------------------------------------------------------------- C2: 3. figure

def robust_change(tw: dict) -> tuple[pd.DataFrame, dict]:
    """Decision change on reassignment, thresholded decisions only, by gender, with intervals that do not rest on the
    20 examiner clusters: a bootstrap over applications within gender (2,000 draws) and a leave-one-twin-out jackknife
    of the female - male difference. Measures: one common ranking with thresholds at the twins' grant rates (the
    benchmark), the leniency-only twins, and the twins with text. Also the female - male difference in the share of
    applications between the strictest and the most lenient twin's threshold, with an application bootstrap."""
    pool, ex = tw["pool"], tw["ex_full"]
    s = pool.shared.to_numpy()
    fem = pool.female.to_numpy() == 1
    Dl = np.column_stack([decide(tw["Pl"][:, j]) for j in range(tw["Pl"].shape[1])]).astype(float)
    Df = np.column_stack([decide(tw["Pf"][:, j]) for j in range(tw["Pf"].shape[1])]).astype(float)
    r_tw = ex.grant_rate.to_numpy()
    n = Dl.shape[1]

    def per_app(keep):
        return {"len_twin": pair_dec(thresholds(s, r_tw[keep])), "dec": pair_dec(Dl[:, keep]),
                "dec_text": pair_dec(Df[:, keep])}
    full = per_app(np.arange(n))
    rng = np.random.default_rng(20261101 + 45)
    fi, mi = np.where(fem)[0], np.where(~fem)[0]
    BB = 2000
    rows = []
    for k, v in full.items():
        bf = np.array([v[rng.choice(fi, len(fi))].mean() for _ in range(BB)])
        bm = np.array([v[rng.choice(mi, len(mi))].mean() for _ in range(BB)])
        jk = []
        for j in range(n):
            vv = per_app(np.delete(np.arange(n), j))[k]
            jk.append(vv[fem].mean() - vv[~fem].mean())
        jk = np.array(jk)
        rows.append(dict(key=k, female=v[fem].mean(), male=v[~fem].mean(), f_minus_m=v[fem].mean() - v[~fem].mean(),
                         female_lo=np.percentile(bf, 2.5), female_hi=np.percentile(bf, 97.5),
                         male_lo=np.percentile(bm, 2.5), male_hi=np.percentile(bm, 97.5),
                         f_minus_m_lo=np.percentile(bf - bm, 2.5), f_minus_m_hi=np.percentile(bf - bm, 97.5),
                         f_minus_m_se_jackknife=np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())))
    lo, hi = np.quantile(s, 1 - r_tw.max()), np.quantile(s, 1 - r_tw.min())
    between = ((s >= lo) & (s < hi)).astype(float)
    bt = np.array([between[rng.choice(fi, len(fi))].mean() - between[rng.choice(mi, len(mi))].mean()
                   for _ in range(BB)])
    Z = dict(between_f_minus_m=between[fem].mean() - between[~fem].mean(), between_f_minus_m_lo=np.percentile(bt, 2.5),
             between_f_minus_m_hi=np.percentile(bt, 97.5))
    return pd.DataFrame(rows), Z


def figure(gt: pd.DataFrame, ch: pd.DataFrame, extra: dict, S: dict, R: pd.DataFrame) -> None:
    """Two figures, no mark for one examiner. fig_c2_twins_ab: (a) group means of twins' against examiners' grant
    rates, (b) pairs of twins binned by the gap in their grant rates. fig_c2_twins_cd: (c) decision change on
    reassignment, thresholded decisions only (benchmark: one ranking with the twins' own grant rates), by gender, 95%
    CI from an application bootstrap; (d) where women's and men's applications sit relative to the twins' thresholds.
    Decisions drawn from the twins' probabilities are in the table, not here: they mix in the same twin drawing twice.
    Legends sit above the plot areas."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    load_module("jf20_style", "20_journal_figures.py")   # 17, loaded by build_twins, resets matplotlib's style
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig = J.figure(150, 78)
    g4 = gt[gt.group != "All 20"]
    ax = J.axes(fig, 14, 11, 52, 53)
    lim = (35, 100)
    ax.plot(lim, lim, color=J.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(97, 93, "equal", ha="right", va="top", fontsize=5.5, color=J.INK2, rotation=45, rotation_mode="anchor")
    x, y = 100 * g4.leniency_mean.to_numpy(), 100 * g4.grant_rate_mean.to_numpy()
    ax.plot(x, y, color=J.DEC, lw=0.7, zorder=2)
    ax.plot(x, 100 * g4.grant_rate_text_mean.to_numpy(), marker="o", ms=3.4, color="white", mec=J.DEC, mew=0.8,
            ls="none", zorder=3)
    ax.plot(x, y, marker="o", ms=4.0, color=J.DEC, mec="white", mew=0.6, ls="none", zorder=4)
    for xi, yi, lab in zip(x, y, ["strictest\nfive", "second\nfive", "third\nfive", "most lenient\nfive"]):
        ax.text(xi + 1.8, yi - 3.0, lab, ha="left", va="top", fontsize=5.6, color=J.INK2, linespacing=0.95)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Examiners' grant rate, leave-one-out (%)", labelpad=1.5)
    ax.set_ylabel("Twins' grant rate, common applications (%)", labelpad=2)
    ax.text(0.97, 0.04, f"means of four groups of five;\nr = {S['corr_rates']:.2f} across the 20", ha="right",
            va="bottom", transform=ax.transAxes, fontsize=5.6, color=J.INK2)
    J.hgrid(ax, "both")
    ha = [Line2D([], [], marker="o", ms=4.0, color=J.DEC, mec="white", mew=0.6, ls="none"),
          Line2D([], [], marker="o", ms=3.4, color="white", mec=J.DEC, mew=0.8, ls="none")]
    ax.legend(ha, ["leniency-only twins", "twins with text"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              fontsize=5.8, handletextpad=0.2, columnspacing=1.0, frameon=False, borderaxespad=0.3)
    J.letter(fig, 1, 76, "a", "Twins' grant rates against examiners' grant rates")
    b = extra["binned"]
    ax = J.axes(fig, 92, 11, 54, 53)
    ax.plot([0, 50], [0, 50], color=J.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for twins, filled in [("leniency only", True), ("with text", False)]:
        t = b[b.twins == twins]
        for col, mk in [("decisions_differ", "o"), ("decisions_differ_drawn", "s")]:
            kw = dict(color=J.DEC, mec="white", mew=0.5) if filled else dict(color="white", mec=J.DEC, mew=0.8)
            ax.plot(100 * t.grant_rate_gap_mean, 100 * t[col], color=J.DEC, lw=0.6 if filled else 0.5,
                    ls="-" if filled else (0, (1, 1)), zorder=2)
            ax.plot(100 * t.grant_rate_gap_mean, 100 * t[col], marker=mk, ms=3.2 if mk == "o" else 2.8, ls="none",
                    zorder=3, **kw)
    hs = [Line2D([], [], marker="o", ms=3.2, color=J.DEC, mec="white", mew=0.5, ls="-", lw=0.6),
          Line2D([], [], marker="s", ms=2.8, color=J.DEC, mec="white", mew=0.5, ls="-", lw=0.6),
          Line2D([], [], marker="o", ms=3.2, color="white", mec=J.DEC, mew=0.8, ls=(0, (1, 1)), lw=0.5),
          Line2D([], [], color=J.MUTED, lw=0.6, ls=(0, (2, 2)))]
    ax.legend(hs, ["thresholded decisions", "drawn from probabilities", "hollow: twins with text",
                   "leniency alone (one ranking)"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              fontsize=5.6, handlelength=1.8, columnspacing=1.0, frameon=False, borderaxespad=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Gap between two twins' grant rates (pp)", labelpad=1.5)
    ax.set_ylabel("Decisions that differ (%)", labelpad=2)
    J.hgrid(ax, "both")
    J.letter(fig, 79, 76, "b", "Decisions that differ, by gap in grant rates")
    J.save(fig, "fig_c2_twins_ab")

    fig = J.figure(150, 76)
    ax = J.axes(fig, 36, 11, 50, 54)
    items = [("Thresholds only,\ntwins' grant rates", "len_twin"), ("Twins' decisions", "dec"),
             ("Twins' decisions,\nwith text", "dec_text")]
    Rk = R.set_index("key")
    for i, (lab, key) in enumerate(items):
        r = Rk.loc[key]
        yy = len(items) - i
        for g, col, dy in [("female", J.INK, 0.15), ("male", J.MUTED, -0.15)]:
            J.dot(ax, 100 * r[g], yy + dy, col, 100 * r[g + "_lo"], 100 * r[g + "_hi"], size=3.0, lw=0.8)
        ax.text(-0.03, yy, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.8,
                color=J.INK, linespacing=0.95)
        ax.text(1.0, yy, f"F$-$M {100 * r.f_minus_m:+.1f}\n[{100 * r.f_minus_m_lo:.1f}, {100 * r.f_minus_m_hi:.1f}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.4, color=J.INK2)
    ax.set_ylim(0.4, len(items) + 0.6)
    ax.set_xlim(0, 45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    J.hgrid(ax, "x")
    ax.set_xlabel("Probability the decision changes (%)", labelpad=1.5)
    hs = [Line2D([], [], marker="o", ms=3.0, color=col, mec="white", mew=0.6, ls="-", lw=0.8) for col in
          [J.INK, J.MUTED]]
    ax.legend(hs, ["female first inventor", "male first inventor"], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, fontsize=5.8, handlelength=1.4, columnspacing=1.2, frameon=False, borderaxespad=0.3)
    J.letter(fig, 1, 74, "c", "Reassignment to another twin (thresholded decisions)")
    ax = J.axes(fig, 112, 11, 34, 44)
    zones = [("below", "#cfcdc6", "rejected by every twin"), ("between", J.DEC, "depends on the twin"),
             ("above", "#9fb7cf", "granted by every twin")]
    for i, (g, lab) in enumerate([("female", "Women"), ("male", "Men")]):
        left = 0
        for zone, col, _ in zones:
            w = 100 * S[f"{zone}_{g}"]
            ax.barh(1 - i, w, left=left, color=col, height=0.55, edgecolor="white", lw=0.5)
            ax.text(left + w / 2, 1 - i, f"{w:.0f}%", ha="center", va="center", fontsize=5.8,
                    color="white" if zone == "between" else J.INK)
            left += w
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Women", "Men"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of common applications (%)", labelpad=1.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    hz = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in zones]
    ax.legend(hz, [lab for _, _, lab in zones], loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=1, fontsize=5.4,
              frameon=False, borderaxespad=0.3, handlelength=1.0)
    J.letter(fig, 96, 74, "d", "Where applications sit, twins' thresholds")
    J.save(fig, "fig_c2_twins_cd")


# ---------------------------------------------------------------- C4



COV = [("female", "Female first inventor (pp)", 100), ("small_entity", "Small entity (pp)", 100),
       ("n_inventors", "Inventors", 1), ("g_score", "Predicted grant score (log-odds)", 1),
       ("pred_cites", "Predicted citations", 1)]








# ---------------------------------------------------------------- output helpers

def tabular(path: pathlib.Path, colspec: str, header: list, rows: list, groups: dict | None = None) -> None:
    """A booktabs tabular as 13 writes it. rows: lists of LaTeX strings; groups: {row index: panel title}."""
    L = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if isinstance(r, str):  # a raw line, e.g. \\midrule
            L.append(r)
            continue
        if groups and i in groups:
            if i:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{len(header)}}}{{@{{}}l}}{{\textit{{{groups[i]}}}}} \\")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def num(x: float, nd: int = 1, s: float = 100) -> str:
    return f"{s * x:.{nd}f}".replace("-", "$-$")


def ci(b: float, se: float, nd: int = 1, s: float = 100) -> str:
    return f"[{num(b - 1.96 * se, nd, s)}, {num(b + 1.96 * se, nd, s)}]"


def md(x: float, nd: int = 1, s: float = 100) -> str:
    return f"{s * x:.{nd}f}"


def mdci(b: float, se: float, nd: int = 1, s: float = 100) -> str:
    return f"{md(b, nd, s)} [{md(b - 1.96 * se, nd, s)}, {md(b + 1.96 * se, nd, s)}]"


# ---------------------------------------------------------------- main

def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    N, L = {}, []

    # ---- C2
    cache = os.environ.get("CFE_TWINS_CACHE")  # development only: a pickle of the rebuilt twins
    if cache and pathlib.Path(cache).exists():
        tw = pd.read_pickle(cache)
    else:
        tw = build_twins()
        if cache:
            pd.to_pickle(tw, cache)
    dev = check_rebuild(tw)
    print(f"twins rebuilt in {time.time() - t0:.0f}s; largest difference from 17's tables: {dev:.2e}", flush=True)
    assert dev < 5e-3, "the rebuilt twins differ from 17's published ones"
    gt, G = group_table(tw)
    ex = tw["ex_full"]
    S_corr = float(np.corrcoef(ex.human_leniency_loo, ex.grant_rate)[0, 1])
    ch, S, extra = change_table(tw)
    S["corr_rates"] = S_corr
    w = women(tw, ch, S, G)
    # published without examiner-level values: a median over five examiners is one examiner's value, and the four
    # groups of five can be rebuilt from PatEx, so their own and their twins' gender gaps stay unpublished
    pub = gt.drop(columns=[c for c in gt.columns if c.endswith("_median")])
    pub.loc[pub.group != "All 20", [c for c in pub.columns if "gap" in c and not c.endswith("_n")]] = np.nan
    pub.to_csv(T / "c2x_groups.csv", index=False)
    ch.to_csv(T / "c2x_decision_change.csv", index=False)
    extra["binned"].to_csv(T / "c2x_pairs_binned.csv", index=False)
    w.to_csv(T / "c2x_women.csv", index=False)
    pd.DataFrame([{**S, **G}]).to_csv(T / "c2x_summary.csv", index=False)
    Rr, Zr = robust_change(tw)
    Rr.to_csv(T / "c2x_decision_change_robust.csv", index=False)
    pd.DataFrame([Zr]).to_csv(T / "c2x_between_robust.csv", index=False)
    figure(gt, ch, extra, S, Rr)

    # Table A30: the pooled row only (rows for the groups of five would attach gender gaps to small groups of
    # identifiable examiners)
    rows = []
    for r in gt.itertuples():
        rows.append([r.group, f"{100 * r.leniency_mean:.0f}", f"{100 * r.grant_rate_mean:.0f}", f"{r.auc_mean:.2f}",
                     num(r.own_gap_pooled), num(r.twin_gap_prob_mean), num(r.twin_gap_dec_mean)])
    tabular(TEX / "c2x_tab_examiners.tex", "lrrrrrr",
            ["", "Examiners' leniency (\\%)", "Twins' grant rate (\\%)", "Twins' AUC", "Examiners' gap",
             "Twins' gap, probabilities", "Twins' gap, decisions"], rows[4:])
    # table A19/A20 v2: rows = measures, columns = all / female / male / F − M
    c = ch.set_index("key")
    groups, rows = {}, []
    spec = [("Thresholded decisions: each twin grants its top share (as in 17)", [
                ("Leniency alone, thresholds at the examiners' grant rates", "len_exam"),
                ("Leniency alone, thresholds at the twins' grant rates", "len_twin"),
                ("Leniency-only twins", "dec"),
                ("Twins with 32 text components", "dec_text")]),
            ("Decisions drawn independently from each twin's grant probability", [
                ("Grant rates alone (no text)", "rate"),
                ("\\quad of which: difference across twins", "rate_between"),
                ("Leniency-only twins", "prob"),
                ("\\quad of which: same twin drawing twice", "prob_same"),
                ("\\quad of which: difference across twins", "prob_between"),
                ("Leniency-only twins, weakest penalty ($C = 1$)", "prob_c1"),
                ("\\quad of which: difference across twins", "prob_c1_between"),
                ("Twins with 32 text components", "prob_text"),
                ("\\quad of which: difference across twins", "prob_text_between")]),
            ("Earlier layout: how many of the 20 twins grant (thresholded decisions)", [
                ("All 20", "all"), ("None", "none"), ("Depends on the twin (1--19)", "depends")])]
    for title, items in spec:
        groups[len(rows)] = title
        for lab, key in items:
            r = c.loc[key]
            fm = (f"{num(r.f_minus_m)} {ci(r.f_minus_m, r.f_minus_m_se)}" if r.f_minus_m_se > 1e-12 else
                  f"{num(r.f_minus_m)}")
            rows.append([lab, num(r["all"]), num(r.female), num(r.male), fm])
    tabular(TEX / "c2x_tab_lottery.tex", "lrrrr",
            ["", f"All ({S['n_common']:,})", f"Female ({S['n_female']:,})", f"Male ({S['n_male']:,})",
             "Female $-$ male [95\\% CI]"], rows, groups)

    # macros (C2)
    for key, name in [("len_exam", "LenExam"), ("len_twin", "LenTwin"), ("dec", "Dec"), ("prob", "Prob"),
                      ("prob_same", "Same"), ("prob_between", "Between"), ("dec_text", "DecText"),
                      ("prob_text", "ProbText"), ("prob_text_between", "BetweenText"), ("dec_c1", "DecCOne"),
                      ("prob_c1", "ProbCOne"), ("prob_c1_same", "SameCOne"), ("prob_c1_between", "BetweenCOne"),
                      ("rate", "Rate"), ("rate_between", "RateBetween"), ("depends", "Depends"),
                      ("depends_text", "DependsText"), ("all", "All"), ("none", "None")]:
        r = c.loc[key]
        N[f"xTtCh{name}"] = num(r["all"])
        N[f"xTtCh{name}F"], N[f"xTtCh{name}M"] = num(r.female), num(r.male)
        N[f"xTtCh{name}FM"], N[f"xTtCh{name}FMCI"] = num(r.f_minus_m), ci(r.f_minus_m, r.f_minus_m_se)
        N[f"xTtCh{name}CI"] = ci(r["all"], r.all_se)
        N[f"xTtCh{name}FCI"] = ci(r.female, r.female_se)
    N["xTtRatesPairsTwin"], N["xTtRatesPairsExam"] = num(S["rates_twins_pairs"]), num(S["rates_exam_pairs"])
    N["xTtRangeTwin"], N["xTtRangeExam"] = num(S["range_twins"]), num(S["range_exam"])
    N["xTtRankCorrMin"] = f"{S['twin_rank_corr_min']:.3f}"
    N["xTtCorrRates"] = f"{S_corr:.2f}"
    # robust intervals for the gender comparison of decision change (application bootstrap; twin jackknife SE)
    Rk = Rr.set_index("key")
    for key, name in [("len_twin", "LenTwin"), ("dec", "Dec"), ("dec_text", "DecText")]:
        r = Rk.loc[key]
        N[f"xTtRob{name}F"], N[f"xTtRob{name}M"] = num(r.female), num(r.male)
        N[f"xTtRob{name}FCI"] = f"[{num(r.female_lo)}, {num(r.female_hi)}]"
        N[f"xTtRob{name}MCI"] = f"[{num(r.male_lo)}, {num(r.male_hi)}]"
        N[f"xTtRob{name}FM"] = num(r.f_minus_m)
        N[f"xTtRob{name}FMCI"] = f"[{num(r.f_minus_m_lo)}, {num(r.f_minus_m_hi)}]"
        N[f"xTtRob{name}FMSeJk"] = num(r.f_minus_m_se_jackknife)
    N["xTtBetweenFM"] = num(Zr["between_f_minus_m"])
    N["xTtBetweenFMCI"] = f"[{num(Zr['between_f_minus_m_lo'])}, {num(Zr['between_f_minus_m_hi'])}]"
    for g, k in [("female", "F"), ("male", "M")]:
        N[f"xTtBetween{k}"], N[f"xTtBelow{k}"] = num(S[f"between_{g}"]), num(S[f"below_{g}"])
        N[f"xTtAbove{k}"], N[f"xTtScorePct{k}"] = num(S[f"above_{g}"]), num(S[f"score_pct_{g}"], 0)
    al = gt[gt.group == "All 20"].iloc[0]
    N["xTtGapDecMedian"], N["xTtGapProbMedian"] = num(al.twin_gap_dec_median), num(al.twin_gap_prob_median)
    N["xTtGapProbCOneMedian"], N["xTtGapDecCOneMedian"] = (num(al.twin_gap_prob_c1_median),
                                                          num(al.twin_gap_dec_c1_median))
    N["xTtThresholdShareCOne"] = num(1 - al.twin_gap_prob_c1_median / al.twin_gap_dec_median, 0)
    N["xTtStrongPenalty"], N["xTtFlat"] = f"{G['n_strong_penalty']}", f"{G['n_flat']}"
    N["xTtFlatCOne"] = f"{G['n_flat_c1']}"
    N["xTtPSDMedian"], N["xTtPSDCOneMedian"] = num(G["p_sd_median"]), num(G["p_sd_c1_median"])
    N["xTtGapDecTextMedian"], N["xTtGapProbTextMedian"] = (num(al.twin_gap_dec_text_median),
                                                          num(al.twin_gap_prob_text_median))
    N["xTtOwnGapMedian"], N["xTtOwnGapPooled"] = num(al.own_gap_median), num(al.own_gap_pooled)
    N["xTtOwnGapPooledCI"] = ci(al.own_gap_pooled, al.own_gap_pooled_se)
    N["xTtOwnGapAllYears"], N["xTtOwnGapAllYearsCI"] = (num(al.own_gap_all_years_pooled),
                                                        ci(al.own_gap_all_years_pooled, al.own_gap_all_years_pooled_se))
    N["xTtHumanGap"], N["xTtHumanGapCI"] = num(G["human_gap"]), ci(G["human_gap"], G["human_gap_se"])
    N["xTtHumanGapSE"] = num(G["human_gap_se"])
    N["xTtThresholdShare"] = num(1 - al.twin_gap_prob_median / al.twin_gap_dec_median, 0)
    N["xTtFemaleExMedian"] = f"{G['n_female_examiner_median']:.0f}"
    N["xTtFemaleExLtTen"] = f"{G['n_examiners_lt10_female']}"
    N["xTtFemaleExNone"] = f"{G['n_examiners_no_female']}"
    wi = w.set_index("sample")
    com = wi.iloc[0]
    N["xTtCommon"], N["xTtFemale"], N["xTtMale"] = f"{S['n_common']:,}", f"{S['n_female']:,}", f"{S['n_male']:,}"
    N["xTtFemaleShare"] = num(com.female_share)
    N["xTtFemaleShareTC"] = num(wi.loc["TC 3600, 2010–2014"].female_share)
    N["xTtFemaleShareAll"] = num(wi.loc["Eight centers, 2010–2014"].female_share)
    N["xTtTCApps"] = f"{int(wi.loc['TC 3600, 2010–2014'].applications):,}"
    N["xTtCommonShareTC"] = num(S["n_common"] / wi.loc["TC 3600, 2010–2014"].applications)
    st2 = wi.iloc[-1]
    N["xTtStageTwoApps"], N["xTtStageTwoFemale"] = f"{int(st2.applications):,}", f"{int(st2.female):,}"
    N["xTtStageTwoHalfWidth"] = f"{st2.gap_ci_halfwidth_scaled_pp:.1f}"
    N["xTtHalfWidth"] = f"{100 * 1.96 * G['human_gap_se']:.1f}"
    shares = w[w["sample"].str.startswith("TC ")]
    N_rank = int((shares.female_share < wi.loc["TC 3600, 2010–2014"].female_share).sum())
    N["xTtFemaleShareRank"] = ["lowest", "second-lowest", "third-lowest", "fourth-lowest", "fifth-lowest",
                               "sixth-lowest", "seventh-lowest", "highest"][N_rank]
    N["xTtFemaleShareMinTC"] = shares.loc[shares.female_share.idxmin(), "sample"][3:7]
    N["xTtFemaleShareMin"] = num(shares.female_share.min())
    N["xTtFemaleShareMax"] = num(shares.female_share.max())
    N["xTtFemaleShareMaxTC"] = shares.loc[shares.female_share.idxmax(), "sample"][3:7]

    # The submitted appendix uses C2 only.
    (TEX / "c2c4x_numbers.tex").write_text(
        "% Generated by script 45; do not edit by hand.\n" + "".join(
            f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    (OUT / "c2_twins.md").write_text(
        "# AI examiner twins (C2)\n\n" + gt[gt.group == "All 20"].to_markdown(index=False)
        + "\n\n" + ch[ch.key.isin(["len_twin", "dec", "dec_text"])].to_markdown(index=False) + "\n")
    print(f"C2 complete in {time.time() - t0:.0f}s; no career-stage analysis is run.")


if __name__ == "__main__":
    main()
