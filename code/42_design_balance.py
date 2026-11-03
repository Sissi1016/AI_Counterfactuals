"""Assumptions and tests on the main sample: balance, monotonicity, exclusion, and calibration in all eight centers.

The appendix's design checks were first run on the pilot (TC 3600 and 1600): balance in 12 (characteristics fixed at
filing) and 21 (predicted citations), monotonicity in 12, exclusion in 12 and 27, calibration in 25. This script reruns
them with the same definitions on the main sample of 38: non-continuation applications filed 2010–2014 in the eight
utility technology centers (1600 1700 2100 2400 2600 2800 3600 3700) with a U.S. first inventor of inferred gender.
Cells are art unit × filing year × USPC class (au_class); leniency is the examiner's leave-one-out grant rate (04);
standard errors are clustered by examiner. Every estimate is reported by center, for the eight centers pooled ("All
centers"), and for TC 3600 and 1600 pooled ("3600+1600", the pilot's "Both", for comparison only). The LaTeX tables
show the eight centers pooled and TC 3600, with the number of centers whose 95% CI lies below / above zero; the csv
files and output/design_balance.md have every center.

1. Balance (12, 21). Each characteristic fixed at filing on leniency within cells, without controls; coefficient × 0.1
   (a 10-point more lenient examiner): female first inventor and small entity (shares; pp in the table), number of
   inventors, the grant-trained text model's cross-fitted score (log-odds), and each evaluator's predicted citations
   (its score mapped to the citation scale out of fold by a Poisson regression fitted on the granted applications of
   the other folds within each center: cfe.check.recalibrate, as in 21).
2. Monotonicity (12). Grant on leniency within cells, controlling for small entity, female first inventor, and number
   of inventors, by group: all, female, male, small and large entity, and quartiles of the grant-trained score within
   each center. Reverse instruments: the examiner's grant rate on the other group's applications only (male
   applications for female ones, large-entity applications for small-entity ones, and so on), computed from all of the
   examiner's 2004–2018 applications in applications_leniency and used when it rests on at least 10 applications.
3. Exclusion (12, 27). Among granted patents: claims at grant (PatentsView g_patent) and pendency from filing to grant
   (years) on leniency within cells with the same controls (pyfixest, CRV1 by examiner); coefficient × 0.1 and mean.
4. Calibration (25, part 2; Kleinberg et al. 2018, Figure IV). An examiner is strict or lenient if below or above the
   mean leniency of the application's cell. A ridge regression of log(1 + citations) on the text embeddings is trained
   only on the lenient examiners' grants of the other folds, within each center (penalty by inner examiner-grouped
   cross-validation). The residual (actual − predicted log citations) of every grant is regressed on strict within cells
   with the same controls: a negative coefficient means strict examiners' grants are cited less than the model predicts.

Eight-center equivalents of pilot-only inputs: the grant-, first-action- and citation-trained evaluators are 38's
cross-fit (all_scores: the specifications of 12 and 21, refitted on the eight centers in 38's folds), the embeddings are
emb/all_s0.npy (in all_text row order; pilot applications keep their pilot vectors), the folds are 38's
(cfe.check.assign_folds over the eight centers), and the zero-shot LLM is 14's score for pilot applications and 40's
elsewhere (log P(Yes) − log P(No)). The covariates, leniency, cells, claims, and pendency are the pilot's data, so those
rows for TC 3600 and 1600 reproduce the pilot exactly; rows built on model scores differ only through re-cross-fitting.
output/design_balance.md compares both with the pilot's tables.

Runs on CPUs only, with CFE_THREADS threads (default 4); about 2 minutes, mostly the ridge regressions of 4.
Run: taskset -c 0-15 uv run python code/42_design_balance.py
Input: data/processed/{all_sample,all_text,all_scores,applications_leniency,app_gender,llm_scores_allpilot,
       llm_scores_allcenters}.parquet, data/processed/emb/all_s0.npy, data/patentsview/tsv/g_patent.tsv;
       for the pilot comparison only: output/tables/{marginal_balance,citations_balance,marginal_first_stage,
       citation_robustness_exclusion,draft_tests_calibration}.csv
Output: output/latex/design_tab_{balance,first_stage,exclusion,calibration}.tex (tabular bodies),
        output/latex/balance_numbers.tex (macros \\xBa...), output/tables/design_{balance,first_stage,exclusion,
        calibration,calibration_penalty}.csv, output/design_balance.md
"""
import os

THREADS = int(os.environ.get("CFE_THREADS", "4"))
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS",
           "RAYON_NUM_THREADS"]:
    os.environ.setdefault(_v, str(THREADS))

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from cfe.check import GROUPS, PATENTS, assign_folds, ols_coef, recalibrate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
PV = ROOT / "data/patentsview/tsv"
OUT = ROOT / "output"
T = OUT / "tables"
TEX = OUT / "latex"
SEED, K_INNER = 20261028, 4
ALPHAS = [1e2, 1e3, 1e4, 1e5]  # as in 25
MIN_REV = 10  # minimum applications behind a reverse-sample leniency (12)
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
POOLED, PILOT, NO3700 = "All centers", "3600+1600", "All centers except 3700"
OUTCOME = "cites_ns"
DESIGN = dataclasses.replace(
    PATENTS, units=tuple(TCS), outcomes=((OUTCOME, "Citations within five years of grant, excluding self-citations"),),
    samples=tuple((t, (t,)) for t in TCS) + ((POOLED, tuple(TCS)), (PILOT, ("3600", "1600"))))
# balance only: also pooled without TC 3700, whose heavy-tailed citations stretch the Poisson-recalibrated predictions
BAL_DESIGN = dataclasses.replace(DESIGN, samples=DESIGN.samples + ((NO3700, tuple(t for t in TCS if t != "3700")),))
CONTROLS = list(PATENTS.controls)  # small entity, female first inventor, number of inventors
GRANT_SCORE = "Train on grant"
EVALUATORS = ["Train on grant", "Train on grant + names", "Train on first action", "Train on citations (granted only)",
              "Train on citations + names", "LLM, zero-shot"]
COVARIATES = [("female", "Female first inventor"), ("small_entity", "Small entity"), ("n_inventors", "Inventors"),
              (GRANT_SCORE, "Predicted grant score (text, log-odds)")]
OTHER = {"Female": "male", "Male": "female", "Small entity": "large-entity", "Large entity": "small-entity"}
TC_WORD = {"1600": "OneSix", "1700": "OneSeven", "2100": "TwoOne", "2400": "TwoFour", "2600": "TwoSix",
           "2800": "TwoEight", "3600": "ThreeSix", "3700": "ThreeSeven"}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- data

def llm_zero_shot() -> pd.DataFrame:
    """Qwen3-8B zero-shot log-odds of "Yes": 14's scores for pilot applications, 40's for the other centers."""
    old = pd.read_parquet(P / "llm_scores_allpilot.parquet")
    sc = pd.concat([old[old.cond == "zero"], pd.read_parquet(P / "llm_scores_allcenters.parquet")])
    sc = sc[sc.cond == "zero"]
    floor = 1e-12  # as in 10, 14, 21 and 40
    sc["LLM, zero-shot"] = np.log(sc.p_yes.clip(lower=floor)) - np.log(sc.p_no.clip(lower=floor))
    return sc[["application_number", "LLM, zero-shot"]]


def extras(d: pd.DataFrame) -> pd.DataFrame:
    """Claims at grant and pendency (exclusion), reverse-sample leniencies (monotonicity); the SQL of 12's extras."""
    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}; SET memory_limit = '64GB'")
    con.register("smp", d[["application_number", "examiner_id"]])
    x = con.execute(f"""
    WITH a AS (
      SELECT l.examiner_id, l.granted, l.small_entity_indicator IN ('SMALL', 'MICRO') AS small,
             g.first_inventor_gender AS gender
      FROM '{P}/applications_leniency.parquet' l
      LEFT JOIN '{P}/app_gender.parquet' g USING (application_number)
    ), ex AS (
      SELECT examiner_id,
             avg(granted) FILTER (WHERE gender = 'F') AS gr_f, count(*) FILTER (WHERE gender = 'F') AS n_f,
             avg(granted) FILTER (WHERE gender = 'M') AS gr_m, count(*) FILTER (WHERE gender = 'M') AS n_m,
             avg(granted) FILTER (WHERE small) AS gr_s, count(*) FILTER (WHERE small) AS n_s,
             avg(granted) FILTER (WHERE NOT small) AS gr_l, count(*) FILTER (WHERE NOT small) AS n_l
      FROM a GROUP BY 1
    ), gp AS (
      SELECT patent_id, TRY_CAST(num_claims AS INTEGER) AS num_claims
      FROM read_csv('{PV}/g_patent.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)
    )
    SELECT s.application_number, l.filing_date, l.patent_issue_date, gp.num_claims,
           ex.gr_f, ex.n_f, ex.gr_m, ex.n_m, ex.gr_s, ex.n_s, ex.gr_l, ex.n_l
    FROM smp s
    JOIN '{P}/applications_leniency.parquet' l ON l.application_number = s.application_number
    LEFT JOIN ex ON ex.examiner_id = s.examiner_id
    LEFT JOIN gp ON gp.patent_id = l.patent_number
    """).df()
    x = d[["application_number"]].merge(x, on="application_number", how="left", validate="1:1")
    out = pd.DataFrame({"application_number": d.application_number})
    g = (d.granted == 1).to_numpy()
    days = (pd.to_datetime(x.patent_issue_date) - pd.to_datetime(x.filing_date)).dt.days.to_numpy(float)
    out["pendency_years"] = np.where(g, days / 365.25, np.nan)
    out["num_claims"] = np.where(g, x.num_claims.to_numpy(float), np.nan)

    def rate(r, n):
        return np.where(x[n].fillna(0) >= MIN_REV, x[r], np.nan)
    fem, small = (d.female == 1).to_numpy(), (d.small_entity == 1).to_numpy()
    out["lenient_rev_gender"] = np.where(fem, rate("gr_m", "n_m"), rate("gr_f", "n_f"))
    out["lenient_rev_entity"] = np.where(small, rate("gr_l", "n_l"), rate("gr_s", "n_s"))
    return out


def load() -> pd.DataFrame:
    """The main sample in the row order of emb/all_s0.npy (all_text order), with scores and the variables above."""
    keys = pd.read_parquet(P / "all_text.parquet", columns=["application_number"])
    d = keys.merge(pd.read_parquet(P / "all_sample.parquet"), on="application_number")
    d = d.sort_values("application_number").reset_index(drop=True)
    assert len(d) == len(keys) and (d.application_number.to_numpy() == keys.application_number.to_numpy()).all()
    d = d.merge(pd.read_parquet(P / "all_scores.parquet"), on="application_number", how="left", validate="1:1")
    d = d.merge(llm_zero_shot(), on="application_number", how="left", validate="1:1")
    d = d.merge(extras(d), on="application_number", how="left", validate="1:1")
    assert (d.application_number.to_numpy() == keys.application_number.to_numpy()).all()
    for c in EVALUATORS + ["lenient_grant"]:
        assert d[c].notna().all(), f"missing {c}"
    return d


# ---------------------------------------------------------------- 1. balance

def balance(d: pd.DataFrame, fold: np.ndarray, design=DESIGN, evaluators=EVALUATORS,
            grant_score: str = GRANT_SCORE) -> pd.DataFrame:
    """Characteristics fixed at filing and predicted citations on leniency within cells (no controls, as in 12/21)."""
    dd = recalibrate(d, list(evaluators), fold, design)
    covs = [(c if c != GRANT_SCORE else grant_score, lab, "Characteristics at filing") for c, lab in COVARIATES]
    covs += [(f"{OUTCOME}::{m}", m, "Predicted citations") for m in evaluators]
    rows = []
    for sname, units in design.samples:
        s = dd[dd.tech_center.isin(units) & dd.lenient_grant.notna()]
        for c, lab, panel in covs:
            b, se = ols_coef(s, c, "lenient_grant", design=design)
            rows.append(dict(sample=sname, panel=panel, variable=lab, coef_per_10pp=0.1 * b, se=0.1 * se,
                             mean=s[c].mean(), sd=s[c].std(), n=len(s)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. monotonicity

def first_stage(d: pd.DataFrame, design=DESIGN, grant_score: str = GRANT_SCORE) -> pd.DataFrame:
    """Grant on leniency within cells with controls, by group; and with reverse-sample instruments (12)."""
    d = d.copy()
    d["q"] = d.groupby("tech_center")[grant_score].transform(lambda v: pd.qcut(v, 4, labels=False) + 1)
    groups = [(g, (lambda x, g=g: x if GROUPS[g] is None else x[x[GROUPS[g][0]] == GROUPS[g][1]])) for g in GROUPS]
    groups += [(f"Predicted grant, quartile {q}", (lambda x, q=q: x[x.q == q])) for q in range(1, 5)]
    rows = []
    for sname, units in design.samples:
        ds = d[d.tech_center.isin(units)]
        for gname, f in groups:
            s = f(ds)
            b, se = ols_coef(s, "granted", "lenient_grant", controls=CONTROLS, design=design)
            rows.append(dict(sample=sname, group=gname, instrument="Leave-one-out grant rate", coef=b, se=se,
                             n=len(s), n_examiners=s.examiner_id.nunique()))
        for gname, zc in [("Female", "lenient_rev_gender"), ("Male", "lenient_rev_gender"),
                          ("Small entity", "lenient_rev_entity"), ("Large entity", "lenient_rev_entity")]:
            g = GROUPS[gname]
            s = ds[(ds[g[0]] == g[1]) & ds[zc].notna()]
            b, se = ols_coef(s, "granted", zc, controls=CONTROLS, design=design)
            rows.append(dict(sample=sname, group=gname, instrument=f"Grant rate on {OTHER[gname]} applications "
                             "(reverse)", coef=b, se=se, n=len(s), n_examiners=s.examiner_id.nunique()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. exclusion

def exclusion(d: pd.DataFrame, design=DESIGN) -> pd.DataFrame:
    """Claims at grant and pendency on leniency among granted patents, within cells with controls (12, 27)."""
    import pyfixest as pf
    rows = []
    for sname, units in design.samples:
        g = d.loc[d.tech_center.isin(units) & (d.granted == 1),
                  ["num_claims", "pendency_years", "lenient_grant", "au_class", "examiner_id"] + CONTROLS]
        g = g.dropna(subset=["num_claims", "pendency_years"])
        for y, lab in [("num_claims", "Claims at grant"), ("pendency_years", "Pendency to grant (years)")]:
            fit = pf.feols(f"{y} ~ lenient_grant + {' + '.join(CONTROLS)} | au_class", data=g,
                           vcov={"CRV1": "examiner_id"})
            rows.append(dict(sample=sname, variable=lab, mean=g[y].mean(), coef_per_10pp=0.1 * fit.coef()["lenient_grant"],
                             se=0.1 * fit.se()["lenient_grant"], n=fit._N, granted=len(g)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. calibration

def ridge_task(X, y, ex, tr, te, seed):
    """25's ridge_task: log(1 + citations) on the embedding, penalty by inner examiner-grouped CV."""
    with threadpool_limits(1):
        mse = {a: 0.0 for a in ALPHAS}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            sc = StandardScaler().fit(X[tr[ia]])
            Xa, Xb = sc.transform(X[tr[ia]]), sc.transform(X[tr[ib]])
            for a in ALPHAS:
                mse[a] += ((Ridge(alpha=a).fit(Xa, y[tr[ia]]).predict(Xb) - y[tr[ib]]) ** 2).sum()
        best = min(mse, key=mse.get)
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=best).fit(sc.transform(X[tr]), y[tr])
        return dict(alpha=best, pred_log=m.predict(sc.transform(X[te])))


def calibration(d: pd.DataFrame, X: np.ndarray, fold: np.ndarray, design=DESIGN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A citation model trained on lenient examiners' grants, applied to strict examiners' grants (25)."""
    cell = pd.factorize(d.tech_center + "|" + d.au_class)[0]
    z = d.lenient_grant.to_numpy(float)
    lenient_half = ((z - pd.Series(z).groupby(cell).transform("mean").to_numpy()) > 0).astype(float)
    granted = (d.granted == 1).to_numpy()
    y_log = np.log1p(d[OUTCOME].fillna(0).to_numpy(float))
    train = (lenient_half == 1) & granted
    ex = pd.factorize(d.examiner_id)[0]
    jobs = []
    for tc in design.units:
        rows = np.where((d.tech_center == tc).to_numpy())[0]
        for k in np.unique(fold[rows]):
            jobs.append((tc, k, rows[(fold[rows] != k) & train[rows]], rows[fold[rows] == k]))
    out = Parallel(n_jobs=THREADS, max_nbytes="1M")(delayed(ridge_task)(X, y_log, ex, tr, te, SEED + i)
                                                    for i, (_, _, tr, te) in enumerate(jobs))
    pred_log = np.full(len(d), np.nan)
    for (_, _, _, te), r in zip(jobs, out):
        pred_log[te] = r["pred_log"]
    alphas = pd.DataFrame([dict(tech_center=tc, fold=k, train=len(tr), alpha=r["alpha"])
                           for (tc, k, tr, _), r in zip(jobs, out)])
    d = d.assign(resid=np.where(granted, y_log - pred_log, np.nan), strict_half=1 - lenient_half)
    rows = []
    for sname, units in design.samples:
        s = d[d.tech_center.isin(units) & granted & d.lenient_grant.notna()]
        b, se = ols_coef(s, "resid", "strict_half", controls=CONTROLS, design=design)
        rows.append(dict(sample=sname, strict_minus_lenient_resid=b, se=se,
                         strict_grants=int((s.strict_half == 1).sum()), lenient_grants=int((s.strict_half == 0).sum()),
                         mean_resid_strict=s.loc[s.strict_half == 1, "resid"].mean(),
                         mean_resid_lenient=s.loc[s.strict_half == 0, "resid"].mean()))
    return pd.DataFrame(rows), alphas


# ---------------------------------------------------------------- formatting

def num(x: float, nd: int = 2) -> str:
    """As 29's num: typographic minus, no minus on a value that rounds to zero."""
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def cin(b: float, se: float, nd: int = 2, scale: float = 1.0) -> str:
    return f"{num(scale * b, nd)} [{num(scale * (b - 1.96 * se), nd)}, {num(scale * (b + 1.96 * se), nd)}]"


def ci_only(b: float, se: float, nd: int = 2, scale: float = 1.0) -> str:
    return cin(b, se, nd, scale).split(" ", 1)[1]


def plain(s: str) -> str:
    """Markdown version of a LaTeX number string."""
    return s.replace("$-$", "−")


def by_center(t: pd.DataFrame, col: str = "coef", se: str = "se") -> pd.DataFrame:
    x = t[t["sample"].isin(TCS)]
    return x.assign(lo=x[col] - 1.96 * x[se], hi=x[col] + 1.96 * x[se])


# The appendix tables show the eight centers pooled and TC 3600, and the number of centers whose 95% CI lies below /
# above zero; every center's estimate is in the csv files and output/design_balance.md.
TC_SHOW = "3600"
SIG_HEAD = "Centers, CI $<$0 / $>$0 (of 8)"


def sig_count(c: pd.DataFrame) -> str:
    """c: by_center() rows of one estimate. Number of centers with the 95% CI below / above zero."""
    assert len(c) == len(TCS)
    return f"{int((c.hi < 0).sum())} / {int((c.lo > 0).sum())}"


BAL_FMT = {"Female first inventor": (100, 2), "Small entity": (100, 2), "Inventors": (1, 3),
           "Predicted grant score (text, log-odds)": (1, 3)}  # scale, decimals (as 29)


def bal_label(v: str) -> str:
    return (v.replace("Predicted grant score (text, log-odds)", "Pred. grant score (log-odds)")
            .replace("Train on ", "Trained on ").replace(" (granted only)", ""))


def fs_labels(g: str, z: str) -> tuple[str, str]:
    zl = z.replace("Leave-one-out grant rate", "Leave-one-out").replace("Grant rate on ", "Reverse: ")
    return g.replace("Predicted grant, quartile", "Pred. grant, quartile"), zl.replace(" applications (reverse)", "")


# ---------------------------------------------------------------- tables, macros, report

def write_outputs(bal, fs, exc, cal, alphas, d, runtime) -> dict:
    M13 = load_module("m13", "13_figures_tables.py")
    tabular, esc = M13.tabular, M13.esc
    N = {}
    ncen = len(TCS)

    # balance
    rows, groups = [], {}
    groups[0] = r"Cells: Art unit $\times$ year $\times$ USPC class"
    for panel in ["Characteristics at filing", "Predicted citations"]:
        if panel == "Predicted citations":
            groups[len(rows)] = "Predicted citations (class cells)"
        for v in bal[bal.panel == panel].variable.unique():
            t = bal[(bal.panel == panel) & (bal.variable == v)]
            sc, nd = BAL_FMT.get(v, (1, 3))
            a = t[t["sample"] == POOLED].iloc[0]
            b = t[t["sample"] == TC_SHOW].iloc[0]
            c = by_center(t, "coef_per_10pp")
            rows.append([esc(bal_label(v)), cin(a.coef_per_10pp, a.se, nd, sc), cin(b.coef_per_10pp, b.se, nd, sc),
                         sig_count(c)])
    tabular(TEX / "design_tab_balance.tex", "lrrc", ["", POOLED, f"TC {TC_SHOW}", SIG_HEAD], rows, groups)

    def bal_macros(v, key, sc, nd):
        t = bal[bal.variable == v]
        a = t[t["sample"] == POOLED].iloc[0]
        c = by_center(t, "coef_per_10pp")
        N[f"xBa{key}"], N[f"xBa{key}CI"] = num(sc * a.coef_per_10pp, nd), ci_only(a.coef_per_10pp, a.se, nd, sc)
        N[f"xBa{key}Neg"], N[f"xBa{key}Pos"] = str(int((c.hi < 0).sum())), str(int((c.lo > 0).sum()))
        N[f"xBa{key}Min"], N[f"xBa{key}Max"] = num(sc * c.coef_per_10pp.min(), nd), num(sc * c.coef_per_10pp.max(), nd)
    for v, key in [("Female first inventor", "Female"), ("Small entity", "Small"), ("Inventors", "Inventors"),
                   ("Predicted grant score (text, log-odds)", "Score")]:
        bal_macros(v, key, *BAL_FMT[v])
    for m, key in zip(EVALUATORS, ["Grant", "GrantNames", "FirstAction", "Citations", "CitationsNames", "LLM"]):
        bal_macros(m, f"Cit{key}", 1, 3)
    pc = bal[(bal.panel == "Predicted citations")]
    pa = pc[pc["sample"] == POOLED]
    N["xBaCitPooledSig"] = str(int(((pa.coef_per_10pp - 1.96 * pa.se > 0) | (pa.coef_per_10pp + 1.96 * pa.se < 0)).sum()))
    N["xBaCitEvaluators"] = str(len(pa))
    cc = by_center(pc, "coef_per_10pp")
    N["xBaCitCenterSig"] = str(int(((cc.lo > 0) | (cc.hi < 0)).sum()))
    N["xBaCitCenterTests"] = str(len(cc))
    i = pa.coef_per_10pp.abs().idxmax()
    N["xBaCitMaxAbs"] = num(abs(pa.loc[i, "coef_per_10pp"]), 3)
    N["xBaCitMaxAbsEvaluator"] = esc(bal_label(pa.loc[i, "variable"])).replace("Trained on", "trained on")
    c = cc.loc[cc.coef_per_10pp.abs().idxmax()]
    N["xBaCitCenterMax"], N["xBaCitCenterMaxCI"] = num(c.coef_per_10pp, 3), ci_only(c.coef_per_10pp, c.se, 3)
    N["xBaCitCenterMaxTC"] = c["sample"]
    N["xBaCitCenterMaxEvaluator"] = esc(bal_label(c.variable)).replace("Trained on", "trained on")
    N["xBaCitCenterMaxPct"] = num(100 * abs(c.coef_per_10pp) / c["mean"], 1)  # % of that center's mean prediction
    px = pc[pc["sample"] == NO3700]
    N["xBaCitPooledSigNoThreeSeven"] = str(int(((px.coef_per_10pp - 1.96 * px.se > 0) |
                                                (px.coef_per_10pp + 1.96 * px.se < 0)).sum()))
    N["xBaCitMaxAbsNoThreeSeven"] = num(px.coef_per_10pp.abs().max(), 3)
    a = bal[(bal["sample"] == POOLED) & (bal.variable == "Predicted grant score (text, log-odds)")].iloc[0]
    N["xBaScoreSD"] = num(a.sd, 2)
    N["xBaScoreSDPct"] = num(100 * a.coef_per_10pp / a.sd, 1)  # % of a standard deviation per 10 points
    for v, key in [("Female first inventor", "Female"), ("Small entity", "Small")]:
        N[f"xBa{key}Mean"] = num(100 * bal[(bal["sample"] == POOLED) & (bal.variable == v)]["mean"].iloc[0], 1)
    g37 = d.loc[(d.granted == 1) & (d.tech_center == "3700"), OUTCOME]
    N["xBaMeanCitesThreeSeven"], N["xBaMaxCitesThreeSeven"] = num(g37.mean(), 1), f"{int(g37.max()):,}"
    mc = d.loc[d.granted == 1, OUTCOME].mean()
    N["xBaMeanCites"] = num(mc, 2)
    N["xBaCitMaxAbsPct"] = num(100 * abs(pa.loc[i, "coef_per_10pp"]) / mc, 1)

    # monotonicity
    rows = []
    for (g, z), t in fs.groupby(["group", "instrument"], sort=False):
        a = t[t["sample"] == POOLED].iloc[0]
        b = t[t["sample"] == TC_SHOW].iloc[0]
        c = by_center(t)
        gl, zl = fs_labels(g, z)
        rows.append([esc(gl), esc(zl), cin(a.coef, a.se, 3), cin(b.coef, b.se, 3), sig_count(c)])
    tabular(TEX / "design_tab_first_stage.tex", "llrrc",
            ["Group", "Instrument", POOLED, f"TC {TC_SHOW}", SIG_HEAD], rows)
    a = fs[(fs["sample"] == POOLED) & (fs.group == "All") & fs.instrument.str.startswith("Leave")].iloc[0]
    N["xBaFsAll"], N["xBaFsAllCI"] = num(a.coef, 2), ci_only(a.coef, a.se, 2)
    fp = fs[fs["sample"] == POOLED]
    m = fp.loc[fp.coef.idxmin()]
    gl, zl = fs_labels(m.group, m.instrument)
    N["xBaFsMin"], N["xBaFsMinCI"] = num(m.coef, 2), ci_only(m.coef, m.se, 2)
    N["xBaFsMinGroup"], N["xBaFsMinInstrument"] = esc(gl), esc(zl)
    c = by_center(fs)
    m = c.loc[c.coef.idxmin()]
    gl, zl = fs_labels(m.group, m.instrument)
    N["xBaFsMinCenter"], N["xBaFsMinCenterCI"] = num(m.coef, 2), ci_only(m.coef, m.se, 2)
    N["xBaFsMinCenterGroup"], N["xBaFsMinCenterInstrument"], N["xBaFsMinCenterTC"] = esc(gl), esc(zl), m["sample"]
    N["xBaFsPositive"], N["xBaFsTests"] = str(int((c.lo > 0).sum())), str(len(c))
    N["xBaFsPooledMax"] = num(fp.coef.max(), 2)

    # exclusion
    xa, xb = (exc[exc["sample"] == s].set_index("variable") for s in (POOLED, TC_SHOW))
    rows = [["Granted patents", f"{int(xa.granted.iloc[0]):,}", f"{int(xb.granted.iloc[0]):,}", ""]]
    for v in ["Claims at grant", "Pendency to grant (years)"]:
        rows.append([v] + [f"{cin(t.loc[v, 'coef_per_10pp'], t.loc[v, 'se'])} ({num(t.loc[v, 'mean'], 1)})"
                           for t in (xa, xb)] + [sig_count(by_center(exc[exc.variable == v], "coef_per_10pp"))])
    tabular(TEX / "design_tab_exclusion.tex", "lrrc", ["", POOLED, f"TC {TC_SHOW}", SIG_HEAD], rows)
    for v, key, sgn in [("Claims at grant", "Claims", 1), ("Pendency to grant (years)", "Pendency", -1)]:
        t = exc[exc.variable == v]
        a = t[t["sample"] == POOLED].iloc[0]
        b, lo, hi = sgn * a.coef_per_10pp, sgn * (a.coef_per_10pp - 1.96 * a.se), sgn * (a.coef_per_10pp + 1.96 * a.se)
        N[f"xBa{key}"], N[f"xBa{key}CI"] = num(b), f"[{num(min(lo, hi))}, {num(max(lo, hi))}]"
        N[f"xBa{key}Mean"] = num(a["mean"], 1)
        c = t[t["sample"].isin(TCS)]
        N[f"xBa{key}Min"], N[f"xBa{key}Max"] = num((sgn * c.coef_per_10pp).min()), num((sgn * c.coef_per_10pp).max())
        cs = by_center(c, "coef_per_10pp")
        N[f"xBa{key}Sig"] = str(int(((cs.lo > 0) if sgn > 0 else (cs.hi < 0)).sum()))
    N["xBaPendencyMonths"] = num(-12 * exc[(exc["sample"] == POOLED) & (exc.variable == "Pendency to grant (years)")]
                                 .coef_per_10pp.iloc[0], 1)

    # calibration
    ca, cb = (cal[cal["sample"] == s].iloc[0] for s in (POOLED, TC_SHOW))
    rows = [["Strict examiners' grants", f"{int(ca.strict_grants):,}", f"{int(cb.strict_grants):,}", ""],
            ["Lenient examiners' grants", f"{int(ca.lenient_grants):,}", f"{int(cb.lenient_grants):,}", ""],
            ["Strict $-$ lenient residual (log points)", cin(ca.strict_minus_lenient_resid, ca.se, 3),
             cin(cb.strict_minus_lenient_resid, cb.se, 3), sig_count(by_center(cal, "strict_minus_lenient_resid"))]]
    tabular(TEX / "design_tab_calibration.tex", "lrrc", ["", POOLED, f"TC {TC_SHOW}", SIG_HEAD], rows)
    a = cal[cal["sample"] == POOLED].iloc[0]
    N["xBaCal"], N["xBaCalCI"] = num(a.strict_minus_lenient_resid, 3), ci_only(a.strict_minus_lenient_resid, a.se, 3)
    c = by_center(cal, "strict_minus_lenient_resid")
    sig = []
    for r in c.itertuples():
        k = TC_WORD[r.sample]
        N[f"xBaCal{k}"], N[f"xBaCal{k}CI"] = num(r.strict_minus_lenient_resid, 3), ci_only(
            r.strict_minus_lenient_resid, r.se, 3)
        if r.lo > 0 or r.hi < 0:
            sig.append(f"TC {r.sample}")
    N["xBaCalSigN"] = str(len(sig))
    N["xBaCalSigCenters"] = ("none" if not sig else sig[0] if len(sig) == 1 else
                             ", ".join(sig[:-1]) + (" and " if len(sig) == 2 else ", and ") + sig[-1])
    N["xBaCalNeg"], N["xBaCalPos"] = str(int((c.hi < 0).sum())), str(int((c.lo > 0).sum()))
    N["xBaCalMin"], N["xBaCalMax"] = num(c.strict_minus_lenient_resid.min(), 3), num(c.strict_minus_lenient_resid.max(), 3)
    # TC 3600, the tables' second column
    for v, key in [("Female first inventor", "Female"), ("Small entity", "Small"), ("Inventors", "Inventors"),
                   ("Predicted grant score (text, log-odds)", "Score")] + [
            (m, f"Cit{k}") for m, k in zip(EVALUATORS, ["Grant", "GrantNames", "FirstAction", "Citations",
                                                         "CitationsNames", "LLM"])]:
        sc, nd = BAL_FMT.get(v, (1, 3))
        r = bal[(bal["sample"] == TC_SHOW) & (bal.variable == v)].iloc[0]
        N[f"xBa{key}ThreeSix"], N[f"xBa{key}ThreeSixCI"] = num(sc * r.coef_per_10pp, nd), ci_only(
            r.coef_per_10pp, r.se, nd, sc)
    r = fs[(fs["sample"] == TC_SHOW) & (fs.group == "All") & fs.instrument.str.startswith("Leave")].iloc[0]
    N["xBaFsAllThreeSix"], N["xBaFsAllThreeSixCI"] = num(r.coef, 2), ci_only(r.coef, r.se, 2)
    r = fs[fs["sample"] == TC_SHOW].loc[lambda x: x.coef.idxmin()]
    gl, zl = fs_labels(r.group, r.instrument)
    N["xBaFsMinThreeSix"], N["xBaFsMinThreeSixCI"] = num(r.coef, 2), ci_only(r.coef, r.se, 2)
    N["xBaFsMinThreeSixGroup"], N["xBaFsMinThreeSixInstrument"] = esc(gl), esc(zl)
    for v, key, sgn in [("Claims at grant", "Claims", 1), ("Pendency to grant (years)", "Pendency", -1)]:
        r = exc[(exc["sample"] == TC_SHOW) & (exc.variable == v)].iloc[0]
        lo, hi = sgn * (r.coef_per_10pp - 1.96 * r.se), sgn * (r.coef_per_10pp + 1.96 * r.se)
        N[f"xBa{key}ThreeSix"], N[f"xBa{key}ThreeSixCI"] = num(sgn * r.coef_per_10pp), \
            f"[{num(min(lo, hi))}, {num(max(lo, hi))}]"
        N[f"xBa{key}ThreeSixMean"] = num(r["mean"], 1)
    N["xBaCenters"] = str(ncen)
    N["xBaApplications"] = f"{len(d):,}"

    lines = ["% Generated by code/42_design_balance.py; do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items()]
    (TEX / "balance_numbers.tex").write_text("\n".join(lines) + "\n")
    report(bal, fs, exc, cal, alphas, N, runtime)
    return N


def pilot_comparison(bal, fs, exc, cal) -> list:
    """Markdown rows: the pilot's estimates (12, 21, 25, 27) against this run's TC 3600, TC 1600, and 3600+1600."""
    L = ["## Comparison with the pilot tables", "",
         "Pilot: the CSVs behind cit_tab_balance, tab_first_stage, cit_tab_exclusion and cit_tab_calibration. "
         "This run: the same samples cut from the main sample. Covariates, leniency, cells, claims and pendency are "
         "the same data, so those rows must agree to rounding; rows built on model scores differ because the "
         "evaluators were re-cross-fitted on the eight centers (38's folds) and recalibrated in those folds.", "",
         "| Table | Row | Pilot: 3600 | Now: 3600 | Pilot: 1600 | Now: 1600 | Pilot: both | Now: 3600+1600 |",
         "|---|---|---|---|---|---|---|---|"]
    smp = [("3600", "3600"), ("1600", "1600"), ("Both", PILOT)]
    try:
        mb = pd.read_csv(T / "marginal_balance.csv", dtype={"sample": str})
        mb = mb[mb.cells == "Art unit × year × USPC class"]
        cb = pd.read_csv(T / "citations_balance.csv", dtype={"sample": str})
        mf = pd.read_csv(T / "marginal_first_stage.csv", dtype={"sample": str})
        ce = pd.read_csv(T / "citation_robustness_exclusion.csv", dtype={"sample": str})
        dc = pd.read_csv(T / "draft_tests_calibration.csv", dtype={"tech_center": str})
    except FileNotFoundError:
        return L + ["(pilot tables not found)", ""]

    def pair(old, new, sc, nd):
        return [plain(cin(old[0], old[1], nd, sc)), plain(cin(new[0], new[1], nd, sc))]
    for v, (sc, nd) in BAL_FMT.items():
        cells = []
        for so, sn in smp:
            o = mb[(mb["sample"] == so) & (mb.variable == v)].iloc[0]
            n = bal[(bal["sample"] == sn) & (bal.variable == v)].iloc[0]
            cells += pair((o.coef_per_10pp, o.se), (n.coef_per_10pp, n.se), sc, nd)
        L.append(f"| Balance | {v} | " + " | ".join(cells) + " |")
    for m in ["Train on grant", "Train on first action", "Train on citations (granted only)",
              "Train on citations + names", "LLM, zero-shot"]:
        cells = []
        for so, sn in smp:
            o = cb[(cb["sample"] == so) & (cb.series == m)].iloc[0]
            n = bal[(bal["sample"] == sn) & (bal.variable == m)].iloc[0]
            cells += pair((0.1 * o.coef, 0.1 * o.se), (n.coef_per_10pp, n.se), 1, 3)
        L.append(f"| Balance, predicted citations | {m} | " + " | ".join(cells) + " |")
    for (g, z), _ in mf.groupby(["group", "instrument"], sort=False):
        cells = []
        for so, sn in smp:
            o = mf[(mf["sample"] == so) & (mf.group == g) & (mf.instrument == z)].iloc[0]
            n = fs[(fs["sample"] == sn) & (fs.group == g) & (fs.instrument == z)].iloc[0]
            cells += pair((o.coef, o.se), (n.coef, n.se), 1, 3)
        L.append(f"| Monotonicity | {g}; {z} | " + " | ".join(cells) + " |")
    for v in ["Claims at grant", "Pendency to grant (years)"]:
        cells = []
        for so, sn in smp:
            o = ce[(ce["sample"] == so) & (ce.variable == v)].iloc[0]
            n = exc[(exc["sample"] == sn) & (exc.variable == v)].iloc[0]
            cells += pair((o.coef_per_10pp, o.se), (n.coef_per_10pp, n.se), 1, 2)
        L.append(f"| Exclusion | {v} | " + " | ".join(cells) + " |")
    cells = []
    for so, sn in smp:
        o = dc[dc.tech_center == so]
        n = cal[cal["sample"] == sn].iloc[0]
        cells += ([plain(cin(o.strict_minus_lenient_resid.iloc[0], o.se.iloc[0], 3))] if len(o) else ["—"])
        cells += [plain(cin(n.strict_minus_lenient_resid, n.se, 3))]
    L.append("| Calibration | Strict − lenient residual | " + " | ".join(cells) + " |")
    return L + [""]


def report(bal, fs, exc, cal, alphas, N, runtime) -> None:
    def ci(b, se, nd=2, sc=1.0):
        return plain(cin(b, se, nd, sc))
    L = ["# Assumptions and tests on the main sample (all eight technology centers)", "",
         f"Applications filed 2010–2014 in the eight utility technology centers with the pilot's restrictions: "
         f"{N['xBaApplications']}. Cells: art unit × filing year × USPC class; leniency: the examiner's leave-one-out "
         "grant rate; 95% CIs clustered by examiner. Full results by center: output/tables/design_*.csv.", "",
         "## Summary", "",
         f"- Balance. A 10-point more lenient examiner receives {N['xBaSmall']} pp {N['xBaSmallCI'].replace('$-$', '−')} "
         f"small-entity applications (pooled; CI below zero in {N['xBaSmallNeg']} of 8 centers, above in "
         f"{N['xBaSmallPos']}), and {N['xBaFemale']} pp {N['xBaFemaleCI'].replace('$-$', '−')} female-first-inventor "
         f"applications (CI excludes zero in {int(N['xBaFemaleNeg']) + int(N['xBaFemalePos'])} of 8 centers). "
         f"Predicted citations: pooled CI excludes zero for {N['xBaCitPooledSig']} of {N['xBaCitEvaluators']} "
         f"evaluators; by center, {N['xBaCitCenterSig']} of {N['xBaCitCenterTests']} evaluator × center CIs exclude "
         f"zero; the largest pooled shift is {N['xBaCitMaxAbs']} citations per 10 points of leniency "
         f"({N['xBaCitMaxAbsEvaluator']}; {N['xBaCitMaxAbsPct']}% of the mean of {N['xBaMeanCites']} among granted); "
         f"the largest by center is {N['xBaCitCenterMax']} {N['xBaCitCenterMaxCI']} (TC {N['xBaCitCenterMaxTC']}, "
         f"{N['xBaCitCenterMaxEvaluator']}; {N['xBaCitCenterMaxPct']}% of that center's mean prediction). Without "
         f"TC 3700 the pooled CI excludes zero for {N['xBaCitPooledSigNoThreeSeven']} of {N['xBaCitEvaluators']} "
         f"evaluators (largest pooled shift {N['xBaCitMaxAbsNoThreeSeven']}). The grant score itself moves by "
         f"{N['xBaScore']} log-odds per 10 points ({N['xBaScoreSDPct']}% of its SD of {N['xBaScoreSD']}).",
         f"- Monotonicity. Pooled first stage {N['xBaFsAll']} {N['xBaFsAllCI']}; smallest pooled coefficient "
         f"{N['xBaFsMin']} {N['xBaFsMinCI']} ({N['xBaFsMinGroup']}, {N['xBaFsMinInstrument']}); smallest by center "
         f"{N['xBaFsMinCenter']} {N['xBaFsMinCenterCI']} (TC {N['xBaFsMinCenterTC']}, {N['xBaFsMinCenterGroup']}, "
         f"{N['xBaFsMinCenterInstrument']}); {N['xBaFsPositive']} of {N['xBaFsTests']} group × instrument × center "
         "CIs lie above zero.",
         f"- Exclusion. A 10-point more lenient examiner grants {N['xBaClaims']} {N['xBaClaimsCI']} more claims "
         f"(mean {N['xBaClaimsMean']}) and grants {N['xBaPendency']} {N['xBaPendencyCI']} years sooner "
         f"(mean pendency {N['xBaPendencyMean']} years); by center {N['xBaClaimsMin']} to {N['xBaClaimsMax']} claims "
         f"and {N['xBaPendencyMin']} to {N['xBaPendencyMax']} years sooner.",
         f"- Calibration. Strict − lenient residual, pooled: {N['xBaCal']} {N['xBaCalCI']} log points; CI excludes "
         f"zero in {N['xBaCalSigN']} of 8 centers ({N['xBaCalSigCenters']}).", ""]
    L = [x.replace("$-$", "−") for x in L]
    order = TCS + [POOLED, PILOT]
    border = order + [NO3700]
    L += ["## 1. Balance: coefficient on leniency × 0.1", "",
          "Female first inventor and small entity in pp; inventors and the grant score (log-odds) in units; predicted "
          "citations in citations per patent. The last column pools the centers other than TC 3700, whose heavy-tailed "
          f"citations (mean {N['xBaMeanCitesThreeSeven']} among granted, maximum {N['xBaMaxCitesThreeSeven']}) make "
          "the Poisson-recalibrated predictions convex in "
          "the score, so that a small shift in the score becomes a larger shift in predicted citations.", "",
          "| Variable | " + " | ".join(border) + " |", "|---|" + "---|" * len(border)]
    for v in bal.variable.unique():
        sc, nd = BAL_FMT.get(v, (1, 3))
        t = bal[bal.variable == v].set_index("sample")
        L.append(f"| {v} | " + " | ".join(ci(t.loc[s, 'coef_per_10pp'], t.loc[s, 'se'], nd, sc) for s in border) +
                 " |")
    L += ["", "## 2. Monotonicity: grant on leniency within cells", "",
          "| Group | Instrument | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for (g, z), t in fs.groupby(["group", "instrument"], sort=False):
        t = t.set_index("sample")
        L.append(f"| {g} | {z} | " + " | ".join(ci(t.loc[s, 'coef'], t.loc[s, 'se'], 3) for s in order) + " |")
    L += ["", "## 3. Exclusion: granted patents, coefficient on leniency × 0.1 (mean)", "",
          "| Sample | Granted | Claims at grant | Pendency to grant (years) |", "|---|---|---|---|"]
    for s in order:
        t = exc[exc["sample"] == s].set_index("variable")
        L.append(f"| {s} | {int(t.granted.iloc[0]):,} | " + " | ".join(
            f"{ci(t.loc[v, 'coef_per_10pp'], t.loc[v, 'se'])} ({t.loc[v, 'mean']:.1f})"
            for v in ["Claims at grant", "Pendency to grant (years)"]) + " |")
    L += ["", "## 4. Calibration: citation model trained on lenient examiners' grants", "",
          "Residual = log(1 + citations) − prediction; strict − lenient within cells with controls (log points). "
          "Negative: strict examiners' grants are cited less than the model predicts.", "",
          "| Sample | Strict grants | Lenient grants | Mean residual, strict | Mean residual, lenient | "
          "Strict − lenient (within cells) |", "|---|---|---|---|---|---|"]
    for s in order:
        r = cal[cal["sample"] == s].iloc[0]
        L.append(f"| {s} | {int(r.strict_grants):,} | {int(r.lenient_grants):,} | {r.mean_resid_strict:.3f} | "
                 f"{r.mean_resid_lenient:.3f} | {ci(r.strict_minus_lenient_resid, r.se, 3)} |")
    ch = alphas.groupby("tech_center").alpha.agg(lambda v: ", ".join(f"{a:g}" for a in sorted(set(v))))
    L += ["", "Ridge penalties chosen (by center): " + "; ".join(f"{k}: {v}" for k, v in ch.items()) + ".", ""]
    L += pilot_comparison(bal, fs, exc, cal)
    L += [f"Runtime: {runtime / 60:.1f} minutes on {THREADS} threads.", ""]
    (OUT / "design_balance.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d = load()
    fold = assign_folds(d, DESIGN)  # 38's folds (same rows, same order)
    print(f"loaded {len(d):,} applications in {time.time() - t0:.0f}s", flush=True)
    with threadpool_limits(THREADS):
        bal = balance(d, fold, BAL_DESIGN)
        print(f"balance done ({time.time() - t0:.0f}s)", flush=True)
        fs = first_stage(d)
        print(f"first stage done ({time.time() - t0:.0f}s)", flush=True)
        exc = exclusion(d)
        print(f"exclusion done ({time.time() - t0:.0f}s)", flush=True)
    X = np.load(E / "all_s0.npy").astype(np.float32)
    cal, alphas = calibration(d, X, fold)
    del X
    print(f"calibration done ({time.time() - t0:.0f}s)", flush=True)
    bal.to_csv(T / "design_balance.csv", index=False)
    fs.to_csv(T / "design_first_stage.csv", index=False)
    exc.to_csv(T / "design_exclusion.csv", index=False)
    cal.to_csv(T / "design_calibration.csv", index=False)
    alphas.to_csv(T / "design_calibration_penalty.csv", index=False)
    N = write_outputs(bal, fs, exc, cal, alphas, d, time.time() - t0)
    print(f"{len(N)} macros -> {TEX / 'balance_numbers.tex'}; total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
