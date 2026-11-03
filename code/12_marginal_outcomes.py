"""Marginal outcomes: how did the applications that strict and lenient examiners decide differently turn out, and do
the counterfactual evaluators predict it?

Identification. Applications in the same art unit, filing year, and USPC class are assigned to examiners
quasi-randomly, so strict and lenient examiners face the same mix of applications. (Art unit × year alone is not
enough: within an art unit, examiners specialize by class, and lenient examiners then receive applications with
systematically different text; see the balance table.) The applications a lenient examiner grants and a strict one
rejects are the marginal applications. Their mean outcome Y (renewal, high citations) is identified even though a
rejected application has no outcome:
    E[Y | marginal] = (share granted with Y, lenient − strict) / (grant rate, lenient − strict).
We estimate it by two-stage least squares of granted × Y on granted, instrumented by the examiner's leave-one-out
grant rate (04), with art unit × filing year × USPC class fixed effects, controls for entity size, inventor gender, and
number of inventors, and standard errors clustered by examiner. Comparing it across applicant groups (female vs. male
first inventor, small vs. large entity) is the outcome test of Arnold, Dobbie and Yang (2018): a group whose marginal
applications do better faces a higher bar.

Model analog. Each counterfactual evaluator gives every application a predicted outcome Ŷ. Putting Ŷ in place of Y in
the same formula gives the model's implied outcome for the marginal applications; putting Ŷ − Y gives the model's
signed error on them (positive: the model overstates how the marginal applications did). The formula is valid only if
Ŷ depends on the application and not on the examiner it drew; the balance check regresses Ŷ on leniency. Models whose
inputs are produced during examination (prosecution history) fail it by construction and are reported only there.

Predictions. Five-fold cross-fitting within each technology center, with folds grouped by examiner, so that every
application is scored by a model that never saw its examiner's decisions. Each model is a logistic regression on the
Qwen3-Embedding vectors, optionally with coded fields appended, trained on its own label, with the L2 penalty chosen
by inner cross-validated AUC (as in 09). Its score is mapped to each outcome's scale by a logistic recalibration
fitted on the granted applications of the training folds, the only ones with an observed outcome (the scores used for
this are themselves cross-fitted within the training folds). "No model" predicts the training folds' mean outcome
among granted applications for every application.

Decision–outcome curve. The grant-trained and renewal-trained text models are blended, score = w · grant score +
(1 − w) · renewal score (each standardized), for w from 0 to 1, and each blend is recalibrated as above. For each w we
report the signed error on the marginal applications and the gender gap in decisions made at the human grant rate
(top r% of scores within each technology center and fold; art unit × year fixed effects, as in 09 and 11).

Diagnostics. Balance of covariates and model predictions under the two cell definitions. Monotonicity: the first
stage within applicant groups and within quartiles of the predicted grant score, and reverse-sample instruments (an
examiner's grant rate on the other group's applications). Exclusion: claims at grant and pendency among granted
patents, and the bias they would imply if their cross-sectional associations with the outcomes were causal.
Robustness: coarser and finer assignment cells, and first-action leniency as the instrument.

Input: data/processed/{pilot_inputs,pilot_sample,applications_leniency,app_gender,patex_application_data}.parquet,
       data/processed/emb/, data/patentsview/tsv/g_patent.tsv, data/maintenance_fee/txt/ (file date only)
Output: output/marginal.md, output/tables/marginal_*.csv; cross-fitted predictions in
        data/processed/crossfit_preds.parquet
"""
import glob
import pathlib
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cfe.check import CELL, CONTRASTS, CONTROLS, GROUPS, HUMAN, OUTCOMES, SAMPLES, marginal, ols_coef, series_matrix
from cfe.estimation import cluster_se, demean, tsls

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
PV = ROOT / "data/patentsview/tsv"
OUT = ROOT / "output"
SEED = 20261028
K_OUTER, K_INNER = 5, 4
C_GRID = [0.5, 0.05, 0.005, 0.0005, 0.00005]  # L2 penalties tried by inner cross-validation
N_JOBS, THREADS = 20, 4
PROCESS = ["n_rejections", "n_rce", "appealed"]
TCS = ["3600", "1600"]
LABELS = ["granted", "fa_allow", "renew_4y", "cites_hi"]
SPECS = [  # name, input, label (outcome labels exist only for granted applications)
    ("Train on grant", "s0", "granted"),
    ("Train on grant + names", "s1", "granted"),
    ("Train on grant + entity status", "s0+entity", "granted"),
    ("Train on first action", "s0", "fa_allow"),
    ("Train on renewal (granted only)", "s0", "renew_4y"),
    ("Train on renewal + names", "s1", "renew_4y"),
    ("Train on renewal + entity status", "s0+entity", "renew_4y"),
    ("Train on citations (granted only)", "s0", "cites_hi"),
    ("Train on citations + entity status", "s0+entity", "cites_hi"),
    ("Train on grant + prosecution history", "s0+process", "granted"),
    ("Train on renewal + prosecution history", "s0+process", "renew_4y"),
]
POST_ASSIGNMENT = {"s0+process"}  # inputs produced during examination
NO_MODEL = "No model: mean outcome of granted applications"
BLEND = ("Train on grant", "Train on renewal (granted only)")
WEIGHTS = np.round(np.linspace(0, 1, 11), 1)
CELL_NAMES = {"au_year": "Art unit × year", "au_class": "Art unit × year × USPC class",
              "au_subclass": "Art unit × year × USPC subclass"}
ROBUST = [("au_year", []), ("au_year", CONTROLS), ("au_class", CONTROLS), ("au_subclass", CONTROLS)]
MIN_REV = 10  # minimum applications behind a reverse-sample leniency


# ---------------------------------------------------------------- data

def load() -> tuple[pd.DataFrame, dict]:
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    s = pd.read_parquet(P / "pilot_sample.parquet")
    d = inp.merge(s, on="application_number", how="left")
    g = d[d.granted == 1]
    med = g.groupby(["tech_center", "filing_year"])["cites_5y"].transform("median")
    d["cites_hi"] = np.where(d.granted == 1, (d["cites_5y"] > med.reindex(d.index)).astype(float), np.nan)
    d["renew_4y"] = d["renew_4y"].astype(float)
    d["appealed"] = d["appealed"].astype(int)
    d = d.join(extras(d))
    s0 = np.load(E / "s0.npy").astype(np.float32)
    X = {"s0": s0, "s1": np.load(E / "s1.npy").astype(np.float32),
         "s0+entity": np.hstack([s0, d[["small_entity"]].to_numpy(np.float32)]),
         "s0+process": np.hstack([s0, d[PROCESS].to_numpy(np.float32)])}
    return d, X


def extras(d: pd.DataFrame) -> pd.DataFrame:
    """USPC cells; pendency and claims at grant (exclusion checks); reverse-sample leniencies (monotonicity)."""
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'")
    con.register("pilot", d[["application_number", "examiner_id", "patent_number"]])
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
    SELECT p.application_number, l.filing_date, gp.num_claims,
           coalesce(u.uspc_class, 'none') AS uspc_class, coalesce(u.uspc_subclass, 'none') AS uspc_subclass,
           ex.gr_f, ex.n_f, ex.gr_m, ex.n_m, ex.gr_s, ex.n_s, ex.gr_l, ex.n_l
    FROM pilot p
    JOIN '{P}/applications_leniency.parquet' l ON l.application_number = p.application_number
    LEFT JOIN '{P}/patex_application_data.parquet' u ON u.application_number = p.application_number
    LEFT JOIN ex ON ex.examiner_id = p.examiner_id
    LEFT JOIN gp ON gp.patent_id = p.patent_number
    """).df()
    x = d[["application_number"]].merge(x, on="application_number", how="left")
    out = pd.DataFrame(index=d.index)
    out["au_class"] = d["au_year"] + "_" + x["uspc_class"]
    out["au_subclass"] = out["au_class"] + "/" + x["uspc_subclass"]
    issue = pd.to_datetime(d["patent_issue_date"], errors="coerce")
    out["pendency_years"] = np.where(d.granted == 1, (issue - pd.to_datetime(x["filing_date"])).dt.days / 365.25,
                                     np.nan)
    out["num_claims"] = np.where(d.granted == 1, x["num_claims"], np.nan)

    def rate(r, n):
        return np.where(x[n] >= MIN_REV, x[r], np.nan)
    out["lenient_rev_gender"] = np.where(d.female == 1, rate("gr_m", "n_m"), rate("gr_f", "n_f"))
    out["lenient_rev_entity"] = np.where(d.small_entity == 1, rate("gr_l", "n_l"), rate("gr_s", "n_s"))
    return out


# ---------------------------------------------------------------- cross-fitted models

def calibrate(s_fit: np.ndarray, y_fit: np.ndarray, s_new: np.ndarray) -> np.ndarray:
    m = LogisticRegression(C=1e6, max_iter=1000).fit(s_fit[:, None], y_fit.astype(int))
    return m.predict_proba(s_new[:, None])[:, 1]


def spec_task(X: np.ndarray, y: np.ndarray, granted: np.ndarray, ex: np.ndarray, tr: np.ndarray, te: np.ndarray,
              seed: int) -> dict:
    """One model in one outer fold.

    The penalty is chosen from C_GRID by inner cross-validated AUC on the model's own label (inner folds grouped by
    examiner). Returns the test scores and the inner cross-fitted scores of the granted training applications, which
    are used for recalibration.
    """
    with threadpool_limits(THREADS):
        oof = {c: np.full(len(tr), np.nan) for c in C_GRID}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            fa = tr[ia][~np.isnan(y[tr[ia]])]
            sc = StandardScaler().fit(X[fa])
            Xa, Xb = sc.transform(X[fa]), sc.transform(X[tr[ib]])
            for c in C_GRID:
                oof[c][ib] = LogisticRegression(C=c, max_iter=3000).fit(Xa, y[fa].astype(int)).decision_function(Xb)
        ok = ~np.isnan(y[tr])
        auc = {c: roc_auc_score(y[tr][ok].astype(int), oof[c][ok]) for c in C_GRID}
        best = max(auc, key=auc.get)
        fit = tr[ok]
        sc = StandardScaler().fit(X[fit])
        m = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X[fit]), y[fit].astype(int))
        return dict(C=best, auc_inner=auc[best], s_te=m.decision_function(sc.transform(X[te])),
                    s_in=oof[best][granted[tr] == 1])


def crossfit(d: pd.DataFrame, X: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted scores, recalibrated predictions, and decision–outcome blends for every application."""
    lab = {c: d[c].to_numpy(float) for c in LABELS}
    ex = pd.factorize(d["examiner_id"])[0]
    fold = np.full(len(d), -1)
    folds = []
    for tc in TCS:
        rows = np.where(d.tech_center == tc)[0]
        for k, (a, b) in enumerate(GroupKFold(K_OUTER, shuffle=True, random_state=SEED).split(rows, groups=ex[rows])):
            fold[rows[b]] = k
            folds.append((tc, k, rows[a], rows[b]))
    jobs = [(i, name, info, label) for i in range(len(folds)) for name, info, label in SPECS]
    out = Parallel(n_jobs=N_JOBS)(delayed(spec_task)(X[info], lab[label], lab["granted"], ex, folds[i][2], folds[i][3],
                                                     SEED + i) for i, name, info, label in jobs)
    res = {(i, name): r for (i, name, _, _), r in zip(jobs, out)}
    cols = {}

    def put(key, te, v):
        cols.setdefault(key, np.full(len(d), np.nan))[te] = v
    chosen = []
    for i, (tc, k, tr, te) in enumerate(folds):
        g_tr = tr[lab["granted"][tr] == 1]
        for name, _, _ in SPECS:
            r = res[(i, name)]
            chosen.append(dict(tech_center=tc, fold=k, series=name, C=r["C"], auc_inner=r["auc_inner"]))
            put(f"score::{name}", te, r["s_te"])
            for o in OUTCOMES:
                put(f"{o}::{name}", te, calibrate(r["s_in"], lab[o][g_tr], r["s_te"]))
        for o in OUTCOMES:
            put(f"{o}::{NO_MODEL}", te, np.full(len(te), lab[o][g_tr].mean()))
        z = {}
        for name in BLEND:
            r = res[(i, name)]
            mu, sd = r["s_in"].mean(), r["s_in"].std()
            z[name] = ((r["s_in"] - mu) / sd, (r["s_te"] - mu) / sd)
        for w in WEIGHTS:
            b_in = w * z[BLEND[0]][0] + (1 - w) * z[BLEND[1]][0]
            b_te = w * z[BLEND[0]][1] + (1 - w) * z[BLEND[1]][1]
            put(f"score::blend {w:.1f}", te, b_te)
            for o in OUTCOMES:
                put(f"{o}::blend {w:.1f}", te, calibrate(b_in, lab[o][g_tr], b_te))
    pred = pd.DataFrame({"application_number": d["application_number"], "fold": fold})
    return pd.concat([pred, pd.DataFrame(cols)], axis=1), pd.DataFrame(chosen)


# ---------------------------------------------------------------- estimation

def split_table(d: pd.DataFrame, names: list) -> pd.DataFrame:
    """Draft Table 1 layout: per 100 applications, examiners below vs. above the cell mean leniency."""
    rows = []
    for sname, tcs in SAMPLES.items():
        s = d[d.tech_center.isin(tcs) & d.lenient_grant.notna()]
        zt = demean(s[["lenient_grant"]].to_numpy(float), pd.factorize(s[CELL])[0])[:, 0]
        hi = (zt > 0).astype(float)
        D = s["granted"].to_numpy(float)
        for o in OUTCOMES:
            V = np.column_stack([D, series_matrix(s, o, names)[:, : len(names) + 1]])
            delta, _ = tsls(V, hi, hi, s[CELL], s[CONTROLS].to_numpy(float))
            strict = V[hi == 0].mean(0)
            for j, series in enumerate([HUMAN] + names):
                rows.append(dict(sample=sname, outcome=o, series=series,
                                 strict_granted=100 * strict[0], strict_granted_y=100 * strict[j + 1],
                                 lenient_granted=100 * (strict[0] + delta[0]),
                                 lenient_granted_y=100 * (strict[j + 1] + delta[j + 1]),
                                 marginal_split=100 * delta[j + 1] / delta[0]))
    return pd.DataFrame(rows)


def fit_table(d: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """Held-out AUC of each model on its own label (cross-fitted), and the penalties chosen across folds."""
    rows = []
    for sname, tcs in SAMPLES.items():
        s = d[d.tech_center.isin(tcs)]
        for name, _, label in SPECS:
            ok = s[label].notna()
            cs = chosen[chosen.tech_center.isin(tcs) & (chosen.series == name)]["C"]
            rows.append(dict(sample=sname, series=name, label=label,
                             auc_own=roc_auc_score(s.loc[ok, label].astype(int), s.loc[ok, f"score::{name}"]),
                             C_chosen=", ".join(f"{c:g}" for c in sorted(cs.unique(), reverse=True))))
    return pd.DataFrame(rows)


def robustness(d: pd.DataFrame) -> pd.DataFrame:
    """The human marginal outcome and its group contrasts under other assignment cells and instruments."""
    rows = []
    specs = [(fe, ctrl, "lenient_grant") for fe, ctrl in ROBUST] + [(CELL, CONTROLS, "lenient_fa")]
    for fe, ctrl, z in specs:
        est, con = marginal(d, [], z=z, fe=fe, controls=ctrl)
        for (sname, o), e in est.groupby(["sample", "outcome"], sort=False):
            c = con[(con["sample"] == sname) & (con.outcome == o)].set_index("contrast")
            a = e[e.group == "All"].iloc[0]
            rows.append(dict(cells=CELL_NAMES[fe], controls="yes" if ctrl else "no",
                             instrument="first-action leniency" if z == "lenient_fa" else "grant leniency",
                             sample=sname, outcome=o, first_stage=a.first_stage, marginal=a.estimate, se=a.se,
                             f_minus_m=c.loc["Female − male", "estimate"], f_minus_m_se=c.loc["Female − male", "se"],
                             s_minus_l=c.loc["Small − large", "estimate"], s_minus_l_se=c.loc["Small − large", "se"]))
    return pd.DataFrame(rows)


def diagnostics(d: pd.DataFrame) -> dict:
    out = {}
    # balance: pre-assignment covariates and model predictions should not move with leniency within a cell
    rows = []
    covs = [("female", "Female first inventor"), ("small_entity", "Small entity"), ("n_inventors", "Inventors"),
            ("score::Train on grant", "Predicted grant score (text, log-odds)")]
    covs += [(f"renew_4y::{m}", f"Predicted renewal: {m}") for m, _, _ in SPECS]
    for fe in ["au_year", CELL]:
        for sname, tcs in SAMPLES.items():
            s = d[d.tech_center.isin(tcs)]
            for c, lab in covs:
                b, se = ols_coef(s, c, "lenient_grant", fe=fe)
                post = any(lab.endswith(m) for m, i, _ in SPECS if i in POST_ASSIGNMENT)
                rows.append(dict(cells=CELL_NAMES[fe], sample=sname, variable=lab, coef_per_10pp=0.1 * b,
                                 se=0.1 * se, post_assignment=post))
    out["balance"] = pd.DataFrame(rows)

    # monotonicity: first stage within groups and within quartiles of the predicted grant score
    rows = []
    for sname, tcs in SAMPLES.items():
        ds = d[d.tech_center.isin(tcs)].copy()
        ds["q"] = ds.groupby("tech_center")["score::Train on grant"].transform(
            lambda v: pd.qcut(v, 4, labels=False) + 1)
        cells = [(g, (lambda x, g=g: x if GROUPS[g] is None else x[x[GROUPS[g][0]] == GROUPS[g][1]])) for g in GROUPS]
        cells += [(f"Predicted grant, quartile {q}", (lambda x, q=q: x[x.q == q])) for q in range(1, 5)]
        for gname, f in cells:
            b, se = ols_coef(f(ds), "granted", "lenient_grant", controls=CONTROLS)
            rows.append(dict(sample=sname, group=gname, instrument="Leave-one-out grant rate", coef=b, se=se))
        for gname, zc in [("Female", "lenient_rev_gender"), ("Male", "lenient_rev_gender"),
                          ("Small entity", "lenient_rev_entity"), ("Large entity", "lenient_rev_entity")]:
            g = GROUPS[gname]
            b, se = ols_coef(ds[(ds[g[0]] == g[1]) & ds[zc].notna()], "granted", zc, controls=CONTROLS)
            other = {"Female": "male", "Male": "female", "Small entity": "large-entity",
                     "Large entity": "small-entity"}[gname]
            rows.append(dict(sample=sname, group=gname, instrument=f"Grant rate on {other} applications (reverse)",
                             coef=b, se=se))
    out["first_stage"] = pd.DataFrame(rows)

    # exclusion: do lenient examiners grant broader or faster patents? (granted only, so composition also moves)
    rows, brows = [], []
    for sname, tcs in SAMPLES.items():
        s = d.loc[d.tech_center.isin(tcs), ["granted", "num_claims", "pendency_years", "lenient_grant", CELL,
                                            "examiner_id", "renew_4y", "cites_hi"] + CONTROLS]
        g = s[s.granted == 1].dropna(subset=["num_claims", "pendency_years"])
        a = {}
        for y, lab in [("num_claims", "Claims at grant"), ("pendency_years", "Pendency to grant (years)")]:
            fit = pf.feols(f"{y} ~ lenient_grant + {' + '.join(CONTROLS)} | {CELL}", data=g,
                           vcov={"CRV1": "examiner_id"})
            a[y] = fit.coef()["lenient_grant"]
            rows.append(dict(sample=sname, variable=lab, mean=g[y].mean(), coef_per_10pp=0.1 * a[y],
                             se=0.1 * fit.se()["lenient_grant"], n=fit._N))
        fs, _ = ols_coef(s, "granted", "lenient_grant", controls=CONTROLS)
        for o in OUTCOMES:
            fit = pf.feols(f"{o} ~ num_claims + pendency_years + {' + '.join(CONTROLS)} | {CELL}", data=g,
                           vcov={"CRV1": "examiner_id"})
            cf = fit.coef()
            bias = s.granted.mean() * sum(cf[k] * a[k] for k in a) / fs
            brows.append(dict(sample=sname, outcome=o, assoc_claims=cf["num_claims"],
                              assoc_pendency=cf["pendency_years"], implied_bias=bias))
    out["exclusion"] = pd.DataFrame(rows)
    out["exclusion_bias"] = pd.DataFrame(brows)
    return out


def curve(d: pd.DataFrame) -> pd.DataFrame:
    """Decision–outcome blends: signed error on the marginal applications and the decision gender gap."""
    names = [f"blend {w:.1f}" for w in WEIGHTS]
    d = d.copy()
    for tc in TCS:
        m = (d.tech_center == tc).to_numpy()
        r = d.loc[m, "granted"].mean()
        for nm in names:
            sc = d.loc[m, f"score::{nm}"]
            cut = sc.groupby(d.loc[m, "fold"]).transform(lambda v: v.quantile(1 - r))
            d.loc[m, f"dec::{nm}"] = (sc >= cut).astype(float)
    err, con = marginal(d, names)
    rows = []
    for sname, tcs in SAMPLES.items():
        s = d[d.tech_center.isin(tcs)]
        Y = s[["granted"] + [f"dec::{nm}" for nm in names]].to_numpy(float)
        f = s["female"].to_numpy(float)
        b, psi = tsls(Y, f, f, s["au_year"])
        se = cluster_se(psi, s["examiner_id"])
        for j, nm in enumerate(names):
            row = dict(sample=sname, weight_on_grant=float(nm.split()[1]), gap_decision=b[j + 1], gap_se=se[j + 1],
                       human_gap=b[0], ratio=b[j + 1] / b[0])
            for o in OUTCOMES:
                e = err[(err["sample"] == sname) & (err.outcome == o) & (err.series == nm) & (err.kind == "error")]
                x = e[e.group == "All"].iloc[0]
                row[f"{o}_error"], row[f"{o}_error_se"] = x.estimate, x.se
                for cname, key in [("Female − male", "f_minus_m"), ("Small − large", "s_minus_l")]:
                    c = con[(con["sample"] == sname) & (con.outcome == o) & (con.series == nm)
                            & (con.kind == "error") & (con.contrast == cname)].iloc[0]
                    row[f"{o}_error_{key}"], row[f"{o}_error_{key}_se"] = c.estimate, c.se
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def ci(b: float, se: float, scale: float = 100, nd: int = 1) -> str:
    return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"


def report(est, con, split, fit, rob, diag, cur, n_info) -> str:
    valid = [NO_MODEL] + [m for m, i, _ in SPECS if i not in POST_ASSIGNMENT]
    L = ["# Marginal outcomes: validating counterfactual evaluators with quasi-random examiner assignment", "",
         "Pilot sample (TC 3600 and 1600, filed 2010–2014); \"Both\" pools the two technology centers. " + n_info, "",
         "Marginal applications are those a lenient examiner grants and a strict one rejects. Their mean outcome "
         "comes from 2SLS of granted × outcome on granted, instrumented by the examiner's leave-one-out grant rate, "
         "with art unit × filing year × USPC class fixed effects and controls for entity size, inventor gender, and "
         "number of inventors; standard errors are clustered by examiner and 95% CIs are in brackets. Model "
         "predictions are cross-fitted (5 folds grouped by examiner; L2 penalty chosen by inner cross-validation, "
         "see tables/marginal_fit.csv) and recalibrated to each outcome on the granted "
         "applications of the training folds. Signed error = the model's implied outcome for the marginal "
         "applications − the actual one (positive: the model overstates how they did). All values are percentages "
         "or percentage points unless noted.", ""]

    L += ["## Table 1. Outcomes of the marginal applications, actual and as predicted", "",
          "Strict / lenient: examiners below / above the cell mean leniency, per 100 applications "
          "(lenient = strict + the regression-adjusted difference). Split = the ratio of the two differences; "
          "2SLS uses continuous leniency. Model rows replace each application's outcome with the model's "
          "prediction; the grant columns are the examiners' actual decisions.", ""]
    for o, olab in OUTCOMES.items():
        L += [f"### {olab}", "",
              "| Sample | Row | Strict: granted | Strict: granted with outcome | Lenient: granted | "
              "Lenient: granted with outcome | Marginal (split) | Marginal (2SLS) | Signed error | Held-out AUC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sname in SAMPLES:
            for series in [HUMAN] + valid:
                sp = split[(split["sample"] == sname) & (split.outcome == o) & (split.series == series)].iloc[0]
                e = est[(est["sample"] == sname) & (est.outcome == o) & (est.group == "All") & (est.series == series)]
                imp = e[e.kind.isin(["marginal", "implied"])].iloc[0]
                err = e[e.kind == "error"]
                auc = fit[(fit["sample"] == sname) & (fit.series == series)]
                auc_s = f"{auc.iloc[0].auc_own:.3f}" if len(auc) else "—"
                err_s = ci(err.iloc[0].estimate, err.iloc[0].se) if len(err) else "—"
                L.append(f"| {sname} | {series} | {sp.strict_granted:.1f} | {sp.strict_granted_y:.1f} | "
                         f"{sp.lenient_granted:.1f} | {sp.lenient_granted_y:.1f} | {sp.marginal_split:.1f} | "
                         f"{ci(imp.estimate, imp.se)} | {err_s} | {auc_s} |")
        L.append("")

    L += ["## Table 2. By applicant group: the marginal outcome and each model's signed error", "",
          "The contrasts in the \"Marginal outcome\" row are the outcome test: a group whose marginal applications "
          "do better faces a higher bar. \"Mean among granted\" is the average granted application, for comparison.",
          ""]
    cols = ["All", "Female", "Male", "Female − male", "Small entity", "Large entity", "Small − large"]
    for o, olab in OUTCOMES.items():
        for sname in SAMPLES:
            L += [f"### {olab}, {sname}", "", "| Row | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
            e = est[(est["sample"] == sname) & (est.outcome == o)]
            c = con[(con["sample"] == sname) & (con.outcome == o)]
            mg = e[e.series == HUMAN].set_index("group")["mean_granted"]
            L.append("| Mean among granted | " + " | ".join(
                f"{100 * mg[g]:.1f}" if g in mg else
                f"{100 * (mg[CONTRASTS[g][0]] - mg[CONTRASTS[g][1]]):.1f}" for g in cols) + " |")
            for series, kind in [(HUMAN, "marginal")] + [(m, "error") for m in valid]:
                cells = []
                for g in cols:
                    t = (c[(c.contrast == g) & (c.series == series) & (c.kind == kind)] if g in CONTRASTS else
                         e[(e.group == g) & (e.series == series) & (e.kind == kind)]).iloc[0]
                    cells.append(ci(t.estimate, t.se))
                lab = "Marginal outcome (truth)" if kind == "marginal" else f"Error: {series}"
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## Table 3. Decision–outcome curve", "",
          "Blend of the grant-trained and renewal-trained text models: score = w · grant score + (1 − w) · renewal "
          "score. Decision gap: female − male difference in decisions made at the human grant rate (art unit × year "
          "fixed effects, as in probes.md); ratio = model gap / human grant gap. Errors are on the marginal "
          "applications.", "",
          "| Sample | Weight on grant | Decision gap | Ratio | Renewal error | Renewal error, F − M | "
          "Renewal error, S − L | Citation error | Citation error, F − M |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in cur.iterrows():
        L.append(f"| {r['sample']} | {r.weight_on_grant:.1f} | {ci(r.gap_decision, r.gap_se)} | {r.ratio:.2f} | "
                 f"{ci(r.renew_4y_error, r.renew_4y_error_se)} | "
                 f"{ci(r.renew_4y_error_f_minus_m, r.renew_4y_error_f_minus_m_se)} | "
                 f"{ci(r.renew_4y_error_s_minus_l, r.renew_4y_error_s_minus_l_se)} | "
                 f"{ci(r.cites_hi_error, r.cites_hi_error_se)} | "
                 f"{ci(r.cites_hi_error_f_minus_m, r.cites_hi_error_f_minus_m_se)} |")
    hg = cur.drop_duplicates("sample").set_index("sample")["human_gap"]
    L += ["", "Human grant gap: " + "; ".join(f"{k} {100 * v:.1f}" for k, v in hg.items()) + ".", ""]

    L += ["## Diagnostics", "", "### Balance: does a 10 pp more lenient examiner receive different applications?", "",
          "Coefficient on leniency × 0.1 within assignment cells. Shares and predicted renewal in percentage points. "
          "Rows marked † use inputs produced during examination, so they move with the examiner by construction "
          "and the marginal-outcome formula does not apply to them.", "",
          "| Variable | Cells | " + " | ".join(SAMPLES) + " |", "|---|---|" + "---|" * len(SAMPLES)]
    bal = diag["balance"]
    for (v, cells), t in bal.groupby(["variable", "cells"], sort=False):
        sc, nd = (1, 3) if v in ("Inventors", "Predicted grant score (text, log-odds)") else (100, 2)
        mark = " †" if t.post_assignment.iloc[0] else ""
        L.append(f"| {v}{mark} | {cells} | " + " | ".join(
            ci(t[t["sample"] == s].iloc[0].coef_per_10pp, t[t["sample"] == s].iloc[0].se, sc, nd)
            for s in SAMPLES) + " |")

    L += ["", "### First stage within groups (monotonicity)", "",
          "Coefficient of grant on the instrument. Monotonicity implies a positive first stage in every group; the "
          "reverse-sample rows measure leniency only on the other group's applications.", "",
          "| Group | Instrument | " + " | ".join(SAMPLES) + " |", "|---|---|" + "---|" * len(SAMPLES)]
    fs = diag["first_stage"]
    for (g, z), t in fs.groupby(["group", "instrument"], sort=False):
        L.append(f"| {g} | {z} | " + " | ".join(
            ci(t[t["sample"] == s].iloc[0].coef, t[t["sample"] == s].iloc[0].se, 1, 3) for s in SAMPLES) + " |")

    L += ["", "### Exclusion: claims and pendency among granted patents", "",
          "Coefficient on leniency × 0.1 among granted patents (mean in parentheses). Lenient examiners also grant "
          "more marginal applications, so these differences mix a direct effect with a change in composition.", "",
          "| Variable | " + " | ".join(f"{s} (mean)" for s in SAMPLES) + " |", "|---|" + "---|" * len(SAMPLES)]
    ex = diag["exclusion"]
    for v, t in ex.groupby("variable", sort=False):
        L.append(f"| {v} | " + " | ".join(
            f"{ci(t[t['sample'] == s].iloc[0].coef_per_10pp, t[t['sample'] == s].iloc[0].se, 1, 3)} "
            f"({t[t['sample'] == s].iloc[0]['mean']:.2f})" for s in SAMPLES) + " |")
    L += ["", "Implied bias in the marginal outcome if the cross-sectional associations of claims and pendency with "
          "the outcome (among granted patents, within cells) were causal and the differences above were entirely "
          "direct effects of the examiner:", "",
          "| Outcome | " + " | ".join(SAMPLES) + " |", "|---|" + "---|" * len(SAMPLES)]
    eb = diag["exclusion_bias"]
    for o, olab in OUTCOMES.items():
        t = eb[eb.outcome == o].set_index("sample")
        L.append(f"| {olab} | " + " | ".join(f"{100 * t.loc[s, 'implied_bias']:+.1f}" for s in SAMPLES) + " |")

    L += ["", "### Robustness of the marginal outcome", "",
          "The human marginal outcome and its group contrasts under other assignment cells, without controls, and "
          "with first-action leniency as the instrument (different marginal applications: those whose grant depends "
          "on the first action). The main specification is the third row of each block.", "",
          "| Outcome | Sample | Cells | Controls | Instrument | First stage | Marginal | Female − male | "
          "Small − large |", "|---|---|---|---|---|---|---|---|---|"]
    for (o, sname), t in rob.groupby(["outcome", "sample"], sort=False):
        for _, r in t.iterrows():
            L.append(f"| {OUTCOMES[o]} | {sname} | {r.cells} | {r.controls} | {r.instrument} | "
                     f"{r.first_stage:.3f} | {ci(r.marginal, r.se)} | {ci(r.f_minus_m, r.f_minus_m_se)} | "
                     f"{ci(r.s_minus_l, r.s_minus_l_se)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    d, X = load()
    cf, chosen = crossfit(d, X)
    cf.to_parquet(P / "crossfit_preds.parquet", index=False)
    d = pd.concat([d, cf.drop(columns="application_number")], axis=1)
    names_all = [NO_MODEL] + [m for m, _, _ in SPECS]
    est, con = marginal(d, names_all)
    split = split_table(d, names_all)
    fit = fit_table(d, chosen)
    rob = robustness(d)
    diag = diagnostics(d)
    cur = curve(d)
    for name, t in [("estimates", est), ("contrasts", con), ("split", split), ("fit", fit), ("robustness", rob),
                    ("curve", cur), ("balance", diag["balance"]), ("first_stage", diag["first_stage"]),
                    ("exclusion", diag["exclusion"]), ("exclusion_bias", diag["exclusion_bias"])]:
        t.to_csv(OUT / "tables" / f"marginal_{name}.csv", index=False)
    issue = pd.to_datetime(d.loc[d.granted == 1, "patent_issue_date"], errors="coerce")
    mf = sorted(glob.glob(str(ROOT / "data/maintenance_fee/txt/MaintFeeEvents_2*.txt")))[-1]
    open_since = pd.Timestamp(pathlib.Path(mf).stem.split("_")[1]) - pd.DateOffset(years=4)
    n_info = (f"{len(d):,} applications, {d.examiner_id.nunique():,} examiners, {d[CELL].nunique():,} assignment "
              f"cells. Of the grants, {(issue > '2020-12-31').mean():.1%} issued after 2020 (5-year citation window "
              f"incomplete) and {(issue > open_since).mean():.1%} after {open_since:%Y-%m-%d} (4th-year renewal "
              "window still open in the maintenance-fee file used).")
    (OUT / "marginal.md").write_text(report(est, con, split, fit, rob, diag, cur, n_info))
    print((OUT / "marginal.md").read_text())


if __name__ == "__main__":
    main()
