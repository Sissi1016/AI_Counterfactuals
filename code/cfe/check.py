"""The marginal-case check for counterfactual evaluators.

When cases are assigned to decision makers quasi-randomly within cells, the cases a lenient decision maker accepts and
a strict one rejects (the marginal cases) have an identified mean outcome:
    E[Y | marginal] = 2SLS of accepted × Y on accepted, instrumented by the decision maker's leniency, within cells.
Putting an evaluator's predicted outcome Ŷ in place of Y gives its implied outcome for the marginal cases; putting
Ŷ − Y gives its signed error there. Both are computed by applicant group, with standard errors clustered by decision
maker. The predictions must not depend on which decision maker a case drew: balance() checks this.

A Design names the columns of the decision record; PATENTS is the U.S. patent pilot of this repository. Another
setting (courts, grant panels, interviews) needs only a record with the same roles and its own Design.
"""
import dataclasses
import pathlib

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import GroupKFold

from cfe.estimation import cluster_se, tsls

HUMAN = "Human record (truth)"
NO_MODEL = "No model: mean outcome of accepted cases"
SEED, K_FOLDS = 20261028, 5


@dataclasses.dataclass(frozen=True)
class Design:
    case: str = "application_number"
    decider: str = "examiner_id"  # cluster for standard errors; folds are grouped by it
    decision: str = "granted"  # 1 = accepted
    leniency: str = "lenient_grant"  # decider's leave-one-out acceptance rate
    cell: str = "au_class"  # assignment cell: comparisons are made only within it
    controls: tuple = ("small_entity", "female", "n_inventors")
    unit: str = "tech_center"  # column whose values define the samples
    units: tuple = ("3600", "1600")
    samples: tuple = (("3600", ("3600",)), ("1600", ("1600",)), ("Both", ("3600", "1600")))
    outcomes: tuple = (("renew_4y", "Renewal at year 4"), ("cites_hi", "High citations (above the median of granted)"))
    groups: tuple = (("All", None), ("Female", ("female", 1)), ("Male", ("female", 0)),
                     ("Small entity", ("small_entity", 1)), ("Large entity", ("small_entity", 0)))
    contrasts: tuple = (("Female − male", ("Female", "Male")), ("Small − large", ("Small entity", "Large entity")))
    gap_group: str = "female"  # group indicator for the decision gap
    gap_cell: str = "au_year"  # cells for the decision gap (as in 09 and 11)


PATENTS = Design()
# module-level names used by scripts 12 and 13
SAMPLES = dict(PATENTS.samples)
OUTCOMES = dict(PATENTS.outcomes)
GROUPS = dict(PATENTS.groups)
CONTRASTS = dict(PATENTS.contrasts)
CELL = PATENTS.cell
CONTROLS = list(PATENTS.controls)


# ---------------------------------------------------------------- the record

def load_record(root: pathlib.Path) -> pd.DataFrame:
    """The patent pilot record, one row per application in the row order of pilot_inputs (as in 09 and 12)."""
    P = root / "data/processed"
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    d = inp.merge(pd.read_parquet(P / "pilot_sample.parquet"), on="application_number", how="left")
    g = d[d.granted == 1]
    med = g.groupby(["tech_center", "filing_year"])["cites_5y"].transform("median")
    d["cites_hi"] = np.where(d.granted == 1, (d["cites_5y"] > med.reindex(d.index)).astype(float), np.nan)
    d["renew_4y"] = d["renew_4y"].astype(float)
    con = duckdb.connect()
    con.register("pilot", d[["application_number"]])
    u = con.execute(f"""
        SELECT p.application_number, coalesce(u.uspc_class, 'none') AS uspc_class
        FROM pilot p LEFT JOIN '{P}/patex_application_data.parquet' u USING (application_number)""").df()
    d = d.merge(u, on="application_number", how="left")
    d["au_class"] = d["au_year"] + "_" + d["uspc_class"]
    return d


def assign_folds(d: pd.DataFrame, design: Design = PATENTS) -> np.ndarray:
    """Folds grouped by decision maker within each unit (the same folds as 12)."""
    ex = pd.factorize(d[design.decider])[0]
    fold = np.full(len(d), -1)
    for u in design.units:
        rows = np.where(d[design.unit] == u)[0]
        for k, (_, b) in enumerate(GroupKFold(K_FOLDS, shuffle=True, random_state=SEED).split(rows, groups=ex[rows])):
            fold[rows[b]] = k
    return fold


def recalibrate(d: pd.DataFrame, cols: list, fold: np.ndarray, design: Design = PATENTS) -> pd.DataFrame:
    """Map each score column to each outcome's scale out of fold: within a unit, a logistic regression of the outcome
    on the score (a Poisson regression when the outcome is a count), fitted on the accepted cases of the other folds.
    Adds columns '{outcome}::{name}' and the no-model baseline (the other folds' mean outcome among accepted cases)."""
    d = d.copy()
    acc = (d[design.decision] == 1).to_numpy()
    for o, _ in design.outcomes:
        for c in cols + [NO_MODEL]:
            d[f"{o}::{c}"] = np.nan
        y = d[o].to_numpy(float)
        binary = bool(np.isin(y[acc & ~np.isnan(y)], [0.0, 1.0]).all())
        for u in design.units:
            in_u = (d[design.unit] == u).to_numpy()
            for k in np.unique(fold[in_u]):
                fit = in_u & acc & (fold != k)
                new = in_u & (fold == k)
                d.loc[new, f"{o}::{NO_MODEL}"] = y[fit].mean()
                for c in cols:
                    s = d[c].to_numpy(float)
                    if binary:
                        m = LogisticRegression(C=1e6, max_iter=1000).fit(s[fit][:, None], y[fit].astype(int))
                        d.loc[new, f"{o}::{c}"] = m.predict_proba(s[new][:, None])[:, 1]
                    else:
                        m = PoissonRegressor(alpha=0.0, max_iter=1000).fit(s[fit][:, None], y[fit])
                        d.loc[new, f"{o}::{c}"] = m.predict(s[new][:, None])
    return d


# ---------------------------------------------------------------- estimates

def series_matrix(s: pd.DataFrame, o: str, names: list, decision: str = "granted") -> np.ndarray:
    """Columns: accepted × Y, then accepted × Ŷ for each evaluator, then accepted × (Ŷ − Y) for each evaluator."""
    D = s[decision].to_numpy(float)
    y = np.where(D == 1, s[o].to_numpy(float), 0.0)
    yh = np.column_stack([np.where(D == 1, s[f"{o}::{m}"].to_numpy(float), 0.0) for m in names] or
                         [np.empty((len(s), 0))])
    return np.column_stack([y, yh, yh - y[:, None]])


def marginal(d: pd.DataFrame, names: list, z: str | None = None, fe: str | None = None,
             controls: list | None = None, design: Design = PATENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marginal outcome (truth), each evaluator's implied outcome and signed error, by sample, outcome, and group;
    plus group contrasts. z, fe, and controls override the design's leniency, cell, and controls."""
    z = design.leniency if z is None else z
    fe = design.cell if fe is None else fe
    controls = list(design.controls) if controls is None else list(controls)
    rows, crows = [], []
    labels = [(HUMAN, "marginal")] + [(m, "implied") for m in names] + [(m, "error") for m in names]
    for sname, units in design.samples:
        ds = d[d[design.unit].isin(units) & d[z].notna()]
        for o, _ in design.outcomes:
            psi_full, est = {}, {}
            for gname, g in design.groups:
                s = ds if g is None else ds[ds[g[0]] == g[1]]
                W = s[controls].to_numpy(float) if controls else None
                Y = series_matrix(s, o, names, design.decision)
                D, Z = s[design.decision].to_numpy(float), s[z].to_numpy(float)
                b, psi = tsls(Y, D, Z, s[fe], W)
                se = cluster_se(psi, s[design.decider])
                fs, _ = tsls(D[:, None], Z, Z, s[fe], W)
                full = np.zeros((len(ds), Y.shape[1]))
                full[ds.index.get_indexer(s.index)] = psi
                psi_full[gname], est[gname] = full, b
                mg = s.loc[s[design.decision] == 1, o].mean()
                for j, (series, kind) in enumerate(labels):
                    rows.append(dict(sample=sname, outcome=o, group=gname, series=series, kind=kind,
                                     estimate=b[j], se=se[j], n=len(s), n_examiners=s[design.decider].nunique(),
                                     first_stage=fs[0], mean_granted=mg))
            for cname, (ga, gb) in design.contrasts:
                se = cluster_se(psi_full[ga] - psi_full[gb], ds[design.decider])
                diff = est[ga] - est[gb]
                for j, (series, kind) in enumerate(labels):
                    crows.append(dict(sample=sname, outcome=o, contrast=cname, series=series, kind=kind,
                                      estimate=diff[j], se=se[j]))
    return pd.DataFrame(rows), pd.DataFrame(crows)


def ols_coef(s: pd.DataFrame, y: str, x: str, fe: str | None = None, controls: list = (),
             design: Design = PATENTS) -> tuple[float, float]:
    """Coefficient of y on x within cells (default: the design's cell), clustered by decision maker."""
    fe = design.cell if fe is None else fe
    W = s[list(controls)].to_numpy(float) if controls else None
    b, psi = tsls(s[[y]].to_numpy(float), s[x].to_numpy(float), s[x].to_numpy(float), s[fe], W)
    return b[0], cluster_se(psi, s[design.decider])[0]


def balance(d: pd.DataFrame, names: list, design: Design = PATENTS) -> pd.DataFrame:
    """Predictions should not move with the decision maker's leniency within cells; coefficient per unit of
    leniency (multiply by 0.1 for a 10-point more lenient decision maker)."""
    rows = []
    for sname, units in design.samples:
        s = d[d[design.unit].isin(units) & d[design.leniency].notna()]
        for o, _ in design.outcomes:
            for m in names:
                b, se = ols_coef(s, f"{o}::{m}", design.leniency, design=design)
                rows.append(dict(sample=sname, outcome=o, series=m, coef=b, se=se))
    return pd.DataFrame(rows)


def decision_gaps(d: pd.DataFrame, cols: list, design: Design = PATENTS) -> pd.DataFrame:
    """Group gap in decisions made at the human acceptance rate: within each unit, accept the top r% of scores, where
    r is the actual acceptance rate; regress the decisions on the group indicator within gap cells."""
    d = d.copy()
    for u in design.units:
        m = (d[design.unit] == u).to_numpy()
        r = d.loc[m, design.decision].mean()
        for c in cols:
            d.loc[m, f"dec::{c}"] = (d.loc[m, c] >= d.loc[m, c].quantile(1 - r)).astype(float)
    rows = []
    for sname, units in design.samples:
        s = d[d[design.unit].isin(units)]
        Y = s[[design.decision] + [f"dec::{c}" for c in cols]].to_numpy(float)
        g = s[design.gap_group].to_numpy(float)
        b, psi = tsls(Y, g, g, s[design.gap_cell])
        se = cluster_se(psi, s[design.decider])
        for j, c in enumerate(["Human decision makers"] + cols):
            rows.append(dict(sample=sname, series=c, gap=b[j], se=se[j], ratio=b[j] / b[0]))
    return pd.DataFrame(rows)


def check(record: pd.DataFrame, scores: pd.DataFrame, design: Design = PATENTS) -> dict:
    """Run the check for every evaluator in `scores`.

    scores: one row per case, the case column, and either score columns (any monotone scale, higher = more favorable;
    recalibrated out of fold to each outcome) or columns '{outcome}::{name}' holding predicted outcome probabilities.
    """
    if scores[design.case].duplicated().any():
        raise ValueError(f"duplicate {design.case} in the predictions")
    d = record.merge(scores, on=design.case, how="left", validate="1:1")
    raw = [c for c in scores.columns if c != design.case and "::" not in c]
    given = sorted({c.split("::", 1)[1] for c in scores.columns if "::" in c})
    for c in raw + [f"{o}::{n}" for o, _ in design.outcomes for n in given]:
        if c not in d.columns:
            raise ValueError(f"missing column {c}")
        if d[c].isna().any():
            raise ValueError(f"{d[c].isna().sum():,d} cases have no value in column {c}")
    d = recalibrate(d, raw, assign_folds(d, design), design)
    names = [NO_MODEL] + raw + [n for n in given if n not in raw]
    est, con = marginal(d, names, design=design)
    return dict(estimates=est, contrasts=con, balance=balance(d, names[1:], design),
                gaps=decision_gaps(d, raw, design) if raw else pd.DataFrame(), names=names, design=design)


# ---------------------------------------------------------------- report

def fmt(b: float, se: float, scale: float = 100, nd: int = 1) -> str:
    return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"


def report(res: dict, title: str) -> str:
    """Markdown report of check()."""
    design, est, con, names = res["design"], res["estimates"], res["contrasts"], res["names"]
    groups = [g for g, _ in design.groups]
    contr = [c for c, _ in design.contrasts]
    cols = []
    for g in groups:
        cols.append(g)
        for c, (a, b) in design.contrasts:
            if b == g:
                cols.append(c)
    L = [f"# {title}", "",
         "Marginal cases: accepted by a lenient decision maker, rejected by a strict one. \"Truth\" is their mean "
         "outcome from 2SLS of accepted × outcome on accepted, instrumented by leniency within assignment cells, with "
         "controls, standard errors clustered by decision maker. \"Implied\" replaces each case's outcome with the "
         "evaluator's out-of-fold prediction; the signed error is implied − truth (positive: the evaluator overstates "
         "how the marginal cases did). Percentages / percentage points, 95% CIs in brackets.", ""]
    for o, olab in design.outcomes:
        for sname, _ in design.samples:
            e = est[(est["sample"] == sname) & (est.outcome == o)]
            c = con[(con["sample"] == sname) & (con.outcome == o)]

            def cell(series, kind, g):
                t = (c[(c.series == series) & (c.kind == kind) & (c.contrast == g)] if g in contr else
                     e[(e.series == series) & (e.kind == kind) & (e.group == g)]).iloc[0]
                return fmt(t.estimate, t.se)
            L += [f"## {olab}, {sname}", "", "| Row | Implied (all) | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
            L.append("| Truth: marginal outcome | — | " + " | ".join(cell(HUMAN, "marginal", g) for g in cols) + " |")
            for n in names:
                imp = e[(e.series == n) & (e.kind == "implied") & (e.group == groups[0])].iloc[0]
                L.append(f"| Error: {n} | {100 * imp.estimate:.1f} | " +
                         " | ".join(cell(n, "error", g) for g in cols) + " |")
            L.append("")
    if len(res["gaps"]):
        L += ["## Group gap in decisions at the human acceptance rate", "",
              f"Gap: {design.gap_group} − others in decisions, within {design.gap_cell} cells; ratio = evaluator gap / "
              "human gap.", "", "| Sample | Row | Gap | Ratio |", "|---|---|---|---|"]
        for _, r in res["gaps"].iterrows():
            L.append(f"| {r['sample']} | {r.series} | {fmt(r.gap, r.se)} | {r.ratio:.2f} |")
        L.append("")
    L += ["## Balance: predicted outcome on leniency", "",
          "Coefficient × 0.1 (a 10-point more lenient decision maker), in percentage points. It should be near zero: "
          "an evaluator whose predictions move with the decision maker it drew cannot be checked this way.", "",
          "| Sample | Outcome | Evaluator | Coefficient |", "|---|---|---|---|"]
    for _, r in res["balance"].iterrows():
        L.append(f"| {r['sample']} | {r.outcome} | {r.series} | {fmt(0.1 * r.coef, 0.1 * r.se, 100, 2)} |")
    return "\n".join(L) + "\n"
