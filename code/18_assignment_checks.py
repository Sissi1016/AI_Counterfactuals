"""Random-assignment checks in the style of Kleinberg et al. (2018): a permutation balance test across within-cell
leniency quintiles, and a residualized leniency as an alternative instrument.

1. Permutation balance test. Within each assignment cell with at least five examiners, examiners are split into
   leniency quintiles. An index of each application's characteristics fixed at filing is formed by projecting the
   outcome on them: grant on the cross-fitted text score of 12 and the coded fields (female first inventor, small
   entity, number of inventors); and renewal, for granted applications, on the renewal score and the same fields.
   In each cell the index is regressed on the quintile indicators, and the F statistic is compared with its
   distribution when the quintiles are permuted across the cell's applications (1,000 draws). Under as-good-as-random
   assignment the cell p-values are uniform. Cells: art unit × year × USPC class (the main design), the finer
   art unit × year × USPC subclass, and, for comparison, the coarser art unit × year.
2. Residualized leniency. The leave-one-out leniency of 04 is an examiner's grant rate over the whole career, which
   mixes the examiner's bar with the mix of applications the examiner received. Here each application in the
   2004–2018 base of 04 is first taken relative to the grant rate of its art unit × year × USPC class cell and entity
   size, and an examiner's leniency is the leave-one-out mean of these residuals. The permutation test (quintiles of
   this leniency), the first stage, the balance regressions, and the marginal check are repeated with it; the
   marginal check is also repeated with the grant index as an added control.
3. Finer cells. The first stage, balance, and marginal check are repeated with art unit × year × USPC subclass
   fixed effects in place of the class cells.

Runs on CPUs only (DuckDB limited to 16 threads).
Input: data/processed/{applications_leniency,patex_application_data,crossfit_preds}.parquet, the record of cfe.check
Output: output/assignment_checks.md, output/tables/assignment_*.csv, output/figures/fig4_balance_pvalues.{pdf,png}
"""
import pathlib
import warnings

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from cfe.check import CONTROLS, HUMAN, load_record, marginal, ols_coef  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
SEED, M, MIN_EXAMINERS, Q = 20261028, 1000, 5, 5
CELLS = {"au_class": "Art unit × year × USPC class", "au_subclass": "Art unit × year × USPC subclass",
         "au_year": "Art unit × year"}
ZLAB = {"lenient_grant": "leave-one-out", "lenient_resid": "residualized"}
SPECS = [("lenient_grant", "Leave-one-out grant rate (04)", "au_class"),
         ("lenient_resid", "Residualized leave-one-out (cell × entity size)", "au_class"),
         ("lenient_grant", "Leave-one-out, art unit × year × USPC subclass cells", "au_subclass")]
INK, INK2, RULE, BAND, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d6d5d0", "#ebeae6", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK, "pdf.fonttype": 42,
})
NAMES = {"Train on grant": "Train on grant", "Train on renewal (granted only)": "Train on renewal (granted only)",
         "Train on citations (granted only)": "Train on citations (granted only)",
         "No model: mean outcome of granted applications": "No model"}


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    d = load_record(ROOT)
    cf = pd.read_parquet(P / "crossfit_preds.parquet")
    keep = ["application_number", "score::Train on grant", "score::Train on renewal (granted only)"]
    keep += [f"{o}::{n}" for o in ["renew_4y", "cites_hi"] for n in NAMES]
    d = d.merge(cf[keep], on="application_number", how="left")
    d = d.merge(residualized_leniency(d), on="application_number", how="left")
    d["au_subclass"] = d["au_class"] + "/" + d["uspc_subclass"]
    # indices of characteristics fixed at filing, projected on the outcome within each technology center
    for tc in ["3600", "1600"]:
        m = (d.tech_center == tc).to_numpy()
        X = d.loc[m, ["score::Train on grant", "female", "small_entity", "n_inventors"]].to_numpy(float)
        d.loc[m, "index_grant"] = LogisticRegression(max_iter=2000).fit(X, d.loc[m, "granted"]).predict_proba(X)[:, 1]
        g = m & (d.granted == 1).to_numpy()
        cols = ["score::Train on renewal (granted only)", "female", "small_entity", "n_inventors"]
        Xr = d.loc[m, cols].to_numpy(float)
        fit = LogisticRegression(max_iter=2000).fit(Xr[g[m]], d.loc[g, "renew_4y"].astype(int))
        d.loc[m, "index_renew"] = fit.predict_proba(Xr)[:, 1]
    return d


def residualized_leniency(d: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean of (granted − mean grant of the application's cell × entity size) over the examiner's
    applications in the 2004–2018 base of 04."""
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    con.register("pilot", d[["application_number"]])
    return con.execute(f"""
        WITH b AS (
          SELECT l.application_number, l.examiner_id, l.granted,
                 l.art_unit || '_' || l.filing_year::VARCHAR || '_' || coalesce(u.uspc_class, 'none') AS cell,
                 l.small_entity_indicator IN ('SMALL', 'MICRO') AS small
          FROM '{P}/applications_leniency.parquet' l
          LEFT JOIN '{P}/patex_application_data.parquet' u USING (application_number)
        ), r AS (
          SELECT application_number, examiner_id, granted - avg(granted) OVER (PARTITION BY cell, small) AS resid
          FROM b
        ), e AS (SELECT examiner_id, sum(resid) AS s, count(*) AS n FROM r GROUP BY 1)
        SELECT r.application_number, (e.s - r.resid) / (e.n - 1) AS lenient_resid,
               coalesce(u.uspc_subclass, 'none') AS uspc_subclass
        FROM r JOIN e USING (examiner_id) JOIN pilot USING (application_number)
        LEFT JOIN '{P}/patex_application_data.parquet' u USING (application_number)""").df()


# ---------------------------------------------------------------- permutation balance test

def f_stat(y: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    """One-way ANOVA F statistic of y across groups g; g may be a matrix of label permutations (draws × n)."""
    g = np.atleast_2d(g)
    n = y.shape[0]
    rows = np.repeat(np.arange(g.shape[0]), n)
    idx = rows * k + g.ravel()
    sums = np.bincount(idx, weights=np.tile(y, g.shape[0]), minlength=g.shape[0] * k).reshape(-1, k)
    cnt = np.bincount(idx, minlength=g.shape[0] * k).reshape(-1, k)
    means = np.divide(sums, cnt, out=np.zeros_like(sums), where=cnt > 0)
    ssb = (cnt * (means - y.mean()) ** 2).sum(1)
    ssw = ((y - y.mean()) ** 2).sum() - ssb
    kk = (cnt > 0).sum(1)
    return (ssb / np.maximum(kk - 1, 1)) / np.maximum(ssw / np.maximum(n - kk, 1), 1e-12)


def permutation_test(d: pd.DataFrame, cell: str, index: str, rng: np.random.Generator,
                     z: str = "lenient_grant") -> pd.DataFrame:
    rows = []
    sub = d[d[index].notna() & d[z].notna()]
    for (tc, c), x in sub.groupby(["tech_center", cell], sort=False):
        ex = x.groupby("examiner_id")[z].mean().sort_values(kind="mergesort")
        if len(ex) < MIN_EXAMINERS:
            continue
        quint = pd.Series(np.arange(len(ex)) * Q // len(ex), index=ex.index)
        g = x.examiner_id.map(quint).to_numpy()
        y = x[index].to_numpy(float)
        f0 = f_stat(y, g, Q)[0]
        perms = np.argsort(rng.random((M, len(g))), axis=1)
        fp = f_stat(y, g[perms], Q)
        rows.append(dict(tech_center=tc, cell=c, n=len(x), examiners=len(ex), f=f0,
                         p=(1 + (fp >= f0 - 1e-12).sum()) / (1 + M)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def ci(b: float, se: float, scale: float = 100, nd: int = 1) -> str:
    return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"


def main() -> None:
    warnings.filterwarnings("ignore")
    d = load()
    rng = np.random.default_rng(SEED)
    perm, summ = [], []
    combos = [(c, i, "lenient_grant") for c in CELLS for i in ["index_grant", "index_renew"]]
    combos += [(c, "index_grant", "lenient_resid") for c in ["au_class", "au_subclass"]]
    for cell, index, z in combos:
        r = permutation_test(d, cell, index, rng, z)
        r.insert(0, "leniency", z)
        r.insert(0, "index", index)
        r.insert(0, "cells", cell)
        perm.append(r)
        for tc, x in [("Both", r)] + list(r.groupby("tech_center")):
            n_tc = d.tech_center.isin(["3600", "1600"] if tc == "Both" else [tc]).sum()
            summ.append(dict(cells=CELLS[cell], leniency=ZLAB[z], index=index, tech_center=tc, n_cells=len(x),
                             share_of_pilot=x.n.sum() / n_tc, mean_examiners=x.examiners.mean(),
                             p_below_05=(x.p < 0.05).mean(), p_below_10=(x.p < 0.10).mean(),
                             p_above_95=(x.p > 0.95).mean(), mean_p=x.p.mean(),
                             ks_p=stats.kstest(x.p, "uniform").pvalue))
    perm, summ = pd.concat(perm), pd.DataFrame(summ)
    perm.drop(columns="cell").to_csv(OUT / "tables/assignment_permutation_pvalues.csv", index=False)
    summ.to_csv(OUT / "tables/assignment_permutation_summary.csv", index=False)

    # residualized leniency: first stage, balance, marginal check, side by side with the main instrument
    names = list(NAMES)
    rows, frows = [], []
    for z, zlab, fe in SPECS:
        for sname, tcs in [("3600", ["3600"]), ("1600", ["1600"]), ("Both", ["3600", "1600"])]:
            s = d[d.tech_center.isin(tcs) & d[z].notna()]
            fs, fse = ols_coef(s, "granted", z, fe=fe, controls=CONTROLS)
            row = dict(instrument=zlab, sample=sname, first_stage=fs, first_stage_se=fse,
                       sd_within=s[z].sub(s.groupby(fe)[z].transform("mean")).std())
            for v in ["female", "small_entity", "n_inventors", "index_grant"]:
                b, se = ols_coef(s, v, z, fe=fe)
                row[f"bal_{v}"], row[f"bal_{v}_se"] = b * row["sd_within"], se * row["sd_within"]
            frows.append(row)
        for ctrl, suffix in [(CONTROLS, ""), (CONTROLS + ["index_grant"], " + grant index control")]:
            est, con = marginal(d, names, z=z, fe=fe, controls=ctrl)
            est["instrument"], con["instrument"] = zlab + suffix, zlab + suffix
            rows.append((est, con))
    fs = pd.DataFrame(frows)
    est = pd.concat([r[0] for r in rows])
    con = pd.concat([r[1] for r in rows])
    fs.to_csv(OUT / "tables/assignment_resid_first_stage.csv", index=False)
    est.to_csv(OUT / "tables/assignment_resid_estimates.csv", index=False)
    con.to_csv(OUT / "tables/assignment_resid_contrasts.csv", index=False)

    # figure: p-value histograms from coarse to fine cells (grant index, both centers)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, cell, tag in zip(axes, ["au_year", "au_class", "au_subclass"], "abc"):
        p = perm[(perm.cells == cell) & (perm["index"] == "index_grant") & (perm.leniency == "lenient_grant")].p
        ax.hist(p, bins=np.linspace(0, 1, 21), weights=np.full(len(p), 20 / len(p)), color=BLUE, edgecolor="white",
                lw=0.6)
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"({tag}) {CELLS[cell].replace('Art unit × year', 'AU × year')}", pad=4)
        ax.text(0.98, 0.97, f"{len(p):,} cells\n{100 * (p < 0.05).mean():.0f}% below 0.05", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=6.5)
        ax.set_xlabel("Permutation p-value")
        for sd in ["top", "right"]:
            ax.spines[sd].set_visible(False)
        ax.grid(axis="y", color=BAND, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Density (uniform = 1)")
    axes[0].text(0.99, 1.04, "uniform", transform=axes[0].get_yaxis_transform(), ha="right", va="bottom",
                 color=INK2, fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2)
    for ext, meta in [("pdf", {"CreationDate": None}), ("png", {"Software": None})]:
        fig.savefig(OUT / "figures" / f"fig4_balance_pvalues.{ext}", metadata=meta, dpi=300)
    plt.close(fig)

    # markdown
    L = ["# Random-assignment checks (after Kleinberg et al. 2018)", "",
         f"Permutation balance test: cells with at least {MIN_EXAMINERS} examiners; examiners split into leniency "
         "quintiles within each cell; index of characteristics fixed at filing regressed on the quintiles; "
         f"F statistic against {M:,} within-cell permutations. Under as-good-as-random assignment the p-values are "
         "uniform (about 5% below 0.05 and 5% above 0.95, mean 0.5). In small cells few distinct permutations exist, "
         "so p-values pile up at 1 and the KS test rejects from the top; the lower tail is the informative part.", "",
         "| Cells | Quintiles of | Index | TC | Cells tested | Share of pilot | Examiners per cell | p < 0.05 | "
         "p < 0.10 | p > 0.95 | Mean p | KS test p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r.cells} | {r.leniency} | {r['index'].replace('index_', '')} | {r.tech_center} | "
                 f"{r.n_cells:,} | {100 * r.share_of_pilot:.1f}% | {r.mean_examiners:.1f} | "
                 f"{100 * r.p_below_05:.1f}% | {100 * r.p_below_10:.1f}% | {100 * r.p_above_95:.1f}% | "
                 f"{r.mean_p:.2f} | {r.ks_p:.3f} |")
    L += ["", "## Residualized leniency and finer cells", "",
          "First stage and balance (coefficient for a one-standard-deviation more lenient examiner within cells; "
          "shares in pp).", "",
          "| Instrument | Sample | First stage | Female (pp) | Small entity (pp) | Inventors | Grant index (pp) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in fs.iterrows():
        L.append(f"| {r.instrument} | {r['sample']} | {ci(r.first_stage, r.first_stage_se, 1, 3)} | "
                 f"{ci(r.bal_female, r.bal_female_se, 100, 2)} | "
                 f"{ci(r.bal_small_entity, r.bal_small_entity_se, 100, 2)} | "
                 f"{ci(r.bal_n_inventors, r.bal_n_inventors_se, 1, 3)} | "
                 f"{ci(r.bal_index_grant, r.bal_index_grant_se, 100, 2)} |")
    L += ["", "Marginal renewal (truth) and signed errors (pp), TC 3600 and both centers:", "",
          "| Instrument | Sample | Row | All | Female | Female − male | Small − large |",
          "|---|---|---|---|---|---|---|"]
    for zlab in est.instrument.unique():
        for sname in ["3600", "Both"]:
            e = est[(est.instrument == zlab) & (est["sample"] == sname) & (est.outcome == "renew_4y")]
            c = con[(con.instrument == zlab) & (con["sample"] == sname) & (con.outcome == "renew_4y")]
            for series, kind in [(HUMAN, "marginal")] + [(n, "error") for n in names]:
                a = e[(e.series == series) & (e.kind == kind) & (e.group == "All")].iloc[0]
                f = e[(e.series == series) & (e.kind == kind) & (e.group == "Female")].iloc[0]
                fm = c[(c.series == series) & (c.kind == kind) & (c.contrast == "Female − male")].iloc[0]
                sl = c[(c.series == series) & (c.kind == kind) & (c.contrast == "Small − large")].iloc[0]
                lab = "Truth" if kind == "marginal" else f"Error: {NAMES[series]}"
                L.append(f"| {zlab} | {sname} | {lab} | {ci(a.estimate, a.se)} | {ci(f.estimate, f.se)} | "
                         f"{ci(fm.estimate, fm.se)} | {ci(sl.estimate, sl.se)} |")
    (OUT / "assignment_checks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
