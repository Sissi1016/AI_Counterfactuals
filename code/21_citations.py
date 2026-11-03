"""Citations as the outcome: the marginal check with citations within five years of grant, excluding self-citations.

Outcome: cites_ns, the number of citations from other U.S. patents granted within five years of the cited patent's
grant, excluding self-citations (06b); defined for granted applications only.

1. A citation-trained evaluator: ridge regression of log(1 + cites_ns) on the Qwen3-Embedding vectors of the granted
   applications (text only, text with names, text with entity status), cross-fitted with the folds of 12 (grouped by
   examiner within each technology center), with the penalty chosen by inner examiner-grouped cross-validation.
2. The check of cfe.check for every evaluator: the grant-, first-action- and citation-trained evaluators of 12 and
   above, Qwen3-8B's log-odds of "Yes" (14), and blends of the grant- and citation-trained scores. Each score is mapped
   to the citation scale out of fold by a Poisson regression fitted on the granted applications of the other folds.
   Truth: 2SLS of granted × citations on granted, instrumented by leave-one-out leniency within art unit × year × USPC
   class cells, with controls; errors clustered by examiner.
3. Strict and lenient examiners per 100 applications (below and above the cell mean leniency), and the citations of
   the patents they grant.
4. As in Kleinberg et al. (2018, Table II): ordinary least squares against gradient boosting for the citations of
   granted patents (same inputs: 64 principal components of the embedding and the coded fields; cross-fitted): overlap
   of the two models' top predictions and the actual citations of the patents only one of them ranks at the top.
5. Robustness to the 2012–2014 changes in patent-eligibility law (Mayo 2012 and Myriad 2013 for biotechnology, Alice
   2014 for business methods): without USPC class 705 (business methods) in TC 3600 and without classes 435, 436 and
   536 (molecular biology, diagnostics, nucleic acids) in TC 1600; and with first-action-year × cell fixed effects.

Runs on CPUs only. Input: the record of cfe.check, data/processed/{cites_noself,crossfit_preds,llm_scores_allpilot,
applications_2004_2018}.parquet, data/processed/emb/{s0,s1}.npy
Output: output/citations.md, output/tables/citations_*.csv, data/processed/citation_scores.parquet
"""
import dataclasses
import pathlib
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cfe.check import HUMAN, NO_MODEL, PATENTS, check, load_record, marginal, ols_coef
from cfe.estimation import tsls

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
SEED, K_INNER = 20261028, 4
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
N_JOBS, THREADS = 8, 2
OUTCOME = "cites_ns"
CIT = dataclasses.replace(PATENTS, outcomes=((OUTCOME, "Citations within five years of grant, excluding "
                                                       "self-citations"),))
CF = {"Train on grant": "score::Train on grant", "Train on first action": "score::Train on first action",
      "Train on grant + names": "score::Train on grant + names",
      "Train on grant + entity status": "score::Train on grant + entity status"}
COUNT = {"Train on citations (granted only)": "s0", "Train on citations + names": "s1",
         "Train on citations + entity status": "s0+entity"}
LLM = {"zero": "LLM, zero-shot", "few_lenient": "LLM, lenient examiners' examples",
       "few_strict": "LLM, strict examiners' examples"}
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
ALICE_CLASSES = {"3600": ["705"]}
MAYO_CLASSES = {"1600": ["435", "436", "536"]}


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    d = load_record(ROOT)
    c = pd.read_parquet(P / "cites_noself.parquet", columns=["application_number", "cites_5y_noself"])
    d = d.merge(c, on="application_number", how="left")
    d[OUTCOME] = np.where(d.granted == 1, d.cites_5y_noself.fillna(0), np.nan)
    a = pd.read_parquet(P / "applications_2004_2018.parquet", columns=["application_number", "first_action_date"])
    d = d.merge(a, on="application_number", how="left")
    d["fa_year"] = pd.to_datetime(d.first_action_date).dt.year.fillna(0).astype(int).astype(str)
    d["au_class_fa"] = d["au_class"] + "_" + d["fa_year"]
    cf = pd.read_parquet(P / "crossfit_preds.parquet", columns=["application_number", "fold"] + list(CF.values()))
    d = d.merge(cf, on="application_number", how="left")
    return d


# ---------------------------------------------------------------- the citation-trained evaluator

def ridge_task(X: np.ndarray, y: np.ndarray, ex: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> dict:
    with threadpool_limits(THREADS):
        mse = {a: 0.0 for a in ALPHAS}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            sc = StandardScaler().fit(X[tr[ia]])
            Xa, Xb = sc.transform(X[tr[ia]]), sc.transform(X[tr[ib]])
            for a in ALPHAS:
                mse[a] += ((Ridge(alpha=a).fit(Xa, y[tr[ia]]).predict(Xb) - y[tr[ib]]) ** 2).sum()
        best = min(mse, key=mse.get)
        sc = StandardScaler().fit(X[tr])
        return dict(alpha=best, s_te=Ridge(alpha=best).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])))


def count_scores(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb = {"s0": np.load(P / "emb/s0.npy").astype(np.float32), "s1": np.load(P / "emb/s1.npy").astype(np.float32)}
    ent = d[["small_entity"]].to_numpy(np.float32)
    X = {"s0": emb["s0"], "s1": emb["s1"], "s0+entity": np.hstack([emb["s0"], ent])}
    y = np.log1p(d[OUTCOME].to_numpy(float))
    ex = pd.factorize(d.examiner_id)[0]
    fold = d.fold.to_numpy()
    jobs = []
    for tc in CIT.units:
        rows = np.where(d.tech_center == tc)[0]
        for k in np.unique(fold[rows]):
            tr = rows[(fold[rows] != k) & (d.granted.to_numpy()[rows] == 1)]
            te = rows[fold[rows] == k]
            for name, inp in COUNT.items():
                jobs.append((name, inp, tc, k, tr, te))
    out = Parallel(n_jobs=N_JOBS)(delayed(ridge_task)(X[inp], y, ex, tr, te, SEED + i)
                                  for i, (name, inp, tc, k, tr, te) in enumerate(jobs))
    sc = {name: np.full(len(d), np.nan) for name in COUNT}
    chosen = []
    for (name, inp, tc, k, tr, te), r in zip(jobs, out):
        sc[name][te] = r["s_te"]
        chosen.append(dict(series=name, tech_center=tc, fold=k, alpha=r["alpha"]))
    return pd.DataFrame(sc), pd.DataFrame(chosen)


def llm_scores() -> pd.DataFrame:
    sc = pd.read_parquet(P / "llm_scores_allpilot.parquet")
    floor = 1e-12  # as in 10 and 14
    sc["score"] = np.log(sc.p_yes.clip(lower=floor)) - np.log(sc.p_no.clip(lower=floor))
    w = sc.pivot(index="application_number", columns="cond", values="score")[list(LLM)].rename(columns=LLM)
    return w.reset_index()


# ---------------------------------------------------------------- strict vs lenient

def split_table(d: pd.DataFrame, names: list) -> pd.DataFrame:
    """Per 100 applications: examiners below (strict) and above (lenient) the mean leniency of their cell."""
    rows = []
    for sname, units in CIT.samples:
        s = d[d.tech_center.isin(units) & d.lenient_grant.notna()]
        g = pd.factorize(s[CIT.cell])[0]
        z = s.lenient_grant.to_numpy(float)
        zt = z - pd.Series(z).groupby(g).transform("mean").to_numpy()
        hi = (zt > 0).astype(float)
        D = s.granted.to_numpy(float)
        V = np.column_stack([D, np.where(D == 1, s[OUTCOME].fillna(0), 0)]
                            + [np.where(D == 1, s[f"{OUTCOME}::{m}"], 0) for m in names])
        delta, _ = tsls(V, hi, hi, s[CIT.cell], s[list(CIT.controls)].to_numpy(float))
        strict = V[hi == 0].mean(0)
        for j, series in enumerate([HUMAN] + names):
            rows.append(dict(sample=sname, series=series, strict_granted=100 * strict[0],
                             lenient_granted=100 * (strict[0] + delta[0]), strict_cites=100 * strict[j + 1],
                             lenient_cites=100 * (strict[j + 1] + delta[j + 1]), marginal=delta[j + 1] / delta[0]))
    return pd.DataFrame(rows)


def granted_comparison(d: pd.DataFrame) -> pd.DataFrame:
    """Among granted patents: citations by examiner leniency within cells (per 10 points of leniency, and lenient −
    strict halves), with controls."""
    rows = []
    for sname, units in CIT.samples:
        s = d[d.tech_center.isin(units) & d.lenient_grant.notna() & (d.granted == 1)].copy()
        g = pd.factorize(s[CIT.cell])[0]
        z = s.lenient_grant.to_numpy(float)
        s["hi"] = ((z - pd.Series(z).groupby(g).transform("mean").to_numpy()) > 0).astype(float)
        b10, se10 = ols_coef(s, OUTCOME, "lenient_grant", controls=list(CIT.controls))
        bh, seh = ols_coef(s, OUTCOME, "hi", controls=list(CIT.controls))
        rows.append(dict(sample=sname, n_granted=len(s), mean_cites=s[OUTCOME].mean(), per_10pp=0.1 * b10,
                         per_10pp_se=0.1 * se10, lenient_minus_strict=bh, lenient_minus_strict_se=seh))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- OLS vs gradient boosting (Kleinberg et al. Table II)

def tail_task(Xtr, ytr, Xte, seed):
    with threadpool_limits(THREADS):
        pca = PCA(64, random_state=seed).fit(Xtr[:, :-3])
        f = lambda X: np.hstack([pca.transform(X[:, :-3]), X[:, -3:]])  # noqa: E731
        A, B = f(Xtr), f(Xte)
        ols = LinearRegression().fit(A, ytr).predict(B)
        gbm = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=500, early_stopping=True,
                                            validation_fraction=0.15, random_state=seed).fit(A, ytr).predict(B)
        return ols, gbm


def tail_comparison(d: pd.DataFrame) -> pd.DataFrame:
    emb = np.load(P / "emb/s0.npy").astype(np.float32)
    X = np.hstack([emb, d[list(CIT.controls)].to_numpy(np.float32)])
    y = d[OUTCOME].to_numpy(float)
    fold = d.fold.to_numpy()
    g = (d.granted == 1).to_numpy()
    jobs = []
    for tc in CIT.units:
        rows = np.where((d.tech_center == tc).to_numpy() & g)[0]
        for k in np.unique(fold[rows]):
            jobs.append((tc, rows[fold[rows] != k], rows[fold[rows] == k]))
    out = Parallel(n_jobs=N_JOBS)(delayed(tail_task)(X[tr], y[tr], X[te], SEED + i)
                                  for i, (tc, tr, te) in enumerate(jobs))
    pred = pd.DataFrame(index=d.index[g], columns=["ols", "gbm"], dtype=float)
    for (tc, tr, te), (o, b) in zip(jobs, out):
        pred.loc[d.index[te], "ols"], pred.loc[d.index[te], "gbm"] = o, b
    rows = []
    for tc in CIT.units:
        s = d.loc[g & (d.tech_center == tc).to_numpy()].join(pred)
        for q in [0.01, 0.05, 0.10, 0.25]:
            top_o = s.ols >= s.ols.quantile(1 - q)
            top_g = s.gbm >= s.gbm.quantile(1 - q)
            rows.append(dict(tech_center=tc, top=q, n_top=int(top_g.sum()),
                             overlap=(top_o & top_g).sum() / top_g.sum(),
                             cites_both=s.loc[top_o & top_g, OUTCOME].mean(),
                             cites_gbm_only=s.loc[top_g & ~top_o, OUTCOME].mean(),
                             cites_ols_only=s.loc[top_o & ~top_g, OUTCOME].mean(),
                             cites_all_gbm=s.loc[top_g, OUTCOME].mean(), cites_all_ols=s.loc[top_o, OUTCOME].mean(),
                             cites_mean=s[OUTCOME].mean(),
                             corr_ols=np.corrcoef(s.ols, s[OUTCOME])[0, 1],
                             corr_gbm=np.corrcoef(s.gbm, s[OUTCOME])[0, 1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report helpers

def ci(b, se, nd=2):
    return f"{b:.{nd}f} [{b - 1.96 * se:.{nd}f}, {b + 1.96 * se:.{nd}f}]"


def main() -> None:
    warnings.filterwarnings("ignore")
    d = load()
    cs, chosen = count_scores(d)
    chosen.to_csv(OUT / "tables/citations_ridge_penalty.csv", index=False)
    scores = pd.concat([d[["application_number"]].reset_index(drop=True), cs.reset_index(drop=True)], axis=1)
    for name, col in CF.items():
        scores[name] = d[col].to_numpy()
    scores = scores.merge(llm_scores(), on="application_number", how="left")
    # decision–outcome blends of the grant- and citation-trained scores (standardized within technology center)
    z = {}
    for name in ["Train on grant", "Train on citations (granted only)"]:
        v = scores[name].to_numpy(float)
        z[name] = np.empty_like(v)
        for tc in CIT.units:
            m = (d.tech_center == tc).to_numpy()
            z[name][m] = (v[m] - v[m].mean()) / v[m].std()
    for w in WEIGHTS:
        scores[f"blend {w:.1f}"] = w * z["Train on grant"] + (1 - w) * z["Train on citations (granted only)"]
    scores.to_parquet(P / "citation_scores.parquet", index=False)

    rec = d.drop(columns=list(CF.values()))
    res = check(rec, scores, CIT)
    for k in ["estimates", "contrasts", "balance", "gaps"]:
        res[k].to_csv(OUT / f"tables/citations_{k}.csv", index=False)

    # recalibrated predictions, for the split table and the robustness checks
    from cfe.check import assign_folds, recalibrate
    raw = [c for c in scores.columns if c != "application_number"]
    dd = recalibrate(rec.merge(scores, on="application_number", how="left"), raw, assign_folds(rec, CIT), CIT)
    names = [NO_MODEL, "Train on grant", "Train on first action", "Train on citations (granted only)",
             "LLM, zero-shot"]
    split = split_table(dd, names)
    split.to_csv(OUT / "tables/citations_split.csv", index=False)
    grc = granted_comparison(dd)
    grc.to_csv(OUT / "tables/citations_granted_by_leniency.csv", index=False)
    tail = tail_comparison(dd)
    tail.to_csv(OUT / "tables/citations_ols_vs_boosting.csv", index=False)

    # robustness to the 2012–2014 eligibility decisions
    rob = []
    excl = dd[~(((dd.tech_center == "3600") & dd.uspc_class.isin(ALICE_CLASSES["3600"]))
                | ((dd.tech_center == "1600") & dd.uspc_class.isin(MAYO_CLASSES["1600"])))]
    for lab, s, fe in [("Main design", dd, None), ("Without the classes most affected by Mayo, Myriad and Alice", excl,
                                                   None),
                       ("Cells × year of first action", dd, "au_class_fa")]:
        e, c = marginal(s, names, fe=fe, design=CIT)
        e["design"], c["design"] = lab, lab
        e["share_kept"] = len(s) / len(dd)
        rob.append((e, c))
    robe = pd.concat([r[0] for r in rob])
    robc = pd.concat([r[1] for r in rob])
    robe.to_csv(OUT / "tables/citations_legal_robustness.csv", index=False)
    robc.to_csv(OUT / "tables/citations_legal_robustness_contrasts.csv", index=False)
    aff = dd[dd.granted.notna()].assign(
        affected=lambda x: ((x.tech_center == "3600") & x.uspc_class.isin(ALICE_CLASSES["3600"]))
        | ((x.tech_center == "1600") & x.uspc_class.isin(MAYO_CLASSES["1600"])),
        fa_after=lambda x: np.where(x.tech_center == "3600", pd.to_datetime(x.first_action_date) >= "2014-06-19",
                                    pd.to_datetime(x.first_action_date) >= "2012-03-20"))
    legal = aff.groupby("tech_center").agg(share_affected_classes=("affected", "mean"),
                                           share_first_action_after_decision=("fa_after", "mean")).reset_index()
    legal.to_csv(OUT / "tables/citations_legal_exposure.csv", index=False)

    # markdown
    est, con, gaps = res["estimates"], res["contrasts"], res["gaps"]
    L = ["# Citations as the outcome", "",
         "Outcome: citations from U.S. patents granted within five years of the cited patent's grant, excluding "
         "self-citations (shared inventor or applicant organization; 06b). Truth: 2SLS of granted × citations on "
         "granted, instrumented by leave-one-out leniency within art unit × year × USPC class cells, with controls; "
         "errors clustered by examiner. Evaluators' scores are mapped to the citation scale out of fold by a Poisson "
         "regression on the granted applications of the other folds. Citations per patent; 95% CIs.", ""]
    for sname, _ in CIT.samples:
        e = est[(est["sample"] == sname)]
        c = con[(con["sample"] == sname)]
        L += [f"## {sname}", "", "| Row | Implied (all) | All | Female | Male | Female − male | Small | Large | "
              "Small − large |", "|---|---|---|---|---|---|---|---|---|"]
        for series in res["names"] + [HUMAN]:
            kind = "marginal" if series == HUMAN else "error"
            if series.startswith("blend"):
                continue

            def cell(g):
                t = (c[(c.series == series) & (c.kind == kind) & (c.contrast == g)] if "−" in g else
                     e[(e.series == series) & (e.kind == kind) & (e.group == g)]).iloc[0]
                return ci(t.estimate, t.se)
            imp = "—" if kind == "marginal" else \
                f"{e[(e.series == series) & (e.kind == 'implied') & (e.group == 'All')].iloc[0].estimate:.2f}"
            lab = "Truth: marginal citations" if kind == "marginal" else f"Error: {series}"
            L.append(f"| {lab} | {imp} | " + " | ".join(cell(g) for g in ["All", "Female", "Male", "Female − male",
                                                                         "Small entity", "Large entity",
                                                                         "Small − large"]) + " |")
        L.append("")
    L += ["## Gender gap in decisions at the human grant rate", "", "| Sample | Evaluator | Gap (pp) | Ratio |",
          "|---|---|---|---|"]
    for _, r in gaps.iterrows():
        L.append(f"| {r['sample']} | {r.series} | {ci(100 * r.gap, 100 * r.se, 1)} | {r.ratio:.2f} |")
    L += ["", "## Strict and lenient examiners, per 100 applications", "",
          "| Sample | Series | Granted: strict | Granted: lenient | Citations: strict | Citations: lenient | "
          "Marginal citations per patent |", "|---|---|---|---|---|---|---|"]
    for _, r in split.iterrows():
        L.append(f"| {r['sample']} | {r.series} | {r.strict_granted:.1f} | {r.lenient_granted:.1f} | "
                 f"{r.strict_cites:.1f} | {r.lenient_cites:.1f} | {r.marginal:.2f} |")
    L += ["", "## Citations of granted patents by examiner leniency (within cells, with controls)", "",
          "| Sample | Granted | Mean citations | Per 10 pp of leniency | Lenient − strict half |",
          "|---|---|---|---|---|"]
    for _, r in grc.iterrows():
        L.append(f"| {r['sample']} | {r.n_granted:,} | {r.mean_cites:.2f} | {ci(r.per_10pp, r.per_10pp_se, 3)} | "
                 f"{ci(r.lenient_minus_strict, r.lenient_minus_strict_se, 3)} |")
    L += ["", "## OLS vs gradient boosting for the citations of granted patents (after Kleinberg et al. 2018, "
          "Table II)", "", "| TC | Top | Overlap | Both | Boosting only | OLS only | All boosting | All OLS | Mean |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tail.iterrows():
        L.append(f"| {r.tech_center} | {100 * r.top:.0f}% | {100 * r.overlap:.1f}% | {r.cites_both:.2f} | "
                 f"{r.cites_gbm_only:.2f} | {r.cites_ols_only:.2f} | {r.cites_all_gbm:.2f} | {r.cites_all_ols:.2f} | "
                 f"{r.cites_mean:.2f} |")
    L += ["", "## Robustness to the 2012–2014 eligibility decisions (TC 3600)", "",
          "| Design | Share kept | Truth | Error: grant | Error: citations | Female − male (truth) |",
          "|---|---|---|---|---|---|"]
    for lab in robe.design.unique():
        e = robe[(robe.design == lab) & (robe["sample"] == "3600")]
        c = robc[(robc.design == lab) & (robc["sample"] == "3600")]

        def g(series, kind):
            t = e[(e.series == series) & (e.kind == kind) & (e.group == "All")].iloc[0]
            return ci(t.estimate, t.se)
        fm = c[(c.series == HUMAN) & (c.kind == "marginal") & (c.contrast == "Female − male")].iloc[0]
        L.append(f"| {lab} | {e.share_kept.iloc[0]:.2f} | {g(HUMAN, 'marginal')} | {g('Train on grant', 'error')} | "
                 f"{g('Train on citations (granted only)', 'error')} | {ci(fm.estimate, fm.se)} |")
    L += ["", "Exposure: " + "; ".join(f"TC {r.tech_center}: {100 * r.share_affected_classes:.0f}% in the affected "
                                       f"classes, {100 * r.share_first_action_after_decision:.0f}% first action after "
                                       "the decision" for r in legal.itertuples()), ""]
    (OUT / "citations.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
