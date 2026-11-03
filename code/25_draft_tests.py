"""The tests as the proposal draft describes them, next to the check.

1. C1 as drafted ("rank matching"). Examiners are split into strict and lenient (below and above the mean leniency of
   their art unit × year × USPC class cell). Lenient examiners grant a share Δ of applications that strict ones
   reject. The draft takes, within each cell, the lowest-cited Δ / (lenient grant rate) of the lenient examiners'
   grants as the counterfactual of the strict examiners' rejections, trains a model on the strict examiners' grants
   plus these "counterfactual patents" and their citations, and predicts citations for every application. We report
   (a) the mean citations of the counterfactual patents, i.e., what the construction implies for the marginal
   applications, and (b) the trained model's implied value and signed error in the marginal check of 21, against the
   identified mean (2SLS).
2. The draft's second assumption (lenient and strict examiners do not differ in what they see beyond the text), tested
   as in Kleinberg et al. (2018, Figure IV): a citation model trained only on the lenient examiners' grants, applied to
   the strict examiners' grants. If strict examiners screened on information the text lacks, their grants would be
   cited more than the model predicts. Residual (actual − predicted log citations), strict − lenient, within cells.

Models: ridge regression of log(1 + citations) on the Qwen3-Embedding vectors, cross-fitted with the folds of 12
(grouped by examiner), penalty by inner examiner-grouped cross-validation; predictions in citations use Duan's
smearing factor.

Runs on CPUs only. Input: as 21, plus data/processed/citation_scores.parquet
Output: output/draft_tests.md, output/tables/draft_tests_*.csv
"""
import dataclasses
import pathlib
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cfe.check import HUMAN, NO_MODEL, PATENTS, assign_folds, load_record, marginal, ols_coef, recalibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
SEED, K_INNER = 20261028, 4
ALPHAS = [1e2, 1e3, 1e4, 1e5]
N_JOBS, THREADS = 8, 2
OUTCOME = "cites_ns"
CIT = dataclasses.replace(PATENTS, outcomes=((OUTCOME, "Citations within five years of grant, excluding "
                                                       "self-citations"),))


def load() -> pd.DataFrame:
    d = load_record(ROOT)
    c = pd.read_parquet(P / "cites_noself.parquet", columns=["application_number", "cites_5y_noself"])
    d = d.merge(c, on="application_number", how="left")
    d[OUTCOME] = np.where(d.granted == 1, d.cites_5y_noself.fillna(0), np.nan)
    d = d.merge(pd.read_parquet(P / "crossfit_preds.parquet", columns=["application_number", "fold"]),
                on="application_number", how="left")
    g = pd.factorize(d.tech_center + "|" + d.au_class)[0]
    z = d.lenient_grant.to_numpy(float)
    d["lenient_half"] = ((z - pd.Series(z).groupby(g).transform("mean").to_numpy()) > 0).astype(float)
    d["cell_id"] = g
    return d


def ridge_task(X, y, ex, tr, te, seed):
    with threadpool_limits(THREADS):
        mse = {a: 0.0 for a in ALPHAS}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            sc = StandardScaler().fit(X[tr[ia]])
            Xa, Xb = sc.transform(X[tr[ia]]), sc.transform(X[tr[ib]])
            for a in ALPHAS:
                mse[a] += ((Ridge(alpha=a).fit(Xa, y[tr[ia]]).predict(Xb) - y[tr[ib]]) ** 2).sum()
        best = min(mse, key=mse.get)
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=best).fit(sc.transform(X[tr]), y[tr])
        smear = np.mean(np.exp(y[tr] - m.predict(sc.transform(X[tr]))))
        return dict(alpha=best, pred_log=m.predict(sc.transform(X[te])), smear=smear)


def crossfit(d: pd.DataFrame, X: np.ndarray, train_mask: np.ndarray, y_log: np.ndarray) -> tuple:
    ex = pd.factorize(d.examiner_id)[0]
    fold = d.fold.to_numpy()
    jobs = []
    for tc in CIT.units:
        rows = np.where((d.tech_center == tc).to_numpy())[0]
        for k in np.unique(fold[rows]):
            jobs.append((rows[(fold[rows] != k) & train_mask[rows]], rows[fold[rows] == k]))
    out = Parallel(n_jobs=N_JOBS)(delayed(ridge_task)(X, y_log, ex, tr, te, SEED + i)
                                  for i, (tr, te) in enumerate(jobs))
    pred_log, pred = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for (tr, te), r in zip(jobs, out):
        pred_log[te] = r["pred_log"]
        pred[te] = r["smear"] * np.exp(r["pred_log"]) - 1
    return pred_log, np.clip(pred, 0, None)


def main() -> None:
    warnings.filterwarnings("ignore")
    d = load()
    X = np.load(P / "emb/s0.npy").astype(np.float32)
    rng = np.random.default_rng(SEED)
    d["tiebreak"] = rng.random(len(d))

    # 1. the draft's construction
    rows, cf = [], np.zeros(len(d), bool)
    for tc in CIT.units:
        s = d[(d.tech_center == tc) & d.lenient_grant.notna()]
        gs = s.loc[s.lenient_half == 0, "granted"].mean()
        gl = s.loc[s.lenient_half == 1, "granted"].mean()
        f = (gl - gs) / gl
        for _, x in s[(s.lenient_half == 1) & (s.granted == 1)].groupby("cell_id"):
            x = x.sort_values([OUTCOME, "tiebreak"])
            k = int(round(f * len(x)))
            cf[d.index.get_indexer(x.index[:k])] = True
        cfs = d[cf & (d.tech_center == tc)]
        rows.append(dict(tech_center=tc, strict_grant_rate=gs, lenient_grant_rate=gl, share_taken=f,
                         counterfactual_patents=len(cfs), counterfactual_mean_cites=cfs[OUTCOME].mean(),
                         counterfactual_zero_share=(cfs[OUTCOME] == 0).mean(),
                         counterfactual_female_share=cfs.female.mean(),
                         lenient_granted_mean_cites=s.loc[(s.lenient_half == 1) & (s.granted == 1), OUTCOME].mean()))
    construction = pd.DataFrame(rows)
    d["draft_counterfactual"] = cf
    train = ((d.lenient_half == 0) & (d.granted == 1)).to_numpy() | cf
    y_log = np.log1p(d[OUTCOME].fillna(0).to_numpy(float))
    _, d["Draft C1 model (rank matching)"] = crossfit(d, X, train, y_log)
    # the same model trained on all granted applications, for comparison
    _, d["Model trained on all granted"] = crossfit(d, X, (d.granted == 1).to_numpy(), y_log)
    names = ["Draft C1 model (rank matching)", "Model trained on all granted"]
    for n in names:  # predictions are already in citations: use them as given
        d[f"{OUTCOME}::{n}"] = d[n]
    dd = recalibrate(d, [], assign_folds(d, CIT), CIT)  # adds the no-model baseline
    e, c = marginal(dd, [NO_MODEL] + names, design=CIT)
    e.to_csv(OUT / "tables/draft_tests_c1_estimates.csv", index=False)
    c.to_csv(OUT / "tables/draft_tests_c1_contrasts.csv", index=False)
    construction.to_csv(OUT / "tables/draft_tests_c1_construction.csv", index=False)

    # 2. the draft's second assumption: calibration of a lenient-trained model on strict examiners' grants
    lp_log, _ = crossfit(d, X, ((d.lenient_half == 1) & (d.granted == 1)).to_numpy(), y_log)
    d["resid"] = np.where(d.granted == 1, y_log - lp_log, np.nan)
    d["strict_half"] = 1 - d.lenient_half
    cal = []
    for tc in CIT.units:
        s = d[(d.tech_center == tc) & (d.granted == 1) & d.lenient_grant.notna()]
        b, se = ols_coef(s, "resid", "strict_half", controls=list(CIT.controls), design=CIT)
        s = s.assign(decile=pd.qcut(lp_log[s.index], 10, labels=False))
        by = s.groupby(["decile", "strict_half"]).agg(pred=("resid", "size"), actual=(OUTCOME, "mean")).reset_index()
        pred_mean = s.groupby(["decile", "strict_half"]).apply(lambda x: np.expm1(lp_log[x.index]).mean()).values
        by["predicted_mean_cites"] = pred_mean
        by["tech_center"] = tc
        cal.append((dict(tech_center=tc, strict_minus_lenient_resid=b, se=se,
                         strict_grants=int((s.strict_half == 1).sum()), lenient_grants=int((s.strict_half == 0).sum())),
                    by))
    calt = pd.DataFrame([x[0] for x in cal])
    calc = pd.concat([x[1] for x in cal]).rename(columns={"pred": "n"})
    calt.to_csv(OUT / "tables/draft_tests_calibration.csv", index=False)
    calc.to_csv(OUT / "tables/draft_tests_calibration_deciles.csv", index=False)

    def ci(b, se, nd=2):
        return f"{b:.{nd}f} [{b - 1.96 * se:.{nd}f}, {b + 1.96 * se:.{nd}f}]"
    L = ["# The tests as drafted", "", "## C1 by rank matching", "",
         "| TC | Strict grant rate | Lenient grant rate | Share of lenient grants taken | Counterfactual patents | "
         "Their mean citations | Share with zero citations | Mean citations, all lenient grants |",
         "|---|---|---|---|---|---|---|---|"]
    for r in construction.itertuples():
        L.append(f"| {r.tech_center} | {100 * r.strict_grant_rate:.1f}% | {100 * r.lenient_grant_rate:.1f}% | "
                 f"{100 * r.share_taken:.1f}% | {r.counterfactual_patents:,} | {r.counterfactual_mean_cites:.2f} | "
                 f"{100 * r.counterfactual_zero_share:.0f}% | {r.lenient_granted_mean_cites:.2f} |")
    L += ["", "Marginal check (citations per patent; truth from 2SLS):", "",
          "| TC | Row | Implied | All | Female | Male |", "|---|---|---|---|---|---|"]
    for tc in ["3600", "1600"]:
        x = e[e["sample"] == tc]
        t = x[(x.series == HUMAN) & (x.kind == "marginal")].set_index("group")
        L.append(f"| {tc} | Truth | — | {ci(t.loc['All', 'estimate'], t.loc['All', 'se'])} | "
                 f"{ci(t.loc['Female', 'estimate'], t.loc['Female', 'se'])} | "
                 f"{ci(t.loc['Male', 'estimate'], t.loc['Male', 'se'])} |")
        for n in [NO_MODEL] + names:
            er = x[(x.series == n) & (x.kind == "error")].set_index("group")
            im = x[(x.series == n) & (x.kind == "implied") & (x.group == "All")].iloc[0].estimate
            L.append(f"| {tc} | Error: {n} | {im:.2f} | {ci(er.loc['All', 'estimate'], er.loc['All', 'se'])} | "
                     f"{ci(er.loc['Female', 'estimate'], er.loc['Female', 'se'])} | "
                     f"{ci(er.loc['Male', 'estimate'], er.loc['Male', 'se'])} |")
    L += ["", "## Second assumption: a model trained on lenient examiners' grants, applied to strict ones", "",
          "Residual = log(1 + actual) − predicted, within art unit × year × class cells with controls; positive: "
          "strict examiners' grants are cited more than the text predicts.", "",
          "| TC | Strict grants | Lenient grants | Strict − lenient residual (log points) |", "|---|---|---|---|"]
    for r in calt.itertuples():
        L.append(f"| {r.tech_center} | {r.strict_grants:,} | {r.lenient_grants:,} | "
                 f"{ci(r.strict_minus_lenient_resid, r.se, 3)} |")
    (OUT / "draft_tests.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
