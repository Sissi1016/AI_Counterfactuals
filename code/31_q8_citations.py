"""A larger model's representation with citations as the outcome: the check of 21 on Qwen3-8B's own representation.

Evaluators: the grant- and first-action-trained models on the Qwen3-8B representation of 15 (crossfit_q8), and a
citation-trained ridge regression of log(1 + citations) on the same representation (as in 21, same folds and
penalty search). Each score is mapped to citations by the Poisson recalibration of cfe.check; truth and errors as in 21.
Compared with the Qwen3-Embedding-0.6B evaluators of 21.

Runs on CPUs only. Input: data/processed/emb/q8.npy, crossfit_q8.parquet, citation_scores.parquet, the inputs of 21
Output: output/q8_citations.md, output/tables/q8_citations_*.csv
"""
import importlib.util
import pathlib
import warnings

import numpy as np
import pandas as pd

from cfe.check import HUMAN, check

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    warnings.filterwarnings("ignore")
    m21 = load_module("m21", "21_citations.py")
    m21.N_JOBS, m21.THREADS = 8, 2
    d = m21.load()
    X = np.load(P / "emb/q8.npy").astype(np.float32)
    y = np.log1p(d[m21.OUTCOME].to_numpy(float))
    ex = pd.factorize(d.examiner_id)[0]
    fold = d.fold.to_numpy()
    jobs = []
    for tc in m21.CIT.units:
        rows = np.where(d.tech_center == tc)[0]
        for k in np.unique(fold[rows]):
            jobs.append((rows[(fold[rows] != k) & (d.granted.to_numpy()[rows] == 1)], rows[fold[rows] == k]))
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=m21.N_JOBS)(delayed(m21.ridge_task)(X, y, ex, tr, te, m21.SEED + i)
                                      for i, (tr, te) in enumerate(jobs))
    cites = np.full(len(d), np.nan)
    for (tr, te), r in zip(jobs, out):
        cites[te] = r["s_te"]
    q8 = pd.read_parquet(P / "crossfit_q8.parquet",
                         columns=["application_number", "score::Train on grant", "score::Train on first action"])
    sc = pd.DataFrame({"application_number": d.application_number,
                       "Qwen3-8B: Train on citations (granted only)": cites})
    sc = sc.merge(q8.rename(columns={"score::Train on grant": "Qwen3-8B: Train on grant",
                                     "score::Train on first action": "Qwen3-8B: Train on first action"}),
                  on="application_number", how="left")
    small = pd.read_parquet(P / "citation_scores.parquet",
                            columns=["application_number", "Train on grant", "Train on first action",
                                     "Train on citations (granted only)"])
    sc = sc.merge(small.rename(columns=lambda c: c if c == "application_number" else f"0.6B: {c}"),
                  on="application_number", how="left")
    rec = d.drop(columns=list(m21.CF.values()))
    res = check(rec, sc, m21.CIT)
    for k in ["estimates", "contrasts", "gaps"]:
        res[k].to_csv(OUT / f"tables/q8_citations_{k}.csv", index=False)
    from sklearn.metrics import roc_auc_score
    rows = []
    for tc in m21.CIT.units:
        m = (d.tech_center == tc).to_numpy()
        g = m & (d.granted == 1).to_numpy()
        for size in ["0.6B", "Qwen3-8B"]:
            rows.append(dict(tech_center=tc, model=size,
                             auc_grant=roc_auc_score(d.granted[m], sc[f"{size}: Train on grant"][m]),
                             corr_cites=np.corrcoef(sc[f"{size}: Train on citations (granted only)"][g],
                                                    y[g])[0, 1]))
    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / "tables/q8_citations_fit.csv", index=False)

    def ci(b, se, nd=2, scale=1.0):
        return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"
    est, gaps = res["estimates"], res["gaps"]
    L = ["# Qwen3-8B representation vs Qwen3-Embedding-0.6B, citations as the outcome", "",
         "| TC | Model | AUC for grant (grant-trained) | Correlation with log citations (citation-trained) |",
         "|---|---|---|---|"]
    for r in fit.itertuples():
        L.append(f"| {r.tech_center} | {r.model} | {r.auc_grant:.3f} | {r.corr_cites:.3f} |")
    L += ["", "| TC | Evaluator | Error, all | Error, female | Gender gap in decisions (pp) | × human |",
          "|---|---|---|---|---|---|"]
    for tc in m21.CIT.units:
        e = est[(est["sample"] == tc) & (est.kind == "error")]
        for s in [c for c in sc.columns if c != "application_number"]:
            a = e[(e.series == s) & (e.group == "All")].iloc[0]
            f = e[(e.series == s) & (e.group == "Female")].iloc[0]
            g = gaps[(gaps["sample"] == tc) & (gaps.series == s)].iloc[0]
            L.append(f"| {tc} | {s} | {ci(a.estimate, a.se)} | {ci(f.estimate, f.se)} | "
                     f"{ci(g.gap, g.se, 1, 100)} | {g.ratio:.2f} |")
    t = est[(est["sample"] == "3600") & (est.series == HUMAN) & (est.kind == "marginal") & (est.group == "All")].iloc[0]
    L += ["", f"Truth (TC 3600): {ci(t.estimate, t.se)} citations."]
    (OUT / "q8_citations.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
