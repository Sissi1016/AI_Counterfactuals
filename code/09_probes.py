"""Counterfactual evaluators (linear-probe version): on the same text embeddings, vary whom the model learns from,
what it learns, and what it sees, and compare the direction of the resulting gender gap.

Method: train logistic regressions on Qwen3-Embedding vectors in the training set and score a common test set. The L2
penalty of each model is chosen from C_GRID by cross-validated AUC within the training set (folds grouped by examiner);
a fixed weak penalty overfits the outcome labels and makes the scores noisier, which shrinks their gaps. Each
model's scores are turned into grant decisions at the human grant rate (top r% within a technology center, where r
is the actual test-set grant rate). We then estimate the decision gap for female first-named inventors (art unit ×
year fixed effects, clustered by examiner) and compare it with the human examiners' grant gap on the same test set:
ratio > 1 amplifies, 0–1 attenuates, < 0 reverses.

Examiner types are computed on the training set only, to avoid leakage:
- Leniency: an examiner's training-set grant rate net of the art unit × year mean, split at the application-weighted
  median into lenient and strict.
- Individual gender gap: after the same demeaning, the mean difference between female and male applications;
  computed only for examiners with at least MIN_PER_GENDER of each, and split at the median into high-gap and
  low-gap.

Output: output/tables/probes_{tc}.csv, output/probes.md; per-application test predictions in
        data/processed/probe_preds_{tc}.parquet (used by 11 for confidence intervals)
"""
import pathlib

import numpy as np
import pandas as pd
import pyfixest as pf
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
OUT = ROOT / "output"
SEED = 20261028
MIN_PER_GENDER = 5
C_GRID = [0.5, 0.05, 0.005, 0.0005, 0.00005]  # L2 penalties tried by inner cross-validation
K_INNER = 4
N_JOBS, THREADS = 20, 4
PROCESS = ["n_rejections", "n_rce", "appealed"]


def load() -> tuple[pd.DataFrame, dict]:
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number", "test"])
    s = pd.read_parquet(P / "pilot_sample.parquet")
    d = inp.merge(s, on="application_number", how="left")
    g = d[d.granted == 1]
    med = g.groupby(["tech_center", "filing_year"])["cites_5y"].transform("median")
    d["cites_hi"] = np.where(d.granted == 1, (d["cites_5y"] > med.reindex(d.index)).astype(float), np.nan)
    d["renew_4y"] = d["renew_4y"].astype(float)
    d["appealed"] = d["appealed"].astype(int)
    emb = {v: np.load(E / f"{v}.npy").astype(np.float32) for v in ["s0", "s1", "s1_f", "s1_m"]}
    return d, emb


def examiner_types(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.copy()
    tr["r"] = tr["granted"] - tr.groupby("au_year")["granted"].transform("mean")
    ex = tr.groupby("examiner_id").agg(n=("r", "size"), len_=("r", "mean"))
    w = np.repeat(ex["len_"].to_numpy(), ex["n"].to_numpy())
    ex["lenient"] = ex["len_"] > np.median(w)
    gg = tr.groupby(["examiner_id", "female"])["r"].agg(["mean", "size"]).unstack("female")
    ok = (gg[("size", 1)] >= MIN_PER_GENDER) & (gg[("size", 0)] >= MIN_PER_GENDER)
    gap = (gg[("mean", 1)] - gg[("mean", 0)])[ok]
    ex["gap"] = gap
    ex["high_gap"] = np.where(ex["gap"].notna(), ex["gap"] < ex["gap"].median(), np.nan)  # more negative = worse for women
    ex["rand_half"] = np.random.default_rng(SEED).random(len(ex)) < 0.5
    return ex


def _inner_score(Xtr, ytr, a, b, c):
    with threadpool_limits(THREADS):
        sc = StandardScaler().fit(Xtr[a])
        m = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(Xtr[a]), ytr[a])
        return c, b, m.decision_function(sc.transform(Xtr[b]))


def fit(Xtr, ytr, groups):
    """Standardize and fit a logistic regression whose L2 penalty is chosen from C_GRID by inner cross-validated AUC
    (inner folds grouped by examiner). Returns the scaler, the model, and the chosen C."""
    folds = GroupKFold(K_INNER, shuffle=True, random_state=SEED).split(Xtr, groups=groups)
    out = Parallel(n_jobs=N_JOBS)(delayed(_inner_score)(Xtr, ytr, a, b, c) for a, b in folds for c in C_GRID)
    oof = {c: np.empty(len(ytr)) for c in C_GRID}
    for c, b, s in out:
        oof[c][b] = s
    best = max(C_GRID, key=lambda c: roc_auc_score(ytr, oof[c]))
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(C=best, max_iter=3000).fit(sc.transform(Xtr), ytr), best


def fit_predict(Xtr, ytr, Xte, groups):
    sc, m, c = fit(Xtr, ytr, groups)
    return m.predict_proba(sc.transform(Xte))[:, 1], c


def gap(df: pd.DataFrame, y: str) -> tuple[float, float]:
    fit = pf.feols(f"{y} ~ female | au_year", data=df, vcov={"CRV1": "examiner_id"})
    return float(fit.coef()["female"]), float(fit.se()["female"])


def run_tc(d: pd.DataFrame, emb: dict, tc: str) -> pd.DataFrame:
    mask = (d.tech_center == tc).to_numpy()
    te_all = d["test"].to_numpy()
    tr_idx, te_idx = np.where(mask & ~te_all)[0], np.where(mask & te_all)[0]
    tr, te = d.iloc[tr_idx].copy(), d.iloc[te_idx].copy()
    test_pos = {r: i for i, r in enumerate(np.where(te_all)[0])}  # test row -> row in the name-swap embeddings
    te_audit = np.array([test_pos[r] for r in te_idx])
    ex = examiner_types(tr)
    tr = tr.join(ex[["lenient", "high_gap", "rand_half"]], on="examiner_id")
    rate = te["granted"].mean()
    preds = {}

    h_grant = gap(te, "granted")
    h_fa = gap(te, "fa_allow")
    h_cites = gap(te[te.granted == 1], "cites_hi")
    rows = [dict(spec="Human examiners", info="—", label="granted", train="—", n_train=np.nan, auc_own=np.nan,
                 auc_granted=np.nan, gap_decision=h_grant[0], se=h_grant[1], ratio=1.0,
                 audit_f_minus_m=np.nan, audit_se=np.nan),
            dict(spec="Human examiners (first action)", info="—", label="fa_allow", train="—", n_train=np.nan,
                 auc_own=np.nan, auc_granted=np.nan, gap_decision=h_fa[0], se=h_fa[1], ratio=np.nan,
                 audit_f_minus_m=np.nan, audit_se=np.nan),
            dict(spec="High citations among granted (outcome)", info="—", label="cites_hi", train="granted", n_train=np.nan,
                 auc_own=np.nan, auc_granted=np.nan, gap_decision=h_cites[0], se=h_cites[1], ratio=np.nan,
                 audit_f_minus_m=np.nan, audit_se=np.nan)]

    def X(info, idx):
        if info == "s0":
            return emb["s0"][idx]
        if info == "s1":
            return emb["s1"][idx]
        if info == "s0+process":
            return np.hstack([emb["s0"][idx], d.iloc[idx][PROCESS].to_numpy(np.float32)])
        raise ValueError(info)

    specs = [
        ("Train on first action", "s0", "fa_allow", "all"),
        ("Train on grant", "s0", "granted", "all"),
        ("Train on citations (granted only)", "s0", "cites_hi", "granted"),
        ("Train on renewal (granted only)", "s0", "renew_4y", "granted"),
        ("Train on grant + prosecution history", "s0+process", "granted", "all"),
        ("Train on first action + names", "s1", "fa_allow", "all"),
        ("Train on grant + names", "s1", "granted", "all"),
        ("Train on citations + names", "s1", "cites_hi", "granted"),
        ("Train on grant, lenient examiners", "s0", "granted", "lenient"),
        ("Train on grant, strict examiners", "s0", "granted", "strict"),
        ("Train on grant, random half of examiners (control)", "s0", "granted", "rand_half"),
        ("Train on grant, high-gap examiners", "s0", "granted", "high_gap"),
        ("Train on grant, low-gap examiners", "s0", "granted", "low_gap"),
    ]
    for name, info, label, train in specs:
        sel = tr[label].notna()
        if train == "granted":
            sel &= tr.granted == 1
        elif train == "lenient":
            sel &= tr.lenient == True  # noqa: E712
        elif train == "strict":
            sel &= tr.lenient == False  # noqa: E712
        elif train == "rand_half":
            sel &= tr.rand_half == True  # noqa: E712
        elif train == "high_gap":
            sel &= tr.high_gap == 1
        elif train == "low_gap":
            sel &= tr.high_gap == 0
        idx_tr = tr_idx[sel.to_numpy()]
        sc, m, c_best = fit(X(info, idx_tr), d.iloc[idx_tr][label].to_numpy().astype(int),
                            d.iloc[idx_tr]["examiner_id"].to_numpy())
        p = m.predict_proba(sc.transform(X(info, te_idx)))[:, 1]
        te["p"] = p
        te["dec"] = (p >= np.quantile(p, 1 - rate)).astype(int)
        preds[f"dec::{name}"] = te["dec"].to_numpy()
        preds[f"p::{name}"] = p
        g = gap(te, "dec")
        own = te[label].notna().to_numpy()
        auc_own = roc_auc_score(te.loc[own, label].astype(int), p[own])
        auc_gr = roc_auc_score(te["granted"], p)
        a_f = a_se = np.nan
        if info == "s1":
            pf_ = m.predict_proba(sc.transform(emb["s1_f"][te_audit]))[:, 1]
            pm_ = m.predict_proba(sc.transform(emb["s1_m"][te_audit]))[:, 1]
            diff = pf_ - pm_
            a_f, a_se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(dict(spec=name, info=info, label=label, train=train, n_train=len(idx_tr), C=c_best,
                         auc_own=auc_own,
                         auc_granted=auc_gr, gap_decision=g[0], se=g[1], ratio=g[0] / h_grant[0],
                         audit_f_minus_m=a_f, audit_se=a_se))
        print(f"[{tc}] {name}: C={c_best:g} gap={g[0]:+.4f} ({g[1]:.4f}) ratio={g[0] / h_grant[0]:+.2f} auc={auc_own:.3f}",
              flush=True)

    # proxy leakage: can text without names predict inventor gender?
    pf_, c_best = fit_predict(X("s0", tr_idx), tr["female"].to_numpy(), X("s0", te_idx), tr["examiner_id"].to_numpy())
    rows.append(dict(spec="Text predicts inventor gender (proxy leakage)", info="s0", label="female", train="all",
                     n_train=len(tr_idx), C=c_best, auc_own=roc_auc_score(te["female"], pf_), auc_granted=np.nan,
                     gap_decision=np.nan, se=np.nan, ratio=np.nan, audit_f_minus_m=np.nan, audit_se=np.nan))
    res = pd.DataFrame(rows)
    res.insert(0, "tech_center", tc)
    res["n_test"] = len(te)
    res["test_grant_rate"] = rate
    keep = te[["application_number", "tech_center", "au_year", "examiner_id", "female", "granted", "fa_allow",
               "cites_hi"]].reset_index(drop=True)
    pd.concat([keep, pd.DataFrame(preds)], axis=1).to_parquet(P / f"probe_preds_{tc}.parquet", index=False)
    return res


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    d, emb = load()
    allres = []
    for tc in ["3600", "1600"]:
        r = run_tc(d, emb, tc)
        r.to_csv(OUT / "tables" / f"probes_{tc}.csv", index=False)
        allres.append(r)
    res = pd.concat(allres)
    cols = ["tech_center", "spec", "n_train", "C", "auc_own", "auc_granted", "gap_decision", "se", "ratio",
            "audit_f_minus_m", "audit_se"]
    with open(OUT / "probes.md", "w") as f:
        f.write("# Counterfactual evaluators: linear-probe results\n\n"
                "gap_decision: grant gap for applications with a female first-named inventor when decisions are made at "
                "the human grant rate (art unit × year fixed effects).\n"
                "ratio: model gap / human examiner grant gap. >1 amplifies, 0–1 attenuates, <0 reverses.\n"
                "audit_f_minus_m: difference in predicted probability when only the first inventor's first name is "
                "swapped to a female vs. a male name.\n"
                "C: L2 penalty chosen by cross-validated AUC within the training set (folds grouped by examiner).\n\n")
        f.write(res[cols].round(4).to_markdown(index=False) + "\n")
    print((OUT / "probes.md").read_text())


if __name__ == "__main__":
    main()
