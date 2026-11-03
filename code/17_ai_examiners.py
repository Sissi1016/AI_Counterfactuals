"""AI examiners: one evaluator per human examiner, and what changes when an application meets a different one (C2).

Examiners: the 20 with the most pilot applications in TC 3600. Each "AI examiner" learns its examiner's own grant
decisions: all of the examiner's non-continuation applications in HUPD 2004–2018 (16), a median of about 870. It
starts from a common score that never saw any of the 20 examiners' decisions (a grant model trained on the other TC 3600
examiners' pilot applications) and adds the main dimensions of the application text (principal components of the
embedding), with a penalty chosen by cross-validated AUC on the examiner's own pilot decisions (compared at three
decimals; ties go to the weakest penalty, C_GRID being ordered from weak to strong). A leniency-only AI
examiner (the common score with the examiner's own intercept and slope) separates differences in the bar from
differences in what examiners look for.

All 20 AI examiners then decide the same applications: the 3,972 pilot applications of the 20 examiners (an AI
examiner's predictions for its own examiner's pilot applications are out of fold). Each grants as many as its
probabilities imply, choosing those it rates highest. Reported:
- held-out AUC of each AI examiner on its examiner's own pilot decisions;
- its grant rate on the common applications, against the human examiner's leave-one-out grant rate (04);
- its gender gap on the common applications (art unit × year fixed effects);
- pairwise agreement, and how many applications would be decided differently depending on the AI examiner drawn,
  overall and by inventor gender.
Examiners appear only as AI examiner 1–20, ordered by human leniency.

Run: uv run python code/17_ai_examiners.py [--features q8|s0]   (default: q8 if available, else s0)
     CFE_AI_PCS sets the number of principal components (default 32)
Input: data/processed/ai_examiner_apps.parquet and emb/ai_ex_{s0,q8}.npy (16), data/processed/emb/{s0,q8}.npy,
       and the record of cfe.check
Output: output/ai_examiners.md, output/tables/ai_examiners_*.csv, output/figures/fig3_ai_examiners.{pdf,png}
"""
import argparse
import pathlib
import warnings

import matplotlib
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from cfe.check import load_record  # noqa: E402
from cfe.estimation import cluster_se, tsls  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
SEED = 20261028
TC, N_EXAMINERS, K = "3600", 20, 5
N_PC = int(__import__("os").environ.get("CFE_AI_PCS", "32"))
C_GRID = [1.0, 0.1, 0.01, 0.001, 0.0001]
SHARED_C = [0.05, 0.005, 0.0005]
INK, INK2, RULE, BAND, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d6d5d0", "#ebeae6", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK, "pdf.fonttype": 42,
})


def fit_shared(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Common grant model on other examiners' applications; penalty by examiner-grouped cross-validated AUC."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    folds = list(GroupKFold(4, shuffle=True, random_state=SEED).split(Xs, y, groups))

    def run(c, a, b):
        return c, b, LogisticRegression(C=c, max_iter=3000).fit(Xs[a], y[a]).decision_function(Xs[b])
    out = Parallel(n_jobs=len(SHARED_C) * len(folds))(delayed(run)(c, a, b) for c in SHARED_C for a, b in folds)
    oof = {c: np.empty(len(y)) for c in SHARED_C}
    for c, b, v in out:
        oof[c][b] = v
    best = max(SHARED_C, key=lambda c: roc_auc_score(y, oof[c]))
    m = LogisticRegression(C=best, max_iter=3000).fit(Xs, y)
    return (lambda E: m.decision_function(sc.transform(E))), best, roc_auc_score(y, oof[best])


def fit_cv(Xx: np.ndarray, yx: np.ndarray, Xo: np.ndarray, yo: np.ndarray, seed: int):
    """Penalty by cross-validated AUC on the examiner's own pilot decisions (Xo, yo), always also training on the
    examiner's other decisions (Xx, yx). Returns out-of-fold probabilities for the pilot decisions, the final model and
    scaler, and the cross-validated AUC."""
    folds = list(StratifiedKFold(K, shuffle=True, random_state=seed).split(Xo, yo))
    best = None
    for c in C_GRID:
        oof = np.empty(len(yo))
        for a, b in folds:
            Xt, yt = np.vstack([Xx, Xo[a]]), np.concatenate([yx, yo[a]])
            sc = StandardScaler().fit(Xt)
            oof[b] = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(Xt), yt).predict_proba(
                sc.transform(Xo[b]))[:, 1]
        # AUC barely depends on the penalty when the model has one feature (leniency-only examiners), so compare it
        # at three decimals and keep the weakest penalty among ties: otherwise floating-point noise from the number
        # of threads picks the penalty
        auc = roc_auc_score(yo, oof)
        if best is None or round(auc, 3) > round(best[1], 3):
            best = (c, auc, oof)
    Xt, yt = np.vstack([Xx, Xo]), np.concatenate([yx, yo])
    sc = StandardScaler().fit(Xt)
    return best[2], LogisticRegression(C=best[0], max_iter=3000).fit(sc.transform(Xt), yt), sc, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["q8", "s0"])
    a = ap.parse_args()
    feat = a.features or ("q8" if (P / "emb/ai_ex_q8.npy").exists() and (P / "emb/q8.npy").exists() else "s0")
    warnings.filterwarnings("ignore")
    d = load_record(ROOT)
    emb = np.load(P / "emb" / f"{feat}.npy", mmap_mode="r")
    xa = pd.read_parquet(P / "ai_examiner_apps.parquet", columns=["application_number", "examiner_id", "granted"])
    xemb = np.load(P / "emb" / f"ai_ex_{feat}.npy").astype(np.float32)
    tc_rows = np.where(d.tech_center == TC)[0]
    counts = d.iloc[tc_rows].groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort")
    top = counts.head(N_EXAMINERS).index.tolist()
    others = tc_rows[~d.examiner_id.iloc[tc_rows].isin(top).to_numpy()]
    E_tc = np.asarray(emb[tc_rows], dtype=np.float32)
    shared, shared_c, shared_auc = fit_shared(np.asarray(emb[others], dtype=np.float32),
                                              d.granted.to_numpy()[others].astype(int),
                                              pd.factorize(d.examiner_id.iloc[others])[0])
    pool = d[d.examiner_id.isin(top)].copy()
    pool_rows = pool.index.to_numpy()
    pool["shared"] = shared(np.asarray(emb[pool_rows], dtype=np.float32))
    xa["shared"] = shared(xemb)
    sc_all = StandardScaler().fit(E_tc)
    Z = PCA(N_PC, random_state=SEED).fit(sc_all.transform(E_tc)) if N_PC > 0 else None

    def feats(shared_col: np.ndarray, E: np.ndarray) -> np.ndarray:
        return np.column_stack([shared_col] + ([Z.transform(sc_all.transform(E))] if Z is not None else []))
    X_full = feats(pool.shared.to_numpy(), np.asarray(emb[pool_rows], dtype=np.float32))
    X_len = pool[["shared"]].to_numpy()
    Xx_full = feats(xa.shared.to_numpy(), xemb)
    Xx_len = xa[["shared"]].to_numpy()

    probs_full, probs_len, rows = {}, {}, []
    for i, e in enumerate(top):
        own = (pool.examiner_id == e).to_numpy()
        ext = (xa.examiner_id == e).to_numpy()
        y, yx = pool.granted.to_numpy()[own].astype(int), xa.granted.to_numpy()[ext].astype(int)
        res = {}
        for kind, X, Xx in [("full", X_full, Xx_full), ("leniency", X_len, Xx_len)]:
            oof, m, sc, auc = fit_cv(Xx[ext], yx, X[own], y, SEED + i)
            p = m.predict_proba(sc.transform(X))[:, 1]
            p[own] = oof
            (probs_full if kind == "full" else probs_len)[e] = p
            res[kind] = auc
        g = pool[own]
        rows.append(dict(examiner=e, n=own.sum(), n_train=own.sum() + ext.sum(), human_grant_rate=y.mean(),
                         human_leniency_loo=g.lenient_grant.iloc[0] if g.lenient_grant.notna().any() else np.nan,
                         human_female_share=g.female.mean(),
                         human_gap_raw=g.loc[g.female == 1, "granted"].mean() - g.loc[g.female == 0, "granted"].mean(),
                         n_female=int(g.female.sum()), auc_full=res["full"], auc_leniency_only=res["leniency"],
                         main_art_unit=g.art_unit.mode().iloc[0]))
    ex = pd.DataFrame(rows)
    # gap and grant rate of each AI examiner on the common applications
    f = pool.female.to_numpy(float)
    for kind, probs in [("full", probs_full), ("leniency", probs_len)]:
        Pm = np.column_stack([probs[e] for e in top])
        Dm = np.column_stack([(Pm[:, j] >= np.quantile(Pm[:, j], 1 - Pm[:, j].mean())) for j in range(Pm.shape[1])])
        b, psi = tsls(Dm.astype(float), f, f, pool.au_year)
        se = cluster_se(psi, pool.examiner_id)
        ex[f"ai_grant_rate_{kind}"] = Pm.mean(0)
        ex[f"ai_gap_{kind}"], ex[f"ai_gap_{kind}_se"] = b, se
    hb, hpsi = tsls(pool[["granted"]].to_numpy(float), f, f, pool.au_year)
    human_gap, human_gap_se = hb[0], cluster_se(hpsi, pool.examiner_id)[0]
    ex = ex.sort_values("human_leniency_loo").reset_index(drop=True)
    ex.insert(0, "ai_examiner", np.arange(1, len(ex) + 1))
    order = ex.examiner.tolist()

    # decisions: each AI examiner grants as many common applications as its probabilities imply (its expected grant
    # rate), choosing those it rates highest; leniency differences set how many, differences in taste set which
    def decide(p: np.ndarray) -> np.ndarray:
        return (p >= np.quantile(p, 1 - p.mean())).astype(int)
    # main: leniency-only AI examiners (their text dimensions do not improve held-out AUC); DL: with text dimensions
    D = np.column_stack([decide(probs_len[e]) for e in order])
    DL = np.column_stack([decide(probs_full[e]) for e in order])
    agree = np.array([[np.mean(D[:, i] == D[:, j]) for j in range(len(order))] for i in range(len(order))])
    k = D.sum(1)
    pool["n_ai_grant"] = k
    lottery = []
    for lab, mask in [("All", np.ones(len(pool), bool)), ("Female", pool.female.to_numpy() == 1),
                      ("Male", pool.female.to_numpy() == 0)]:
        kk = k[mask]
        lottery.append(dict(group=lab, n=int(mask.sum()), granted_by_all=np.mean(kk == len(order)),
                            granted_by_none=np.mean(kk == 0),
                            depends_on_examiner=np.mean((kk > 0) & (kk < len(order))),
                            depends_with_text=np.mean((DL[mask].sum(1) > 0) & (DL[mask].sum(1) < len(order))),
                            mean_share_granting=kk.mean() / len(order)))
    lottery = pd.DataFrame(lottery)
    corr_p = np.corrcoef(ex.human_leniency_loo, ex.ai_grant_rate_leniency)[0, 1]
    corr_s = pd.Series(ex.human_leniency_loo).rank().corr(pd.Series(ex.ai_grant_rate_leniency).rank())
    corr_gap = ex.human_gap_raw.corr(ex.ai_gap_leniency)
    n_gap = int(ex.human_gap_raw.notna().sum())
    off = agree[~np.eye(len(order), dtype=bool)]

    # nothing that could identify an examiner, and no examiner's own gender gap, leaves this script (top-20 lists can be
    # rebuilt from public PatEx data, so pseudonyms alone do not protect individual-level estimates)
    pub = ex.drop(columns=["examiner", "main_art_unit", "human_gap_raw", "n_female", "human_female_share"])
    pub.to_csv(OUT / "tables/ai_examiners.csv", index=False)
    lottery.to_csv(OUT / "tables/ai_examiners_lottery.csv", index=False)
    pd.DataFrame([dict(features=feat, n_examiners=len(order), n_common=len(pool), n_train_median=ex.n_train.median(),
                       auc_leniency_median=ex.auc_leniency_only.median(), auc_text_median=ex.auc_full.median(),
                       corr_leniency=corr_p, spearman_leniency=corr_s, ai_grant_min=ex.ai_grant_rate_leniency.min(),
                       ai_grant_max=ex.ai_grant_rate_leniency.max(), agreement_median=np.median(off),
                       ai_gap_median=ex.ai_gap_leniency.median(), ai_gap_min=ex.ai_gap_leniency.min(),
                       ai_gap_max=ex.ai_gap_leniency.max(), human_gap=human_gap, human_gap_se=human_gap_se,
                       corr_individual_gap=corr_gap, n_individual_gap=n_gap)]).to_csv(
        OUT / "tables/ai_examiners_summary.csv", index=False)
    pd.DataFrame(agree, index=ex.ai_examiner, columns=ex.ai_examiner).to_csv(OUT / "tables/ai_examiners_agreement.csv")
    # how many AI examiners would grant each common application, by inventor gender (counts of applications)
    fem = pool.female.to_numpy() == 1
    pd.DataFrame({"ai_examiners_granting": np.arange(len(order) + 1),
                  "female": np.bincount(k[fem].astype(int), minlength=len(order) + 1),
                  "male": np.bincount(k[~fem].astype(int), minlength=len(order) + 1)}).to_csv(
        OUT / "tables/ai_examiners_count_distribution.csv", index=False)

    # figure: AI vs human leniency; how many of the 20 AI examiners would grant each application
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.scatter(100 * ex.human_leniency_loo, 100 * ex.ai_grant_rate_leniency, s=22, color=BLUE, edgecolor="white",
               lw=0.6,
               zorder=3)
    ax.set_xlabel("Human examiner's grant rate, leave-one-out (%)")
    ax.set_ylabel("AI examiner's grant rate on\nthe common applications (%)")
    ax.set_title(f"(a) AI and human leniency (r = {corr_p:.2f})", pad=4)
    ax = axes[1]
    bins = np.arange(len(order) + 2) - 0.5
    for lab, col, mask in [("Male first inventor", BLUE, pool.female.to_numpy() == 0),
                           ("Female first inventor", ORANGE, pool.female.to_numpy() == 1)]:
        ax.hist(k[mask], bins=bins, density=True, histtype="step", lw=1.6, color=col, label=lab)
    ax.set_xlabel(f"AI examiners (of {len(order)}) that would grant")
    ax.set_ylabel("Share of applications")
    ax.set_title(f"(b) One application, {len(order)} AI examiners", pad=4)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center")
    for ax in axes:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(color=BAND, lw=0.5)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.18)
    for ext, meta in [("pdf", {"CreationDate": None}), ("png", {"Software": None})]:
        fig.savefig(OUT / "figures" / f"fig3_ai_examiners.{ext}", metadata=meta, dpi=300)
    plt.close(fig)

    def pct(x):
        return f"{100 * x:.1f}"
    L = [f"# AI examiners: the {len(order)} examiners with the most pilot applications in TC {TC}", "",
         f"Features: {'Qwen3-8B representation (15)' if feat == 'q8' else 'Qwen3-Embedding-0.6B (08)'}; a common "
         f"grant score trained on the other TC {TC} examiners (C = {shared_c:g}, examiner-grouped AUC "
         f"{shared_auc:.3f}) plus {N_PC} principal components of the embedding. Each AI examiner trains on its "
         f"examiner's decisions in HUPD 2004–2018 (median {int(ex.n_train.median())}). Common applications: the "
         f"{len(pool):,} pilot applications of these examiners ({ex.n.min()}–{ex.n.max()} each, "
         f"{pool.art_unit.nunique()} art units). AI decisions: each AI examiner grants as many of the common "
         "applications as its probabilities imply, choosing those it rates highest.",
         "",
         "Main AI examiners: the common score with each examiner's own intercept and slope (\"leniency only\"); "
         f"adding {N_PC} text dimensions does not improve their held-out AUC, so it is reported only for comparison.",
         "",
         f"- Held-out AUC on the examiner's own pilot decisions: median {ex.auc_leniency_only.median():.2f} "
         f"(range {ex.auc_leniency_only.min():.2f}–{ex.auc_leniency_only.max():.2f}); with text dimensions: median "
         f"{ex.auc_full.median():.2f}.",
         f"- AI and human leniency: Pearson r = {corr_p:.2f}, Spearman = {corr_s:.2f} across the {len(order)} "
         f"examiners; on the same applications the AI examiners grant {pct(ex.ai_grant_rate_leniency.min())}–"
         f"{pct(ex.ai_grant_rate_leniency.max())}%.",
         f"- Pairwise agreement of AI decisions: median {pct(np.median(off))}% (range {pct(off.min())}–"
         f"{pct(off.max())}%).",
         f"- Applications whose decision depends on which AI examiner they meet: "
         f"{pct(lottery.loc[0, 'depends_on_examiner'])}% (with text dimensions: "
         f"{pct(lottery.loc[0, 'depends_with_text'])}%); female first inventor "
         f"{pct(lottery.loc[1, 'depends_on_examiner'])}%, male {pct(lottery.loc[2, 'depends_on_examiner'])}%.",
         f"- Gender gap of the AI examiners on the common applications: median {pct(ex.ai_gap_leniency.median())} pp "
         f"(range {pct(ex.ai_gap_leniency.min())} to {pct(ex.ai_gap_leniency.max())}); human examiners on the same "
         f"applications: {pct(human_gap)} pp [{pct(human_gap - 1.96 * human_gap_se)}, "
         f"{pct(human_gap + 1.96 * human_gap_se)}]. Correlation with each human examiner's own raw gap: "
         f"{corr_gap:.2f} across the {n_gap} examiners with female applications (human gaps rest on "
         f"{ex.n_female.min()}–{ex.n_female.max()} female applications each, so they are very noisy).", "",
         "| AI examiner | Pilot applications | Decisions learned | Human grant rate | Human leniency (LOO) | AUC | "
         "AUC, with text | AI grant rate | AI gender gap (pp) |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ex.iterrows():
        L.append(f"| {r.ai_examiner} | {r.n} | {r.n_train} | {pct(r.human_grant_rate)} | {pct(r.human_leniency_loo)} | "
                 f"{r.auc_leniency_only:.2f} | {r.auc_full:.2f} | {pct(r.ai_grant_rate_leniency)} | "
                 f"{pct(r.ai_gap_leniency)} [{pct(r.ai_gap_leniency - 1.96 * r.ai_gap_leniency_se)}, "
                 f"{pct(r.ai_gap_leniency + 1.96 * r.ai_gap_leniency_se)}] |")
    L += ["", "## How many AI examiners would grant", "", lottery.round(3).to_markdown(index=False), ""]
    (OUT / "ai_examiners.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    # single-threaded linear algebra: the twins' penalties are chosen among near-ties (see fit_cv), and thread counts
    # can flip a choice; 45 rebuilds the published twins exactly under the same limit
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
