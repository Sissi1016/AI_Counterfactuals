"""C1: how accurate are the counterfactual scores on rejected applications? Errors at the examiners' margin.

A rejected application's citations had it been granted, Y(1), are never observed, so its counterfactual score s cannot
be checked one application at a time. But within an art unit × year × class cell applications are assigned to
examiners as good as at random, and an application that a stricter quintile of examiners rejects and the most lenient
quintile grants (a "marginal" application, a complier) is granted, with its Y(1) observed, in the most lenient
quintile. Under random assignment, monotonicity and exclusion (the assumptions behind 35's levels at the margin, the
2.98 citations), the mean over marginal applications of ANY function h(s, Y(1)) is identified (Imbens and Rubin
1997; Abadie 2002, 2003):

    E[h(s, Y(1)) | marginal] = ( E[h·D | quintile 5] − E[h·D | quintile k] ) / ( g5 − gk ),

where D is the grant and h·D = 0 for rejected applications, because s depends only on the application's text, fixed
at filing. With h = Y this is 35's 2.98; with h = |s − Y| it is the score's mean absolute error on the marginal
applications; with h = (s − Y)², s², Y², s·Y it gives the mean squared error, the variance of Y(1), the correlation
between s and Y(1), and the calibration slope among marginal applications; with h = 1{|s − Y| ≤ t} it gives the
distribution of the errors. A benchmark that knows nothing about the application ("guessing"), the mean citations of
granted applications of the same center and filing year in the training set, is checked the same way.

As in 35's margin_levels: the test and imputation sets, cells with at least five examiners, quintiles of equal
caseload by leave-one-out leniency; the most lenient quintile against each stricter one, summed over the four
comparisons and over cells (cells weighted by size); 95% intervals from 500 bootstrap draws of art units, the same
draws for every quantity. Citations are heavy-tailed, so everything is also reported on the log scale, log(1 + .),
and with citations and scores top-coded at the 99th percentile of granted applications of the same center and year.

Check: s is fixed at filing, so its mean over marginal applications computed from the granted side (above) must equal
the one computed from the rejected side, E[s·(1 − D) | k] − E[s·(1 − D) | 5] over g5 − gk (35's computation).

Limits: only marginal applications are covered (those that some examiners grant and others reject); applications every
examiner would reject are not. The score that also uses the application's own prior-art citations cannot be checked
this way: its input changes with the grant, so the score a rejected application has and its Y(1) are never observed
together.

Per application: every rejected application with its score, its percentile among granted applications of the same
center and year, and 50% and 80% ranges derived from the error distribution at the margin (log scale, by tercile of
the score), which hold on average for marginal applications, not for each.

Input: 35's cached predictions (data/processed/c1_preds_all_all.parquet) and the sample (35's load_all)
Guesses: the mean citations of granted training applications of the same center and filing year (raw scale), their
mean log citations (log scale), the mean of their top-coded citations (top-coded scale, so the guess is top-coded like
the outcome), and their median (the best constant under absolute error). Ranges (F50, F80) are read off a fine grid
of factors (steps of 0.05; 0.1 by tercile), for the score and for the guess, with the same bootstrap draws.

Output: output/c1_margin_validation.md, output/tables/c1mv_{metrics,cdf,bands,levels,ratio_distribution}.csv,
        output/tables/c1_counterfactual_scores_validated.csv, output/latex/c1mv_numbers.tex (macros \\xMv...),
        output/figures/journal/fig_c1_score_check.{pdf,png} (appendix Figure A8)
"""
import importlib.util
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "output"
T, TEX = OUT / "tables", OUT / "latex"
SEED, B = 20261101, int(os.environ.get("CFE_B", "500"))
POIS = os.environ.get("CFE_LOSS", "poisson") == "poisson"   # 35's Poisson variant (its cached predictions)
VSFX, MAC = ("_poisson", "xMvPois") if POIS else ("", "xMv")
ABS_T = [0.5, 1, 2, 3, 5, 10, 20]                 # thresholds for |s − Y| (citations)
LOG_F = [1.25, 1.5, 2, 3, 5, 10, 20]                     # thresholds for the ratio (1 + s) / (1 + Y), either way


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build():
    m35 = load_module("m35", "35_c1_kllm.py")
    d = m35.load_all()
    x = m35.cached(d, "all")
    g = x.granted == 1
    cap = x[g].groupby(["tech_center", "filing_year"]).cites.quantile(0.99).rename("cap")
    x = x.join(cap, on=["tech_center", "filing_year"])
    # guessing: a benchmark that knows only the center and filing year, from granted training-set applications of the
    # same center and year: their mean citations (raw scale), mean log citations (log scale), mean of citations
    # TOP-CODED like the outcome (top-coded scale), and median citations (the best constant under absolute error)
    tr = x[g & (x.part == "train")]
    key = [tr.tech_center, tr.filing_year]
    guess = tr.cites.groupby(key).mean().rename("guess")
    lguess = np.log1p(tr.cites).groupby(key).mean().rename("lguess")
    tguess = np.minimum(tr.cites, tr.cap).groupby(key).mean().rename("tguess")
    mguess = tr.cites.groupby(key).median().rename("mguess")
    for gs in [guess, lguess, tguess, mguess]:
        x = x.join(gs, on=["tech_center", "filing_year"])
    # the score recalibrated on the log scale: OLS of log(1 + citations) on log(1 + score), granted training set
    ls_tr, ly_tr = np.log1p(np.maximum(tr.cf.to_numpy(float), 0)), np.log1p(tr.cites.to_numpy(float))
    b1 = np.cov(ls_tr, ly_tr)[0, 1] / np.var(ls_tr, ddof=1)
    b0 = ly_tr.mean() - b1 * ls_tr.mean()
    x["lcal"] = b0 + b1 * np.log1p(np.maximum(x.cf.to_numpy(float), 0))
    print(f"log recalibration on the training set: intercept {b0:.3f}, slope {b1:.3f}")
    x["cf_pct"] = m35.percentile_among_granted(x, "cf")
    ev = x[x.part.isin(["test", "impute"])].copy()
    ev["q"] = m35.quintiles(ev)
    ev = ev[ev.q.notna()].copy()
    return m35, x, ev


def functions(e: pd.DataFrame, tercile_edges) -> dict:
    """h(s, Y) for every application (used only where granted)."""
    s, y, c = e.cf.to_numpy(float), np.nan_to_num(e.cites.to_numpy(float)), e.guess.to_numpy(float)
    cap = e.cap.to_numpy(float)
    ls, ly = np.log1p(np.maximum(s, 0)), np.log1p(y)
    lc, lk = e.lguess.to_numpy(float), e.lcal.to_numpy(float)
    st, yt = np.minimum(np.maximum(s, 0), cap), np.minimum(y, cap)
    H = {"y": y, "y2": y ** 2, "s": s, "s2": s ** 2, "sy": s * y, "g": c, "gt": e.tguess.to_numpy(float),
         "ae": np.abs(s - y), "se": (s - y) ** 2, "ae_g": np.abs(c - y), "se_g": (c - y) ** 2,
         "ly": ly, "ly2": ly ** 2, "ls": ls, "ls2": ls ** 2, "lsly": ls * ly,
         "lae": np.abs(ls - ly), "lse": (ls - ly) ** 2, "lae_g": np.abs(lc - ly), "lse_g": (lc - ly) ** 2,
         "lk": lk, "lk2": lk ** 2, "lkly": lk * ly, "lkae": np.abs(lk - ly), "lkse": (lk - ly) ** 2,
         "yt": yt, "yt2": yt ** 2, "st": st, "st2": st ** 2, "styt": st * yt,
         "tae": np.abs(st - yt), "tse": (st - yt) ** 2, "tae_g": np.abs(e.tguess.to_numpy(float) - yt),
         "tse_g": (e.tguess.to_numpy(float) - yt) ** 2,
         "ae_gm": np.abs(e.mguess.to_numpy(float) - y),
         "tae_gm": np.abs(np.minimum(e.mguess.to_numpy(float), cap) - yt)}
    for t in ABS_T:
        H[f"cdf_ae_{t}"] = (np.abs(s - y) <= t).astype(float)
    for f in LOG_F:
        H[f"cdf_lr_{f}"] = (np.abs(ls - ly) <= np.log(f)).astype(float)
    # by tercile of the score (edges from all rejected applications): membership and log-ratio distribution within
    b = np.digitize(s, tercile_edges)
    for k in range(3):
        H[f"bin{k}"] = (b == k).astype(float)
        for f in LOG_F:
            H[f"bin{k}_lr_{f}"] = ((b == k) & (np.abs(ls - ly) <= np.log(f))).astype(float)
    return H


def margin_sums(ev: pd.DataFrame, H: dict):
    """Per cell: sum over the four comparisons of the change in grant rate (D) and of E[h·D] (V[h]); and, for the
    check, of the rejected-side mean of s."""
    cells = list(ev.groupby("au_class", sort=True))
    pos = pd.Series(np.arange(len(ev)), index=ev.index)
    n = np.array([len(c) for _, c in cells], float)
    au = np.array([c.art_unit.iloc[0] for _, c in cells])
    D = np.zeros(len(cells))
    V = {k: np.zeros(len(cells)) for k in H}
    Vrej = np.zeros(len(cells))
    for i, (_, c) in enumerate(cells):
        p = pos[c.index].to_numpy()
        gr = c.granted.to_numpy(float)
        q = c.q.to_numpy(int)
        a = q == 5
        for k in [1, 2, 3, 4]:
            bk = q == k
            if not bk.any():
                continue
            D[i] += gr[a].mean() - gr[bk].mean()
            for h, v in H.items():
                hv = v[p] * gr
                V[h][i] += hv[a].mean() - hv[bk].mean()
            s = H["s"][p] * (1 - gr)
            Vrej[i] += s[bk].mean() - s[a].mean()
    return n, au, D, V, Vrej


def boot(au: np.ndarray, rng) -> np.ndarray:
    u, inv = np.unique(au, return_inverse=True)
    mult = np.ones((B + 1, len(au)))
    for b in range(1, B + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
    return mult


FINE = np.round(np.arange(1.05, 8.0001, 0.05), 2)        # factors for the 50% and 80% ranges (fine grid)
GRID = np.round(np.arange(-3.0, 3.0001, 0.25), 2)         # log(1 + Y) - log(1 + prediction), for the distribution


def margins_fine(ev: pd.DataFrame, edges, mult: np.ndarray) -> dict:
    """Vectorized margin means for the figure and the ranges: per stricter quintile (top-coded levels of the
    citations, the score and the top-coded guess; raw levels pooled), the share of marginal applications within each
    factor of the fine grid (score: overall and by tercile; guess: overall), and the distribution of the log ratio
    for the score and the guess. Same identification and the same bootstrap draws (mult) as margin_sums."""
    y = np.nan_to_num(ev.cites.to_numpy(float))
    s = np.maximum(ev.cf.to_numpy(float), 0)
    c, cap = ev.guess.to_numpy(float), ev.cap.to_numpy(float)
    lr_s, lr_c = np.log1p(y) - np.log1p(s), np.log1p(y) - np.log1p(c)
    b = np.digitize(s, edges)
    cols = {"y": y, "s": s, "c": c, "yt": np.minimum(y, cap), "st": np.minimum(s, cap),
            "gt": ev.tguess.to_numpy(float)}
    for f in FINE:
        cols[f"abs_s_{f}"] = (np.abs(lr_s) <= np.log(f)).astype(float)
        cols[f"abs_c_{f}"] = (np.abs(lr_c) <= np.log(f)).astype(float)
        if round(f * 20) % 2 == 0:                          # terciles on a grid of 0.1
            for k in range(3):
                cols[f"bin{k}_abs_s_{f}"] = ((b == k) & (np.abs(lr_s) <= np.log(f))).astype(float)
    for k in range(3):
        cols[f"bin{k}"] = (b == k).astype(float)
    for r in GRID:
        cols[f"cdf_s_{r}"] = (lr_s <= r).astype(float)
        cols[f"cdf_c_{r}"] = (lr_c <= r).astype(float)
    keys = list(cols)
    Hm = np.column_stack([cols[k] for k in keys]).astype(np.float32)
    cells = list(ev.groupby("au_class", sort=True))
    pos = pd.Series(np.arange(len(ev)), index=ev.index)
    n = np.array([len(cc) for _, cc in cells], float)
    D = np.zeros((4, len(cells)))
    V = np.zeros((4, len(cells), len(keys)), dtype=np.float32)
    for i, (_, cc) in enumerate(cells):
        p = pos[cc.index].to_numpy()
        gr = cc.granted.to_numpy(float)
        q = cc.q.to_numpy(int)
        a = q == 5                                          # most lenient quintile (35's coding)
        hv = Hm[p] * gr[:, None]
        for j, k in enumerate([4, 3, 2, 1]):               # stricter quintiles 2, 3, 4, 5 (1 = most lenient)
            bk = q == k
            if not bk.any() or not a.any():
                continue
            D[j, i] = gr[a].mean() - gr[bk].mean()
            V[j, i] = hv[a].mean(0) - hv[bk].mean(0)
    W = mult * n

    def mean(js):
        return (sum(W @ V[j] for j in js)) / (sum(W @ D[j] for j in js))[:, None]
    out = {"keys": {k: i for i, k in enumerate(keys)}, "pooled": mean([0, 1, 2, 3])}
    for j, qn in enumerate(["2", "3", "4", "5"]):
        out[qn] = mean([j])
    return out


def factor_fine(M: np.ndarray, K: dict, prefix: str, share: float, grid=FINE, denom=None) -> np.ndarray:
    """Per draw: the smallest factor f of the grid with a share >= share of marginal applications within f."""
    prof = np.column_stack([M[:, K[f"{prefix}{f}"]] for f in grid])
    if denom is not None:
        prof = prof / np.maximum(denom[:, None], 1e-9)
    prof = np.maximum.accumulate(prof, axis=1)
    hit = prof >= share
    return np.where(hit.any(1), np.asarray(grid)[hit.argmax(1)], np.nan)


def score_check_figure(F: dict, met: pd.DataFrame, summaries: pd.DataFrame | None = None) -> None:
    """Figure A8: the counterfactual scores checked against the record. (a) right on average, and so is a guess;
    (b) case by case, no better than a fair guess; (c) right as rankings, weakly (deciles within center x year, from
    35's cf_deciles; within-group rank correlations from 43 --within); (d) a single score is a range."""
    J = load_module("jf20", "20_journal_figures.py")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.0})
    K = F["keys"]
    m = met.set_index("quantity")

    def est(M, k):
        if summaries is not None:
            q = next(q for q in ["2", "3", "4", "5", "pooled"] if F[q] is M)
            r = summaries[(summaries.quintile.astype(str) == q) & (summaries.quantity == k)].iloc[0]
            return r.value, r.lo, r.hi
        v = M[:, K[k]]
        return v[0], np.nanpercentile(v[1:], 2.5), np.nanpercentile(v[1:], 97.5)
    fig = J.figure(180, 128)
    ax = J.axes(fig, 12, 74, 72, 44)
    for off, (kq, kp, col, mk, lab) in zip([-0.2, 0, 0.2], [
            ("yt", "y", J.INK, "o", "Observed citations"), ("st", "s", J.OUTC, "s", "Model's score"),
            ("gt", "c", J.MUTED, "D", "Guess: mean of granted patents, same center and year")]):
        for i, q in enumerate(["2", "3", "4", "5", "pooled"]):
            v, lo, hi = est(F[q], kq if q != "pooled" else kp)
            ax.errorbar(i + off, v, yerr=[[v - lo], [hi - v]], fmt=mk, ms=3.0, color=col,
                        mfc=col if q != "pooled" else "white", mec=col, elinewidth=0.8, capsize=0,
                        label=lab if i == 0 else None, zorder=3)
    ax.axvline(3.5, color=J.GRID, lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["2", "3", "4", "5", "2–5\n(raw)"])
    ax.set_xlabel("Stricter quintile (top-coded; 2–5 pooled, raw)")
    ax.set_ylabel("Citations per marginal application")
    ax.set_ylim(0, 6)
    ax.legend(loc="upper left", frameon=False)
    J.hgrid(ax, "y")
    J.letter(fig, 2, 124, "a", "Right on average, and so is a guess")
    ax = J.axes(fig, 104, 74, 72, 44)
    comp = [("top-coded: share of guessing's absolute error removed", "Absolute error,\ntop-coded"),
            ("top-coded: R2 against guessing", "Squared error,\ntop-coded"),
            ("log, recalibrated: R2 against guessing", "Squared error,\nlog scale")]
    for i, (q, lab) in enumerate(comp):
        r = m.loc[q]
        ax.errorbar(i, 100 * r.value, yerr=[[100 * (r.value - r.lo)], [100 * (r.hi - r.value)]], fmt="s", ms=3.4,
                    color=J.OUTC, elinewidth=0.9, capsize=0, zorder=3)
        ax.text(i + 0.12, 100 * r.value, f"{100 * r.value:+.0f}%".replace("-", "\u2212"), fontsize=6, color=J.INK2,
                va="center")
    ax.axhline(0, color=J.INK, lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab in comp], fontsize=6)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-25, 25)
    ax.set_ylabel("Guess's error the model removes (%)")
    J.hgrid(ax, "y")
    J.letter(fig, 94, 124, "b", "Case by case, no better than a fair guess")
    dec = pd.read_csv(T / f"c1_kllm_all{VSFX}_cf_deciles.csv")
    wr = pd.read_csv(T / f"c1x{VSFX}_within.csv").set_index(["grouping", "key"])
    ax = J.axes(fig, 15, 11, 69, 40)
    for gflag, col, mk, lab in [(1, J.OUTC, "o", "Granted"), (0, J.DEC, "s", "Rejected")]:
        z = dec[dec.granted == gflag].sort_values("dec")
        ax.plot(z.dec, z.prior_art, marker=mk, ms=2.8, color=col, lw=1.0, label=lab)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 3, 4, 6, 8])
    ax.set_yticklabels(["2", "3", "4", "6", "8"])
    ax.set_ylim(1.4, 11)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Score decile, within center × filing year")
    ax.set_ylabel("Prior-art citations (log scale)")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.05, "Rank correlation within center × year:\n"
            f"rejected {wr.loc[('TcYear', 'rej'), 'value']:.2f}, granted {wr.loc[('TcYear', 'gr'), 'value']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=J.INK2)
    J.hgrid(ax, "y")
    J.letter(fig, 2, 61, "c", "Right as rankings, weakly; a guess cannot rank")
    ax = J.axes(fig, 104, 11, 72, 40)
    Mp = F["pooled"]
    for who, kind in [("s", "bar"), ("c", "step")]:
        cdf = np.array([Mp[0, K[f"cdf_{who}_{r}"]] for r in GRID])
        dens = np.diff(np.maximum.accumulate(np.clip(cdf, 0, 1))) / np.diff(GRID)
        mid = (GRID[:-1] + GRID[1:]) / 2
        if kind == "bar":
            ax.bar(mid, dens, width=0.25, color=J.OUTC, alpha=0.75, label="Model's score", zorder=2)
        else:
            ax.step(np.r_[GRID[:-1], GRID[-1]], np.r_[dens, dens[-1]], where="post", color=J.MUTED, lw=1.0,
                    label="Guess", zorder=3)
    from matplotlib.lines import Line2D
    f50, f80 = F["F_s_50"][0], F["F_s_80"][0]
    for f, ls_ in [(f50, "-"), (f80, (0, (3, 2)))]:
        for sgn in [-1, 1]:
            ax.axvline(sgn * np.log(f), color=J.INK, lw=0.7, ls=ls_, zorder=4)
    fl = [f"{v:.2f}".rstrip("0").rstrip(".") for v in (f50, f80)]
    hd, lb = ax.get_legend_handles_labels()
    hd += [Line2D([], [], color=J.INK, lw=0.7), Line2D([], [], color=J.INK, lw=0.7, ls=(0, (3, 2)))]
    lb += [f"50% within ×{fl[0]}", f"80% within ×{fl[1]}"]
    ax.set_xticks(np.log([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]))
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Observed ÷ predicted, (1 + citations) / (1 + prediction), log scale")
    ax.set_ylabel("Density, marginal applications")
    ax.legend(hd, lb, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, borderaxespad=0.3,
              columnspacing=1.2)
    J.hgrid(ax, "y")
    J.letter(fig, 94, 61, "d", "A single score is a range, about as wide as a guess's")
    J.save(fig, f"fig_c1_score_check{VSFX}")


def main() -> None:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    m35, x, ev = build()
    rej_scores = x.loc[x.granted != 1, "cf"].to_numpy(float)
    edges = np.quantile(rej_scores, [1 / 3, 2 / 3])
    H = functions(ev, edges)
    n, au, D, V, Vrej = margin_sums(ev, H)
    mult = boot(au, rng)
    W = mult * n                                            # draws × cells
    den = W @ D
    M = {h: (W @ v) / den for h, v in V.items()}            # draws: means over marginal applications
    s_rej = (W @ Vrej) / den

    def var(a2, a):
        return M[a2] - M[a] ** 2

    def cov(ab, a, b):
        return M[ab] - M[a] * M[b]
    R = {}
    R["Mean citations, marginal applications (35's 2.98)"] = M["y"]
    R["Mean score, from the granted side"] = M["s"]
    R["Mean score, from the rejected side (check)"] = s_rej
    R["Mean score − mean citations (bias)"] = M["s"] - M["y"]
    R["Mean guess, marginal applications"] = M["g"]
    R["Mean guess, top-coded, marginal applications"] = M["gt"]
    for lab, ae, se, aeg, seg, yy, y2, ss, s2, sy in [
            ("raw", "ae", "se", "ae_g", "se_g", "y", "y2", "s", "s2", "sy"),
            ("log", "lae", "lse", "lae_g", "lse_g", "ly", "ly2", "ls", "ls2", "lsly"),
            ("log, recalibrated", "lkae", "lkse", "lae_g", "lse_g", "ly", "ly2", "lk", "lk2", "lkly"),
            ("top-coded", "tae", "tse", "tae_g", "tse_g", "yt", "yt2", "st", "st2", "styt")]:
        vy, vs, cv = var(y2, yy), var(s2, ss), cov(sy, ss, yy)
        R[f"{lab}: mean absolute error, score"] = M[ae]
        R[f"{lab}: mean absolute error, guessing"] = M[aeg]
        R[f"{lab}: guessing − score (absolute error)"] = M[aeg] - M[ae]
        R[f"{lab}: share of guessing's absolute error removed"] = 1 - M[ae] / M[aeg]
        R[f"{lab}: mean squared error, score"] = M[se]
        R[f"{lab}: mean squared error, guessing"] = M[seg]
        R[f"{lab}: R2 against the best constant"] = 1 - M[se] / vy
        R[f"{lab}: R2 against guessing"] = 1 - M[se] / M[seg]
        R[f"{lab}: correlation of score and citations"] = cv / np.sqrt(np.clip(vy, 1e-12, None) *
                                                                       np.clip(vs, 1e-12, None))
        R[f"{lab}: calibration slope (citations on score)"] = cv / np.clip(vs, 1e-12, None)
    # the median guess (the best constant under absolute error)
    R["raw: share of the median guess's absolute error removed"] = 1 - M["ae"] / M["ae_gm"]
    R["top-coded: share of the median guess's absolute error removed"] = 1 - M["tae"] / M["tae_gm"]
    # ranges on a fine grid of factors (same draws): the factor within which (1 + citations) lies of (1 + score)
    # for 50% and 80% of marginal applications, for the score and for the guess
    F = margins_fine(ev, edges, mult)
    K, Mp = F["keys"], F["pooled"]
    for who, lab in [("s", "score"), ("c", "guess")]:
        for share in [0.5, 0.8]:
            F[f"F_{who}_{int(100 * share)}"] = factor_fine(Mp, K, f"abs_{who}_", share)
            R[f"factor holding {int(100 * share)}% of marginal applications, {lab}"] = F[f"F_{who}_{int(100 * share)}"]
    grid10 = FINE[np.round(FINE * 20).astype(int) % 2 == 0]
    for k in range(3):
        for share in [0.5, 0.8]:
            R[f"factor holding {int(100 * share)}%, score tercile {k + 1}"] = factor_fine(
                Mp, K, f"bin{k}_abs_s_", share, grid=grid10, denom=Mp[:, K[f"bin{k}"]])
    rows = [dict(quantity=k, value=v[0], lo=np.nanpercentile(v[1:], 2.5), hi=np.nanpercentile(v[1:], 97.5))
            for k, v in R.items()]
    met = pd.DataFrame(rows)
    # the same quantities among granted test-set applications in the same cells (ordinary validation, for contrast)
    gr = ev[(ev.granted == 1)]
    s_, y_ = gr.cf.to_numpy(float), gr.cites.to_numpy(float)
    ls_, ly_ = np.log1p(np.maximum(s_, 0)), np.log1p(y_)
    c_ = gr.guess.to_numpy(float)
    lc_, lk_ = gr.lguess.to_numpy(float), gr.lcal.to_numpy(float)
    G = {"raw: mean absolute error, score": np.mean(np.abs(s_ - y_)),
         "raw: mean absolute error, guessing": np.mean(np.abs(c_ - y_)),
         "log: mean absolute error, score": np.mean(np.abs(ls_ - ly_)),
         "log: mean absolute error, guessing": np.mean(np.abs(lc_ - ly_)),
         "log, recalibrated: mean absolute error, score": np.mean(np.abs(lk_ - ly_)),
         "log, recalibrated: R2 against guessing": 1 - np.mean((lk_ - ly_) ** 2) / np.mean((lc_ - ly_) ** 2),
         "log, recalibrated: correlation of score and citations": np.corrcoef(lk_, ly_)[0, 1],
         "log: correlation of score and citations": np.corrcoef(ls_, ly_)[0, 1],
         "raw: correlation of score and citations": np.corrcoef(s_, y_)[0, 1]}
    met["granted applications (same cells)"] = met.quantity.map(G)
    met.to_csv(T / f"c1mv{VSFX}_metrics.csv", index=False)

    # error distribution at the margin (monotone by rearrangement), overall and by tercile of the score
    cdf = []
    for t in ABS_T:
        cdf.append(dict(kind="|score − citations| ≤ t", t=t, share=M[f"cdf_ae_{t}"][0],
                        lo=np.nanpercentile(M[f"cdf_ae_{t}"][1:], 2.5), hi=np.nanpercentile(M[f"cdf_ae_{t}"][1:], 97.5)))
    for f in LOG_F:
        cdf.append(dict(kind="(1 + score) / (1 + citations) within a factor of t", t=f, share=M[f"cdf_lr_{f}"][0],
                        lo=np.nanpercentile(M[f"cdf_lr_{f}"][1:], 2.5),
                        hi=np.nanpercentile(M[f"cdf_lr_{f}"][1:], 97.5)))
    cdf = pd.DataFrame(cdf)
    for kind, grp in cdf.groupby("kind"):
        cdf.loc[grp.index, "share"] = np.maximum.accumulate(grp.share.clip(0, 1).to_numpy())
    cdf.to_csv(T / f"c1mv{VSFX}_cdf.csv", index=False)

    # bands: the factor F such that a share p of marginal applications has (1 + Y) within F of (1 + s), by tercile
    def factor(shares: np.ndarray, p: float) -> float:
        shares = np.maximum.accumulate(np.clip(shares, 0, 1))
        if shares[-1] < p:
            return np.nan                                   # beyond the grid
        lf = np.log(LOG_F)
        j = int(np.argmax(shares >= p))
        if j == 0:
            return float(LOG_F[0]) if shares[0] >= p else np.nan
        w = (p - shares[j - 1]) / max(shares[j] - shares[j - 1], 1e-9)
        return float(np.exp(lf[j - 1] + w * (lf[j] - lf[j - 1])))
    # F50 / F80 from the fine grid (draw 0); the coarse-grid interpolation (factor) is kept for comparison only
    mq = met.set_index("quantity").value
    bands = []
    for k in range(3):
        pb = M[f"bin{k}"][0]
        sh = np.array([M[f"bin{k}_lr_{f}"][0] for f in LOG_F]) / max(pb, 1e-9)
        bands.append(dict(tercile=k + 1, share_of_marginal=pb, F50=mq[f"factor holding 50%, score tercile {k + 1}"],
                          F80=mq[f"factor holding 80%, score tercile {k + 1}"],
                          F50_coarse=factor(sh, 0.5), F80_coarse=factor(sh, 0.8),
                          **{f"within_{f}": v for f, v in zip(LOG_F, sh)}))
    allsh = np.array([M[f"cdf_lr_{f}"][0] for f in LOG_F])
    bands.append(dict(tercile="all", share_of_marginal=1.0,
                      F50=mq["factor holding 50% of marginal applications, score"],
                      F80=mq["factor holding 80% of marginal applications, score"],
                      F50_coarse=factor(allsh, 0.5), F80_coarse=factor(allsh, 0.8),
                      **{f"within_{f}": v for f, v in zip(LOG_F, allsh)}))
    bands = pd.DataFrame(bands)
    bands.to_csv(T / f"c1mv{VSFX}_bands.csv", index=False)

    # every rejected application, with its range
    rej = x[x.granted != 1].copy()
    b = np.digitize(rej.cf.to_numpy(float), edges)
    F50 = bands.set_index("tercile").F50
    F80 = bands.set_index("tercile").F80
    s1 = 1 + np.maximum(rej.cf.to_numpy(float), 0)
    f50 = np.array([F50.get(k + 1, np.nan) for k in b])
    f80 = np.array([F80.get(k + 1, np.nan) for k in b])
    titles = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "invention_title"])
    out = pd.DataFrame(dict(application_number=rej.application_number, tech_center=rej.tech_center,
                            filing_year=rej.filing_year, set=rej.part, counterfactual_citations=rej.cf.round(2),
                            percentile_among_granted=rej.cf_pct.round(1), score_tercile=b + 1,
                            range50_low=np.maximum(s1 / f50 - 1, 0).round(2), range50_high=(s1 * f50 - 1).round(2),
                            range80_low=np.maximum(s1 / f80 - 1, 0).round(2), range80_high=(s1 * f80 - 1).round(2),
                            prior_art_citations=rej.prior_art))
    out = out.merge(titles, on="application_number", how="left")
    out.to_csv(T / f"c1_counterfactual_scores_validated{VSFX}.csv", index=False)

    # macros and report
    m = met.set_index("quantity")
    N = {}

    def put(name, q, nd=2, s=1.0):
        r = m.loc[q]
        N[name] = f"{s * r.value:.{nd}f}".replace("-", "$-$")
        N[name + "CI"] = f"[{s * r.lo:.{nd}f}, {s * r.hi:.{nd}f}]".replace("-", "$-$")
    put(MAC + "MeanY", "Mean citations, marginal applications (35's 2.98)")
    put(MAC + "MeanS", "Mean score, from the granted side")
    put(MAC + "MeanSRej", "Mean score, from the rejected side (check)")
    put(MAC + "Bias", "Mean score − mean citations (bias)")
    for lab, mac in [("raw", "Raw"), ("log", "Log"), ("log, recalibrated", "Cal"), ("top-coded", "Trim")]:
        put(f"{MAC}{mac}MAE", f"{lab}: mean absolute error, score")
        put(f"{MAC}{mac}MAEGuess", f"{lab}: mean absolute error, guessing")
        put(f"{MAC}{mac}Gain", f"{lab}: guessing − score (absolute error)")
        put(f"{MAC}{mac}GainShare", f"{lab}: share of guessing's absolute error removed", 0, 100)
        put(f"{MAC}{mac}Rsq", f"{lab}: R2 against the best constant")
        put(f"{MAC}{mac}RsqGuess", f"{lab}: R2 against guessing")
        put(f"{MAC}{mac}Corr", f"{lab}: correlation of score and citations")
        put(f"{MAC}{mac}Slope", f"{lab}: calibration slope (citations on score)")
    put(MAC + "MeanGuess", "Mean guess, marginal applications")
    put(MAC + "MeanGuessTrim", "Mean guess, top-coded, marginal applications")
    put(MAC + "RawGainShareMed", "raw: share of the median guess's absolute error removed", 0, 100)
    put(MAC + "TrimGainShareMed", "top-coded: share of the median guess's absolute error removed", 0, 100)
    put(MAC + "TrimRsqGuessPct", "top-coded: R2 against guessing", 0, 100)      # share of its squared error removed
    put(MAC + "CalRsqGuessPct", "log, recalibrated: R2 against guessing", 0, 100)

    def fac(v):                                             # factors on a grid of 0.05: 2.30 -> 2.3, 3.65 -> 3.65
        return f"{v:.2f}".rstrip("0").rstrip(".")
    for k, r in bands.iterrows():
        tag = {0: "One", 1: "Two", 2: "Three", 3: "All"}[k]
        N[f"{MAC}FFifty{tag}"], N[f"{MAC}FEighty{tag}"] = fac(r.F50), fac(r.F80)
    for who, tag in [("score", "All"), ("guess", "Guess")]:
        for share, nm in [(50, "FFifty"), (80, "FEighty")]:
            r = m.loc[f"factor holding {share}% of marginal applications, {who}"]
            N[f"{MAC}{nm}{tag}"] = fac(r.value)
            N[f"{MAC}{nm}{tag}CI"] = f"[{fac(r.lo)}, {fac(r.hi)}]"
    N[MAC + "NRej"] = f"{len(out):,}"
    (TEX / f"c1mv{VSFX}_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    L = ["# C1: the counterfactual scores at the examiners' margin", "",
         f"{len(ev):,} test and imputation applications in {ev.au_class.nunique():,} cells; marginal applications: "
         "granted by the most lenient quintile, rejected by a stricter one (four comparisons, pooled). 95% CI from "
         f"{B} bootstrap draws of art units. Guessing: mean citations (log scale: mean log citations) of granted "
         "training-set applications of the same center and filing year. Recalibrated: log(1 + score) mapped to the "
         "log scale by an OLS fitted on granted training-set applications. Raw-scale second moments (mean squared "
         "error, R2, correlation) are left out: with citations this heavy-tailed they are not estimable "
         "(see c1mv_metrics.csv)." + (" Scores: 35's Poisson variant." if POIS else ""), "", "| Quantity | Marginal applications | 95% CI | Granted applications |",
         "|---|---|---|---|"]
    # second moments on the raw scale are not usable: with citations this heavy-tailed, the difference between two
    # quintiles' means of squared citations is dominated by a few patents (estimates can be negative); csv only
    shown = met[~met.quantity.str.match(r"raw: (mean squared|R2|correlation|calibration)")]
    for r in shown.itertuples():
        gcol = "" if pd.isna(r[-1]) else f"{r[-1]:.3f}"
        L.append(f"| {r.quantity} | {r.value:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {gcol} |")
    L += ["", "Error distribution at the margin (share of marginal applications)", "",
          "| Measure | t | Share | 95% CI |", "|---|---|---|---|"]
    for r in cdf.itertuples():
        L.append(f"| {r.kind} | {r.t} | {r.share:.3f} | [{r.lo:.3f}, {r.hi:.3f}] |")
    L += ["", "Ranges by tercile of the score: (1 + citations) lies within a factor F50 (F80) of (1 + score) for 50% "
              "(80%) of marginal applications", "", "| Tercile | Share of marginal | F50 | F80 |", "|---|---|---|---|"]
    for r in bands.itertuples():
        L.append(f"| {r.tercile} | {r.share_of_marginal:.2f} | {r.F50:.2f} | {r.F80:.2f} |")
    (OUT / f"c1_margin_validation{VSFX}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # levels per stricter quintile and the ratio distribution, for the figure
    lv = []
    for qn in ["2", "3", "4", "5", "pooled"]:
        for kq in ["y", "s", "c", "yt", "st", "gt"]:
            v = F[qn][:, K[kq]]
            lv.append(dict(quintile=qn, quantity=kq, value=v[0], lo=np.nanpercentile(v[1:], 2.5),
                           hi=np.nanpercentile(v[1:], 97.5)))
    pd.DataFrame(lv).to_csv(T / f"c1mv{VSFX}_levels.csv", index=False)
    dist = []
    for who in ["s", "c"]:
        cdf_ = np.maximum.accumulate(np.clip([Mp[0, K[f"cdf_{who}_{r}"]] for r in GRID], 0, 1))
        for i in range(len(GRID) - 1):
            dist.append(dict(predictor={"s": "score", "c": "guess"}[who], left=GRID[i], right=GRID[i + 1],
                             mass=cdf_[i + 1] - cdf_[i]))
    pd.DataFrame(dist).to_csv(T / f"c1mv{VSFX}_ratio_distribution.csv", index=False)
    if (T / f"c1_kllm_all{VSFX}_cf_deciles.csv").exists() and (T / f"c1x{VSFX}_within.csv").exists():
        score_check_figure(F, met)


if __name__ == "__main__":
    main()
