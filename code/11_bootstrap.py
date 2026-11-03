"""Examiner-clustered bootstrap: confidence intervals for the gender gaps of the models in 09.

The trained models and their test-set decisions are held fixed, so the intervals reflect test-sample sampling error
only. Each draw resamples examiners with replacement; an application's weight is the number of times its examiner is
drawn. After weighted demeaning within art unit × year, we compute the coefficient of the decision on female.

Reported
- Each model's gap with its 95% interval; the difference from the human grant gap (percentage points) with its
  interval; the ratio to the human gap with its interval (meaningful only where the human gap is significant).
- The key contrasts in CONTRASTS, e.g. train on grant − train on citations, and high-gap − low-gap examiners.

Input: data/processed/probe_preds_{3600,1600}.parquet (output of 09)
Output: output/tables/probes_bootstrap.csv, output/probes_ci.md
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
B, SEED = 1000, 20261028
CONTRASTS = [("Train on grant", "Train on citations (granted only)"),
             ("Train on grant", "Train on first action"),
             ("Train on grant", "Train on grant + prosecution history"),
             ("Train on grant, high-gap examiners", "Train on grant, low-gap examiners"),
             ("Train on grant, lenient examiners", "Train on grant, strict examiners")]


def fe_gaps(Y: np.ndarray, f: np.ndarray, grp: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted within-group regression coefficients of each column of Y on f.

    Y and f are demeaned within groups grp using weights w. Y is n x K; returns a vector of length K.
    """
    G = grp.max() + 1
    sw = np.bincount(grp, weights=w, minlength=G)
    ok = sw > 0
    fbar = np.zeros(G)
    fbar[ok] = np.bincount(grp, weights=w * f, minlength=G)[ok] / sw[ok]
    fd = f - fbar[grp]
    Ybar = np.zeros((G, Y.shape[1]))
    for k in range(Y.shape[1]):
        Ybar[ok, k] = np.bincount(grp, weights=w * Y[:, k], minlength=G)[ok] / sw[ok]
    Yd = Y - Ybar[grp]
    return (w * fd) @ Yd / np.sum(w * fd * fd)


def run_tc(tc: str) -> pd.DataFrame:
    d = pd.read_parquet(P / f"probe_preds_{tc}.parquet")
    specs = [c.split("::", 1)[1] for c in d.columns if c.startswith("dec::")]
    cols = ["granted", "fa_allow"] + [f"dec::{s}" for s in specs]
    names = ["Human examiners", "Human examiners (first action)"] + specs
    Y = d[cols].to_numpy(float)
    f = d["female"].to_numpy(float)
    grp = pd.factorize(d["au_year"])[0]
    ex = pd.factorize(d["examiner_id"])[0]
    n_ex = ex.max() + 1
    point = fe_gaps(Y, f, grp, np.ones(len(d)))
    rng = np.random.default_rng(SEED)
    draws = np.empty((B, len(names)))
    for b in range(B):
        m = np.bincount(rng.integers(0, n_ex, n_ex), minlength=n_ex)
        draws[b] = fe_gaps(Y, f, grp, m[ex].astype(float))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    diff = draws - draws[:, [0]]
    ratio = draws / draws[:, [0]]
    rows = []
    for k, nm in enumerate(names):
        rows.append(dict(tech_center=tc, spec=nm, gap=point[k], gap_lo=lo[k], gap_hi=hi[k],
                         diff_vs_human=point[k] - point[0],
                         diff_lo=np.percentile(diff[:, k], 2.5), diff_hi=np.percentile(diff[:, k], 97.5),
                         ratio=point[k] / point[0],
                         ratio_lo=np.percentile(ratio[:, k], 2.5), ratio_hi=np.percentile(ratio[:, k], 97.5)))
    idx = {nm: k for k, nm in enumerate(names)}
    for a, b in CONTRASTS:
        if a in idx and b in idx:
            dd = draws[:, idx[a]] - draws[:, idx[b]]
            rows.append(dict(tech_center=tc, spec=f"Contrast: {a} − {b}", gap=point[idx[a]] - point[idx[b]],
                             gap_lo=np.percentile(dd, 2.5), gap_hi=np.percentile(dd, 97.5)))
    return pd.DataFrame(rows)


def main() -> None:
    res = pd.concat([run_tc(tc) for tc in ["3600", "1600"]], ignore_index=True)
    res.to_csv(OUT / "tables" / "probes_bootstrap.csv", index=False)
    pp = lambda x: "" if pd.isna(x) else f"{100 * x:+.1f}"
    rr = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    t = pd.DataFrame({
        "TC": res.tech_center, "Model": res.spec,
        "Gender gap (pp)": res.gap.map(pp),
        "95% CI": [f"[{pp(a)}, {pp(b)}]" for a, b in zip(res.gap_lo, res.gap_hi)],
        "Diff. vs. human": res.diff_vs_human.map(pp),
        "Diff. 95% CI": ["" if pd.isna(a) else f"[{pp(a)}, {pp(b)}]" for a, b in zip(res.diff_lo, res.diff_hi)],
        "Ratio": res.ratio.map(rr),
        "Ratio 95% CI": ["" if pd.isna(a) else f"[{rr(a)}, {rr(b)}]" for a, b in zip(res.ratio_lo, res.ratio_hi)],
    })
    with open(OUT / "probes_ci.md", "w") as fh:
        fh.write(f"# Confidence intervals for probe results (examiner-clustered bootstrap, {B} draws)\n\n"
                 "Models and decisions are held fixed; intervals reflect test-sample sampling error only. In TC 1600 the "
                 "human gap is close to 0, so ratios are not meaningful; read the gaps and differences instead.\n\n")
        fh.write(t.to_markdown(index=False) + "\n")
    print((OUT / "probes_ci.md").read_text())


if __name__ == "__main__":
    main()
