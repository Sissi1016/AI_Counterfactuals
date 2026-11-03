"""C1, their Table II in both tails: does the boosted model pick different applications than the linear benchmark, and
are its picks better? With uncertainty and the heavy tail in view.

Kleinberg et al. (2018, Table II) compare the defendants that only the machine-learning model, or only a logistic
regression, places in the highest-risk tail. Their decision rule detains from that tail. Ours rejects from the LOWEST
predicted citations, so here both tails are shown: among test applications, the top and bottom 1%, 5%, 10%, and 25%
by each model's prediction (35's Poisson boosted model, ml_cites, and its Poisson GLM benchmark with the same inputs and
objective, lin_cites), the overlap of the two sets, and the mean citations of the GRANTED applications flagged by only
one model; for the top, higher is better, for the bottom, lower is better. Each mean is also given with citations
top-coded at the 99th percentile of granted applications of the same center and filing year, and without each set's
five most-cited patents; the difference between the two models comes with a 95% CI from 1,000 bootstrap draws of art
units. The top 1% also shows which centers the picks come from.

Inputs: data/processed/{all_sample,c1_preds_all_poisson_all}.parquet. Outputs: output/tables/tb_table2_tails.csv,
output/latex/tb_tab_table2.tex, output/latex/tb_numbers.tex (macros \\xTb...), output/c1_table2_tails.md.
CPU only, under a minute: taskset -c 0-15 uv run python code/51_c1_table2_tails.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, T, TEX = ROOT / "data/processed", ROOT / "output/tables", ROOT / "output/latex"
SEED, B = 20261101 + 51, 1000
CUTS = [0.01, 0.05, 0.10, 0.25]
WORD = {0.01: "One", 0.05: "Five", 0.10: "Ten", 0.25: "Quart"}


def main() -> None:
    d = pd.read_parquet(P / "all_sample.parquet")
    pr = pd.read_parquet(P / "c1_preds_all_poisson_all.parquet",
                         columns=["application_number", "part", "ml_cites", "lin_cites"])
    x = d.merge(pr, on="application_number")
    cap = x[x.granted == 1].groupby(["tech_center", "filing_year"]).cites_ns.quantile(0.99).rename("cap").reset_index()
    t = x[x.part == "test"].merge(cap, on=["tech_center", "filing_year"], how="left")
    t["cites_tc"] = np.minimum(t.cites_ns, t.cap)
    n = len(t)
    rng = np.random.default_rng(SEED)
    aus = np.sort(t.art_unit.unique())
    W = np.stack([np.bincount(rng.integers(0, len(aus), len(aus)), minlength=len(aus)) for _ in range(B)])  # B x AU

    def boot(g: pd.DataFrame, col: str) -> np.ndarray:
        """Bootstrap of mean(ml_only) - mean(lin_only) over art units (resampled with replacement)."""
        idx = np.searchsorted(aus, g.art_unit.to_numpy())
        out = {}
        for k_ in ["ml_only", "lin_only"]:
            m = (g.grp == k_).to_numpy()
            sm = np.bincount(idx[m], weights=g[col].to_numpy()[m], minlength=len(aus))
            ct = np.bincount(idx[m], minlength=len(aus)).astype(float)
            out[k_] = (W @ sm) / np.maximum(W @ ct, 1e-9)
        return out["ml_only"] - out["lin_only"]
    # citations as a percentile among granted test applications of the same center and filing year (mid-ranks)
    t["tcy"] = t.tech_center.astype(str) + "_" + t.filing_year.astype(str)
    gr = t.granted == 1
    t["cpct"] = np.nan
    t.loc[gr, "cpct"] = 100 * t[gr].groupby("tcy").cites_ns.rank(pct=True, method="average")
    rows, N = [], {}
    for scope in ["pooled", "within"]:
        for tail in ["top", "bottom"]:
            asc = tail == "bottom"
            for c in CUTS:
                if scope == "pooled":
                    k = int(round(c * n))
                    ml = t.ml_cites.rank(ascending=asc, method="first") <= k
                    ln = t.lin_cites.rank(ascending=asc, method="first") <= k
                else:   # each model flags the top (bottom) share within every center x filing year
                    size = t.groupby("tcy").tcy.transform("size")
                    kk = (c * size).round()
                    ml = t.groupby("tcy").ml_cites.rank(ascending=asc, method="first") <= kk
                    ln = t.groupby("tcy").lin_cites.rank(ascending=asc, method="first") <= kk
                    k = int(ml.sum())
                grp = np.select([ml & ~ln, ~ml & ln], ["ml_only", "lin_only"], "")
                g = t.assign(grp=grp)
                g = g[(g.grp != "") & (g.granted == 1)]
                st = g.groupby("grp").agg(n=("cites_ns", "size"), mean=("cites_ns", "mean"),
                                          median=("cites_ns", "median"), mean_tc=("cites_tc", "mean"),
                                          pct=("cpct", "mean"))
                trim5 = {k_: np.sort(v.cites_ns.to_numpy())[::-1][5:].mean() for k_, v in g.groupby("grp")}
                dd, dt, dp = boot(g, "cites_ns"), boot(g, "cites_tc"), boot(g, "cpct")
                r = dict(scope=scope, tail=tail, cut=c, flagged=k, overlap=float((ml & ln).sum() / k),
                         n_ml=int(st.loc["ml_only", "n"]), n_lin=int(st.loc["lin_only", "n"]),
                         ml_mean=st.loc["ml_only", "mean"], lin_mean=st.loc["lin_only", "mean"],
                         ml_median=st.loc["ml_only", "median"], lin_median=st.loc["lin_only", "median"],
                         ml_mean_tc=st.loc["ml_only", "mean_tc"], lin_mean_tc=st.loc["lin_only", "mean_tc"],
                         ml_pct=st.loc["ml_only", "pct"], lin_pct=st.loc["lin_only", "pct"],
                         ml_mean_trim5=trim5["ml_only"], lin_mean_trim5=trim5["lin_only"],
                         diff=st.loc["ml_only", "mean"] - st.loc["lin_only", "mean"],
                         diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5),
                         diff_tc=st.loc["ml_only", "mean_tc"] - st.loc["lin_only", "mean_tc"],
                         diff_tc_lo=np.percentile(dt, 2.5), diff_tc_hi=np.percentile(dt, 97.5),
                         diff_pct=st.loc["ml_only", "pct"] - st.loc["lin_only", "pct"],
                         diff_pct_lo=np.percentile(dp, 2.5), diff_pct_hi=np.percentile(dp, 97.5))
                if scope == "pooled" and tail == "top" and c == 0.01:
                    for k_, v in g.groupby("grp"):
                        r[f"{k_}_top5"] = ", ".join(str(int(u)) for u in np.sort(v.cites_ns.to_numpy())[::-1][:5])
                    sel = t.assign(grp=np.select([ml & ~ln, ~ml & ln, ml & ln], ["ml_only", "lin_only", "both"], ""))
                    for k_ in ["ml_only", "lin_only", "both"]:
                        r[f"{k_}_tc3700"] = float((sel[sel.grp == k_].tech_center.astype(str) == "3700").mean())
                    r["ml_max"], r["lin_max"] = t.ml_cites.max(), t.lin_cites.max()
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(T / "tb_table2_tails.csv", index=False)

    def f(v, nd=1):
        return f"{v:.{nd}f}".replace("-", "$-$")
    body = []
    for scope, tail, lab in [("pooled", "bottom", "Lowest predicted (the tail a rule rejects from; fewer is better)"),
                             ("pooled", "top", "Highest predicted (more is better)"),
                             ("within", "bottom", "Lowest predicted within each center $\\times$ filing year"),
                             ("within", "top", "Highest predicted within each center $\\times$ filing year")]:
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        for r in R[(R["tail"] == tail) & (R.scope == scope)].itertuples():
            body.append(" & ".join([f"{int(100 * r.cut)}\\%", f"{100 * r.overlap:.0f}",
                                    f"{f(r.ml_mean, 2)} / {f(r.lin_mean, 2)}", f"{f(r.ml_median, 0)} / {f(r.lin_median, 0)}",
                                    f"{f(r.ml_mean_tc, 2)} / {f(r.lin_mean_tc, 2)}",
                                    f"{f(r.diff, 2)} {{\\footnotesize [{f(r.diff_lo, 2)}, {f(r.diff_hi, 2)}]}}",
                                    f"{f(r.diff_tc, 2)} {{\\footnotesize [{f(r.diff_tc_lo, 2)}, {f(r.diff_tc_hi, 2)}]}}"])
                        + r" \\")
        body.append(r"\addlinespace")
    hdr = (r"\begin{tabular}{lcccccc}" "\n" r"\toprule" "\n"
           r" & & \multicolumn{3}{c}{Citations, granted applications flagged by one model only: boosted / linear} & "
           r"\multicolumn{2}{c}{Boosted $-$ linear} \\" "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}" "\n"
           r"Share of test applications & Overlap (\%) & Mean & Median & Mean, top-coded & Mean & Mean, top-coded \\"
           "\n" r"\midrule" "\n")
    (TEX / "tb_tab_table2.tex").write_text(hdr + "\n".join(body[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    for r in R.itertuples():
        w = ("W" if r.scope == "within" else "") + ("Top" if r.tail == "top" else "Bot") + WORD[r.cut]
        N[f"xTb{w}Overlap"] = f"{100 * r.overlap:.0f}"
        for k_, v in [("Ml", r.ml_mean), ("Lin", r.lin_mean), ("MlTc", r.ml_mean_tc), ("LinTc", r.lin_mean_tc),
                      ("MlTrim", r.ml_mean_trim5), ("LinTrim", r.lin_mean_trim5)]:
            N[f"xTb{w}{k_}"] = f(v, 1 if r.tail == "top" else 2)
        N[f"xTb{w}MlPct"], N[f"xTb{w}LinPct"] = f(r.ml_pct, 1), f(r.lin_pct, 1)
        for k_, v, lo, hi in [("Diff", r.diff, r.diff_lo, r.diff_hi), ("DiffTc", r.diff_tc, r.diff_tc_lo, r.diff_tc_hi),
                              ("DiffPct", r.diff_pct, r.diff_pct_lo, r.diff_pct_hi)]:
            N[f"xTb{w}{k_}"] = f(v, 2)
            N[f"xTb{w}{k_}CI"] = f"[{f(lo, 2)}, {f(hi, 2)}]"
    top1 = R[(R["tail"] == "top") & (R.cut == 0.01) & (R.scope == "pooled")].iloc[0]
    N["xTbTopOneLinTopFive"] = top1.lin_only_top5
    N["xTbTopOneMlTopFive"] = top1.ml_only_top5
    N["xTbTopOneMlTcThreeSeven"] = f"{100 * top1.ml_only_tc3700:.0f}"
    N["xTbTopOneLinTcThreeSeven"] = f"{100 * top1.lin_only_tc3700:.0f}"
    N["xTbTopOneBothTcThreeSeven"] = f"{100 * top1.both_tc3700:.0f}"
    N["xTbMlMax"], N["xTbLinMax"] = f"{top1.ml_max:,.0f}", f"{top1.lin_max:,.0f}"
    for k in list(N):
        if N[k].startswith("$-$"):
            N[k + "Abs"] = N[k][3:]
    (TEX / "tb_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    md = ["# C1, Table II in both tails (script 51)", "", R.round(3).to_markdown(index=False)]
    (ROOT / "output/c1_table2_tails.md").write_text("\n".join(md) + "\n")
    print(R[["tail", "cut", "overlap", "ml_mean", "lin_mean", "ml_mean_tc", "lin_mean_tc", "ml_mean_trim5",
             "lin_mean_trim5", "diff", "diff_lo", "diff_hi", "diff_tc", "diff_tc_lo", "diff_tc_hi"]].round(2).to_string())


if __name__ == "__main__":
    main()
