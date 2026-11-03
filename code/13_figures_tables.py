"""Figures and LaTeX tables for the proposal and the technical appendix.

Reads the result tables written by 09–19 and the pilot sample, and writes
- output/figures/fig1_training_target.{pdf,png}: gender gap in decisions by training signal and by teacher (TC 3600);
- output/figures/fig2_decision_outcome.{pdf,png}: the decision–outcome curve (TC 3600);
- output/latex/tab_*.tex: table bodies (tabular environments; captions live in the documents);
- output/latex/numbers.tex: macros for the numbers quoted in the text, so text and tables come from the same run.

Input: output/tables/{probes_bootstrap,probes_3600,probes_1600,llm_prompting_full,llm_prompting_ci_full}.csv,
       output/tables/{marginal,llm_marginal,llm_features,ai_examiners,assignment,contraction}_*.csv,
       data/processed/{pilot_inputs,pilot_sample}.parquet
Output: output/figures/, output/latex/
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
T = ROOT / "output/tables"
FIG = ROOT / "output/figures"
TEX = ROOT / "output/latex"

# palette: validated categorical slots 1–2 (blue, orange) on a light surface; text and rules in neutral inks
INK, INK2, RULE, BAND = "#0b0b0b", "#52514e", "#d6d5d0", "#ebeae6"
BLUE = "#2a78d6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK,
    "xtick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "pdf.fonttype": 42, "savefig.dpi": 300,
})

# no creation timestamp, so that reruns produce identical files
META = {"pdf": {"CreationDate": None}, "png": {"Software": None}}
HUMAN = "Human record (truth)"
NO_MODEL = "No model: mean outcome of granted applications"
LLM_NO_MODEL = "No model: mean outcome of accepted cases"
LLM_ROWS = {"Zero-shot": "LLM, zero-shot", "Few-shot, lenient examiners' decisions": "LLM, lenient examiners' examples",
            "Few-shot, strict examiners' decisions": "LLM, strict examiners' examples"}


# ---------------------------------------------------------------- helpers

def pp(x: float, nd: int = 1, scale: float = 100) -> str:
    """Percentage points with a typographic minus, for LaTeX (no minus on a value that rounds to zero)."""
    s = f"{scale * x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def ci(b: float, se: float, nd: int = 1, scale: float = 100) -> str:
    return f"{pp(b, nd, scale)} [{pp(b - 1.96 * se, nd, scale)}, {pp(b + 1.96 * se, nd, scale)}]"


def ci_lohi(b: float, lo: float, hi: float, nd: int = 1, scale: float = 100) -> str:
    return f"{pp(b, nd, scale)} [{pp(lo, nd, scale)}, {pp(hi, nd, scale)}]"


def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("−", "--")


def tabular(path: pathlib.Path, colspec: str, header: list, rows: list, groups: dict | None = None) -> None:
    """Write a booktabs tabular. rows: list of lists of LaTeX strings; groups: {row index: panel title}."""
    L = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    ncol = len(header)
    for i, r in enumerate(rows):
        if groups and i in groups:
            if i:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{groups[i]}}}}} \\")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def spec_label(spec: str) -> str:
    """Compact probe names for tables: 'Train on grant, lenient examiners' -> 'Grant, lenient examiners'."""
    x = spec.replace("Train on ", "").replace(" (granted only)", "").replace(" of examiners (control)", " (control)")
    x = x.replace(" − ", " $-$ ")
    return x[0].upper() + x[1:]


def short(series: str) -> tuple[str, str]:
    """Split an evaluator name into (training label, inputs) for tables."""
    if series == NO_MODEL:
        return "None (mean renewal of granted)", "--"
    s = series.replace("Train on ", "").replace(" (granted only)", "")
    label, _, extra = s.partition(" + ")
    inputs = "Text" + (f" + {extra}" if extra else "")
    return label[0].upper() + label[1:], inputs


# ---------------------------------------------------------------- figures

def fig_training_target(boot: pd.DataFrame) -> None:
    b = boot[boot.tech_center == "3600"].set_index("spec")
    human = b.loc["Human examiners"]
    panels = [
        ("(a) What the evaluator is trained on", [
            ("Human examiners: final grant", "Human examiners", True),
            ("Human examiners: first action", "Human examiners (first action)", True),
            ("First action", "Train on first action", False),
            ("Final grant", "Train on grant", False),
            ("Final grant + prosecution history", "Train on grant + prosecution history", False),
            ("Renewal (granted only)", "Train on renewal (granted only)", False),
            ("Citations (granted only)", "Train on citations (granted only)", False)]),
        ("(b) Whom it learns from (all trained on final grant)", [
            ("All examiners", "Train on grant", False),
            ("Lenient examiners", "Train on grant, lenient examiners", False),
            ("Strict examiners", "Train on grant, strict examiners", False),
            ("Examiners with a large own gap", "Train on grant, high-gap examiners", False),
            ("Examiners with a small own gap", "Train on grant, low-gap examiners", False),
            ("Random half of examiners", "Train on grant, random half of examiners (control)", False)]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.1), sharex=True,
                             gridspec_kw={"height_ratios": [len(p[1]) for p in panels], "hspace": 0.35})
    for ax, (title, rows) in zip(axes, panels):
        ax.axvspan(100 * human.gap_lo, 100 * human.gap_hi, color=BAND, zorder=0, lw=0)
        ax.axvline(100 * human.gap, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        for i, (lab, key, is_human) in enumerate(rows):
            r = b.loc[key]
            y = len(rows) - 1 - i
            col = INK2 if is_human else BLUE
            ax.errorbar(100 * r.gap, y, xerr=[[100 * (r.gap - r.gap_lo)], [100 * (r.gap_hi - r.gap)]], fmt="o",
                        ms=4, color=col, ecolor=col, elinewidth=1.4, capsize=0, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)
            if not is_human:
                ax.text(1.005, y, f"{r.ratio:.2f}×".replace("-", "−"), transform=ax.get_yaxis_transform(),
                        va="center", ha="left",
                        color=INK2, fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_ylim(-1.1 if ax is axes[0] else -0.6, len(rows) - 0.4)
        ax.set_title(title, pad=4)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=BAND, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].text(1.005, 1.0, "× human", transform=axes[0].transAxes, va="bottom", ha="left", color=INK2, fontsize=7)
    axes[0].text(100 * human.gap_hi, -0.85, " gray band: human grant gap, 95% CI", color=INK2, fontsize=6.5,
                 va="center", ha="left")
    axes[-1].set_xlabel("Gender gap in grant decisions, female − male first inventor (percentage points)")
    fig.subplots_adjust(left=0.30, right=0.92, top=0.95, bottom=0.11)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig1_training_target.{ext}", metadata=META[ext])
    plt.close(fig)


def fig_decision_outcome(cur: pd.DataFrame) -> None:
    c = cur[cur["sample"] == "3600"].sort_values("weight_on_grant")
    w = c.weight_on_grant.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    g, se = 100 * c.gap_decision.to_numpy(), 100 * c.gap_se.to_numpy()
    ax.fill_between(w, g - 1.96 * se, g + 1.96 * se, color=BLUE, alpha=0.15, lw=0)
    ax.plot(w, g, color=BLUE, lw=1.6, marker="o", ms=3.2, markeredgecolor="white", markeredgewidth=0.6)
    hg = 100 * c.human_gap.iloc[0]
    ax.axhline(hg, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.0, hg - 0.25, f"Human examiners ({hg:.1f})".replace("-", "−"), color=INK2, fontsize=6.5,
            va="top", ha="right")
    ax.set_ylim(min(g - 1.96 * se) - 0.5, hg + 1.0)
    ax.set_ylabel("Gender gap in decisions (pp)")
    ax.set_title("(a) Decisions at the human grant rate", pad=4)

    ax = axes[1]
    e, s = 100 * c["renew_4y_error"].to_numpy(), 100 * c["renew_4y_error_se"].to_numpy()
    ax.fill_between(w, e - 1.96 * s, e + 1.96 * s, color=BLUE, alpha=0.15, lw=0)
    ax.plot(w, e, color=BLUE, lw=1.6, marker="o", ms=3.2, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.0, 0.25, "No error", color=INK2, fontsize=6.5, va="bottom", ha="right")
    ax.set_ylim(min(e - 1.96 * s) - 0.5, 1.5)
    ax.set_ylabel("Predicted − actual renewal (pp)")
    ax.set_title("(b) Error on the marginal applications", pad=4)

    for ax in axes:
        ax.set_xlabel("Weight on the grant decision in the training signal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0\nrenewal", "0.25", "0.5", "0.75", "1\ngrant"])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=BAND, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", colors=INK2, width=0.6, size=2.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.25)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig2_decision_outcome.{ext}", metadata=META[ext])
    plt.close(fig)


# ---------------------------------------------------------------- tables

def tables(boot, probes, llm, llm_ci, est, con, split, fit, rob, bal, fs, exc, exb, cur, sample, lm) -> dict:
    N = {}

    # Table 1: validation on the marginal applications (TC 3600, renewal)
    o, smp = "renew_4y", "3600"
    sp = split[(split["sample"] == smp) & (split.outcome == o) & (split.series == HUMAN)].iloc[0]
    e = est[(est["sample"] == smp) & (est.outcome == o)]
    c = con[(con["sample"] == smp) & (con.outcome == o)]

    def get(series, kind, group="All"):
        return e[(e.series == series) & (e.kind == kind) & (e.group == group)].iloc[0]

    def contrast(series, kind, name):
        return c[(c.series == series) & (c.kind == kind) & (c.contrast == name)].iloc[0]
    h = get(HUMAN, "marginal")
    rows_a = [["Strict examiners", pp(sp.strict_granted / 100), pp(sp.strict_granted_y / 100)],
              ["Lenient examiners", pp(sp.lenient_granted / 100), pp(sp.lenient_granted_y / 100)],
              ["Difference", pp((sp.lenient_granted - sp.strict_granted) / 100),
               pp((sp.lenient_granted_y - sp.strict_granted_y) / 100)]]
    tabular(TEX / "tab_validation_a.tex", "lrr", ["", "Granted", "Granted and renewed"], rows_a)
    N["splitMarginal"] = pp(sp.marginal_split / 100)
    N["splitStrictGranted"], N["splitLenientGranted"] = pp(sp.strict_granted / 100), pp(sp.lenient_granted / 100)
    N["splitStrictRenewed"], N["splitLenientRenewed"] = pp(sp.strict_granted_y / 100), pp(sp.lenient_granted_y / 100)

    order = [NO_MODEL, "Train on first action", "Train on grant", "Train on grant + names",
             "Train on grant + entity status", "Train on renewal (granted only)", "Train on renewal + names",
             "Train on renewal + entity status", "Train on citations (granted only)"]
    f = fit[fit["sample"] == smp].set_index("series")
    rows_b = [["Human record", "--", ci(h.estimate, h.se), "--", "--", "--"]]
    for s in order:
        lab, inp = short(s)
        name = "No model (mean of granted)" if s == NO_MODEL else f"{lab} ({inp[0].lower() + inp[1:]})"
        imp, err = get(s, "implied"), get(s, "error")
        fe = get(s, "error", "Female")
        sl = contrast(s, "error", "Small − large")
        auc = f"{f.loc[s, 'auc_own']:.2f}" if s in f.index else "--"
        rows_b.append([esc(name), auc, pp(imp.estimate), ci(err.estimate, err.se), ci(fe.estimate, fe.se),
                       ci(sl.estimate, sl.se)])
    groups_b = {0: "Truth from quasi-random assignment", 1: "Text-embedding evaluators"}
    le, lc = lm["estimates"], lm["contrasts"]
    le, lc = le[(le["sample"] == smp) & (le.outcome == o)], lc[(lc["sample"] == smp) & (lc.outcome == o)]
    la = lm["auc"][lm["auc"].tech_center == smp].set_index("evaluator")
    groups_b[len(rows_b)] = "LLM evaluators (Qwen3-8B, asked whether the application will be granted)"
    for s, name in LLM_ROWS.items():
        imp = le[(le.series == s) & (le.kind == "implied") & (le.group == "All")].iloc[0]
        err = le[(le.series == s) & (le.kind == "error") & (le.group == "All")].iloc[0]
        fe = le[(le.series == s) & (le.kind == "error") & (le.group == "Female")].iloc[0]
        sl = lc[(lc.series == s) & (lc.kind == "error") & (lc.contrast == "Small − large")].iloc[0]
        rows_b.append([name, f"{la.loc[s, 'auc_grant']:.2f}", pp(imp.estimate), ci(err.estimate, err.se),
                       ci(fe.estimate, fe.se), ci(sl.estimate, sl.se)])
    tabular(TEX / "tab_validation_b.tex", "@{}lrrrrr@{}",
            ["Trained on (inputs)", "AUC", "Marginal renewal", "Signed error", "Error, female",
             "Error, small $-$ large"], rows_b, groups=groups_b)

    N["margRenew"], N["margRenewCI"] = pp(h.estimate), ci(h.estimate, h.se).split(" ", 1)[1]
    N["meanRenewGranted"] = pp(h.mean_granted)
    for key, s in [("Grant", "Train on grant"), ("First", "Train on first action"),
                   ("Renew", "Train on renewal (granted only)"), ("Cites", "Train on citations (granted only)"),
                   ("None", NO_MODEL), ("RenewEntity", "Train on renewal + entity status"),
                   ("GrantEntity", "Train on grant + entity status")]:
        err = get(s, "error")
        N[f"err{key}"], N[f"err{key}CI"] = pp(err.estimate), ci(err.estimate, err.se).split(" ", 1)[1]
        N[f"imp{key}"] = pp(get(s, "implied").estimate)
        fe, fm = get(s, "error", "Female"), contrast(s, "error", "Female − male")
        N[f"err{key}Female"], N[f"err{key}FemaleCI"] = pp(fe.estimate), ci(fe.estimate, fe.se).split(" ", 1)[1]
        N[f"err{key}FM"], N[f"err{key}FMCI"] = pp(fm.estimate), ci(fm.estimate, fm.se).split(" ", 1)[1]
        sl = contrast(s, "error", "Small − large")
        N[f"err{key}SL"], N[f"err{key}SLCI"] = pp(sl.estimate), ci(sl.estimate, sl.se).split(" ", 1)[1]
    for g, key in [("Female", "Female"), ("Male", "Male"), ("Small entity", "Small"), ("Large entity", "Large")]:
        N[f"marg{key}"] = pp(get(HUMAN, "marginal", g).estimate)
    for name, key in [("Female − male", "FM"), ("Small − large", "SL")]:
        t = contrast(HUMAN, "marginal", name)
        N[f"marg{key}"], N[f"marg{key}CI"] = pp(t.estimate), ci(t.estimate, t.se).split(" ", 1)[1]

    # appendix: marginal outcome and signed errors by group, both outcomes and centers
    cols = ["All", "Female", "Male", "Female − male", "Small entity", "Large entity", "Small − large"]
    for o, oname in [("renew_4y", "renew"), ("cites_hi", "cites")]:
        rows, groups = [], {}
        for smp in ["3600", "1600"]:
            e = est[(est["sample"] == smp) & (est.outcome == o)]
            c = con[(con["sample"] == smp) & (con.outcome == o)]
            groups[len(rows)] = f"Technology center {smp}"
            mg = e[e.series == HUMAN].set_index("group")["mean_granted"]
            rows.append(["Mean among granted"] + [
                pp(mg[g]) if g in mg else pp(mg[{"Female − male": "Female", "Small − large": "Small entity"}[g]]
                                             - mg[{"Female − male": "Male", "Small − large": "Large entity"}[g]])
                for g in cols])
            for s, kind in [(HUMAN, "marginal")] + [(s, "error") for s in order]:
                cells = []
                for g in cols:
                    if g in ("Female − male", "Small − large"):
                        t = c[(c.series == s) & (c.kind == kind) & (c.contrast == g)].iloc[0]
                    else:
                        t = e[(e.series == s) & (e.kind == kind) & (e.group == g)].iloc[0]
                    cells.append(ci(t.estimate, t.se))
                if kind == "marginal":
                    lab = "Marginal outcome (truth)"
                elif s == NO_MODEL:
                    lab = "Error: no model"
                else:
                    t, inp = short(s)
                    lab = f"Error: {t.lower()} ({inp[0].lower() + inp[1:]})"
                rows.append([esc(lab)] + cells)
        tabular(TEX / f"tab_groups_{oname}.tex", "l" + "r" * len(cols),
                ["", "All", "Female", "Male", "F $-$ M", "Small", "Large", "S $-$ L"], rows, groups)

    # appendix: probes with bootstrap intervals
    rows, groups = [], {}
    for tc in ["3600", "1600"]:
        b = boot[boot.tech_center == tc]
        p = probes[tc].set_index("spec")
        groups[len(rows)] = f"Technology center {tc}"
        for _, r in b.iterrows():
            if r.spec.startswith("Contrast: "):
                lab = "Contrast: " + spec_label(r.spec[len("Contrast: "):])
                rows.append([esc(lab), "", "", ci_lohi(r.gap, r.gap_lo, r.gap_hi), "", ""])
                continue
            auc = f"{p.loc[r.spec, 'auc_own']:.3f}" if r.spec in p.index and pd.notna(p.loc[r.spec, "auc_own"]) else ""
            cval = f"{p.loc[r.spec, 'C']:g}" if r.spec in p.index and pd.notna(p.loc[r.spec, "C"]) else ""
            ratio = "" if r.spec == "Human examiners" or "first action)" in r.spec else ci_lohi(
                r.ratio, r.ratio_lo, r.ratio_hi, 2, 1)
            rows.append([esc(spec_label(r.spec)), cval, auc, ci_lohi(r.gap, r.gap_lo, r.gap_hi),
                         ci_lohi(r.diff_vs_human, r.diff_lo, r.diff_hi) if r.spec != "Human examiners" else "",
                         ratio])
    tabular(TEX / "tab_probes.tex", r">{\raggedright\arraybackslash}p{4.7cm}rrrrr",
            ["", "$C$", "AUC", "Gender gap", "Difference from human", "Ratio"],
            rows, groups)
    b3 = boot[boot.tech_center == "3600"].set_index("spec")
    for key, s in [("Grant", "Train on grant"), ("First", "Train on first action"),
                   ("Renew", "Train on renewal (granted only)"), ("Cites", "Train on citations (granted only)"),
                   ("Process", "Train on grant + prosecution history"), ("Names", "Train on grant + names")]:
        r = b3.loc[s]
        N[f"ratio{key}"] = f"{r.ratio:.2f}".replace("-", "$-$")
        N[f"ratio{key}CI"] = f"[{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]".replace("-", "$-$")
        N[f"gap{key}"] = pp(r.gap)
    hb = b3.loc["Human examiners"]
    N["humanGap"], N["humanGapCI"] = pp(hb.gap), f"[{pp(hb.gap_lo)}, {pp(hb.gap_hi)}]"
    hf = b3.loc["Human examiners (first action)"]
    N["humanGapFirst"] = pp(hf.gap)
    for key, s in [("GrantCites", "Contrast: Train on grant − Train on citations (granted only)"),
                   ("Teachers", "Contrast: Train on grant, lenient examiners − Train on grant, strict examiners"),
                   ("GapTeachers", "Contrast: Train on grant, high-gap examiners − Train on grant, low-gap examiners"),
                   ("GrantProcess", "Contrast: Train on grant − Train on grant + prosecution history")]:
        r = b3.loc[s]
        N[f"contrast{key}"], N[f"contrast{key}CI"] = pp(r.gap), f"[{pp(r.gap_lo)}, {pp(r.gap_hi)}]"
    leak = probes["3600"].set_index("spec").loc["Text predicts inventor gender (proxy leakage)"]
    N["proxyAUC"] = f"{leak.auc_own:.2f}"
    nm = probes["3600"].set_index("spec").loc["Train on grant + names"]
    N["auditGrantNames"] = pp(nm.audit_f_minus_m, 2)

    # appendix: LLM prompting
    lab = {"Human examiners": "Human examiners", "zero_noname": "Zero-shot, no names",
           "zero_f": "Zero-shot, female first name", "zero_m": "Zero-shot, male first name",
           "few_lenient": "Few-shot, examples from lenient examiners",
           "few_strict": "Few-shot, examples from strict examiners"}
    rows, groups = [], {}
    for tc in ["3600", "1600"]:
        x = llm[llm.tech_center == tc]
        groups[len(rows)] = f"Technology center {tc} ({int(probes[tc].n_test.iloc[0]):,} test applications)"
        for _, r in x.iterrows():
            if r.cond in lab:
                rows.append([esc(lab[r.cond]), f"{r.auc_granted:.3f}" if pd.notna(r.auc_granted) else "",
                             ci(r.gap_decision, r.se), f"{r.ratio:.2f}".replace("-", "$-$")])
            else:
                unit = "log-odds" if "log" in r.cond else "probability"
                nd = 3 if unit == "log-odds" else 4
                rows.append([esc(f"Name swap, female $-$ male ({unit})").replace(r"\$", "$"), "",
                             ci(r.mean_score, r.se, nd, 1), ""])
    tabular(TEX / "tab_llm.tex", "lrrr", ["", "AUC (grant)", "Gender gap or difference", "Ratio"], rows, groups)
    lc = llm_ci[llm_ci.tech_center == "3600"].set_index("contrast")
    r = lc.loc["few_lenient − human examiners"]
    N["llmLenientVsHuman"], N["llmLenientVsHumanCI"] = f"{r['diff']:.1f}", f"[{r.lo:.1f}, {r.hi:.1f}]"
    r = lc.loc["Strict-examiner examples − lenient-examiner examples"]
    N["llmStrictVsLenient"] = f"{r['diff']:.1f}".replace("-", "$-$")
    N["llmStrictVsLenientCI"] = f"[{r.lo:.1f}, {r.hi:.1f}]".replace("-", "$-$")
    x = llm[llm.tech_center == "3600"].set_index("cond")
    N["llmZeroGap"] = pp(x.loc["zero_noname", "gap_decision"])
    N["llmZeroAUC"] = f"{x.loc['zero_noname', 'auc_granted']:.2f}"
    swap = x[x.index.str.contains("log")].iloc[0]
    N["llmSwapLogOdds"] = f"{swap.mean_score:.2f}"
    N["llmN"] = f"{int(probes['3600'].n_test.iloc[0]) + int(probes['1600'].n_test.iloc[0]):,}"

    # appendix: decision–outcome curve
    rows, groups = [], {}
    for smp in ["3600", "1600"]:
        x = cur[cur["sample"] == smp].sort_values("weight_on_grant")
        groups[len(rows)] = f"Technology center {smp} (human grant gap {pp(x.human_gap.iloc[0])} pp)"
        for _, r in x.iterrows():
            rows.append([f"{r.weight_on_grant:.1f}", ci(r.gap_decision, r.gap_se), f"{r.ratio:.2f}".replace("-", "$-$"),
                         ci(r.renew_4y_error, r.renew_4y_error_se),
                         ci(r.renew_4y_error_f_minus_m, r.renew_4y_error_f_minus_m_se),
                         ci(r.renew_4y_error_s_minus_l, r.renew_4y_error_s_minus_l_se)])
    tabular(TEX / "tab_curve.tex", "rrrrrr", ["Weight on grant", "Decision gap", "Ratio", "Renewal error",
                                              "Error, F $-$ M", "Error, S $-$ L"], rows, groups)
    x = cur[cur["sample"] == "3600"].sort_values("weight_on_grant")
    N["curveRatioRenew"], N["curveRatioGrant"] = f"{x.ratio.iloc[0]:.2f}", f"{x.ratio.iloc[-1]:.2f}"
    N["curveRatioMax"] = f"{x.ratio.max():.2f}"
    N["curveErrRenew"], N["curveErrGrant"] = pp(x.renew_4y_error.iloc[0]), pp(x.renew_4y_error.iloc[-1])
    N["curveErrMin"] = pp(x.renew_4y_error.min())

    # appendix: balance
    rows, groups = [], {}
    vars_ = ["Female first inventor", "Small entity", "Inventors", "Predicted grant score (text, log-odds)",
             "Predicted renewal: Train on grant", "Predicted renewal: Train on renewal (granted only)",
             "Predicted renewal: Train on renewal + entity status",
             "Predicted renewal: Train on grant + prosecution history"]
    for cells in ["Art unit × year", "Art unit × year × USPC class"]:
        groups[len(rows)] = f"Cells: {cells}"
        for v in vars_:
            t = bal[(bal.cells == cells) & (bal.variable == v)].set_index("sample")
            sc, nd = (1, 3) if v in ("Inventors", "Predicted grant score (text, log-odds)") else (100, 2)
            name = v.replace("Predicted renewal: Train on ", "Pred. renewal: ").replace(" (granted only)", "")
            name = name.replace("Predicted grant score (text, log-odds)", "Pred. grant score (log-odds)")
            name = name + r"$^\dagger$" if "prosecution" in v else name
            rows.append([esc(name).replace(r"\$", "$")] + [ci(t.loc[s, "coef_per_10pp"], t.loc[s, "se"], nd, sc)
                                                         for s in ["3600", "1600", "Both"]])
    tabular(TEX / "tab_balance.tex", "lrrr", ["", "TC 3600", "TC 1600", "Both"], rows, groups)
    for cells, key in [("Art unit × year", "AU"), ("Art unit × year × USPC class", "Cls")]:
        for v, vk, sc, nd in [("Small entity", "Small", 100, 2), ("Female first inventor", "Female", 100, 2),
                              ("Predicted grant score (text, log-odds)", "Score", 1, 3)]:
            t = bal[(bal.cells == cells) & (bal.variable == v) & (bal["sample"] == "3600")].iloc[0]
            N[f"bal{vk}{key}"] = pp(t.coef_per_10pp, nd, sc)

    # appendix: first stage
    rows = []
    for (g, z), t in fs.groupby(["group", "instrument"], sort=False):
        t = t.set_index("sample")
        zl = z.replace("Leave-one-out grant rate", "Leave-one-out").replace("Grant rate on ", "Reverse: ")
        zl = zl.replace(" applications (reverse)", "")
        rows.append([esc(g.replace("Predicted grant, quartile", "Pred. grant, quartile")), esc(zl)] +
                    [ci(t.loc[s, "coef"], t.loc[s, "se"], 3, 1) for s in ["3600", "1600", "Both"]])
    tabular(TEX / "tab_first_stage.tex", "llrrr", ["Group", "Instrument", "TC 3600", "TC 1600", "Both"], rows)
    t = fs[fs["sample"] == "3600"]
    N["fsMin"] = f"{t.coef.min():.2f}"
    N["fsAll"] = f"{t[(t.group == 'All') & t.instrument.str.startswith('Leave')].coef.iloc[0]:.2f}"

    # appendix: exclusion
    rows = []
    for v, t in exc.groupby("variable", sort=False):
        t = t.set_index("sample")
        rows.append([esc(v)] + [f"{ci(t.loc[s, 'coef_per_10pp'], t.loc[s, 'se'], 2, 1)} ({t.loc[s, 'mean']:.2f})"
                                for s in ["3600", "1600", "Both"]])
    for o, olab in [("renew_4y", "Implied bias: renewal (pp)"), ("cites_hi", "Implied bias: high citations (pp)")]:
        t = exb[exb.outcome == o].set_index("sample")
        rows.append([olab] + [f"+{100 * t.loc[s, 'implied_bias']:.1f}" for s in ["3600", "1600", "Both"]])
    tabular(TEX / "tab_exclusion.tex", "lrrr", ["", "TC 3600", "TC 1600", "Both"], rows)
    t = exc[exc["sample"] == "3600"].set_index("variable")
    N["claimsPerTen"] = f"{t.loc['Claims at grant', 'coef_per_10pp']:.2f}"
    N["claimsMean"] = f"{t.loc['Claims at grant', 'mean']:.1f}"
    N["pendencyPerTen"] = f"{-t.loc['Pendency to grant (years)', 'coef_per_10pp']:.2f}"
    N["pendencyMean"] = f"{t.loc['Pendency to grant (years)', 'mean']:.1f}"
    t = exb[exb["sample"] == "3600"].set_index("outcome")
    N["biasRenew"], N["biasCites"] = f"{100 * t.loc['renew_4y', 'implied_bias']:.1f}", \
        f"{100 * t.loc['cites_hi', 'implied_bias']:.1f}"

    # appendix: robustness
    rows, groups = [], {}
    for (o, smp), t in rob.groupby(["outcome", "sample"], sort=False):
        if smp == "Both":
            continue
        groups[len(rows)] = f"{'Renewal' if o == 'renew_4y' else 'High citations'}, technology center {smp}"
        for _, r in t.iterrows():
            rows.append([esc(r.cells.replace("Art unit × year", "AU × yr").replace(" × USPC ", " × ")),
                         r.controls, "first action" if r.instrument.startswith("first") else "grant",
                         f"{r.first_stage:.2f}", ci(r.marginal, r.se), ci(r.f_minus_m, r.f_minus_m_se),
                         ci(r.s_minus_l, r.s_minus_l_se)])
    tabular(TEX / "tab_robust.tex", "lllrrrr", ["Cells", "Controls", "Leniency", "First stage", "Marginal",
                                                "F $-$ M", "S $-$ L"], rows, groups)

    # appendix: LLM evaluators on the marginal applications (renewal), by group, and their decision gaps
    rows, groups = [], {}
    for smp in ["3600", "1600"]:
        e = lm["estimates"][(lm["estimates"]["sample"] == smp) & (lm["estimates"].outcome == "renew_4y")]
        c = lm["contrasts"][(lm["contrasts"]["sample"] == smp) & (lm["contrasts"].outcome == "renew_4y")]
        groups[len(rows)] = f"Technology center {smp}"
        for s, kind in [(HUMAN, "marginal"), (LLM_NO_MODEL, "error")] + [(s, "error") for s in LLM_ROWS]:
            cells = []
            for g in cols:
                is_contrast = g in ("Female − male", "Small − large")
                t = (c[(c.series == s) & (c.kind == kind) & (c.contrast == g)] if is_contrast
                     else e[(e.series == s) & (e.kind == kind) & (e.group == g)]).iloc[0]
                cells.append(ci(t.estimate, t.se))
            lab = ("Marginal outcome (truth)" if kind == "marginal" else
                   "Error: no model" if s == LLM_NO_MODEL else f"Error: {LLM_ROWS[s][5:]}")
            rows.append([esc(lab)] + cells)
    tabular(TEX / "tab_llm_marginal.tex", "l" + "r" * len(cols),
            ["", "All", "Female", "Male", "F $-$ M", "Small", "Large", "S $-$ L"], rows, groups)
    gp, au = lm["gaps"], lm["auc"]
    rows = []
    for s in ["Human decision makers"] + list(LLM_ROWS):
        g3 = gp[(gp["sample"] == "3600") & (gp.series == s)].iloc[0]
        g1 = gp[(gp["sample"] == "1600") & (gp.series == s)].iloc[0]
        a3 = au[(au.tech_center == "3600") & (au.evaluator == s)]
        name = "Human examiners" if s.startswith("Human") else LLM_ROWS[s][5:].capitalize()
        rows.append([esc(name), ci(g3.gap, g3.se), f"{g3.ratio:.2f}", ci(g1.gap, g1.se),
                     f"{a3.iloc[0].auc_grant:.3f}" if len(a3) else "--",
                     f"{a3.iloc[0].auc_renewal:.3f}" if len(a3) else "--"])
    tabular(TEX / "tab_llm_gaps.tex", "lrrrrr",
            ["", "Gap, TC 3600", "Ratio", "Gap, TC 1600", "AUC: grant", "AUC: renewal"], rows)
    e = lm["estimates"][(lm["estimates"]["sample"] == "3600") & (lm["estimates"].outcome == "renew_4y")]
    for key, s in [("Zero", "Zero-shot"), ("Len", "Few-shot, lenient examiners' decisions"),
                   ("Str", "Few-shot, strict examiners' decisions")]:
        err = e[(e.series == s) & (e.kind == "error") & (e.group == "All")].iloc[0]
        fe = e[(e.series == s) & (e.kind == "error") & (e.group == "Female")].iloc[0]
        N[f"llmM{key}Imp"] = pp(e[(e.series == s) & (e.kind == "implied") & (e.group == "All")].iloc[0].estimate)
        N[f"llmM{key}Err"], N[f"llmM{key}ErrCI"] = pp(err.estimate), ci(err.estimate, err.se).split(" ", 1)[1]
        N[f"llmM{key}Female"], N[f"llmM{key}FemaleCI"] = pp(fe.estimate), ci(fe.estimate, fe.se).split(" ", 1)[1]
        g = gp[(gp["sample"] == "3600") & (gp.series == s)].iloc[0]
        N[f"llmM{key}Ratio"] = f"{g.ratio:.2f}"
        N[f"llmM{key}AUC"] = f"{au[(au.tech_center == '3600') & (au.evaluator == s)].iloc[0].auc_grant:.2f}"

    # appendix: AI examiners (17)
    ax = pd.read_csv(T / "ai_examiners.csv")
    al = pd.read_csv(T / "ai_examiners_lottery.csv").set_index("group")
    asum = pd.read_csv(T / "ai_examiners_summary.csv").iloc[0]
    # no row per examiner (the busiest examiners can be rebuilt from public data): four groups of five twins by the
    # examiner's leniency, ranges rounded to 5 points, medians otherwise
    ax = ax.sort_values("human_leniency_loo").reset_index(drop=True)
    ax["grp"] = np.arange(len(ax)) * 4 // len(ax)
    labels = ["Strictest", "Second", "Third", "Most lenient"]

    def band(v):
        lo, hi = 5 * np.floor(100 * v.min() / 5), 5 * np.ceil(100 * v.max() / 5)
        return f"{lo:.0f}–{hi:.0f}"
    rows = [[labels[k], band(g.human_leniency_loo), f"{g.auc_leniency_only.median():.2f}", f"{g.auc_full.median():.2f}",
             band(g.ai_grant_rate_leniency), pp(g.ai_gap_leniency.median())] for k, g in ax.groupby("grp")]
    tabular(TEX / "tab_ai_examiners.tex", "lrrrrr",
            ["Five twins each", "Examiners' leniency (\\%)", "AUC (median)", "AUC, text (median)",
             "Twins' grant rate (\\%)", "Twins' gender gap (pp, median)"], rows)
    rows = [[g, f"{int(al.loc[g, 'n']):,}", pp(al.loc[g, "granted_by_all"]), pp(al.loc[g, "granted_by_none"]),
             pp(al.loc[g, "depends_on_examiner"]), pp(al.loc[g, "depends_with_text"])]
            for g in ["All", "Female", "Male"]]
    tabular(TEX / "tab_ai_lottery.tex", "lrrrrr",
            ["First inventor", "Applications", "All grant", "None grant", "Depends", "Depends, text"], rows)
    N["aiN"], N["aiCommon"] = f"{int(asum.n_examiners)}", f"{int(asum.n_common):,}"
    N["aiTrainMedian"] = f"{int(asum.n_train_median)}"
    N["aiAUC"], N["aiAUCText"] = f"{asum.auc_leniency_median:.2f}", f"{asum.auc_text_median:.2f}"
    N["aiCorr"], N["aiAgreement"] = f"{asum.corr_leniency:.2f}", pp(asum.agreement_median)
    N["aiGrantMin"], N["aiGrantMax"] = pp(asum.ai_grant_min), pp(asum.ai_grant_max)
    N["aiGapMedian"], N["aiHumanGap"] = pp(asum.ai_gap_median), pp(asum.human_gap)
    N["aiHumanGapCI"] = ci(asum.human_gap, asum.human_gap_se).split(" ", 1)[1]
    for g, key in [("All", ""), ("Female", "Female"), ("Male", "Male")]:
        N[f"aiDepends{key}"] = pp(al.loc[g, "depends_on_examiner"])
        N[f"aiNone{key}"], N[f"aiAll{key}"] = pp(al.loc[g, "granted_by_none"]), pp(al.loc[g, "granted_by_all"])

    # appendix: evaluators on Qwen3-8B representations (15), side by side with Qwen3-Embedding-0.6B (12)
    cq = pd.read_csv(T / "llm_features_comparison.csv", dtype={"tech_center": str})
    gq = pd.read_csv(T / "llm_features_gaps.csv", dtype={"sample": str})
    rows, groups = [], {}
    for tc in ["3600", "1600"]:
        groups[len(rows)] = f"Technology center {tc}"
        for r in cq[cq.tech_center == tc].itertuples():
            g = gq[(gq["sample"] == tc) & (gq.series == f"Qwen3-8B: {r.series}")].iloc[0]
            rows.append([esc(spec_label(r.series)), f"{r.auc_06b:.3f}", f"{r.auc_8b:.3f}",
                         ci(r.renew_err_all_06b, r.renew_err_all_06b_se), ci(r.renew_err_all_8b, r.renew_err_all_8b_se),
                         ci(r.renew_err_female_8b, r.renew_err_female_8b_se), f"{g.ratio:.2f}".replace("-", "$-$")])
    tabular(TEX / "tab_llm_features.tex", "lrrrrrr",
            ["Trained on", "AUC, 0.6B", "AUC, 8B", "Error, 0.6B", "Error, 8B", "Error, female, 8B", "Gap ratio, 8B"],
            rows, groups)
    c3 = cq[cq.tech_center == "3600"].set_index("series")
    for key, s_ in [("First", "Train on first action"), ("Grant", "Train on grant"),
                    ("Renew", "Train on renewal (granted only)"), ("Cites", "Train on citations (granted only)")]:
        N[f"qAUC{key}"] = f"{c3.loc[s_, 'auc_8b']:.2f}"
        N[f"qRatio{key}"] = f"{gq[(gq['sample'] == '3600') & (gq.series == f'Qwen3-8B: {s_}')].iloc[0].ratio:.2f}"
    N["qErrGrant"], N["qErrGrantFemale"] = pp(c3.loc["Train on grant", "renew_err_all_8b"]), \
        pp(c3.loc["Train on grant", "renew_err_female_8b"])
    gain = cq.auc_8b - cq.auc_06b
    N["qGainMin"], N["qGainMax"] = f"{gain.min():.2f}", f"{gain.max():.2f}"

    # appendix: sample (measures in rows, technology centers in columns)
    cols = {}
    for tc, s in sample.groupby("tech_center"):
        g = s[s.granted == 1]
        cols[tc] = [f"{len(s):,}", f"{s.examiner_id.nunique():,}", f"{s.au_year.nunique():,}", pp(s.female.mean()),
                    pp(s.small_entity.mean()), f"{s.n_inventors.mean():.2f}", pp(s.fa_allow.mean()),
                    pp(s.granted.mean()), f"{s.n_rejections.mean():.2f}", pp(g.renew_4y.mean()),
                    f"{g.cites_5y.mean():.2f}"]
    labels = ["Applications", "Examiners", "Art unit $\\times$ filing-year cells", "Female first inventor (\\%)",
              "Small entity (\\%)", "Inventors per application", "Allowed at the first action (\\%)",
              "Granted (\\%)", "Rejections per application", "Renewed at year 4 (\\% of granted)",
              "Citations within 5 years of grant (mean, granted)"]
    rows = [[lab, cols["3600"][i], cols["1600"][i]] for i, lab in enumerate(labels)]
    tabular(TEX / "tab_sample.tex", "lrr", ["", "TC 3600", "TC 1600"], rows)
    N["nApps"] = f"{len(sample):,}"
    N["nExaminers"] = f"{sample.examiner_id.nunique():,}"
    N["nAppsThreeSix"] = f"{(sample.tech_center == '3600').sum():,}"
    N["nAppsOneSix"] = f"{(sample.tech_center == '1600').sum():,}"
    N["femaleShareThreeSix"] = pp(sample[sample.tech_center == "3600"].female.mean())
    return N


def assignment(N: dict) -> None:
    """Appendix: random-assignment checks of 18 (permutation balance test, residualized leniency, subclass cells)."""
    sm = pd.read_csv(T / "assignment_permutation_summary.csv", dtype={"tech_center": str})
    est = pd.read_csv(T / "assignment_resid_estimates.csv", dtype={"sample": str})
    fs = pd.read_csv(T / "assignment_resid_first_stage.csv", dtype={"sample": str})

    def perm(cells: str, lab: str, index: str, tc: str):
        t = sm[(sm.cells == cells) & (sm.leniency == lab) & (sm["index"] == index) & (sm.tech_center == tc)]
        return t.iloc[0] if len(t) else None

    AU, CLS, SUB = "Art unit × year", "Art unit × year × USPC class", "Art unit × year × USPC subclass"
    rows = []
    for cells, lab in [(AU, "leave-one-out"), (CLS, "leave-one-out"), (SUB, "leave-one-out"), (CLS, "residualized")]:
        b = perm(cells, lab, "index_grant", "Both")
        name = esc(cells.replace(AU, "AU × yr").replace(" × USPC ", " × ")).replace("×", r"$\times$")
        r = [name + (", residualized" if lab == "residualized" else ""), f"{int(b.n_cells):,}",
             pp(b.share_of_pilot, 0), f"{b.mean_examiners:.1f}"]
        for index in ["index_grant", "index_renew"]:
            for tc in ["3600", "1600"]:
                t = perm(cells, lab, index, tc)
                r.append(pp(t.p_below_05) if t is not None else "--")
        rows.append(r)
    tabular(TEX / "tab_permutation.tex", "lrrrrrrr",
            ["Cells", "Cells tested", r"Pilot (\%)", "Examiners", "Grant: 3600", "1600", "Renewal: 3600", "1600"],
            rows)
    for key, cells, lab in [("AU", AU, "leave-one-out"), ("Cls", CLS, "leave-one-out"), ("Sub", SUB, "leave-one-out"),
                            ("Resid", CLS, "residualized")]:
        for tk, tc in [("", "Both"), ("ThreeSix", "3600"), ("OneSix", "1600")]:
            N[f"perm{key}{tk}"] = pp(perm(cells, lab, "index_grant", tc).p_below_05)
        if lab != "residualized":
            N[f"perm{key}Renew"] = pp(perm(cells, lab, "index_renew", "Both").p_below_05)
        b = perm(cells, lab, "index_grant", "Both")
        N[f"perm{key}Cells"], N[f"perm{key}Pilot"] = f"{int(b.n_cells):,}", pp(b.share_of_pilot, 0)
        N[f"perm{key}Examiners"], N[f"perm{key}Top"] = f"{b.mean_examiners:.1f}", pp(b.p_above_95)
        N[f"perm{key}MeanP"] = f"{b.mean_p:.2f}"

    # the marginal check under the three designs, renewal
    designs = [("Main", "Leave-one-out grant rate (04)"), ("Resid", "Residualized leave-one-out (cell × entity size)"),
               ("Sub", "Leave-one-out, art unit × year × USPC subclass cells")]
    errs = [("Train on grant", "All", "Error: train on grant"),
            ("Train on grant", "Female", "Error: train on grant, female inventor"),
            ("Train on renewal (granted only)", "All", "Error: train on renewal"),
            ("Train on citations (granted only)", "All", "Error: train on citations"),
            (NO_MODEL, "All", "Error: no model")]
    rows, groups = [], {}
    for smp, slab in [("3600", "Technology center 3600"), ("Both", "Both technology centers")]:
        groups[len(rows)] = slab
        f = fs[fs["sample"] == smp].set_index("instrument")
        e = est[(est["sample"] == smp) & (est.outcome == "renew_4y")]
        def get(z: str, series: str, kind: str, group: str):
            return e[(e.instrument == z) & (e.series == series) & (e.kind == kind) & (e.group == group)].iloc[0]

        def bal(z: str, v: str) -> str:
            return ci(f.loc[z, f"bal_{v}"], f.loc[z, f"bal_{v}_se"], 2)
        rows.append(["First stage"] + [f"{f.loc[z, 'first_stage']:.2f}" for _, z in designs])
        rows.append(["Balance: grant index (pp per SD)"] + [bal(z, "index_grant") for _, z in designs])
        rows.append(["Balance: small entity (pp per SD)"] + [bal(z, "small_entity") for _, z in designs])
        cells = [(HUMAN, "All", "marginal", "Marginal renewal (truth, \\%)")] + [(s_, g, "error", lab)
                                                                               for s_, g, lab in errs]
        for series, g, kind, lab in cells:
            rows.append([lab] + [ci(get(z, series, kind, g).estimate, get(z, series, kind, g).se) for _, z in designs])
        if smp == "3600":
            for key, z in designs:
                N[f"rb{key}BalIdx"], N[f"rb{key}BalIdxCI"] = pp(f.loc[z, "bal_index_grant"], 2), \
                    bal(z, "index_grant").split(" ", 1)[1]
                N[f"rb{key}Truth"] = pp(get(z, HUMAN, "marginal", "All").estimate)
                for ek, (series, g, _) in zip(["ErrGrant", "ErrGrantFemale", "ErrRenew", "ErrCites", "ErrNoModel"],
                                              errs):
                    t = get(z, series, "error", g)
                    N[f"rb{key}{ek}"], N[f"rb{key}{ek}CI"] = pp(t.estimate), ci(t.estimate, t.se).split(" ", 1)[1]
    tabular(TEX / "tab_assignment_robust.tex", "lrrr",
            ["", "Main design", "Residualized leniency", "Subclass cells"], rows, groups)
    # adding the grant index as a control: largest change in any renewal estimate
    base = est[~est.instrument.str.endswith(" + grant index control")]
    ctrl = est[est.instrument.str.endswith(" + grant index control")].copy()
    ctrl["instrument"] = ctrl.instrument.str.replace(" + grant index control", "", regex=False)
    k = ["instrument", "sample", "outcome", "group", "series", "kind"]
    mm = base.merge(ctrl, on=k, suffixes=("", "_c"))
    mm = mm[mm.outcome == "renew_4y"]
    N["idxCtrlMaxChange"] = pp((mm.estimate - mm.estimate_c).abs().max())


def outcome_test(N: dict) -> None:
    """The examiners' own record under each criterion: group differences in the marginal outcome (12)."""
    c = pd.read_csv(T / "marginal_contrasts.csv", dtype={"sample": str})
    c = c[(c.series == HUMAN) & (c.kind == "marginal")]
    for smp, sk in [("3600", "ThreeSix"), ("1600", "OneSix")]:
        for o, ok in [("renew_4y", "Renew"), ("cites_hi", "Cites")]:
            for con, ck in [("Female − male", "FM"), ("Small − large", "SL")]:
                t = c[(c["sample"] == smp) & (c.outcome == o) & (c.contrast == con)].iloc[0]
                N[f"hum{ck}{ok}{sk}"] = pp(t.estimate)
                N[f"hum{ck}{ok}{sk}CI"] = ci(t.estimate, t.se).split(" ", 1)[1]


def contraction(N: dict) -> None:
    """Appendix: contraction of the most lenient examiners' grants (19)."""
    qt = pd.read_csv(T / "contraction_quintiles.csv", dtype={"sample": str})
    et = pd.read_csv(T / "contraction_evaluators.csv", dtype={"sample": str, "quintile": str})
    CLS = "Art unit × year × USPC class"
    names = {"Examiners": "Stricter examiners", "Random": "Random", "Train on grant": "Trained on grant",
             "Train on first action": "Trained on first action",
             "Train on renewal (granted only)": "Trained on renewal",
             "Train on citations (granted only)": "Trained on citations", "LLM, zero-shot": "LLM, zero-shot"}
    keys = {"Examiners": "Exam", "Random": "Random", "Train on grant": "Grant", "Train on first action": "First",
            "Train on renewal (granted only)": "Renew", "Train on citations (granted only)": "Cites",
            "LLM, zero-shot": "LLM"}
    rows, groups = [], {}
    for smp, slab in [("3600", "Technology center 3600"), ("Both", "Both technology centers")]:
        groups[len(rows)] = slab
        e = et[(et.cells == CLS) & (et["sample"] == smp) & (et.quintile == "2–5 (mean)")]
        for ev, lab in names.items():
            t = e[e.evaluator == ev].set_index("measure")
            r, f = t.loc["renew_rate"], t.loc["female_share"]
            if ev == "Examiners":
                cells = [ci_lohi(r.estimate, r.lo, r.hi), "", ci_lohi(f.estimate, f.lo, f.hi), "", ""]
            else:
                nd = t.loc["need_ratio"]
                cells = [ci_lohi(r.estimate, r.lo, r.hi), ci_lohi(r["diff"], r.diff_lo, r.diff_hi),
                         ci_lohi(f.estimate, f.lo, f.hi), ci_lohi(f["diff"], f.diff_lo, f.diff_hi),
                         ci_lohi(nd.estimate, nd.lo, nd.hi, 2, 1)]
            rows.append([lab] + cells)
            if smp == "3600":
                k = keys[ev]
                N[f"ctr{k}Renew"], N[f"ctr{k}Female"] = pp(r.estimate), pp(f.estimate)
                N[f"ctr{k}RenewCI"] = ci_lohi(r.estimate, r.lo, r.hi).split(" ", 1)[1]
                if ev != "Examiners":
                    N[f"ctr{k}RenewDiff"] = pp(r["diff"])
                    N[f"ctr{k}RenewDiffCI"] = ci_lohi(r["diff"], r.diff_lo, r.diff_hi).split(" ", 1)[1]
                    N[f"ctr{k}FemaleDiff"] = pp(f["diff"])
                    N[f"ctr{k}FemaleDiffCI"] = ci_lohi(f["diff"], f.diff_lo, f.diff_hi).split(" ", 1)[1]
                    N[f"ctr{k}Need"] = f"{t.loc['need_ratio'].estimate:.2f}"
    tabular(TEX / "tab_contraction.tex", "lrrrrr",
            ["Extra rejections by", "Renewal rate (\\%)", "vs examiners (pp)", "Female share (\\%)",
             "vs examiners (pp)", "Rejections needed"], rows, groups)
    q = qt[(qt.cells == CLS) & (qt["sample"] == "3600")].set_index("quintile")
    N["ctrCells"], N["ctrPilot"] = f"{int(q.n_cells.iloc[0]):,}", pp(q.share_of_pilot.iloc[0], 0)
    N["ctrGrantQOne"], N["ctrGrantQFive"] = pp(q.loc[1, "grant_rate"]), pp(q.loc[5, "grant_rate"])
    N["ctrDGMin"], N["ctrDGMax"] = pp(q.loc[2:, "dG"].min()), pp(q.loc[2:, "dG"].max())
    N["ctrGapQOne"], N["ctrGapQFive"] = pp(q.loc[1, "gap"]), pp(q.loc[5, "gap"])
    sub = et[(et.cells != CLS) & (et["sample"] == "3600") & (et.quintile == "2–5 (mean)")].set_index(
        ["evaluator", "measure"])
    for ev in ["Train on grant", "Train on renewal (granted only)"]:
        N[f"ctrSub{keys[ev]}RenewDiff"] = pp(sub.loc[(ev, "renew_rate"), "diff"])
        N[f"ctrSub{keys[ev]}FemaleDiff"] = pp(sub.loc[(ev, "female_share"), "diff"])
    N["ctrSubExamRenew"] = pp(sub.loc[("Examiners", "renew_rate"), "estimate"])
    m = pd.read_csv(T / "marginal_estimates.csv", dtype={"sample": str})
    m = m[(m["sample"] == "3600") & (m.outcome == "renew_4y") & (m.group == "All") & (m.kind == "implied")]
    N["ctrImpRenew"] = pp(m[m.series == "Train on renewal (granted only)"].estimate.iloc[0])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    rd = lambda name: pd.read_csv(T / f"{name}.csv", dtype={"tech_center": str, "sample": str})  # noqa: E731
    boot = rd("probes_bootstrap")
    probes = {tc: rd(f"probes_{tc}") for tc in ["3600", "1600"]}
    llm, llm_ci = rd("llm_prompting_full"), rd("llm_prompting_ci_full")
    m = {k: rd(f"marginal_{k}") for k in ["estimates", "contrasts", "split", "fit", "robustness", "balance",
                                          "first_stage", "exclusion", "exclusion_bias", "curve"]}
    lm = {k: rd(f"llm_marginal_{k}") for k in ["estimates", "contrasts", "gaps", "auc", "balance"]}
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    sample = inp.merge(pd.read_parquet(P / "pilot_sample.parquet"), on="application_number", how="left")

    fig_training_target(boot)
    fig_decision_outcome(m["curve"])
    N = tables(boot, probes, llm, llm_ci, m["estimates"], m["contrasts"], m["split"], m["fit"], m["robustness"],
               m["balance"], m["first_stage"], m["exclusion"], m["exclusion_bias"], m["curve"], sample, lm)
    assignment(N)
    contraction(N)
    outcome_test(N)
    lines = ["% Generated by code/13_figures_tables.py; do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items()]
    (TEX / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(N)} macros; tables: {sorted(p.name for p in TEX.glob('tab_*.tex'))}")


if __name__ == "__main__":
    main()
