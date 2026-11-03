"""LaTeX tables and number macros for the extensions of the citation analysis (30–33, 35–37).

Reads output/tables/*.csv written by 30 (fine-tuned LLM evaluators), 31 (Qwen3-8B representation), 32 (AI twins that
see names), 33 (race and ethnicity from surnames), 35 (C1 as in Kleinberg et al. 2018), 36 (geography and industry),
and 37 (C4: the examiner's career stage); each part is skipped if its results are missing. Writes
output/latex/ext_tab_*.tex (tabular bodies) and output/latex/ext_numbers.tex (macros, prefix \\x...).
Runs on CPUs in seconds.
"""
import importlib.util
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "output/tables"
TEX = ROOT / "output/latex"
HUMAN_DM = "Human decision makers"
HUMAN = "Human record (truth)"
# C1's main specification: the Poisson version of 35 (count models with a Poisson loss). The squared-error
# version, the original specification, keeps its outputs (sfx "_all") and its macros under the prefix \xSq...
C1_MAIN = "_all_poisson"
STATES = {"AK": "Alaska", "MT": "Montana", "NY": "New York", "MN": "Minnesota", "CA": "California", "WY": "Wyoming",
          "VT": "Vermont", "ND": "North Dakota", "SD": "South Dakota", "DC": "the District of Columbia"}
TC_NAMES = {"1600": "Biotechnology, organic chemistry", "1700": "Chemical and materials engineering",
            "2100": "Computer architecture, software, information security",
            "2400": "Networking, multiplexing, cable, security", "2600": "Communications",
            "2800": "Semiconductors, electrical and optical systems",
            "3600": "Transportation, construction, e-commerce, business methods",
            "3700": "Mechanical engineering, manufacturing, medical devices"}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M29 = load_module("m29", "29_citation_tables.py")
tabular, num, cin, rd = M29.tabular, M29.num, M29.cin, M29.rd


def pct(x: float, nd: int = 1) -> str:
    return num(100 * x, nd)


def gap_cell(g: pd.Series, ratio: bool = True) -> str:
    """Gap in decisions (pp) and its ratio to the examiners' gap; the ratio is left out where the examiners' gap is
    near zero."""
    return f"{pct(g.gap)} ({num(g.ratio, 2)}$\\times$)" if ratio else pct(g.gap)


def ethnicity(N: dict) -> None:
    sh = rd("ethnicity_shares").set_index("tech_center")
    sm = rd("ethnicity_summary").set_index(["comparison", "tech_center"])
    g = rd("ethnicity_gaps").set_index(["comparison", "sample", "series"])
    labels = {"asian": "Asian", "hispanic": "Hispanic", "black": "Black"}
    ev = [("Train on grant", "grant"), ("Train on citations (granted only)", "citations"), ("LLM, zero-shot", "LLM")]
    rows = []
    for key, lab in labels.items():
        for tc in ["3600", "1600"]:
            r = sm.loc[(key, tc)]
            sig = abs(r.grant_gap) > 1.96 * r.grant_gap_se
            rows.append([lab if tc == "3600" else "", tc, f"{int(r.n_group):,}",
                         cin(r.grant_gap, r.grant_gap_se, 1, 100)]
                        + [gap_cell(g.loc[(key, tc, s)], sig) for s, _ in ev])
    tabular(TEX / "ext_tab_ethnicity.tex", "llrllll",
            ["Group", "TC", "Applications", "Examiners' gap (pp)", "Grant-trained", "Citation-trained", "LLM"], rows)
    N["xMatched"] = pct(sh.loc["3600", "surname_matched"], 0)
    for key, mac in [("asian", "Asian"), ("hispanic", "Hisp"), ("black", "Black")]:
        r = sm.loc[(key, "3600")]
        N[f"xShare{mac}"] = pct(sh.loc["3600", key])
        N[f"xShare{mac}Bio"] = pct(sh.loc["1600", key])
        N[f"xN{mac}"] = f"{int(r.n_group):,}"
        N[f"xGap{mac}"] = pct(r.grant_gap)
        N[f"xGap{mac}CI"] = f"[{pct(r.grant_gap - 1.96 * r.grant_gap_se)}, {pct(r.grant_gap + 1.96 * r.grant_gap_se)}]"
        for s, short in [("Train on grant", "Grant"), ("Train on citations (granted only)", "Cites"),
                         ("LLM, zero-shot", "LLM")]:
            x = g.loc[(key, "3600", s)]
            N[f"xRatio{mac}{short}"] = num(x.ratio, 2)
            N[f"xGap{mac}{short}"] = pct(x.gap)


def q8(N: dict) -> None:
    fit = rd("q8_citations_fit").set_index(["tech_center", "model"])
    g = rd("q8_citations_gaps").set_index(["sample", "series"])
    rows = []
    for tc in ["3600", "1600"]:
        for model in ["0.6B", "Qwen3-8B"]:
            r = fit.loc[(tc, model)]
            name = "Qwen3-Embedding-0.6B" if model == "0.6B" else "Qwen3-8B (hidden states)"
            rows.append([tc, name, num(r.auc_grant, 3), num(r.corr_cites, 3),
                         gap_cell(g.loc[(tc, f"{model}: Train on grant")]),
                         gap_cell(g.loc[(tc, f"{model}: Train on citations (granted only)")])])
    tabular(TEX / "ext_tab_q8.tex", "llrrll",
            ["TC", "Representation", "AUC, grant", "Corr., log citations", "Gap: grant-trained",
             "Gap: citation-trained"], rows)
    N["xQAUC"], N["xQAUCsmall"] = num(fit.loc[("3600", "Qwen3-8B")].auc_grant, 3), num(
        fit.loc[("3600", "0.6B")].auc_grant, 3)
    N["xQCorr"], N["xQCorrSmall"] = num(fit.loc[("3600", "Qwen3-8B")].corr_cites, 2), num(
        fit.loc[("3600", "0.6B")].corr_cites, 2)
    N["xQRatioGrant"] = num(g.loc[("3600", "Qwen3-8B: Train on grant")].ratio, 2)
    N["xQRatioCites"] = num(g.loc[("3600", "Qwen3-8B: Train on citations (granted only)")].ratio, 2)


def finetune(N: dict) -> None:
    fit = rd("finetune_fit").set_index(["tech_center", "evaluator"])
    gaps = rd("finetune_gaps").set_index(["sample", "series"])
    series = [("LLM fine-tuned on grant", "LLM fine-tuned: grant", "FtGrant"),
              ("LLM fine-tuned on citations", "LLM fine-tuned: citations", "FtCites"),
              ("Train on grant", "Embedding: grant", "EmGrant"),
              ("Train on citations (granted only)", "Embedding: citations", "EmCites"),
              ("LLM, zero-shot", "LLM, zero-shot", "ZeroShot")]
    rows = []
    for tc in ["3600", "1600"]:
        for s, lab, mac in series:
            f = fit.loc[(tc, s)]
            gg = gaps.loc[(tc, s)]
            rows.append([tc, lab, num(f.auc_grant, 3), num(f.corr_log_cites, 3), gap_cell(gg)])
            if tc == "3600":
                N[f"x{mac}AUC"], N[f"x{mac}Corr"] = num(f.auc_grant, 3), num(f.corr_log_cites, 2)
                N[f"x{mac}Gap"], N[f"x{mac}Ratio"] = pct(gg.gap), num(gg.ratio, 2)
    tabular(TEX / "ext_tab_finetune.tex", "llrrl",
            ["TC", "Evaluator", "AUC, grant", "Corr., log citations", "Gender gap (pp)"], rows)


def twins_names(N: dict) -> None:
    s = rd("twins_names_summary").set_index("kind")
    tw = rd("twins_names_by_twin")
    rows = []
    for kind, lab in [("leniency", "Leniency twins (main)"), ("text", "Twins with text dimensions")]:
        r = s.loc[kind]
        rows.append([lab, pct(r.gap_hidden_median), pct(r.gap_shown_median), cin(r.dprob_mean, r.dprob_se, 2, 100),
                     cin(r.dgrant_mean, r.dgrant_se, 2, 100), pct(r.changed_by_name_mean, 2),
                     f"{pct(r.changed_when_shown_female)} / {pct(r.changed_when_shown_male)}"])
    tabular(TEX / "ext_tab_twins_names.tex", "lrrllrr",
            ["AI twins", "Gap, names hidden", "Gap, names shown", "$\\Delta$ probability, F $-$ M name",
             "$\\Delta$ grants, F $-$ M name", "Decisions changed by the name (\\%)",
             "Changed by showing names, F / M (\\%)"], rows)
    r = s.loc["leniency"]
    N["xTwGapHidden"], N["xTwGapShown"] = pct(r.gap_hidden_median), pct(r.gap_shown_median)
    N["xTwDProb"] = num(100 * r.dprob_mean, 2)
    N["xTwDProbCI"] = f"[{num(100 * (r.dprob_mean - 1.96 * r.dprob_se), 2)}, " \
                      f"{num(100 * (r.dprob_mean + 1.96 * r.dprob_se), 2)}]"
    N["xTwDGrant"] = num(100 * r.dgrant_mean, 2)
    N["xTwDGrantCI"] = f"[{num(100 * (r.dgrant_mean - 1.96 * r.dgrant_se), 2)}, " \
                       f"{num(100 * (r.dgrant_mean + 1.96 * r.dgrant_se), 2)}]"
    N["xTwChanged"] = pct(r.changed_by_name_mean, 2)
    N["xTwShownF"], N["xTwShownM"] = pct(r.changed_when_shown_female), pct(r.changed_when_shown_male)
    le = tw[tw.kind == "leniency"]
    N["xTwAUCHidden"], N["xTwAUCShown"] = num(le.auc_hidden.median()), num(le.auc_shown.median())
    sig = ((le.dgrant_female_minus_male - 1.96 * le.dgrant_se > 0) |
           (le.dgrant_female_minus_male + 1.96 * le.dgrant_se < 0)).sum()
    N["xTwSig"] = str(int(sig))
    t = s.loc["text"]
    N["xTwTextDGrant"], N["xTwTextChanged"] = num(100 * t.dgrant_mean, 2), pct(t.changed_by_name_mean, 2)


def kllm(N_out: dict, file_sfx: str = "_all", tc0: str = "all", prefix: str = "xK", tables: bool = True) -> None:
    """C1 as in Kleinberg et al. (2018), extensions, and counterfactual scores (35). Main sample: the eight centers
    pooled (sfx "_all", macros \\xK...); pilot TC 3600 with the fine-tuned LLM as a rule (sfx "", macros \\xP...,
    no tables)."""
    N: dict = {}

    def r(name, tc=tc0):
        x = rd(name.replace("c1_kllm", f"c1_kllm{file_sfx}", 1))
        x["tc"] = x.tc.astype(str)
        return x[x.tc == tc].copy()

    def tabular_(*a, **k):
        if tables:
            tabular(*a, **k)

    def c95(v, lo, hi, s=100, nd=1):
        return f"{num(s * v, nd)} [{num(s * lo, nd)}, {num(s * hi, nd)}]"
    ALG, LEN, PAA, PAS = ("Algorithm (trained on citations)", "Algorithm trained on the most lenient quintile's grants",
                          "Algorithm trained on prior-art citations (all applications)",
                          "Algorithm trained on prior-art citations (granted only)")
    LLM, JDG, RND = "LLM fine-tuned on citations (Qwen3-8B)", "Predicted examiner (trained on grants)", "Random"
    MAT, EQU = "Algorithm, examiners' female share", "Algorithm, equal rates for women and men"
    CF, CFP = "Counterfactual score (text)", "Counterfactual score (text + prior-art citations)"
    rules = [(ALG, "Algorithm", "Algo"), (LEN, "Lenient-trained", "Len"), (PAA, "Prior-art-trained", "Pa"),
             (LLM, "LLM, fine-tuned", "Llm"), (JDG, "Predicted examiner", "Judge"), (RND, "Random", "Rnd")]
    q, c, t4 = r("c1_kllm_quintiles"), r("c1_kllm_contraction"), r("c1_kllm_table4")
    rules = [x for x in rules if (c.rule == x[0]).any()]
    for x in (q, c):
        x["quintile"] = x.quintile.astype(str)
    fit, perm, f4 = r("c1_kllm_fit").iloc[0], r("c1_kllm_permutation").iloc[0], r("c1_kllm_fig4")
    qc = q[(q["sample"] == "test + imputation") & (q.outcome == "cites")].set_index("quintile")
    cc = c[(c.outcome == "cites") & (c["sample"] == "test + imputation")]

    # Table III / VIII
    rows = []
    for k in ["2", "3", "4", "5", "2–5 (mean)"]:
        x = qc.loc[k]
        rows.append(["Mean" if k.startswith("2–5") else k, num(100 * x.extra_rejections, 1), num(100 * x.change, 1)]
                    + [num(100 * cc[(cc.rule == rl) & (cc.quintile == k)].change_at_examiner_rejections.iloc[0], 1)
                       for rl, _, _ in rules])
    for samp, lab in [("test + imputation", "Rule $-$ examiners, mean"), ("lock box", "Lock box: rule $-$ examiners")]:
        z = c[(c.outcome == "cites") & (c["sample"] == samp) & (c.quintile == "2–5 (mean)")].set_index("rule")
        rows.append([lab, "", ""] + ["" if rl == RND and samp == "lock box" else
                                     c95(z.loc[rl]["diff"], z.loc[rl].diff_lo, z.loc[rl].diff_hi) for rl, _, _ in rules])
    tabular_(TEX / "ext_tab_kllm_contraction.tex", "lrrrrrrrr",
            ["Stricter quintile", "Extra rejections (pp)", "Examiners", *[lab for _, lab, _ in rules]], rows,
            groups={0: "Change in citations per application relative to the most lenient quintile (\\%)",
                    5: "Rule $-$ examiners (pp, 95\\% CI)"})
    x = qc.drop("2–5 (mean)")
    N["xKExtraLo"], N["xKExtraHi"] = num(100 * x.extra_rejections.min(), 0), num(100 * x.extra_rejections.max(), 0)
    N["xKExLo"], N["xKExHi"] = num(-100 * x.change.max(), 0), num(-100 * x.change.min(), 0)
    rnd = cc[(cc.rule == RND) & (cc.quintile != "2–5 (mean)")].change_at_examiner_rejections
    N["xKRndLo"], N["xKRndHi"] = num(-100 * rnd.max(), 0), num(-100 * rnd.min(), 0)
    N["xKExMean"] = num(-100 * qc.loc["2–5 (mean)"].change, 1)
    m = cc[cc.quintile == "2–5 (mean)"].set_index("rule")
    for rl, _, mac in rules + [(MAT, "", "Match"), (EQU, "", "Equal")]:
        N[f"xK{mac}Mean"] = num(-100 * m.loc[rl].change_at_examiner_rejections, 1)
        N[f"xK{mac}Diff"] = num(100 * m.loc[rl]["diff"], 1)
        N[f"xK{mac}DiffCI"] = f"[{num(100 * m.loc[rl].diff_lo, 1)}, {num(100 * m.loc[rl].diff_hi, 1)}]"
    bc = c[(c.outcome == "cites") & (c["sample"] == "balanced cells") & (c.quintile == "2–5 (mean)")].set_index("rule")
    if len(bc):
        N["xKBalDiff"] = num(100 * bc.loc[ALG]["diff"], 1)
        N["xKBalDiffCI"] = f"[{num(100 * bc.loc[ALG].diff_lo, 1)}, {num(100 * bc.loc[ALG].diff_hi, 1)}]"
        qb = q[(q["sample"] == "balanced cells") & (q.outcome == "cites") & (q.quintile == "2–5 (mean)")].iloc[0]
        N["xKBalEx"] = num(-100 * qb.change, 1)
        N["xKBalRnd"] = num(-100 * bc.loc[RND].change_at_examiner_rejections, 1)
    lb = c[(c.outcome == "cites") & (c["sample"] == "lock box") & (c.quintile == "2–5 (mean)")].set_index("rule")
    N["xKLockDiff"] = num(100 * lb.loc[ALG]["diff"], 1)
    N["xKLockDiffCI"] = f"[{num(100 * lb.loc[ALG].diff_lo, 1)}, {num(100 * lb.loc[ALG].diff_hi, 1)}]"

    # Table V: other outcomes. Columns are "better than the examiners" (pp): for citations, renewals and top-decile
    # patents, fewer lost; for granted patents that nobody cites (a bad outcome), more removed, so the sign is flipped
    other = [("cites", "Citations", ALG), ("uncited", "Granted but never cited (fewer is better)",
                                           "Algorithm (trained on no citations)"),
             ("renew", "Renewed at four years", ALG), ("top10", "Top-decile citations", ALG),
             ("pa_granted", "Prior-art citations of granted applications", ALG)]
    rows = []
    for o, lab, alg in other:
        qq = q[(q["sample"] == "test + imputation") & (q.outcome == o) & (q.quintile == "2–5 (mean)")].iloc[0]
        z = c[(c["sample"] == "test + imputation") & (c.outcome == o) & (c.quintile == "2–5 (mean)")].set_index("rule")
        sg = -1 if o == "uncited" else 1

        def b95(rl):
            d, lo, hi = sg * z.loc[rl]["diff"], sg * z.loc[rl].diff_lo, sg * z.loc[rl].diff_hi
            return d, min(lo, hi), max(lo, hi)
        rows.append([lab, num(100 * qq.change, 1), num(100 * z.loc[alg].change_at_examiner_rejections, 1),
                     c95(*b95(alg)), c95(*b95(JDG))])
        mac = {"cites": "Cites", "uncited": "Unc", "renew": "Renew", "top10": "Top", "pa_granted": "PaG"}[o]
        N[f"xKV{mac}Ex"] = num(-100 * qq.change, 1)
        N[f"xKV{mac}Diff"] = num(100 * z.loc[alg]["diff"], 1)
        N[f"xKV{mac}DiffCI"] = f"[{num(100 * z.loc[alg].diff_lo, 1)}, {num(100 * z.loc[alg].diff_hi, 1)}]"
        d, lo, hi = b95(alg)
        N[f"xKV{mac}Better"], N[f"xKV{mac}BetterCI"] = num(100 * d, 1), f"[{num(100 * lo, 1)}, {num(100 * hi, 1)}]"
        d, lo, hi = b95(JDG)
        N[f"xKV{mac}Judge"], N[f"xKV{mac}JudgeCI"] = num(100 * d, 1), f"[{num(100 * lo, 1)}, {num(100 * hi, 1)}]"
        if RND in z.index:   # random rule = its expected value within each cell (35); 43 gives paired intervals against random
            d, lo, hi = b95(RND)
            N[f"xKV{mac}Rnd"], N[f"xKV{mac}RndCI"] = num(100 * d, 1), f"[{num(100 * lo, 1)}, {num(100 * hi, 1)}]"
        N[f"xKV{mac}Algo"] = num(-100 * z.loc[alg].change_at_examiner_rejections, 1)
    N["xKUncDiff"], N["xKUncDiffCI"] = N["xKVUncDiff"], N["xKVUncDiffCI"]
    tabular_(TEX / "ext_tab_kllm_other.tex", "lrrll",
            ["Outcome", "Examiners (\\%)", "Algorithm (\\%)", "Algorithm better by (pp)",
             "Predicted examiner better by (pp)"], rows)

    # Table IV
    t4["a"] = t4.alpha.astype(str)
    tc4 = t4[t4.outcome == "cites"]
    rows = []
    for rl, lab, mac in [x for x in rules if x[0] != RND]:
        row = [lab]
        for a_ in ["1.0", "1.5", "2.0", "inf"]:
            x = tc4[(tc4.rule == rl) & (tc4.a == a_)].iloc[0]
            row.append(c95(x.outcome_change, x.outcome_change_lo, x.outcome_change_hi))
            N[f"xKRr{mac}{dict(zip(['1.0', '1.5', '2.0', 'inf'], ['One', 'OneHalf', 'Two', 'Inf']))[a_]}"] = num(
                100 * x.outcome_change, 1)
        x = tc4[(tc4.rule == rl) & (tc4.a == "1.25")].iloc[0]
        N[f"xKRr{mac}OneQuarter"] = num(100 * x.outcome_change, 1)
        x = tc4[(tc4.rule == rl) & (tc4.a == "1.0")].iloc[0]
        N[f"xKRr{mac}OneCI"] = f"[{num(100 * x.outcome_change_lo, 1)}, {num(100 * x.outcome_change_hi, 1)}]"
        rows.append(row)
    tabular_(TEX / "ext_tab_kllm_rerank.tex", "lllll", ["Rule", "$\\alpha = 1$", "$\\alpha = 1.5$", "$\\alpha = 2$",
                                                         "$\\alpha = \\infty$"], rows)

    # Extension A: prior-art citations, imputed as in the paper and observed
    tp = t4[t4.outcome == "pa_obs"]
    rows = []
    for rl, lab, mac in [(ALG, "Algorithm (citations)", "Algo"), (PAS, "Algorithm (prior art, granted only)", "Sel"),
                         (PAA, "Algorithm (prior art, all applications)", "All")]:
        row = [lab]
        for a_, sfx in [("1.0", "One"), ("1.5", "OneHalf"), ("inf", "Inf"), ("observed", "Obs")]:
            x = tp[(tp.rule == rl) & (tp.a == a_)].iloc[0]
            row.append(c95(x.outcome_change, x.outcome_change_lo, x.outcome_change_hi))
            N[f"xKPa{mac}{sfx}"] = num(100 * x.outcome_change, 1)
            N[f"xKPa{mac}{sfx}CI"] = f"[{num(100 * x.outcome_change_lo, 1)}, {num(100 * x.outcome_change_hi, 1)}]"
        rows.append(row)
    al = r("c1_kllm_alpha")
    al["decile"] = al.decile.astype(str)
    for pred, mac in [("prior-art model", "Pa"), ("counterfactual citation score", "Cf")]:
        z = al[(al.prediction == pred) & (al.decile == "all")].iloc[0]
        N[f"xKAlpha{mac}"] = num(z.alpha, 2)
        N[f"xKAlpha{mac}Gr"], N[f"xKAlpha{mac}Rej"] = num(z.granted, 2), num(z.rejected, 2)
    tabular_(TEX / "ext_tab_kllm_prior_art.tex", "lllll",
            ["Rule", "Imputed, $\\alpha = 1$", "Imputed, $\\alpha = 1.5$", "Imputed, $\\alpha = \\infty$",
             "Observed"], rows)

    # Tables VI and VII: gender
    t7 = r("c1_kllm_table7").set_index("rule")
    rows = [["\\emph{Contraction (Table VI), mean over the stricter quintiles}", "", "", ""]]
    qx = qc.loc["2–5 (mean)"]
    rows.append(["Stricter examiners", num(100 * qx.change, 1), "--",
                 c95(qx.female_share, qx.female_share_lo, qx.female_share_hi)])
    N["xKFemExam"] = num(100 * qx.female_share, 1)
    N["xKFemExamCI"] = f"[{num(100 * qx.female_share_lo, 1)}, {num(100 * qx.female_share_hi, 1)}]"
    for rl, lab, mac in [(ALG, "Algorithm", "Algo"), (MAT, "Algorithm, examiners' female share", "Match"),
                         (EQU, "Algorithm, equal rates for women and men", "Equal"), (RND, "Random", "Rnd")]:
        z = m.loc[rl]
        rows.append([lab, num(100 * z.change_at_examiner_rejections, 1), "--",
                     c95(z.female_share, z.female_share_lo, z.female_share_hi)])
        N[f"xKFem{mac}"] = num(100 * z.female_share, 1)
        N[f"xKFem{mac}CI"] = f"[{num(100 * z.female_share_lo, 1)}, {num(100 * z.female_share_hi, 1)}]"
    rows.append(["\\emph{Reranking (Table VII), at the examiners' grant rate, $\\alpha = 1$}", "", "", ""])
    for rl, lab, mac in [("Examiners", "Examiners", "Exam"), (ALG, "Algorithm", "Algo"),
                         (MAT, "Algorithm, examiners' female share", "Match"),
                         (EQU, "Algorithm, equal rates for women and men", "Equal")]:
        z = t7.loc[rl]
        rows.append([lab, "--" if rl == "Examiners" else c95(z.outcome_change, z.outcome_change_lo, z.outcome_change_hi),
                     c95(z.gap, z.gap_lo, z.gap_hi), c95(z.female_share_rejected, z.female_share_rejected_lo,
                                                         z.female_share_rejected_hi)])
        N[f"xKSevGap{mac}"] = num(100 * z.gap, 1)
        N[f"xKSevGap{mac}CI"] = f"[{num(100 * z.gap_lo, 1)}, {num(100 * z.gap_hi, 1)}]"
        N[f"xKSevCh{mac}"] = num(100 * z.outcome_change, 1)
        N[f"xKSevCh{mac}CI"] = f"[{num(100 * z.outcome_change_lo, 1)}, {num(100 * z.outcome_change_hi, 1)}]"
    tabular_(TEX / "ext_tab_kllm_gender.tex", "lrll",
            ["Rule", "Citations per application (\\%)", "Female $-$ other grant rate (pp)",
             "Female share of (extra) rejections (\\%)"], rows)

    # Figure VI
    f6 = r("c1_kllm_fig6")
    z = f6[pd.to_numeric(f6.quintile, errors="coerce") == 5]   # rd() reads the column as text
    N["xKSixExam"], N["xKSixAlgo"] = pct(z.examiners[z.bin <= 5].sum(), 0), pct(z.algorithm[z.bin <= 5].sum(), 0)
    N["xKSixExamTop"] = pct(z.examiners[z.bin > 15].sum(), 0)

    # Extension C: levels at the margin
    lv = r("c1_kllm_levels")
    lv["quintile"] = lv.quintile.astype(str)
    rows = []
    series = [("observed citations", "Citations, observed (granted by the most lenient quintile)", "Obs"),
              (CF, "Counterfactual score (text)", "Cf"), (CFP, "Counterfactual score (text + prior art)", "CfPa"),
              ("observed prior-art citations", "Prior-art citations, as rejected (stricter quintile)", "PaObs"),
              ("observed prior-art citations, as granted", "Prior-art citations, as granted (most lenient quintile)",
               "PaGrant")]
    series = [x for x in series if (lv.series == x[0]).any()]
    for ser, lab, mac in series:
        row = [lab]
        for k in ["2", "3", "4", "5", "2–5 (pooled)"]:
            x = lv[(lv.series == ser) & (lv.quintile == k)].iloc[0]
            row.append(f"{num(x.value)} [{num(x.lo)}, {num(x.hi)}]")
        x = lv[(lv.series == ser) & (lv.quintile == "2–5 (pooled)")].iloc[0]
        N[f"xKLv{mac}"], N[f"xKLv{mac}CI"] = num(x.value), f"[{num(x.lo)}, {num(x.hi)}]"
        rows.append(row)
    for ser, mac in [(CF, "Cf"), (CFP, "CfPa")]:
        x = lv[(lv.series == f"{ser} − observed citations")].iloc[0]
        N[f"xKLv{mac}Diff"], N[f"xKLv{mac}DiffCI"] = num(x.value), f"[{num(x.lo)}, {num(x.hi)}]"
    for ser, mac in [("grant effect on prior-art citations", "GrantEffect"),
                     ("grant effect on prior-art citations (ratio)", "GrantRatio")]:
        x = lv[lv.series == ser]
        if len(x):
            x = x.iloc[0]
            N[f"xK{mac}"], N[f"xK{mac}CI"] = num(x.value), f"[{num(x.lo)}, {num(x.hi)}]"
    if "xKGrantRatio" in N:
        a_ = al[(al.prediction == "prior-art model") & (al.decile == "all")].iloc[0].alpha
        N["xKAlphaSel"] = num(a_ / float(N["xKGrantRatio"]), 2)
    tabular_(TEX / "ext_tab_kllm_levels.tex", "llllll",
            ["Mean over the extra rejections", "Quintile 2", "3", "4", "5", "Pooled"], rows)

    # Extension D: counterfactual scores
    rows = []
    for tc in ([tc0] if tc0 == "all" else ["3600", "1600"]):
        v = r("c1_kllm_cf_valid", tc).set_index("score")
        for ser, lab in [(CF, "Text"), (CFP, "Text + prior art")]:
            x = v.loc[ser]
            rows.append(["All" if tc == "all" else tc, lab, num(x.corr_cites_granted, 2), num(x.spearman_prior_art_granted, 2),
                         num(x.spearman_prior_art_rejected, 2), num(x.mean_granted), num(x.mean_rejected),
                         pct(x.share_rejected_above_granted_median, 0), pct(x.share_rejected_top_quarter, 0)])
            if tc == tc0:
                mac = "Cf" if ser == CF else "CfPa"
                N[f"xK{mac}Corr"] = num(x.corr_cites_granted, 2)
                N[f"xK{mac}SpGr"], N[f"xK{mac}SpRej"] = (num(x.spearman_prior_art_granted, 2),
                                                         num(x.spearman_prior_art_rejected, 2))
                N[f"xK{mac}Rej"], N[f"xK{mac}Gr"] = num(x.mean_rejected), num(x.mean_granted)
                N[f"xK{mac}Above"] = pct(x.share_rejected_above_granted_median, 0)
                N[f"xK{mac}TopQ"] = pct(x.share_rejected_top_quarter, 0)
                N["xKPaRej"], N["xKPaGr"] = num(x.mean_prior_art_rejected), num(x.mean_prior_art_granted)
                N["xKMeanCites"] = num(x.mean_cites_granted)
    tabular_(TEX / "ext_tab_kllm_scores.tex", "llrrrrrrr",
            ["TC", "Score", "Corr., citations (granted)", "Spearman, prior art (granted)",
             "Spearman, prior art (rejected)", "Mean, granted", "Mean, rejected", "Rejected above granted median (\\%)",
             "Rejected in granted top quarter (\\%)"], rows)
    sc = pd.read_csv(T / f"c1_counterfactual_scores{file_sfx}.csv", dtype={"application_number": str, "tech_center": str})
    N["xKNRej"] = f"{len(sc):,}"
    N["xKNRejThreeSix"] = f"{int((sc.tech_center == '3600').sum()):,}"
    top = rd(f"c1_kllm{file_sfx}_top_rejected")
    top = (top if tc0 == "all" else top[top.tech_center.astype(str) == tc0]).head(10)
    rows = []
    for x in top.itertuples():
        t = str(x.invention_title).strip().capitalize()
        t = t if len(t) <= 70 else t[:67].rsplit(" ", 1)[0] + "..."
        t = t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
        rows.append([str(x.application_number).zfill(8), str(x.tech_center), str(x.filing_year), t, num(x.cf_cites, 1),
                     num(x.cf_percentile_among_granted, 0), num(x.prior_art_cites, 0)])
    tabular_(TEX / "ext_tab_kllm_examples.tex", "lllp{7.4cm}rrr",
            ["Application", "TC", "Filed", "Title", "Score", "Percentile", "Prior-art cites"], rows)

    # Figures II–IV, Table II
    N["xKCorr"], N["xKCorrLin"], N["xKCorrLLM"] = num(fit.corr_cites, 2), num(fit.corr_cites_linear, 2), num(
        fit.corr_cites_llm, 2)
    N["xKSpear"] = num(fit.spearman_cites, 2)
    N["xKLowGrant"], N["xKGrant"] = num(100 * fit.lowest1_grant_rate, 0), num(100 * fit.grant_rate, 0)
    N["xKAUCGrant"], N["xKAUCJudge"] = num(fit.auc_grant_by_predicted_cites, 2), num(fit.auc_grant_predicted_examiner, 2)
    N["xKPerm"], N["xKPermCells"] = num(100 * perm.share_p_below_05, 1), f"{int(perm.cells):,}"
    g = f4[f4.outcome == "cites"]
    N["xKFourLo"], N["xKFourHi"] = num(g.resid.min(), 2), num(g.resid.max(), 2)
    N["xKFourAllZero"] = "all" if ((g.resid_lo < 0) & (g.resid_hi > 0)).all() else "not all"
    if "resid_rel" in g.columns:   # relative to the most lenient quintile: removes a level error common to all quintiles
        gr = g[g.quintile.astype(str) != "1"]
        N["xKFourLevel"] = num(g[g.quintile.astype(str) == "1"].resid.iloc[0], 2)
        N["xKFourRelLo"], N["xKFourRelHi"] = num(gr.resid_rel.min(), 2), num(gr.resid_rel.max(), 2)
        N["xKFourRelAllZero"] = "all" if ((gr.resid_rel_lo < 0) & (gr.resid_rel_hi > 0)).all() else "not all"
    t2 = r("c1_kllm_table2")
    h = t2[t2.outcome == "cites (highest)"].set_index("top")
    N["xKTopOverlap"] = pct(h.loc[0.01].overlap, 0)
    N["xKTopML"], N["xKTopLin"] = num(h.loc[0.01].ml_only, 1), num(h.loc[0.01].linear_only, 1)
    N["xKTopQuart"] = pct(h.loc[0.25].overlap, 0)
    lo = t2[t2.outcome == "cites"].set_index("top")
    N["xKLowOverlap"] = pct(lo.loc[0.01].overlap, 0)
    N["xKTopFiveML"], N["xKTopFiveLin"] = num(h.loc[0.05].ml_only, 1), num(h.loc[0.05].linear_only, 1)
    # the lowest-predicted 1% (where the contraction rejects): citations of the granted picks flagged by one model only
    N["xKBotML"], N["xKBotLin"] = num(lo.loc[0.01].ml_only, 2), num(lo.loc[0.01].linear_only, 2)
    N["xKBotAllML"], N["xKBotAllLin"] = num(lo.loc[0.01].all_ml, 2), num(lo.loc[0.01].all_linear, 2)
    N_out.update({prefix + k[2:]: v for k, v in N.items()})


def geography(N: dict) -> None:
    s = rd("geography_states")
    N["xGeoN"] = f"{int(s.applications.sum()):,}"
    N["xGeoStates"] = str(len(s))
    N["xGeoExpStrict"], N["xGeoExpLen"] = pct(s.expected_strict.mean()), pct(s.expected_lenient.mean())
    N["xGeoSigStrict"] = str(int((s.z_strict.abs() > 1.96).sum()))
    N["xGeoSigLen"] = str(int((s.z_lenient.abs() > 1.96).sum()))
    N["xGeoMaxZ"] = num(max(s.z_strict.abs().max(), s.z_lenient.abs().max()), 1)
    if (T / "geo_state_leniency.csv").exists():   # Figure A2c: mean leniency drawn, relative to the cell, by state
        g = rd("geo_state_leniency")
        top = g.loc[g.mean_dev_pp.abs().idxmax()]
        N["xGeoLenSig"] = str(int((g.z.abs() >= 1.96).sum()))
        N["xGeoLenMax"], N["xGeoLenMaxZ"] = num(abs(top.mean_dev_pp), 1), num(top.z, 2)
        N["xGeoLenMaxState"], N["xGeoLenMaxN"] = STATES.get(top.state, top.state), f"{int(top.n):,}"
        N["xGeoLenMaxZAll"] = num(g.z.abs().max(), 2)
        N["xGeoLenSD"] = num(g.mean_dev_pp.std(), 2)
    tc = rd("industry_tc").sort_values("median")
    lo, hi = tc.iloc[0], tc.iloc[-1]
    N["xIndStrictTC"], N["xIndStrictMed"] = str(lo.tech_center), pct(lo["median"], 0)
    N["xIndLenTC"], N["xIndLenMed"] = str(hi.tech_center), pct(hi["median"], 0)
    N["xIndSpreadTC"] = str(tc.assign(w=tc.p90 - tc.p10).sort_values("w").iloc[-1].tech_center)
    c = rd("industry_cpc").sort_values("grant_rate")
    for i, mac in [(0, "One"), (1, "Two"), (2, "Three")]:
        N[f"xIndCpcLow{mac}"] = f"{c.iloc[i].cpc} {c.iloc[i]['name'].lower()} ({pct(c.iloc[i].grant_rate, 0)}\\%)"
        r = c.iloc[-1 - i]
        N[f"xIndCpcHigh{mac}"] = f"{r.cpc} {r['name'].lower()} ({pct(r.grant_rate, 0)}\\%)"




def allcenters(N: dict) -> None:
    """The check in all eight centers (38)."""
    S = rd("all_centers_summary")
    S["sample"] = S["sample"].astype(str)
    G, C, GN = "Train on grant", "Train on citations (granted only)", "Train on grant + names"
    llm = None
    if (T / "llm_all_centers.csv").exists():
        llm = rd("llm_all_centers")
        llm["sample"] = llm["sample"].astype(str)
        llm = llm.set_index("sample")
    rows = []
    for _, r in S.iterrows():
        row = [("TC " + r["sample"]) if r["sample"] != "All centers" else "All centers",
               f"{int(r.applications):,}", pct(r.female), cin(r.human_gap, r.human_gap_se, 1, 100),
               cin(r[f"gap::{G}"], r[f"gap_se::{G}"], 1, 100), cin(r[f"gap::{C}"], r[f"gap_se::{C}"], 1, 100)]
        if llm is not None:
            x = llm.loc[r["sample"]]
            row.append(cin(x.llm_gap, x.llm_gap_se, 1, 100))
        rows.append(row)
    head = ["Center", "Applications", "Female (\\%)", "Gap: examiners (pp)", "Gap: grant-trained",
            "Gap: citation-trained"] + (["Gap: LLM, zero-shot"] if llm is not None else [])
    tabular(TEX / "ext_tab_allcenters.tex", "lrrlll" + ("l" if llm is not None else ""), head, rows)
    tc = S[S["sample"] != "All centers"]
    sig = (tc.human_gap.abs() > 1.96 * tc.human_gap_se)
    N["xAllN"] = f"{int(S[S['sample'] == 'All centers'].applications.iloc[0]):,}"
    N["xAllSigCenters"] = str(int(sig.sum()))
    N["xAllAmp"] = str(int(((tc[f"ratio::{G}"] > 1) & sig).sum()))
    N["xAllCitesBelow"] = str(int(((tc[f"ratio::{C}"].abs() < tc[f"ratio::{G}"].abs()) & sig).sum()))
    gsig = (tc[f"gap::{G}"].abs() > 1.96 * tc[f"gap_se::{G}"])
    N["xAllGrantGapSig"] = str(int((gsig & (tc[f"gap::{G}"] < 0)).sum()))
    N["xAllGrantLarger"] = str(int((tc[f"gap::{G}"].abs() > tc.human_gap.abs()).sum()))
    N["xAllCitesSigNeg"] = str(int((tc[f"gap::{C}"] + 1.96 * tc[f"gap_se::{C}"] < 0).sum()))
    N["xAllCitesSigPos"] = str(int((tc[f"gap::{C}"] - 1.96 * tc[f"gap_se::{C}"] > 0).sum()))
    a = S[S["sample"] == "All centers"].iloc[0]
    N["xAllHumanGap"] = pct(a.human_gap)
    N["xAllHumanGapCI"] = f"[{pct(a.human_gap - 1.96 * a.human_gap_se)}, {pct(a.human_gap + 1.96 * a.human_gap_se)}]"
    for k, mac in [(G, "Grant"), (C, "Cites"), (GN, "GrantNames")]:
        N[f"xAllGap{mac}"] = pct(a[f"gap::{k}"])
        N[f"xAllRatio{mac}"] = num(a[f"ratio::{k}"], 2)


def c3checks(N: dict) -> None:
    """Outcome test, permutation-calibrated FLL test, and the name test on the main sample (39)."""
    ot = rd("c3_checks_outcome_test")
    ot["sample"] = ot["sample"].astype(str)

    def q(x, k, nd=2):
        return f"{num(x[k], nd)} [{num(x[k + '_lo'], nd)}, {num(x[k + '_hi'], nd)}]"
    rows, groups = [], {0: "By center (all cells)"}
    for _, x in ot[ot.design == "all cells"].iterrows():
        rows.append([("TC " + x["sample"]) if x["sample"] != "All centers" else "All centers", f"{int(x.n):,}",
                     q(x, "marginal"), q(x, "female"), q(x, "male"), q(x, "f_minus_m")])
    groups[len(rows)] = "Alternative designs"
    for _, x in ot[ot.design != "all cells"].iterrows():
        rows.append([("All centers" if x["sample"] == "All centers" else "TC " + x["sample"]) + ", " + x.design,
                     f"{int(x.n):,}", q(x, "marginal"), q(x, "female"), q(x, "male"), q(x, "f_minus_m")])
    tabular(TEX / "ext_tab_outcome_test.tex", "lrllll",
            ["Sample", "Applications", "All", "Female", "Male", "Female $-$ male"], rows, groups)
    for (des, smp), mac in [(("all cells", "All centers"), "All"),
                            (("cells that pass the permutation test", "All centers"), "Bal"),
                            (("all cells", "3600"), "ThreeSix"),
                            (("subclass cells", "All centers"), "Sub"),
                            (("without classes 705, 435, 436, 536", "All centers"), "NoAlice")]:
        x = ot[(ot.design == des) & (ot["sample"] == smp)].iloc[0]
        N[f"xOT{mac}"] = num(x.f_minus_m)
        N[f"xOT{mac}CI"] = f"[{num(x.f_minus_m_lo)}, {num(x.f_minus_m_hi)}]"
        N[f"xOT{mac}Fem"], N[f"xOT{mac}Male"], N[f"xOT{mac}Marg"] = num(x.female), num(x.male), num(x.marginal)
        N[f"xOT{mac}MargCI"] = f"[{num(x.marginal_lo)}, {num(x.marginal_hi)}]"
    a_ = ot[(ot.design == "all cells") & (ot["sample"] != "All centers")]
    N["xOTSigCenters"] = str(int(((a_.f_minus_m_lo > 0) | (a_.f_minus_m_hi < 0)).sum()))
    N["xOTSigCentersWord"] = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight"][
        int(((a_.f_minus_m_lo > 0) | (a_.f_minus_m_hi < 0)).sum())]
    N["xOTCenterLo"], N["xOTCenterHi"] = num(a_.f_minus_m.min()), num(a_.f_minus_m.max())
    oc = rd("c3_checks_outcome_cohorts")
    oc["tech_center"] = oc.tech_center.astype(str)
    rows = []
    for _, x in oc.iterrows():
        rows.append([("TC " + x.tech_center) if x.tech_center != "All centers" else "All centers", x.cohort,
                     f"{int(x.n):,}", cin(x.grant_gap, x.grant_gap_se, 1, 100), q(x, "marginal"), q(x, "f_minus_m")])
    tabular(TEX / "ext_tab_outcome_cohorts.tex", "llrlll",
            ["Center", "Cohort", "Applications", "Grant gap (pp)", "Extra grants: citations", "Female $-$ male"], rows)
    sig = oc[(oc.tech_center != "All centers") & ((oc.f_minus_m_lo > 0) | (oc.f_minus_m_hi < 0))]
    N["xOTSigCells"] = str(len(sig))
    for coh, mac in [("2004–2009", "Early"), ("2010–2014", "Late")]:
        for tc, m2 in [("All centers", "All"), ("3600", "ThreeSix")]:
            x = oc[(oc.tech_center == tc) & (oc.cohort == coh)].iloc[0]
            N[f"xOT{mac}{m2}"] = num(x.f_minus_m)
            N[f"xOT{mac}{m2}CI"] = f"[{num(x.f_minus_m_lo)}, {num(x.f_minus_m_hi)}]"
    pm = rd("c3_checks_permutation")
    N["xPermAll"] = pct((pm.p < 0.05).mean())
    N["xPermCells"] = f"{len(pm):,}"
    N["xPermDropped"] = f"{int((pm.p < 0.05).sum()):,}"
    N["xBalN"] = f"{int(ot[(ot.design == 'cells that pass the permutation test') & (ot['sample'] == 'All centers')].n.iloc[0]):,}"
    fl = rd("c3_checks_fll").set_index("group")
    rows = [[g, f"{int(fl.loc[g, 'examiners']):,}", num(fl.loc[g, "ratio"]), num(fl.loc[g, "null_ratio"]),
             num(fl.loc[g, "p"])] for g in ["All", "Female", "Male"]]
    tabular(TEX / "ext_tab_fll.tex", "lrrrr",
            ["First inventor", "Examiners", "Misfit / df", "Same, null (mean)", "$p$"], rows)
    for g, mac in [("All", "All"), ("Female", "Fem"), ("Male", "Male")]:
        N[f"xFLL{mac}P"], N[f"xFLL{mac}Ratio"] = num(fl.loc[g, "p"]), num(fl.loc[g, "ratio"])
        N[f"xFLL{mac}Null"], N[f"xFLL{mac}N"] = num(fl.loc[g, "null_ratio"]), f"{int(fl.loc[g, 'examiners']):,}"
    def c(v, se, nd=2, s=1.0):
        return f"{num(s * v, nd)} [{num(s * (v - 1.96 * se), nd)}, {num(s * (v + 1.96 * se), nd)}]"
    nm = rd("c3_checks_names")
    nm["sample"] = nm["sample"].astype(str)
    rows = []
    for _, x in nm.iterrows():
        rows.append([("TC " + x["sample"]) if x["sample"] != "All centers" else "All centers", f"{int(x.n_common):,}",
                     f"{int(x.n_rare):,}", c(x.gap_common, x.gap_common_se, 1, 100), c(x.gap_rare, x.gap_rare_se, 1, 100),
                     c(x["diff"], x.diff_se, 1, 100)])
    tabular(TEX / "ext_tab_names.tex", "lrrlll",
            ["Sample", "Women, common name", "Women, rare name", "Gap: common (pp)", "Gap: rare (pp)",
             "Common $-$ rare (pp)"], rows)
    for smp, mac in [("All centers", "All"), ("3600", "ThreeSix"), ("1600", "OneSix")]:
        x = nm[nm["sample"] == smp].iloc[0]
        N[f"xNm{mac}Common"], N[f"xNm{mac}Rare"] = pct(x.gap_common), pct(x.gap_rare)
        N[f"xNm{mac}Diff"] = pct(x["diff"])
        N[f"xNm{mac}DiffCI"] = f"[{pct(x['diff'] - 1.96 * x.diff_se)}, {pct(x['diff'] + 1.96 * x.diff_se)}]"


def robust(N: dict) -> None:
    """Fewer single-examiner groups and same-period leniency (35 robust, 39 robust): C1's contraction (algorithm −
    examiners) and C3's outcome test (female − male), next to the main design."""
    ALG = "Algorithm (trained on citations)"
    ot = rd("c3_checks_outcome_test")
    ot["sample"] = ot["sample"].astype(str)
    rb = rd("c3_checks_robust")
    rb["sample"] = rb["sample"].astype(str)
    c = rd(f"c1_kllm{C1_MAIN}_contraction")
    c1 = c[(c.outcome == "cites") & (c["sample"] == "test + imputation") & (c.rule == ALG)
           & (c.quintile.astype(str) == "2–5 (mean)")].iloc[0]
    have_c1 = (T / f"c1_kllm{C1_MAIN}_robust.csv").exists()
    if have_c1:
        r1 = rd(f"c1_kllm{C1_MAIN}_robust")
        r1 = r1[(r1.outcome == "cites") & (r1.rule == ALG) & (r1.quintile.astype(str) == "2–5 (mean)")]
        r1 = r1.set_index("sample")
    C1LAB = {"ten or more examiners, quintiles": "Cells with at least ten examiners",
             "fifteen or more examiners, quintiles": "Cells with at least fifteen examiners",
             "leniency from the same period (t - 2 to t + 2)": "Leniency from the same period (filing years t - 2 to t + 2)",
             "docketed to an examiner only once": "Applications docketed to an examiner only once"}
    LAB = {"main": "Main: five or more examiners, quintiles", "ten or more examiners, quintiles": "Ten or more examiners",
           "fifteen or more examiners, quintiles": "Fifteen or more examiners", "terciles": "Terciles",
           "halves": "Halves", "leniency from the same period (t - 2 to t + 2)": "Same-period leniency",
           "docketed to an examiner only once": "Docketed only once",
           "citations top-coded at the 99th percentile": "Citations top-coded at the 99th percentile"}
    MAC = {"main": "Main", "ten or more examiners, quintiles": "Ten", "fifteen or more examiners, quintiles": "Fifteen",
           "terciles": "Ter", "halves": "Half", "leniency from the same period (t - 2 to t + 2)": "Win",
           "docketed to an examiner only once": "Dock", "citations top-coded at the 99th percentile": "Trim"}

    def q(x, k, nd=2):
        return f"{num(x[k], nd)} [{num(x[k + '_lo'], nd)}, {num(x[k + '_hi'], nd)}]"
    rows, fms = [], []
    for des in MAC:
        if des == "main":
            a = ot[(ot.design == "all cells") & (ot["sample"] == "All centers")].iloc[0]
            t = ot[(ot.design == "all cells") & (ot["sample"] == "3600")].iloc[0]
            e = ot[(ot.design == "all cells") & (ot["sample"] == "2800")].iloc[0]
            k1 = c1
        else:
            z = rb[rb.design == des]
            if not len(z):
                continue
            a, t, e = (z[z["sample"] == s_].iloc[0] for s_ in ["All centers", "3600", "2800"])
            k1 = r1.loc[C1LAB[des]] if have_c1 and C1LAB.get(des) in r1.index else None
        c1cell = "" if k1 is None else f"{num(100 * k1['diff'], 1)} [{num(100 * k1.diff_lo, 1)}, {num(100 * k1.diff_hi, 1)}]"
        epg = "" if pd.isna(a.get("examiners_per_group", float("nan"))) else num(a.examiners_per_group, 0)
        sing = "" if pd.isna(a.get("single_examiner_groups", float("nan"))) else pct(a.single_examiner_groups, 0)
        rows.append([LAB[des], f"{int(a.cells):,}", epg, sing, c1cell, q(a, "f_minus_m"), q(t, "f_minus_m"),
                     q(e, "f_minus_m")])
        m = MAC[des]
        N[f"xRb{m}Fm"], N[f"xRb{m}FmCI"] = num(a.f_minus_m), f"[{num(a.f_minus_m_lo)}, {num(a.f_minus_m_hi)}]"
        N[f"xRb{m}ThreeSix"] = num(t.f_minus_m)
        N[f"xRb{m}ThreeSixCI"] = f"[{num(t.f_minus_m_lo)}, {num(t.f_minus_m_hi)}]"
        N[f"xRb{m}TwoEight"] = num(e.f_minus_m)
        N[f"xRb{m}TwoEightCI"] = f"[{num(e.f_minus_m_lo)}, {num(e.f_minus_m_hi)}]"
        N[f"xRb{m}Cells"] = f"{int(a.cells):,}"
        if epg:
            N[f"xRb{m}ExPerGroup"], N[f"xRb{m}Single"] = epg, sing
        if k1 is not None:
            N[f"xRb{m}COne"] = num(100 * k1["diff"], 1)
            N[f"xRb{m}COneCI"] = f"[{num(100 * k1.diff_lo, 1)}, {num(100 * k1.diff_hi, 1)}]"
        fms.append((m, a, t, e))
    tabular(TEX / "ext_tab_robust_groups.tex", "lrrrllll",
            ["Design", "Cells", "Examiners per group", "Single-examiner groups (\\%)",
             "C1: algorithm $-$ examiners (pp)", "Bar test: female $-$ male, all centers", "TC 3600", "TC 2800"], rows)
    lw = pd.read_parquet(ROOT / "data/processed/all_sample.parquet", columns=["application_number", "lenient_grant"]).merge(
        pd.read_parquet(ROOT / "data/processed/leniency_window.parquet"), on="application_number", how="left")
    N["xLenWinCorr"] = num(lw[["lenient_grant", "lenient_win"]].corr().iloc[0, 1], 2)
    N["xLenWinN"] = f"{lw.n_win.mean():,.0f}"
    N["xLenWinCover"] = pct(lw.lenient_win.notna().mean())
    # ranges over the designs that keep every randomly assigned application; the docketed-once subset is selected by
    # the examiners' own first actions (below) and is reported apart
    alt = [x for x in fms if x[0] not in ("Main", "Dock", "Trim")]
    N["xRbFmLo"], N["xRbFmHi"] = num(min(x[1].f_minus_m for x in alt)), num(max(x[1].f_minus_m for x in alt))
    N["xRbThreeSixLo"] = num(min(x[2].f_minus_m for x in alt))
    N["xRbThreeSixHi"] = num(max(x[2].f_minus_m for x in alt))
    N["xRbFmAllNull"] = "all" if all(x[1].f_minus_m_lo < 0 < x[1].f_minus_m_hi for x in alt) else "not all"
    N["xRbThreeSixAllPos"] = "all" if all(x[2].f_minus_m_lo > 0 for x in alt) else "not all"
    N["xRbTwoEightSig"] = str(sum(not (x[3].f_minus_m_lo < 0 < x[3].f_minus_m_hi) for x in alt))
    N["xRbNAlt"] = str(len(alt))
    if have_c1:
        z = r1[r1.index.isin([v for k, v in C1LAB.items() if k != "docketed to an examiner only once"])]
        N["xRbCOneLo"], N["xRbCOneHi"] = num(100 * z["diff"].min(), 1), num(100 * z["diff"].max(), 1)
        N["xRbCOneAllPos"] = "all" if (z.diff_lo > 0).all() else ("none" if (z.diff_lo <= 0).all() else "some")
    # is being docketed once affected by the examiner? within-cell regression on leniency (41's flag)
    dk = ROOT / "data/processed/docketed_once.parquet"
    if dk.exists():
        import pyfixest as pf
        x = pd.read_parquet(ROOT / "data/processed/all_sample.parquet",
                            columns=["application_number", "examiner_id", "au_class", "tech_center", "lenient_grant"])
        x = x.merge(pd.read_parquet(dk, columns=["application_number", "docketed_once"]), on="application_number")
        x["once"] = 100 * x.docketed_once.astype(float)
        f = pf.feols("once ~ lenient_grant | au_class", data=x, vcov={"CRV1": "examiner_id"})
        b, se = 0.1 * f.coef()["lenient_grant"], 0.1 * f.se()["lenient_grant"]
        N["xDockLen"], N["xDockLenCI"] = num(b), f"[{num(b - 1.96 * se)}, {num(b + 1.96 * se)}]"
        pos = neg = 0
        for _, y in x.groupby("tech_center"):
            if y.once.std() == 0:
                continue
            g = pf.feols("once ~ lenient_grant | au_class", data=y, vcov={"CRV1": "examiner_id"})
            lo = g.coef()["lenient_grant"] - 1.96 * g.se()["lenient_grant"]
            hi = g.coef()["lenient_grant"] + 1.96 * g.se()["lenient_grant"]
            pos, neg = pos + (lo > 0), neg + (hi < 0)
        N["xDockLenPos"], N["xDockLenNeg"] = str(int(pos)), str(int(neg))


def main_tables(N: dict) -> None:
    """The two tables of the proposal's main text, from the macros above (so they equal the appendix)."""
    def ci(k):
        return f"{N[k]} {N[k + 'CI']}"
    rows = [["Stricter examiners' extra rejections", N["xKExMean"], "", "", ""],
            ["Random rejections, same number", N["xKRndMean"], ci("xKRndDiff"), ci("xKVRenewRnd"), ci("xKVTopRnd")],
            ["Algorithm (learns citations)", N["xKAlgoMean"], ci("xKAlgoDiff"), ci("xKVRenewBetter"),
             ci("xKVTopBetter")],
            ["Predicted examiner (learns decisions)", N["xKJudgeMean"], ci("xKJudgeDiff"), ci("xKVRenewJudge"),
             ci("xKVTopJudge")]]
    import re as _re
    dn = (TEX / "design_numbers.tex").read_text() if (TEX / "design_numbers.tex").exists() else ""
    m_ = _re.search(r"\\newcommand\{\\xDeUtilityApps\}\{([\d,]+)\}", dn)
    if m_:
        N["xDeUtilityAppsMillion"] = f"{int(m_.group(1).replace(',', '')) / 1e6:.1f}"
    tabular(TEX / "main_tab_c1_outcomes.tex", "lrlll",
            ["", "Citations lost (\\%)", "Citations", "Renewed at 4 years", "Top-decile citations"], rows,
            groups={0: "Within cells, at the stricter quintiles' extra rejection rates; better than the examiners by (pp)"})
    # Table 1 of the proposal: how accurate and for whom (script 43 --forwhom, same loss as C1_MAIN)
    fw = TEX / ("c1x_poisson_numbers.tex" if C1_MAIN.endswith("_poisson") else "c1x_numbers.tex")
    F = dict(_re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", fw.read_text())) if fw.exists() else {}
    if "xCeFwAlgoGap" in F:
        def f(k):
            return f"{F[k]} {F[k + 'CI']}"
        rows = [["Stricter examiners", F["xCeFwExLost"], "", f("xCeFwExFem"), f("xCeFwExGap"),
                 f("xCeFwExOt")],
                ["Random rejections", N["xKRndMean"], ci("xKRndDiff"), "", "", ""],
                ["Model learning citations", F["xCeFwAlgoLost"], f("xCeFwAlgoBetter"), f("xCeFwAlgoFem"),
                 f("xCeFwAlgoGap"), f("xCeFwAlgoOt")],
                ["Model learning decisions", F["xCeFwJudgeLost"], f("xCeFwJudgeBetter"), f("xCeFwJudgeFem"),
                 f("xCeFwJudgeGap"), f("xCeFwJudgeOt")]]
        head = ("\\begin{tabular}{lrlllll}\n\\toprule\n"
                " & \\multicolumn{2}{c}{How accurate} & \\multicolumn{3}{c}{For whom} \\\\\n"
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
                " & Citations lost (\\%) & Gain over examiners (pp) & Women rejected (\\%) & "
                "F $-$ M grants (pp) & F $-$ M citations at margin \\\\\n\\midrule\n")
        body = "".join(" & ".join(r_) + " \\\\\n" for r_ in rows)
        (TEX / "main_tab_c1.tex").write_text(head.replace("lrlllll", "lrllll") + body + "\\bottomrule\n\\end{tabular}\n")
    R = [["Women", N["xOTAllFem"], N["xOTThreeSixFem"]], ["Men", N["xOTAllMale"], N["xOTThreeSixMale"]],
         ["Women $-$ men", ci("xOTAll"), ci("xOTThreeSix")]]
    for lab, m in [("Cells with $\\geq$10 examiners", "Ten"), ("Cells with $\\geq$15 examiners", "Fifteen"),
                   ("Terciles of examiners", "Ter"), ("Halves of examiners", "Half"),
                   ("Leniency from the same period", "Win"),
                   ("Citations top-coded at the 99th percentile", "Trim")]:
        R.append([lab, ci(f"xRb{m}Fm"), ci(f"xRb{m}ThreeSix")])
    tabular(TEX / "main_tab_c3.tex", "lll", ["", "All centers", "TC 3600"], R,
            groups={0: "Citations of the applications at the bar (granted by the most lenient fifth, rejected by stricter ones)",
                    3: "Women $-$ men under other designs"})


def llmall(N: dict) -> None:
    """Qwen3-8B, zero-shot, in all eight centers (40)."""
    x = rd("llm_all_centers")
    x["sample"] = x["sample"].astype(str)
    a = x[x["sample"] == "All centers"].iloc[0]
    N["xLLMAllGap"], N["xLLMAllRatio"], N["xLLMAllAUC"] = pct(a.llm_gap), num(a.ratio, 2), num(a.auc, 3)
    N["xLLMAllGapCI"] = f"[{pct(a.llm_gap - 1.96 * a.llm_gap_se)}, {pct(a.llm_gap + 1.96 * a.llm_gap_se)}]"
    tc = x[x["sample"] != "All centers"]
    N["xLLMAllAUCLo"], N["xLLMAllAUCHi"] = num(tc.auc.min(), 2), num(tc.auc.max(), 2)


def sample_all(N: dict) -> None:
    """The main sample by technology center (38's all_sample, 06c's prior-art citations)."""
    d = pd.read_parquet(ROOT / "data/processed/all_sample.parquet").merge(
        pd.read_parquet(ROOT / "data/processed/prior_art_cites.parquet"), on="application_number", how="left")
    rows = []
    order = d.tech_center.value_counts().index.tolist()   # largest center first
    for tc, x in [(t, d[d.tech_center == t]) for t in order] + [("All centers", d)]:
        g = x.granted == 1
        rows.append([("TC " + tc) if tc != "All centers" else tc, TC_NAMES.get(tc, ""), f"{len(x):,}",
                     f"{x.examiner_id.nunique():,}", pct(x.female.mean()), pct(x.granted.mean()),
                     num(x.cites_ns[g].mean()), num(x.prior_art_cites[g & x.published.fillna(False)].mean()),
                     num(x.prior_art_cites[~g & x.published.fillna(False)].mean())])
    tabular(TEX / "ext_tab_sample_all.tex", "llrrrrrrr",
            ["Center", "Field", "Applications", "Examiners", "Female (\\%)", "Granted (\\%)", "Citations (granted)",
             "Prior-art cit., granted", "Prior-art cit., rejected"], rows)
    N["xSampleN"], N["xSampleExaminers"] = f"{len(d):,}", f"{d.examiner_id.nunique():,}"
    N["xSampleFemale"], N["xSampleGrant"] = pct(d.female.mean()), pct(d.granted.mean())
    # prior-art citations (06c), published applications of the whole sample (all but a handful; 06d)
    g, pub = d.granted == 1, d.published.fillna(False).astype(bool)
    N["xSampleCites"] = num(d.cites_ns[g].mean())
    N["xSampleRej"] = f"{int((~g).sum()):,}"
    N["xSampleUnpub"] = f"{int((~pub).sum()):,}"
    for col, mac in [("prior_art_cites", ""), ("prior_art_cites_examiner", "Ex"), ("prior_art_cites_pub", "Pub")]:
        N[f"xSamplePa{mac}Gr"] = num(d.loc[g & pub, col].mean())
        N[f"xSamplePa{mac}Rej"] = num(d.loc[~g & pub, col].mean())
    diag = T / "prior_art_diagnostics_sources.csv"
    if diag.exists():
        src = rd("prior_art_diagnostics_sources").set_index("decision").loc["all"]
        N["xPaShareExaminer"], N["xPaShareApplicantOnly"] = pct(src.any_examiner, 0), pct(src.applicant_only, 0)
        N["xPaShareInAction"], N["xPaSharePatentOnly"] = pct(src.any_in_action, 0), pct(src.via_patent_only, 0)
        rt = rd("prior_art_diagnostics_ratios")
        rt["filing_year"] = rt.filing_year.astype(str)
        w = rt[rt.filing_year.str.contains("within filing year")].iloc[0]
        N["xPaRatioAll"], N["xPaRatioPub"] = num(w.n_all_ratio), num(w.n_pub_ratio)
        N["xPaRatioPubCenter"] = num(w.n_pub_ratio_within_center)
        yrs = rt[rt.filing_year.str.fullmatch(r"\d{4}")]
        N["xPaRatioPubLo"], N["xPaRatioPubHi"] = num(yrs.n_pub_ratio.min()), num(yrs.n_pub_ratio.max())
        N["xPaRatioAllLo"], N["xPaRatioAllHi"] = num(yrs.n_all_ratio.min()), num(yrs.n_all_ratio.max())
        co = rd("prior_art_diagnostics_cohorts")
        co["filing_year"] = co.filing_year.astype(int)
        for yr, mac in [(2010, "First"), (2014, "Last")]:
            for dec, m2 in [("granted", "Gr"), ("rejected", "Rej")]:
                x = co[(co.filing_year == yr) & (co.decision == dec)].iloc[0]
                N[f"xPa{mac}{m2}"] = num(x.mean_all)
            x = co[(co.filing_year == yr) & (co.decision == "rejected")].iloc[0]
            N[f"xPa{mac}Window"] = num(x.median_years_pub_to_end, 1)


def main() -> None:
    N: dict = {}
    for part, files in [(ethnicity, ["ethnicity_summary"]), (q8, ["q8_citations_fit"]),
                        (finetune, ["finetune_fit"]), (twins_names, ["twins_names_summary"]),
                        (lambda N: kllm(N, C1_MAIN), [f"c1_kllm{C1_MAIN}_contraction"]),
                        (lambda N: kllm(N, "_all", "all", "xSq", False), ["c1_kllm_all_contraction"]),
                        (lambda N: kllm(N, C1_MAIN.replace("_all", ""), "3600", "xP", False),
                         [f"c1_kllm{C1_MAIN.replace('_all', '')}_contraction"]), (geography, ["geography_states", "industry_tc"]),
                        (allcenters, ["all_centers_summary"]),
                        (c3checks, ["c3_checks_outcome_test", "c3_checks_outcome_cohorts", "c3_checks_fll",
                                    "c3_checks_names"]),
                        (llmall, ["llm_all_centers"]), (sample_all, ["all_centers_summary"]),
                        (robust, ["c3_checks_robust", "c3_checks_outcome_test", f"c1_kllm{C1_MAIN}_contraction"]),
                        (main_tables, ["c3_checks_robust", f"c1_kllm{C1_MAIN}_contraction"])]:
        if all((T / f"{f}.csv").exists() for f in files):
            part(N)
            print("done:", part.__name__)
        else:
            print("skipped (no results yet):", part.__name__)
    # the squared-error C1 extras (script 43 without CFE_LOSS) under their own prefix, next to the Poisson main ones
    sq = TEX / "c1x_numbers.tex"
    if C1_MAIN.endswith("_poisson") and sq.exists():
        (TEX / "c1x_sq_numbers.tex").write_text(sq.read_text().replace("\\newcommand{\\xCe", "\\newcommand{\\xSqCe"))
        print("done: c1x_sq_numbers")
    import re as _re
    for k, v in list(N.items()):   # magnitudes, for text that states the direction in words
        if _re.fullmatch(r"\$-\$[\d.,]+", v):
            N[k + "Abs"] = v[3:]
        elif _re.fullmatch(r"[\d.,]+", v):
            N.setdefault(k + "Abs", v)
    (TEX / "ext_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    print(f"{len(N)} macros")


if __name__ == "__main__":
    main()
