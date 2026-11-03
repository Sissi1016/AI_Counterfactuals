"""LaTeX tables and number macros for the citation-based technical appendix (results of 21–27).

Reads output/tables/*.csv written by 17, 18, 21–27 and writes output/latex/cit_tab_*.tex (tabular bodies) and
output/latex/cit_numbers.tex (macros, prefix \\c...). Uses the formatting helpers of 13.
Runs on CPUs in seconds.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "output/tables"
TEX = ROOT / "output/latex"
P = ROOT / "data/processed"
HUMAN = "Human record (truth)"
NO_MODEL = "No model: mean outcome of accepted cases"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M13 = load_module("m13", "13_figures_tables.py")
tabular, esc = M13.tabular, M13.esc


def num(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def cin(b: float, se: float, nd: int = 2, scale: float = 1.0) -> str:
    return f"{num(scale * b, nd)} [{num(scale * (b - 1.96 * se), nd)}, {num(scale * (b + 1.96 * se), nd)}]"


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv", dtype={"sample": str, "tech_center": str, "quintile": str})


def main() -> None:
    N = {}
    # ---------------------------------------------------------------- sample
    s = pd.read_parquet(P / "pilot_sample.parquet")
    c = pd.read_parquet(P / "cites_noself.parquet")
    s = s.merge(c, on="application_number", how="left")
    sc = rd("self_citations_summary").set_index("tech_center")
    cols = {}
    for tc, x in s.groupby("tech_center"):
        g = x[x.granted == 1]
        cols[tc] = [f"{len(x):,}", f"{x.examiner_id.nunique():,}", f"{x.au_year.nunique():,}",
                    num(100 * x.female.mean(), 1), num(100 * x.small_entity.mean(), 1), num(x.n_inventors.mean(), 2),
                    num(100 * x.granted.mean(), 1), num(g.cites_5y_noself.mean(), 2),
                    num(100 * (g.cites_5y_noself == 0).mean(), 1), num(100 * sc.loc[tc, "self_share"], 1)]
    labels = ["Applications", "Examiners", "Art unit $\\times$ filing-year cells", "Female first inventor (\\%)",
              "Small entity (\\%)", "Inventors per application", "Granted (\\%)",
              "Citations within 5 years, excluding self-citations (mean, granted)",
              "Granted patents never cited by others within 5 years (\\%)", "Self-citations (\\% of all citations)"]
    tabular(TEX / "cit_tab_sample.tex", "lrr", ["", "TC 3600", "TC 1600"],
            [[lab, cols["3600"][i], cols["1600"][i]] for i, lab in enumerate(labels)])
    N["cSelfThreeSix"], N["cSelfOneSix"] = num(100 * sc.loc["3600", "self_share"], 0), num(
        100 * sc.loc["1600", "self_share"], 0)
    N["cMeanCitesThreeSix"] = num(s[(s.tech_center == "3600") & (s.granted == 1)].cites_5y_noself.mean())
    N["cMeanCitesOneSix"] = num(s[(s.tech_center == "1600") & (s.granted == 1)].cites_5y_noself.mean())
    N["cMatched"] = num(100 * sc["citing_matched_share"].min(), 0)

    # ---------------------------------------------------------------- human record across centers and cohorts
    h = rd("human_record_by_tc")
    rows = [[r.tech_center, r.cohort, f"{int(r.applications):,}", num(100 * r.female_share, 1),
             num(100 * r.grant_rate, 1), num(r.first_stage), num(r.mean_cites_granted),
             cin(r.marginal, r.marginal_se), cin(r.f_minus_m, r.f_minus_m_se), cin(r.grant_gap, r.grant_gap_se, 1, 100)]
            for r in h.itertuples()]
    tabular(TEX / "cit_tab_human_record.tex", "llrrrrrrrr",
            ["TC", "Filed", "Applications", "Female (\\%)", "Granted (\\%)", "First stage", "Citations, granted",
             "Marginal citations", "F $-$ M, marginal", "Grant gap (pp)"], rows)
    hh = h.set_index(["tech_center", "cohort"])
    r = hh.loc[("3600", "2004–2009")]
    N["cEarlyFM"], N["cEarlyFMCI"] = num(r.f_minus_m), cin(r.f_minus_m, r.f_minus_m_se).split(" ", 1)[1]
    N["cEarlyMarg"], N["cEarlyMean"] = num(r.marginal), num(r.mean_cites_granted)
    sig = ((h.f_minus_m - 1.96 * h.f_minus_m_se > 0) | (h.f_minus_m + 1.96 * h.f_minus_m_se < 0)).sum()
    N["cFMSignificant"] = str(int(sig))
    N["cMargAboveMean"] = str(int((h.marginal >= h.mean_cites_granted).sum()))
    gap = h[h.cohort == "2010–2014"].sort_values("grant_gap")
    N["cLargestGapTC"], N["cLargestGap"] = gap.iloc[0].tech_center, num(100 * gap.iloc[0].grant_gap, 1)

    # ---------------------------------------------------------------- examiners (22)
    ident = rd("examiners_identification")
    multi = rd("examiners_names_in_several_tcs").iloc[0]
    agree = rd("examiners_patex_vs_hupd").iloc[0]
    rea = rd("examiners_reassignment").set_index("tech_center")
    spr = rd("examiners_leniency_spread").set_index("tech_center")
    N["cExaminers"] = f"{int(multi.examiners):,}"
    N["cNoLeniency"] = num(100 * ident.iloc[0].no_leniency, 1)
    N["cMultiTC"], N["cMultiTCApps"] = num(100 * multi.share_names_in_several_tcs, 1), num(
        100 * multi.share_applications_of_those, 1)
    N["cAgree"] = num(100 * agree.patex_equals_hupd, 1)
    N["cDockOnceThreeSix"], N["cDockOnceOneSix"] = num(100 * rea.loc["3600", "docketed_once"], 0), num(
        100 * rea.loc["1600", "docketed_once"], 0)
    for tc, k in [("3600", "ThreeSix"), ("1600", "OneSix")]:
        N[f"cStrictMean{k}"] = num(100 * spr.loc[tc, "strict_mean"], 0)
        N[f"cLenientMean{k}"] = num(100 * spr.loc[tc, "lenient_mean"], 0)
        N[f"cSDWithin{k}"] = num(spr.loc[tc, "sd_within_cells"], 2)
        N[f"cExPerCell{k}"] = num(spr.loc[tc, "examiners_per_cell"], 1)
    rows = [["Examiner names, 2004--2018", N["cExaminers"], ""],
            ["Applications without an examiner name (\\%)", "0.18", ""],
            ["Applications whose examiner has $<$20 other applications (\\%)", N["cNoLeniency"], ""],
            ["Names in more than one technology center (\\% of names)", N["cMultiTC"], ""],
            ["Same examiner in HUPD (2022) and PatEx (2023) (\\% of pilot)", N["cAgree"], ""],
            ["Docketed to an examiner only once (\\% of pilot)", N["cDockOnceThreeSix"], N["cDockOnceOneSix"]],
            ["Deleted from an examiner's docket (\\% of pilot)", "0.0", "0.0"],
            ["Examiners per assignment cell", N["cExPerCellThreeSix"], N["cExPerCellOneSix"]],
            ["SD of leniency within cells", N["cSDWithinThreeSix"], N["cSDWithinOneSix"]],
            ["Leniency of strict examiners (mean, \\%)", N["cStrictMeanThreeSix"], N["cStrictMeanOneSix"]],
            ["Leniency of lenient examiners (mean, \\%)", N["cLenientMeanThreeSix"], N["cLenientMeanOneSix"]]]
    tabular(TEX / "cit_tab_examiners.tex", "lrr", ["", "TC 3600", "TC 1600"], rows)
    cor = rd("examiners_leniency_correlates")
    reg = rd("examiners_leniency_regression").set_index("term")
    pol = rd("examiners_applicant_state").set_index("tech_center")
    rows, groups = [], {}
    for var, lab in [("gender", "Examiner gender"), ("seniority", "Years since first application (2010)"),
                     ("main_tc", "Technology center")]:
        groups[len(rows)] = lab
        for r in cor[cor.variable == var].itertuples():
            rows.append([esc(str(r.value)), f"{int(r.examiners):,}", cin(r.mean_grant_rate, r.se, 1, 100)])
    tabular(TEX / "cit_tab_correlates.tex", "lrr", ["", "Examiners", "Grant rate (\\%)"], rows, groups)
    N["cFemExam"] = num(100 * reg.loc["female_examiner", "coef"], 1)
    N["cFemExamCI"] = cin(reg.loc["female_examiner", "coef"], reg.loc["female_examiner", "se"], 1, 100).split(" ", 1)[1]
    N["cSeniority"] = num(100 * reg.loc["seniority_2010", "coef"], 1)
    p3 = pol.loc["3600"]
    N["cBlue"] = num(100 * p3.grant_gap_blue_minus_red, 1)
    N["cBlueCI"] = cin(p3.grant_gap_blue_minus_red, p3.grant_gap_se, 1, 100).split(" ", 1)[1]
    N["cBlueLen"] = cin(p3.leniency_drawn_blue_minus_red, p3.leniency_se, 2, 100)

    # ---------------------------------------------------------------- balance with citations
    b = rd("marginal_balance")
    cb = rd("citations_balance")
    rows, groups = [], {}
    for cells in ["Art unit × year × USPC class"]:   # the design's cells (art unit × year alone is not used)
        groups[len(rows)] = f"Cells: {cells}"
        for v in ["Female first inventor", "Small entity", "Inventors", "Predicted grant score (text, log-odds)"]:
            t = b[(b.cells == cells) & (b.variable == v)].set_index("sample")
            sc_, nd = (1, 3) if v in ("Inventors", "Predicted grant score (text, log-odds)") else (100, 2)
            rows.append([esc(v.replace("Predicted grant score (text, log-odds)", "Pred. grant score (log-odds)"))]
                        + [cin(t.loc[x, "coef_per_10pp"], t.loc[x, "se"], nd, sc_) for x in ["3600", "1600", "Both"]])
    groups[len(rows)] = "Predicted citations (class cells)"
    for ser in ["Train on grant", "Train on first action", "Train on citations (granted only)",
                "Train on citations + names", "LLM, zero-shot"]:
        t = cb[cb.series == ser].set_index("sample")
        rows.append([esc(ser.replace("Train on ", "Trained on ").replace(" (granted only)", ""))]
                    + [cin(0.1 * t.loc[x, "coef"], 0.1 * t.loc[x, "se"], 3) for x in ["3600", "1600", "Both"]])
    tabular(TEX / "cit_tab_balance.tex", "lrrr", ["", "TC 3600", "TC 1600", "Both"], rows, groups)

    # ---------------------------------------------------------------- permutation (grant index)
    pm = rd("assignment_permutation_summary")
    rows = []
    for cells, lab in [("Art unit × year × USPC class", "leave-one-out"),
                       ("Art unit × year × USPC subclass", "leave-one-out"),
                       ("Art unit × year × USPC class", "residualized")]:
        x = pm[(pm.cells == cells) & (pm.leniency == lab) & (pm["index"] == "index_grant")].set_index("tech_center")
        name = esc(cells.replace("Art unit × year", "AU × yr").replace(" × USPC ", " × ")).replace("×", "$\\times$")
        rows.append([name + (", residualized" if lab == "residualized" else ""), f"{int(x.loc['Both', 'n_cells']):,}",
                     num(100 * x.loc["Both", "share_of_pilot"], 0), num(x.loc["Both", "mean_examiners"], 1),
                     num(100 * x.loc["3600", "p_below_05"], 1), num(100 * x.loc["1600", "p_below_05"], 1)])
    tabular(TEX / "cit_tab_permutation.tex", "lrrrrr",
            ["Cells", "Cells tested", "Pilot (\\%)", "Examiners", "$p<0.05$: TC 3600 (\\%)", "TC 1600 (\\%)"], rows)

    # ---------------------------------------------------------------- calibration (25)
    cal = rd("draft_tests_calibration").set_index("tech_center")
    rows = [[tc, f"{int(cal.loc[tc, 'strict_grants']):,}", f"{int(cal.loc[tc, 'lenient_grants']):,}",
             cin(cal.loc[tc, "strict_minus_lenient_resid"], cal.loc[tc, "se"], 3)] for tc in ["3600", "1600"]]
    tabular(TEX / "cit_tab_calibration.tex", "lrrr",
            ["TC", "Strict examiners' grants", "Lenient examiners' grants", "Strict $-$ lenient residual"], rows)
    for tc, k in [("3600", "ThreeSix"), ("1600", "OneSix")]:
        N[f"cCal{k}"] = num(cal.loc[tc, "strict_minus_lenient_resid"], 3)
        N[f"cCal{k}CI"] = cin(cal.loc[tc, "strict_minus_lenient_resid"], cal.loc[tc, "se"], 3).split(" ", 1)[1]

    # ---------------------------------------------------------------- exclusion and designs (27, 21, 22)
    ex = rd("citation_robustness_exclusion")
    exb = rd("citation_robustness_exclusion_bias").set_index("sample")
    rows = []
    for v in ["Claims at grant", "Pendency to grant (years)"]:
        t = ex[ex.variable == v].set_index("sample")
        rows.append([v] + [f"{cin(t.loc[x, 'coef_per_10pp'], t.loc[x, 'se'])} ({num(t.loc[x, 'mean'], 1)})"
                           for x in ["3600", "1600", "Both"]])
    tabular(TEX / "cit_tab_exclusion.tex", "lrrr", ["", "TC 3600", "TC 1600", "Both"], rows)
    N["cExclBias"] = num(exb.loc["3600", "implied_bias"])
    t = ex[ex.variable == "Claims at grant"].set_index("sample")
    N["cClaims"] = num(t.loc["3600", "coef_per_10pp"])
    t = ex[ex.variable == "Pendency to grant (years)"].set_index("sample")
    N["cPendency"] = num(-t.loc["3600", "coef_per_10pp"])
    de = rd("citation_robustness_designs_estimates")
    dc = rd("citation_robustness_designs_contrasts")
    lr = rd("citations_legal_robustness")
    lc = rd("citations_legal_robustness_contrasts")
    mv = rd("examiners_moved_robustness")
    rows = []

    def cell(e, series, kind, group="All"):
        x = e[(e.series == series) & (e.kind == kind) & (e.group == group)].iloc[0]
        return cin(x.estimate, x.se)
    for lab in de.design.unique():
        e = de[(de.design == lab) & (de["sample"] == "3600")]
        cc = dc[(dc.design == lab) & (dc["sample"] == "3600")]
        fm = cc[(cc.series == HUMAN) & (cc.kind == "marginal") & (cc.contrast == "Female − male")].iloc[0]
        rows.append([esc(lab), cell(e, HUMAN, "marginal"), cin(fm.estimate, fm.se)])
    for lab in lr.design.unique()[1:]:
        e = lr[(lr.design == lab) & (lr["sample"] == "3600")]
        cc = lc[(lc.design == lab) & (lc["sample"] == "3600")]
        fm = cc[(cc.series == HUMAN) & (cc.kind == "marginal") & (cc.contrast == "Female − male")].iloc[0]
        short = ("Without classes 705, 435, 436, 536" if lab.startswith("Without")
                 else "Cells $\\times$ first-action year")
        rows.append([short, cell(e, HUMAN, "marginal"), cin(fm.estimate, fm.se)])
    e = mv[(mv.design == "Applications docketed only once") & (mv["sample"] == "3600")]
    rows.append(["Docketed only once", cell(e, HUMAN, "marginal"), "--"])
    tabular(TEX / "cit_tab_designs.tex", "lrr", ["Design (TC 3600)", "Marginal citations", "F $-$ M"], rows)
    exp_ = rd("citations_legal_exposure").set_index("tech_center")
    N["cAliceShare"] = num(100 * exp_.loc["3600", "share_affected_classes"], 0)
    N["cAfterAlice"] = num(100 * exp_.loc["3600", "share_first_action_after_decision"], 0)
    N["cMayoShare"] = num(100 * exp_.loc["1600", "share_affected_classes"], 0)

    # ---------------------------------------------------------------- C1 (21, 25)
    est, con = rd("citations_estimates"), rd("citations_contrasts")
    de1 = rd("draft_tests_c1_estimates")
    series = [(NO_MODEL, "No model (mean of granted)", est), ("Train on grant", "Trained on grant", est),
              ("Train on first action", "Trained on first action", est),
              ("Train on citations (granted only)", "Trained on citations", est),
              ("Train on citations + entity status", "Trained on citations + entity status", est),
              ("Draft C1 model (rank matching)", "Draft C1: rank matching", de1),
              ("LLM, zero-shot", "LLM, zero-shot", est),
              ("LLM, lenient examiners' examples", "LLM, lenient examples", est),
              ("LLM, strict examiners' examples", "LLM, strict examples", est)]
    rows = []
    for tc in ["3600", "1600"]:
        t = est[(est["sample"] == tc) & (est.series == HUMAN) & (est.kind == "marginal") & (est.group == "All")].iloc[0]
        rows.append([f"\\textit{{TC {tc}}}: truth", "--", cin(t.estimate, t.se)])
        for ser, lab, src in series:
            x = src[(src["sample"] == tc) & (src.series == ser) & (src.group == "All")]
            imp = x[x.kind == "implied"].iloc[0].estimate
            er = x[x.kind == "error"].iloc[0]
            rows.append([esc(lab), num(imp), cin(er.estimate, er.se)])
    tabular(TEX / "cit_tab_c1.tex", "lrr", ["", "Implied", "Signed error (citations)"], rows)
    t = est[(est["sample"] == "3600") & (est.series == HUMAN) & (est.kind == "marginal") & (est.group == "All")].iloc[0]
    N["cMarg"], N["cMargCI"] = num(t.estimate), cin(t.estimate, t.se).split(" ", 1)[1]
    N["cMeanGranted"] = num(t.mean_granted)
    for ser, k, src in [("Train on grant", "Grant", est), ("Train on citations (granted only)", "Cites", est),
                        ("LLM, zero-shot", "LLM", est), ("Draft C1 model (rank matching)", "Draft", de1),
                        ("Model trained on all granted", "AllGranted", de1)]:
        x = src[(src["sample"] == "3600") & (src.series == ser) & (src.group == "All")]
        er = x[x.kind == "error"].iloc[0]
        N[f"cErr{k}"], N[f"cErr{k}CI"] = num(er.estimate), cin(er.estimate, er.se).split(" ", 1)[1]
        N[f"cImp{k}"] = num(x[x.kind == "implied"].iloc[0].estimate)
    cons = rd("draft_tests_c1_construction").set_index("tech_center")
    rows = [[tc, num(100 * cons.loc[tc, "strict_grant_rate"], 1), num(100 * cons.loc[tc, "lenient_grant_rate"], 1),
             num(100 * cons.loc[tc, "share_taken"], 1), f"{int(cons.loc[tc, 'counterfactual_patents']):,}",
             num(cons.loc[tc, "counterfactual_mean_cites"]), num(100 * cons.loc[tc, "counterfactual_zero_share"], 0),
             num(cons.loc[tc, "lenient_granted_mean_cites"])] for tc in ["3600", "1600"]]
    tabular(TEX / "cit_tab_draft_construction.tex", "lrrrrrrr",
            ["TC", "Strict grant (\\%)", "Lenient grant (\\%)", "Share taken (\\%)", "Patents", "Mean citations",
             "Zero citations (\\%)", "All lenient grants"], rows)
    N["cDraftShare"] = num(100 * cons.loc["3600", "share_taken"], 1)
    N["cDraftMean"] = num(cons.loc["3600", "counterfactual_mean_cites"])
    N["cDraftZero"] = num(100 * cons.loc["3600", "counterfactual_zero_share"], 0)
    sp = rd("citations_split")
    x = sp[(sp["sample"] == "3600") & (sp.series == HUMAN)].iloc[0]
    N["cStrictGranted"], N["cLenientGranted"] = num(x.strict_granted, 1), num(x.lenient_granted, 1)
    N["cStrictCites"], N["cLenientCites"] = num(x.strict_cites, 1), num(x.lenient_cites, 1)
    N["cSplitMarg"], N["cExtra"] = num(x.marginal), num(x.lenient_granted - x.strict_granted, 1)
    gl = rd("citations_granted_by_leniency").set_index("sample")
    N["cGrantedLS"] = num(gl.loc["3600", "lenient_minus_strict"])
    N["cGrantedLSCI"] = cin(gl.loc["3600", "lenient_minus_strict"], gl.loc["3600", "lenient_minus_strict_se"]).split(
        " ", 1)[1]
    ob = rd("citations_ols_vs_boosting")
    rows = [[r.tech_center, f"{100 * r.top:.0f}\\%", num(100 * r.overlap, 1), num(r.cites_both), num(r.cites_gbm_only),
             num(r.cites_ols_only), num(r.cites_all_gbm), num(r.cites_all_ols), num(r.cites_mean)]
            for r in ob.itertuples()]
    tabular(TEX / "cit_tab_ols_boosting.tex", "llrrrrrrr",
            ["TC", "Top", "Overlap (\\%)", "Both", "Boosting only", "OLS only", "All boosting", "All OLS", "Mean"],
            rows)
    o1 = ob[(ob.tech_center == "3600") & (ob.top == 0.01)].iloc[0]
    o25 = ob[(ob.tech_center == "3600") & (ob.top == 0.25)].iloc[0]
    N["cOverlap"], N["cBoostOnly"], N["cOLSOnly"] = num(100 * o1.overlap, 0), num(o1.cites_gbm_only, 1), num(
        o1.cites_ols_only, 1)
    N["cOverlapQuart"] = num(100 * o25.overlap, 0)

    # ---------------------------------------------------------------- contraction (26)
    ce = rd("contraction_cites_evaluators")
    rows, groups = [], {}
    for smp, slab in [("3600", "Technology center 3600"), ("Both", "Both technology centers")]:
        groups[len(rows)] = slab
        e = ce[(ce["sample"] == smp) & (ce.quintile == "2–5 (mean)")]
        for ev, lab in [("Examiners", "Stricter examiners"), ("Random", "Random"),
                        ("Train on grant", "Trained on grant"), ("Train on first action", "Trained on first action"),
                        ("Train on citations (granted only)", "Trained on citations"),
                        ("LLM, zero-shot", "LLM, zero-shot")]:
            t = e[e.evaluator == ev].set_index("measure")
            m, f = t.loc["mean_cites"], t.loc["female_share"]
            rows.append([lab, f"{num(m.estimate)} [{num(m.lo)}, {num(m.hi)}]",
                         "" if ev == "Examiners" else f"{num(m['diff'])} [{num(m.diff_lo)}, {num(m.diff_hi)}]",
                         f"{num(100 * f.estimate, 1)} [{num(100 * f.lo, 1)}, {num(100 * f.hi, 1)}]"])
            if smp == "3600":
                k = {"Examiners": "Exam", "Random": "Random", "Train on grant": "Grant",
                     "Train on first action": "First",
                     "Train on citations (granted only)": "Cites", "LLM, zero-shot": "LLM"}[ev]
                N[f"cCtr{k}"] = num(m.estimate)
                if ev != "Examiners":
                    N[f"cCtr{k}Diff"] = num(m["diff"])
                    N[f"cCtr{k}DiffCI"] = f"[{num(m.diff_lo)}, {num(m.diff_hi)}]"
    tabular(TEX / "cit_tab_contraction.tex", "lrrr",
            ["Extra rejections by", "Mean citations", "vs examiners", "Female share (\\%)"], rows, groups)

    # ---------------------------------------------------------------- C3: groups (21, 17, 10)
    rows, groups = [], {}
    for tc in ["3600", "1600"]:
        groups[len(rows)] = f"Technology center {tc}"
        e = est[est["sample"] == tc]
        cc = con[con["sample"] == tc]

        def g3(ser, kind):
            f = e[(e.series == ser) & (e.kind == kind) & (e.group == "Female")].iloc[0]
            m = e[(e.series == ser) & (e.kind == kind) & (e.group == "Male")].iloc[0]
            d_ = cc[(cc.series == ser) & (cc.kind == kind) & (cc.contrast == "Female − male")].iloc[0]
            return [cin(f.estimate, f.se), cin(m.estimate, m.se), cin(d_.estimate, d_.se)]
        rows.append(["Marginal citations (outcome test)"] + g3(HUMAN, "marginal"))
    tabular(TEX / "cit_tab_c3_errors.tex", "lrrr", ["", "Female", "Male", "F $-$ M"], rows, groups)
    e = est[est["sample"] == "3600"]
    cc = con[con["sample"] == "3600"]
    f = e[(e.series == HUMAN) & (e.kind == "marginal") & (e.group == "Female")].iloc[0]
    m = e[(e.series == HUMAN) & (e.kind == "marginal") & (e.group == "Male")].iloc[0]
    d_ = cc[(cc.series == HUMAN) & (cc.kind == "marginal") & (cc.contrast == "Female − male")].iloc[0]
    N["cFem"], N["cMale"], N["cFM"] = num(f.estimate, 1), num(m.estimate, 1), num(d_.estimate, 1)
    N["cFMCI"] = cin(d_.estimate, d_.se, 1).split(" ", 1)[1]
    for ser, k in [("LLM, zero-shot", "LLM"), ("Train on grant", "Grant"), ("Train on citations (granted only)",
                                                                           "Cites"), (NO_MODEL, "None")]:
        x = e[(e.series == ser) & (e.kind == "error") & (e.group == "Female")].iloc[0]
        N[f"cFemErr{k}"], N[f"cFemErr{k}CI"] = num(x.estimate), cin(x.estimate, x.se).split(" ", 1)[1]
    gp = rd("citations_gaps")
    rows = []
    for ser, lab in [("Human decision makers", "Human examiners"), ("Train on grant", "Trained on grant"),
                     ("Train on grant + names", "Trained on grant + names"),
                     ("Train on first action", "Trained on first action"),
                     ("Train on citations (granted only)", "Trained on citations"),
                     ("Train on citations + names", "Trained on citations + names"),
                     ("LLM, zero-shot", "LLM, zero-shot"),
                     ("LLM, lenient examiners' examples", "LLM, lenient examples"),
                     ("LLM, strict examiners' examples", "LLM, strict examples")]:
        r3 = gp[(gp["sample"] == "3600") & (gp.series == ser)].iloc[0]
        r1 = gp[(gp["sample"] == "1600") & (gp.series == ser)].iloc[0]
        rows.append([esc(lab), cin(r3.gap, r3.se, 1, 100), num(r3.ratio), cin(r1.gap, r1.se, 1, 100)])
    tabular(TEX / "cit_tab_c3_gaps.tex", "lrrr", ["", "Gap, TC 3600 (pp)", "$\\times$ human", "Gap, TC 1600 (pp)"],
            rows)
    g3 = gp[gp["sample"] == "3600"].set_index("series")
    N["cHumanGap"] = num(100 * g3.loc["Human decision makers", "gap"], 1)
    for ser, k in [("Train on grant", "Grant"), ("Train on grant + names", "GrantNames"),
                   ("Train on citations (granted only)", "Cites"), ("Train on citations + names", "CitesNames"),
                   ("LLM, zero-shot", "LLM")]:
        N[f"cGap{k}"], N[f"cRatio{k}"] = num(100 * g3.loc[ser, "gap"], 1), num(g3.loc[ser, "ratio"])
    # decision–outcome curve with citations
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        ser = f"blend {w:.1f}"
        g_ = g3.loc[ser]
        rows.append([f"{w:.1f}", cin(g_.gap, g_.se, 1, 100), num(g_.ratio)])
    tabular(TEX / "cit_tab_curve.tex", "lrr", ["Weight on grant", "Gender gap (pp)", "$\\times$ human"], rows)
    N["cCurveLow"], N["cCurveHigh"] = num(g3.loc["blend 0.0", "ratio"]), num(g3.loc["blend 1.0", "ratio"])
    sw = pd.read_csv(T / "llm_prompting_full.csv", dtype={"tech_center": str})
    s3 = sw[(sw.tech_center == "3600") & (sw.cond == "Name swap: female − male (log-odds)")].iloc[0]
    N["cSwap"], N["cSwapCI"] = num(s3.mean_score), cin(s3.mean_score, s3.se).split(" ", 1)[1]

    # ---------------------------------------------------------------- memorization (23)
    mg = rd("memorization_granted_vs_rejected").set_index("tech_center")
    ma = rd("memorization_auc_by_half")
    mgap = rd("memorization_gaps_by_half")
    rows = []
    for half in ["More familiar half", "Less familiar half"]:
        a3 = ma[(ma.half == half) & (ma.tech_center == "3600")].iloc[0]
        a1 = ma[(ma.half == half) & (ma.tech_center == "1600")].iloc[0]
        gg = mgap[(mgap.half == half) & (mgap["sample"] == "3600") & (mgap.series == "LLM, zero-shot")].iloc[0]
        rows.append([half, num(a3.auc_llm_grant, 3), num(a3.auc_grant_model, 3), num(a1.auc_llm_grant, 3),
                     num(a1.auc_grant_model, 3), cin(gg.gap, gg.se, 1, 100)])
    tabular(TEX / "cit_tab_memorization.tex", "lrrrrr",
            ["", "AUC LLM, 3600", "AUC text model, 3600", "AUC LLM, 1600", "AUC text model, 1600",
             "LLM gender gap, 3600 (pp)"], rows)
    N["cMemSD"], N["cMemAUC"] = num(mg.loc["3600", "granted_minus_rejected_sd"]), num(
        mg.loc["3600", "auc_min_k_for_grant"])

    lines = ["% Generated by code/29_citation_tables.py; do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items()]
    (TEX / "cit_numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(N)} macros; tables: {sorted(p.name for p in TEX.glob('cit_tab_*.tex'))}")


if __name__ == "__main__":
    main()
