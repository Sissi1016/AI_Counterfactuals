"""Examiners: how they are identified, how often an application changes examiner, how leniency is measured, and what
goes with it.

1. Identification. PatEx has no examiner ID; an examiner is the string examiner_full_name ("LAST, FIRST MIDDLE"), as
   in 02 and 04. Coverage in the 2004–2018 base of 04 and in the pilot frame (TC 3600 and 1600, filed 2010–2014,
   in HUPD, not a continuation): applications without an examiner name, and applications whose examiner has too few
   other applications for a leave-one-out leniency. Names that appear in more than one technology center are possible
   homonyms or transfers. The examiner recorded in PatEx (June 2023) is compared with HUPD's (February 2022).
2. Reassignment. Docketing events per pilot application: DOCK (docketed to an examiner), DKST (docketed or
   redocketed) and DELD (deleted from an examiner's docket). The examiner in PatEx is the examiner of record at the
   end. Re-docketing is routine (cases return to the docket after replies), so several DOCK events do not mean a new
   examiner; a deletion from a docket is the direct signal. The marginal check with citations (21) is repeated on
   applications docketed only once.
3. Leniency. The examiner's leave-one-out grant rate over the 2004–2018 base (04). Strict and lenient examiners are
   those below and above the mean leniency of their art unit × year × USPC class cell; the 2SLS uses leniency as a
   continuous instrument. Spread of leniency within cells.
4. What goes with leniency, among examiners with at least 50 applications in the main sample (38's all_sample, filed 2010–2014; grant rate on those): the examiner's gender (first
   name, WGND 2.0 U.S. table, share ≥ 0.9), seniority (years since the examiner's first application in PatEx, at
   2010) and technology center. And on the applicant side: the first inventor's state by its 2012 presidential vote.

Runs on CPUs only (DuckDB limited to 16 threads).
Input: data/processed/{applications_2004_2018,applications_leniency,examiners,patex_application_data,
       patex_transactions,patex_all_inventors,pilot_sample,citation_scores}.parquet, data/hupd metadata, data/wgnd
Output: output/examiners.md, output/tables/examiners_*.csv
"""
import dataclasses
import pathlib
import warnings

import duckdb
import numpy as np
import pandas as pd

from cfe.check import HUMAN, NO_MODEL, PATENTS, assign_folds, load_record, marginal, ols_coef, recalibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
H = ROOT / "data/hupd/hupd_metadata_2022-02-22.feather"
W = ROOT / "data/wgnd/wgnd_2_0_name-gender-code.tab"
OUTCOME = "cites_ns"
CIT = dataclasses.replace(PATENTS, outcomes=((OUTCOME, "Citations within five years of grant, excluding "
                                                       "self-citations"),))
# 2012 presidential election: states carried by the Democratic candidate (and DC); the rest by the Republican
BLUE = {"CA", "CO", "CT", "DE", "DC", "FL", "HI", "IL", "IA", "ME", "MD", "MA", "MI", "MN", "NV", "NH", "NJ", "NM",
        "NY", "OH", "OR", "PA", "RI", "VT", "VA", "WA", "WI"}
RED = {"AL", "AK", "AZ", "AR", "GA", "ID", "IN", "KS", "KY", "LA", "MS", "MO", "MT", "NE", "NC", "ND", "OK", "SC",
       "SD", "TN", "TX", "UT", "WV", "WY"}


def ci(b, se, nd=2, scale=1.0):
    return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"


def main() -> None:
    warnings.filterwarnings("ignore")
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    con.register("hupd", pd.read_feather(H, columns=["application_number", "examiner_full_name"]))

    # 1. identification
    ident = con.execute(f"""
    WITH b AS (SELECT a.*, l.lenient_grant FROM '{P}/applications_2004_2018.parquet' a
               LEFT JOIN '{P}/applications_leniency.parquet' l USING (application_number)),
         frame AS (SELECT * FROM b WHERE tech_center IN ('3600', '1600') AND filing_year BETWEEN 2010 AND 2014
                   AND in_hupd AND NOT continuation)
    SELECT 'Base 2004–2018' AS sample, count(*) AS applications,
           avg((examiner_id IS NULL OR examiner_id = '')::DOUBLE) AS no_examiner_name,
           avg((lenient_grant IS NULL)::DOUBLE) AS no_leniency FROM b
    UNION ALL
    SELECT 'Pilot frame (TC 3600, 1600; 2010–2014)', count(*),
           avg((examiner_id IS NULL OR examiner_id = '')::DOUBLE), avg((lenient_grant IS NULL)::DOUBLE) FROM frame
    """).df()
    multi = con.execute(f"""
    WITH e AS (SELECT examiner_id, count(DISTINCT tech_center) AS n_tc, count(*) AS n
               FROM '{P}/applications_2004_2018.parquet' WHERE examiner_id IS NOT NULL GROUP BY 1)
    SELECT count(*) AS examiners, avg((n_tc > 1)::DOUBLE) AS share_names_in_several_tcs,
           sum(n) FILTER (WHERE n_tc > 1) / sum(n) AS share_applications_of_those
    FROM e""").df()
    agree = con.execute(f"""
    SELECT avg((upper(trim(p.examiner_id)) = upper(trim(h.examiner_full_name)))::DOUBLE) AS patex_equals_hupd,
           count(*) AS compared
    FROM '{P}/pilot_sample.parquet' p JOIN hupd h USING (application_number)
    WHERE h.examiner_full_name IS NOT NULL""").df()

    # 2. reassignment
    dock = con.execute(f"""
    WITH t AS (SELECT application_number,
                      count(*) FILTER (WHERE event_code = 'DOCK') AS n_dock,
                      count(*) FILTER (WHERE event_code = 'DKST') AS n_dkst,
                      count(*) FILTER (WHERE event_code = 'DELD') AS n_deld
               FROM '{P}/patex_transactions.parquet'
               WHERE application_number IN (SELECT application_number FROM '{P}/pilot_sample.parquet')
               GROUP BY 1)
    SELECT p.application_number, p.tech_center, coalesce(n_dock, 0) AS n_dock, coalesce(n_dkst, 0) AS n_dkst,
           coalesce(n_deld, 0) AS n_deld
    FROM '{P}/pilot_sample.parquet' p LEFT JOIN t USING (application_number)""").df()
    dock["moved"] = (dock.n_deld > 0) | (dock.n_dock > 1)  # not docketed exactly once
    reassign = dock.groupby("tech_center").agg(applications=("application_number", "size"),
                                               docketed_once=("n_dock", lambda x: (x == 1).mean()),
                                               docketed_more_than_once=("n_dock", lambda x: (x > 1).mean()),
                                               deleted_from_a_docket=("n_deld", lambda x: (x > 0).mean()),
                                               not_docketed_exactly_once=("moved", "mean")).reset_index()

    # the marginal check without applications that may have changed examiner
    d = load_record(ROOT)
    c = pd.read_parquet(P / "cites_noself.parquet", columns=["application_number", "cites_5y_noself"])
    d = d.merge(c, on="application_number", how="left")
    d[OUTCOME] = np.where(d.granted == 1, d.cites_5y_noself.fillna(0), np.nan)
    sc = pd.read_parquet(P / "citation_scores.parquet")
    names = ["Train on grant", "Train on citations (granted only)"]
    dd = recalibrate(d.merge(sc[["application_number"] + names], on="application_number", how="left"), names,
                     assign_folds(d, CIT), CIT)
    dd = dd.merge(dock[["application_number", "moved"]], on="application_number", how="left")
    rows = []
    for lab, s in [("All pilot applications", dd), ("Applications docketed only once", dd[~dd.moved])]:
        e, _ = marginal(s, [NO_MODEL] + names, design=CIT)
        e = e[(e.group == "All") & (e.kind.isin(["marginal", "error"]))]
        e["design"] = lab
        rows.append(e)
    moved_est = pd.concat(rows)

    # 3. leniency: spread within cells
    s = d[d.lenient_grant.notna()].copy()
    s["z_within"] = s.lenient_grant - s.groupby("au_class").lenient_grant.transform("mean")
    ex = s.groupby(["tech_center", "au_class", "examiner_id"]).lenient_grant.mean().reset_index()
    spread = pd.DataFrame([dict(tech_center=tc, sd_leniency=x.lenient_grant.std(),
                                sd_within_cells=x.z_within.std(),
                                p10=x.lenient_grant.quantile(0.1), p90=x.lenient_grant.quantile(0.9),
                                examiners_per_cell=ex[ex.tech_center == tc].groupby("au_class").size().mean(),
                                strict_mean=x.loc[x.z_within <= 0, "lenient_grant"].mean(),
                                lenient_mean=x.loc[x.z_within > 0, "lenient_grant"].mean())
                           for tc, x in s.groupby("tech_center")])

    # 4. what goes with leniency (examiner level)
    con.execute(f"""
    CREATE TEMP TABLE wg AS
    SELECT lower(name) AS name, max(wgt) FILTER (WHERE gender = 'F') AS p_f, max(wgt) FILTER (WHERE gender = 'M') AS p_m
    FROM read_csv('{W}', delim = '\t', header = true, quote = '"') WHERE code = 'US' GROUP BY 1""")
    exl = con.execute(f"""
    WITH first_seen AS (SELECT examiner_full_name AS examiner_id,
                               min(year(CAST(filing_date AS DATE))) AS first_year_all
                        FROM '{P}/patex_application_data.parquet' WHERE examiner_full_name IS NOT NULL GROUP BY 1),
         e AS (SELECT examiner_id, count(*) AS n_e, avg(granted::DOUBLE) AS grant_rate, mode(tech_center) AS main_tc,
                      lower(regexp_extract(examiner_id, ',\\s*([A-Za-z]+)', 1)) AS first_name
               FROM '{P}/all_sample.parquet'
               GROUP BY examiner_id HAVING count(*) >= 50)
    SELECT e.*, f.first_year_all,
           CASE WHEN w.p_f >= 0.9 THEN 'female' WHEN w.p_m >= 0.9 THEN 'male' END AS gender
    FROM e LEFT JOIN first_seen f USING (examiner_id) LEFT JOIN wg w ON w.name = e.first_name""").df()
    exl["seniority_2010"] = (2010 - exl.first_year_all).clip(lower=0)
    exl["seniority"] = pd.cut(exl.seniority_2010, [-1, 2, 5, 10, 100], labels=["0–2", "3–5", "6–10", "over 10"])
    corr = []
    for var in ["gender", "seniority", "main_tc"]:
        for v, x in exl.dropna(subset=[var]).groupby(var, observed=True):
            corr.append(dict(variable=var, value=str(v), examiners=len(x), mean_grant_rate=x.grant_rate.mean(),
                             se=x.grant_rate.std() / np.sqrt(len(x))))
    corr = pd.DataFrame(corr)
    # within technology center: female − male examiners, and per year of seniority
    import pyfixest as pf
    x = exl.dropna(subset=["gender", "seniority_2010"]).copy()
    x["female_examiner"] = (x.gender == "female").astype(float)
    fit = pf.feols("grant_rate ~ female_examiner + seniority_2010 | main_tc", data=x, vcov="hetero")
    t = fit.tidy()
    exreg = pd.DataFrame({"term": t.index, "coef": t["Estimate"].to_numpy(), "se": t["Std. Error"].to_numpy()})

    # applicant side: first inventor's state by its 2012 vote
    st = con.execute(f"""
    SELECT application_number, upper(inventor_region_code) AS state
    FROM '{P}/patex_all_inventors.parquet'
    WHERE application_number IN (SELECT application_number FROM '{P}/pilot_sample.parquet')
      AND inventor_rank = '1'""").df()
    d = d.merge(st, on="application_number", how="left")
    d["blue"] = np.where(d.state.isin(BLUE), 1.0, np.where(d.state.isin(RED), 0.0, np.nan))
    pol = []
    for tc in ["3600", "1600"]:
        s = d[(d.tech_center == tc) & d.blue.notna() & d.lenient_grant.notna()]
        bg, seg = ols_coef(s, "granted", "blue", fe="au_year", design=CIT)
        bl, sel = ols_coef(s, "lenient_grant", "blue", design=CIT)
        pol.append(dict(tech_center=tc, share_blue=s.blue.mean(), grant_gap_blue_minus_red=bg, grant_gap_se=seg,
                        leniency_drawn_blue_minus_red=bl, leniency_se=sel, n=len(s)))
    pol = pd.DataFrame(pol)

    for name, t in [("identification", ident), ("names_in_several_tcs", multi), ("patex_vs_hupd", agree),
                    ("reassignment", reassign), ("moved_robustness", moved_est), ("leniency_spread", spread),
                    ("leniency_correlates", corr), ("leniency_regression", exreg), ("applicant_state", pol)]:
        t.to_csv(OUT / f"tables/examiners_{name}.csv", index=False)

    L = ["# Examiners: identification, reassignment, leniency", "", "## Identification", "",
         "PatEx has no examiner ID; examiners are identified by the full-name string.", "",
         "| Sample | Applications | No examiner name | No leave-one-out leniency |", "|---|---|---|---|"]
    for r in ident.itertuples():
        L.append(f"| {r.sample} | {r.applications:,} | {100 * r.no_examiner_name:.2f}% | {100 * r.no_leniency:.2f}% |")
    r = multi.iloc[0]
    L += ["", f"Examiner names in 2004–2018: {int(r.examiners):,}; {100 * r.share_names_in_several_tcs:.1f}% "
          "appear in more than one technology center (homonyms or transfers), covering "
          f"{100 * r.share_applications_of_those:.1f}% of applications.", "",
          f"Pilot applications whose PatEx examiner (June 2023) equals HUPD's (February 2022): "
          f"{100 * agree.patex_equals_hupd.iloc[0]:.1f}% of {int(agree.compared.iloc[0]):,}.", "",
          "## Reassignment (docketing events)", "",
          "| TC | Applications | Docketed once | Docketed more than once | Deleted from a docket |",
          "|---|---|---|---|---|"]
    for r in reassign.itertuples():
        L.append(f"| {r.tech_center} | {r.applications:,} | {100 * r.docketed_once:.1f}% | "
                 f"{100 * r.docketed_more_than_once:.1f}% | {100 * r.deleted_from_a_docket:.1f}% |")
    L += ["", "Re-docketing is routine (cases return to the docket after replies); a deletion from a docket is the "
          "direct signal of a change of examiner. Marginal check (citations, TC 3600) on applications docketed only "
          "once:", "",
          "| Design | Row | Estimate |", "|---|---|---|"]
    for r in moved_est[moved_est["sample"] == "3600"].itertuples():
        lab = "Truth" if r.series == HUMAN else f"Error: {r.series}"
        if r.kind == "error" or r.series == HUMAN:
            L.append(f"| {r.design} | {lab} | {ci(r.estimate, r.se)} |")
    L += ["", "## Leniency", "", "| TC | SD | SD within cells | P10 | P90 | Examiners per cell | Strict mean | "
          "Lenient mean |", "|---|---|---|---|---|---|---|---|"]
    for r in spread.itertuples():
        L.append(f"| {r.tech_center} | {r.sd_leniency:.3f} | {r.sd_within_cells:.3f} | {r.p10:.3f} | {r.p90:.3f} | "
                 f"{r.examiners_per_cell:.1f} | {r.strict_mean:.3f} | {r.lenient_mean:.3f} |")
    L += ["", "## What goes with leniency (examiners with ≥ 50 applications in the main sample, 2010–2014)", "",
          "| Variable | Value | Examiners | Mean grant rate |", "|---|---|---|---|"]
    for r in corr.itertuples():
        L.append(f"| {r.variable} | {r.value} | {r.examiners:,} | {ci(r.mean_grant_rate, r.se, 1, 100)} |")
    L += ["", "Regression of the examiner's grant rate on gender (female − male), seniority (per year) and technology "
          "center (robust SEs):", "", "| Term | Coefficient (pp) |", "|---|---|"]
    for r in exreg.itertuples():
        L.append(f"| {r.term} | {ci(r.coef, r.se, 2, 100)} |")
    L += ["", "## Applicant side: first inventor's state by its 2012 presidential vote", "",
          "| TC | Share in Democratic states | Grant gap, Democratic − Republican (pp, art unit × year) | "
          "Leniency of the "
          "examiner drawn (pp) |", "|---|---|---|---|"]
    for r in pol.itertuples():
        L.append(f"| {r.tech_center} | {100 * r.share_blue:.1f}% | "
                 f"{ci(r.grant_gap_blue_minus_red, r.grant_gap_se, 2, 100)} | "
                 f"{ci(r.leniency_drawn_blue_minus_red, r.leniency_se, 2, 100)} |")
    (OUT / "examiners.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
