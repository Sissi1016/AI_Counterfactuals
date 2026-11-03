"""The human record in every technology center and in earlier cohorts: is the pilot (TC 3600 and 1600, 2010–2014)
special?

For every technology center and two filing cohorts (2004–2009 and 2010–2014), with the pilot's sample rules
(non-continuation applications in HUPD, first inventor in the U.S. with a gender inferred from the first name), the
check without any evaluator: the first stage of grant on examiner leniency, the mean citations of the marginal
applications (2SLS of granted × citations on granted, instrumented by leave-one-out leniency within art unit × year ×
USPC class cells, with controls; errors clustered by examiner) against the mean of granted patents, the
female − male difference in marginal citations (the outcome test), and the gender gap in grants (art unit × year).
Citations: from U.S. patents granted within five years of the cited patent's grant, excluding self-citations
(shared inventor or applicant organization, as in 06b).

Runs on CPUs only (DuckDB limited to 16 threads).
Input: data/processed/{applications_leniency,app_gender,patex_application_data,patex_all_inventors,
       patex_all_applicants}.parquet, data/patentsview/tsv/{g_patent,g_us_patent_citation}.tsv
Output: data/processed/cites_noself_all.parquet, output/human_record.md, output/tables/human_record_*.csv
"""
import dataclasses
import pathlib
import warnings

import duckdb
import numpy as np
import pandas as pd

from cfe.check import HUMAN, PATENTS, marginal, ols_coef

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
PV = ROOT / "data/patentsview/tsv"
OUT = ROOT / "output"
COHORTS = {"2004–2009": (2004, 2009), "2010–2014": (2010, 2014)}
ORG = r"""trim(regexp_replace(regexp_replace(regexp_replace(upper(applicant_organization), '[^A-Z0-9 ]', ' ', 'g'),
      '\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|L L C|LTD|LIMITED|GMBH|AG|SA|PLC|LP|LLP|KK|BV|NV|THE)\b',
      ' ', 'g'), '\s+', ' ', 'g'))"""


def build(con) -> pd.DataFrame:
    con.execute(f"""
    CREATE TEMP TABLE base AS
    SELECT a.application_number, a.examiner_id, a.art_unit, a.au_year, a.tech_center, a.filing_year, a.granted,
           a.lenient_grant, a.first_action_date,
           (a.small_entity_indicator IN ('SMALL', 'MICRO'))::INT AS small_entity,
           ltrim(a.patent_number, '0') AS patent_number, g.n_inventors,
           (g.first_inventor_gender = 'F')::INT AS female
    FROM '{P}/applications_leniency.parquet' a
    JOIN '{P}/app_gender.parquet' g USING (application_number)
    WHERE a.filing_year BETWEEN 2004 AND 2014 AND a.in_hupd AND NOT a.continuation
      AND g.first_inventor_country = 'US' AND g.first_inventor_gender IN ('F', 'M')
      AND a.lenient_grant IS NOT NULL
      AND a.tech_center IN ('1600', '1700', '2100', '2400', '2600', '2800', '3600', '3700')""")
    cached = P / "cites_noself_all.parquet"
    if not cached.exists():
        con.execute("""CREATE TEMP TABLE pat AS SELECT application_number, patent_number FROM base
                       WHERE granted = 1 AND patent_number IS NOT NULL AND patent_number <> ''""")
        con.execute(f"""
        CREATE TEMP TABLE gp AS
        SELECT patent_id, CAST(patent_date AS DATE) AS patent_date
        FROM read_csv('{PV}/g_patent.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)""")
        con.execute(f"""
        CREATE TEMP TABLE c5 AS
        WITH c AS (
          SELECT DISTINCT patent_id AS citing, citation_patent_id AS cited
          FROM read_csv('{PV}/g_us_patent_citation.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)
          WHERE citation_patent_id IN (SELECT patent_number FROM pat))
        SELECT c.citing, p.application_number AS cited_app
        FROM c JOIN pat p ON p.patent_number = c.cited
        JOIN gp gc ON gc.patent_id = c.citing JOIN gp gd ON gd.patent_id = c.cited
        WHERE gc.patent_date <= gd.patent_date + INTERVAL 5 YEAR""")
        con.execute(f"""
        CREATE TEMP TABLE capp AS
        SELECT DISTINCT c.citing, ltrim(a.patent_number, '0') AS pn, a.application_number AS citing_app
        FROM (SELECT DISTINCT citing FROM c5) c
        JOIN '{P}/patex_application_data.parquet' a ON ltrim(a.patent_number, '0') = c.citing""")
        con.execute("CREATE TEMP TABLE apps AS SELECT application_number FROM pat UNION SELECT citing_app FROM capp")
        con.execute(f"""
        CREATE TEMP TABLE inv AS
        SELECT DISTINCT i.application_number,
               regexp_replace(upper(inventor_name_last), '[^A-Z]', '', 'g') || '|' ||
               coalesce(regexp_extract(upper(inventor_name_first), '[A-Z]+'), '') AS k
        FROM '{P}/patex_all_inventors.parquet' i JOIN apps USING (application_number)
        WHERE inventor_name_last IS NOT NULL""")
        con.execute(f"""
        CREATE TEMP TABLE org AS
        SELECT * FROM (SELECT DISTINCT o.application_number, {ORG} AS k
                       FROM '{P}/patex_all_applicants.parquet' o JOIN apps USING (application_number)
                       WHERE applicant_organization IS NOT NULL) WHERE length(k) > 1""")
        con.execute("""
        CREATE TEMP TABLE selfc AS
        SELECT DISTINCT c5.citing, c5.cited_app FROM c5 JOIN capp USING (citing)
        JOIN inv a ON a.application_number = capp.citing_app JOIN inv b ON b.application_number = c5.cited_app
                   AND a.k = b.k
        UNION
        SELECT DISTINCT c5.citing, c5.cited_app FROM c5 JOIN capp USING (citing)
        JOIN org a ON a.application_number = capp.citing_app JOIN org b ON b.application_number = c5.cited_app
                   AND a.k = b.k""")
        con.execute(f"""
        COPY (SELECT p.application_number,
                     count(DISTINCT c5.citing) AS cites_5y_all,
                     count(DISTINCT c5.citing) FILTER (WHERE s.citing IS NULL AND c5.citing IS NOT NULL)
                       AS cites_5y_noself
              FROM pat p LEFT JOIN c5 ON c5.cited_app = p.application_number
              LEFT JOIN selfc s ON s.citing = c5.citing AND s.cited_app = c5.cited_app
              GROUP BY 1) TO '{cached}' (FORMAT PARQUET)""")
    d = con.execute(f"""
    SELECT b.*, coalesce(u.uspc_class, 'none') AS uspc_class, c.cites_5y_noself
    FROM base b LEFT JOIN '{P}/patex_application_data.parquet' u USING (application_number)
    LEFT JOIN '{cached}' c USING (application_number)""").df()
    d["au_class"] = d.au_year + "_" + d.uspc_class
    d["cites_ns"] = np.where(d.granted == 1, d.cites_5y_noself.fillna(0), np.nan)
    d["cohort"] = np.where(d.filing_year <= 2009, "2004–2009", "2010–2014")
    d["unit"] = d.tech_center + " " + d.cohort
    return d


def main() -> None:
    warnings.filterwarnings("ignore")
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '120GB'")
    d = build(con)
    units = sorted(d.unit.unique())
    design = dataclasses.replace(
        PATENTS, unit="unit", units=tuple(units), samples=tuple((u, (u,)) for u in units),
        outcomes=(("cites_ns", "Citations within five years of grant, excluding self-citations"),),
        groups=(("All", None), ("Female", ("female", 1)), ("Male", ("female", 0))),
        contrasts=(("Female − male", ("Female", "Male")),))
    est, con_ = marginal(d, [], design=design)
    rows = []
    for u in units:
        s = d[d.unit == u]
        e = est[(est["sample"] == u) & (est.series == HUMAN)].set_index("group")
        c = con_[(con_["sample"] == u) & (con_.series == HUMAN)].iloc[0]
        gb, gse = ols_coef(s, "granted", "female", fe="au_year", design=design)
        tc, coh = u.split(" ")
        rows.append(dict(tech_center=tc, cohort=coh, applications=len(s), examiners=s.examiner_id.nunique(),
                         female_share=s.female.mean(), grant_rate=s.granted.mean(),
                         mean_cites_granted=s.cites_ns.mean(), first_stage=e.loc["All", "first_stage"],
                         marginal=e.loc["All", "estimate"], marginal_se=e.loc["All", "se"],
                         marginal_female=e.loc["Female", "estimate"], marginal_female_se=e.loc["Female", "se"],
                         marginal_male=e.loc["Male", "estimate"], marginal_male_se=e.loc["Male", "se"],
                         f_minus_m=c.estimate, f_minus_m_se=c.se, grant_gap=gb, grant_gap_se=gse))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tables/human_record_by_tc.csv", index=False)

    def ci(b, se, nd=2, scale=1.0):
        return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"
    L = ["# The human record in every technology center and cohort", "",
         "Sample rules of the pilot (non-continuation, in HUPD, U.S. first inventor with gender known). Citations "
         "within five years of grant, excluding self-citations. Marginal: 2SLS with leave-one-out leniency within art "
         "unit × year × USPC class cells, controls, errors clustered by examiner. Gender gap in grants: female − male, "
         "art unit × year fixed effects (pp).", "",
         "| TC | Cohort | Applications | Female share | Grant rate | First stage | Mean citations, granted | "
         "Marginal citations | Female − male, marginal | Gender gap in grants (pp) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.tech_center} | {r.cohort} | {r.applications:,} | {100 * r.female_share:.1f}% | "
                 f"{100 * r.grant_rate:.1f}% | {r.first_stage:.2f} | {r.mean_cites_granted:.2f} | "
                 f"{ci(r.marginal, r.marginal_se)} | {ci(r.f_minus_m, r.f_minus_m_se)} | "
                 f"{ci(r.grant_gap, r.grant_gap_se, 1, 100)} |")
    (OUT / "human_record.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
