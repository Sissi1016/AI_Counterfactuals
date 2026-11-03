"""Pilot sample and outcome labels.

Sample: technology centers 3600 (transportation, e-commerce & business methods; main) and 1600 (biotech & organic
chemistry; secondary); filed 2010–2014; in HUPD (full text available); not a continuation; first-named inventor in
the U.S. with known gender; has a first action on the merits; examiner handled at least 20 applications in the sample
period.

Outcome labels (granted patents only)
- cites_5y: citations from other U.S. patents within 5 years of grant (PatentsView); cites_5y_examiner counts only
  examiner-added citations.
- renew_4y: whether the 4th-year maintenance fee was paid (M1551 / M2551 / M3551 / F170).

Input: data/processed/{applications_leniency,app_gender}.parquet, data/patentsview/tsv/*.tsv,
       data/maintenance_fee/txt/MaintFeeEvents_*.txt
Output: data/processed/pilot_sample.parquet
"""
import glob
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
PV = ROOT / "data/patentsview/tsv"
MF = sorted(glob.glob(str(ROOT / "data/maintenance_fee/txt/MaintFeeEvents_2*.txt")))[-1]
TCS = ("3600", "1600")
YEARS = (2010, 2014)
RENEW_CODES = ("M1551", "M2551", "M3551", "F170")


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'")

    con.execute(f"""
    CREATE TEMP TABLE pilot AS
    SELECT a.application_number, a.examiner_id, a.art_unit, a.au_year, a.tech_center, a.filing_year,
           a.main_cpc_label, a.hupd_decision,
           a.fa_allow, a.granted, a.abandoned, a.n_rejections, a.n_rce, a.appealed,
           a.lenient_fa, a.lenient_grant, a.examiner_n,
           (a.small_entity_indicator IN ('SMALL', 'MICRO'))::INT AS small_entity,
           ltrim(a.patent_number, '0') AS patent_number, a.patent_issue_date,
           g.n_inventors, g.n_female, g.n_gender_known,
           (g.first_inventor_gender = 'F')::INT AS female
    FROM '{P}/applications_leniency.parquet' a
    JOIN '{P}/app_gender.parquet' g USING (application_number)
    WHERE a.tech_center IN {TCS}
      AND a.filing_year BETWEEN {YEARS[0]} AND {YEARS[1]}
      AND a.in_hupd AND NOT a.continuation
      AND g.first_inventor_country = 'US' AND g.first_inventor_gender IN ('F', 'M')
    """)

    con.execute(f"""
    CREATE TEMP TABLE gp AS
    SELECT patent_id, CAST(patent_date AS DATE) AS patent_date
    FROM read_csv('{PV}/g_patent.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)
    """)
    con.execute(f"""
    CREATE TEMP TABLE cites AS
    WITH c AS (
      SELECT patent_id AS citing, citation_patent_id AS cited, citation_category
      FROM read_csv('{PV}/g_us_patent_citation.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)
      WHERE citation_patent_id IN (SELECT patent_number FROM pilot WHERE patent_number IS NOT NULL)
    )
    SELECT c.cited AS patent_number,
           count(DISTINCT c.citing) AS cites_5y,
           count(DISTINCT c.citing) FILTER (WHERE c.citation_category = 'cited by examiner') AS cites_5y_examiner
    FROM c
    JOIN gp gc ON gc.patent_id = c.citing
    JOIN gp gd ON gd.patent_id = c.cited
    WHERE gc.patent_date <= gd.patent_date + INTERVAL 5 YEAR
    GROUP BY 1
    """)

    con.execute(f"""
    CREATE TEMP TABLE renew AS
    WITH m AS (
      SELECT string_split(trim(regexp_replace(line, '\\s+', ' ', 'g')), ' ') AS f
      FROM read_csv('{MF}', delim = '\x01', header = false, columns = {{'line': 'VARCHAR'}})
    )
    SELECT ltrim(f[1], '0') AS patent_number,
           max((f[7] IN {RENEW_CODES})::INT) AS renew_4y
    FROM m GROUP BY 1
    """)

    con.execute(f"""
    COPY (
      SELECT p.*,
             CASE WHEN p.granted = 1 THEN coalesce(c.cites_5y, 0) END          AS cites_5y,
             CASE WHEN p.granted = 1 THEN coalesce(c.cites_5y_examiner, 0) END AS cites_5y_examiner,
             CASE WHEN p.granted = 1 THEN coalesce(r.renew_4y, 0) END          AS renew_4y
      FROM pilot p
      LEFT JOIN cites c USING (patent_number)
      LEFT JOIN renew r USING (patent_number)
    ) TO '{P}/pilot_sample.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)

    print(con.execute(f"""
      SELECT tech_center, count(*) n, count(DISTINCT examiner_id) examiners,
             round(avg(female), 3) female, round(avg(fa_allow), 3) fa_allow, round(avg(granted), 3) granted,
             round(avg(n_rejections), 2) rejections, round(avg(cites_5y), 2) cites_5y,
             round(median(cites_5y), 1) med_cites, round(avg(renew_4y), 3) renew_4y
      FROM '{P}/pilot_sample.parquet' GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


if __name__ == "__main__":
    main()
