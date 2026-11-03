"""Citations within five years of grant, excluding self-citations, for the granted pilot applications.

A citing patent is matched to its application in PatEx (by patent number), and a citation is a self-citation when the
citing and the cited application share an inventor (last name and first given name, letters only) or an applicant
organization (upper case, punctuation and legal-form words such as INC, CORP, LLC, LTD removed). PatEx lists
organizations as applicants mainly for applications filed after 16 September 2012 (America Invents Act); for earlier
ones the inventor match carries the definition. Citing patents granted after PatEx's June 2023 cut-off cannot be
matched and count as non-self citations; the share matched is reported.

The five-year window and the citation table are those of 06 (PatentsView g_us_patent_citation, citing patent granted
within five years of the cited patent's grant).

Runs on CPUs only (DuckDB limited to 16 threads).
Input: data/processed/{pilot_sample,patex_application_data,patex_all_inventors,patex_all_applicants}.parquet,
       data/patentsview/tsv/{g_patent,g_us_patent_citation}.tsv
Output: data/processed/cites_noself.parquet (application_number, cites_5y_all, cites_5y_self, cites_5y_noself,
        citing_matched_share), output/tables/self_citations_summary.csv
"""
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
PV = ROOT / "data/patentsview/tsv"
OUT = ROOT / "output/tables"

ORG = r"""trim(regexp_replace(regexp_replace(regexp_replace(upper(applicant_organization), '[^A-Z0-9 ]', ' ', 'g'),
      '\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|L L C|LTD|LIMITED|GMBH|AG|SA|PLC|LP|LLP|KK|BV|NV|THE)\b',
      ' ', 'g'), '\s+', ' ', 'g'))"""


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    con.execute(f"""
    CREATE TEMP TABLE pilot AS
    SELECT application_number, patent_number, tech_center FROM '{P}/pilot_sample.parquet'
    WHERE granted = 1 AND patent_number IS NOT NULL""")
    con.execute(f"""
    CREATE TEMP TABLE gp AS
    SELECT patent_id, CAST(patent_date AS DATE) AS patent_date
    FROM read_csv('{PV}/g_patent.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)""")
    con.execute(f"""
    CREATE TEMP TABLE c5 AS
    WITH c AS (
      SELECT DISTINCT patent_id AS citing, citation_patent_id AS cited
      FROM read_csv('{PV}/g_us_patent_citation.tsv', delim = '\t', header = true, quote = '"', all_varchar = true)
      WHERE citation_patent_id IN (SELECT patent_number FROM pilot))
    SELECT c.citing, p.application_number AS cited_app
    FROM c JOIN pilot p ON p.patent_number = c.cited
    JOIN gp gc ON gc.patent_id = c.citing
    JOIN gp gd ON gd.patent_id = c.cited
    WHERE gc.patent_date <= gd.patent_date + INTERVAL 5 YEAR""")
    con.execute(f"""
    CREATE TEMP TABLE capp AS
    SELECT DISTINCT c5.citing, a.application_number AS citing_app
    FROM (SELECT DISTINCT citing FROM c5) c5
    JOIN '{P}/patex_application_data.parquet' a ON a.patent_number = c5.citing""")
    con.execute(f"""
    CREATE TEMP TABLE apps AS
    SELECT application_number FROM pilot UNION SELECT citing_app FROM capp""")
    con.execute(f"""
    CREATE TEMP TABLE inv AS
    SELECT DISTINCT i.application_number,
           regexp_replace(upper(inventor_name_last), '[^A-Z]', '', 'g') || '|' ||
           coalesce(regexp_extract(upper(inventor_name_first), '[A-Z]+'), '') AS k
    FROM '{P}/patex_all_inventors.parquet' i JOIN apps USING (application_number)
    WHERE inventor_name_last IS NOT NULL""")
    con.execute(f"""
    CREATE TEMP TABLE org AS
    SELECT * FROM (
      SELECT DISTINCT o.application_number, {ORG} AS k
      FROM '{P}/patex_all_applicants.parquet' o JOIN apps USING (application_number)
      WHERE applicant_organization IS NOT NULL) WHERE length(k) > 1""")
    con.execute("""
    CREATE TEMP TABLE selfc AS
    SELECT DISTINCT c5.citing, c5.cited_app
    FROM c5 JOIN capp USING (citing)
    JOIN inv a ON a.application_number = capp.citing_app
    JOIN inv b ON b.application_number = c5.cited_app AND a.k = b.k
    UNION
    SELECT DISTINCT c5.citing, c5.cited_app
    FROM c5 JOIN capp USING (citing)
    JOIN org a ON a.application_number = capp.citing_app
    JOIN org b ON b.application_number = c5.cited_app AND a.k = b.k""")
    df = con.execute("""
    WITH t AS (
      SELECT c5.cited_app AS application_number, c5.citing,
             s.citing IS NOT NULL AS is_self, c5.citing IN (SELECT citing FROM capp) AS matched
      FROM c5 LEFT JOIN selfc s ON s.citing = c5.citing AND s.cited_app = c5.cited_app)
    SELECT p.application_number,
           coalesce(count(DISTINCT t.citing), 0) AS cites_5y_all,
           coalesce(count(DISTINCT t.citing) FILTER (WHERE t.is_self), 0) AS cites_5y_self,
           coalesce(count(DISTINCT t.citing) FILTER (WHERE NOT t.is_self), 0) AS cites_5y_noself,
           avg(t.matched::DOUBLE) AS citing_matched_share
    FROM pilot p LEFT JOIN t USING (application_number)
    GROUP BY 1""").df()
    df.to_parquet(P / "cites_noself.parquet", index=False)
    summ = con.execute("""
    SELECT p.tech_center, count(*) AS granted, sum(cites_5y_all) AS citations, sum(cites_5y_self) AS self_citations,
           sum(cites_5y_self) / sum(cites_5y_all) AS self_share, avg(cites_5y_all) AS mean_all,
           avg(cites_5y_noself) AS mean_noself, median(cites_5y_noself) AS median_noself,
           avg((cites_5y_noself = 0)::DOUBLE) AS share_zero,
           sum(citing_matched_share * cites_5y_all) / sum(cites_5y_all) AS citing_matched_share
    FROM df JOIN pilot p USING (application_number) GROUP BY 1 ORDER BY 1""").df()
    summ.to_csv(OUT / "self_citations_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
