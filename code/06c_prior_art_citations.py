"""Prior-art citations: how often later applications cite an application, granted or not.

Patent citations exist only for granted applications. But an application is published about 18 months after filing,
and the publication is prior art whether or not the application is later granted. The USPTO Office Action Research
Dataset lists the prior art cited in the examination of later applications: by the examiner (Form PTO-892 or in the
text of an office action) or by the applicant (Form PTO-1449, information disclosure). We count, for each pilot
application, the later applications that cite its pre-grant publication or, once granted, its patent. Unlike patent
citations, this outcome is observed for rejected applications too.

Definitions:
- the citing application is filed after the cited one and is not the cited application itself;
- self-citations (citing and cited application share an inventor or an applicant organization; the rules of 06b) are
  excluded;
- prior_art_cites: distinct citing applications; prior_art_cites_examiner: those where the examiner cites it (892 or
  in the action); prior_art_cites_pub: citing the publication number only.
The dataset covers office actions mailed from 2008 to June 2017, so later cohorts have shorter windows; the analyses
compare applications within art unit × filing year cells.

Runs on CPUs only (DuckDB limited to 16 threads). The zip members are extracted to a temporary directory under
data/processed and removed at the end.
Input: data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip (citations.csv),
       data/processed/{pilot_sample,all_sample,patex_application_data,patex_all_inventors,patex_all_applicants}.parquet
Output: data/processed/prior_art_cites.parquet, output/tables/prior_art_cites_summary.csv
"""
import pathlib
import shutil
import zipfile

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output/tables"
ZIP = ROOT / "data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip"
ORG = r"""trim(regexp_replace(regexp_replace(regexp_replace(upper(applicant_organization), '[^A-Z0-9 ]', ' ', 'g'),
      '\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|L L C|LTD|LIMITED|GMBH|AG|SA|PLC|LP|LLP|KK|BV|NV|THE)\b',
      ' ', 'g'), '\s+', ' ', 'g'))"""


def main() -> None:
    tmp = P / "_oa_tmp"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extract("citations.csv", tmp)
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    focal = [f"SELECT application_number FROM '{P}/pilot_sample.parquet'"]
    if (P / "all_sample.parquet").exists():
        focal.append(f"SELECT application_number FROM '{P}/all_sample.parquet'")
    con.execute(f"""
    CREATE TEMP TABLE focal AS
    SELECT a.application_number, CAST(a.filing_date AS DATE) AS filing_date,
           regexp_extract(a.earliest_pgpub_number, '([0-9]{{11}})', 1) AS pub,
           CAST(a.earliest_pgpub_date AS DATE) AS pub_date,
           ltrim(a.patent_number, '0') AS pat
    FROM '{P}/patex_application_data.parquet' a
    WHERE a.application_number IN ({' UNION '.join(focal)})""")
    con.execute("""
    CREATE TEMP TABLE keys AS
    SELECT application_number, pub AS k, true AS is_pub FROM focal WHERE pub IS NOT NULL AND pub <> ''
    UNION ALL
    SELECT application_number, pat AS k, false AS is_pub FROM focal WHERE pat IS NOT NULL AND pat <> ''""")
    con.execute(f"""
    CREATE TEMP TABLE c AS
    SELECT DISTINCT lpad(CAST(c.app_id AS VARCHAR), 8, '0') AS citing_app, k.application_number AS cited_app,
           k.is_pub, (c.form892 = '1' OR c.citation_in_oa = '1') AS by_examiner
    FROM read_csv('{tmp}/citations.csv', header = true, all_varchar = true) c
    JOIN keys k ON k.k = ltrim(c.parsed, '0')
    WHERE lpad(CAST(c.app_id AS VARCHAR), 8, '0') <> k.application_number""")
    con.execute(f"""
    CREATE TEMP TABLE c2 AS
    SELECT c.* FROM c
    JOIN '{P}/patex_application_data.parquet' a ON a.application_number = c.citing_app
    JOIN focal f ON f.application_number = c.cited_app
    WHERE CAST(a.filing_date AS DATE) > f.filing_date""")
    con.execute("""
    CREATE TEMP TABLE apps AS
    SELECT application_number FROM focal UNION SELECT DISTINCT citing_app FROM c2""")
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
    SELECT DISTINCT c2.citing_app, c2.cited_app FROM c2
    JOIN inv a ON a.application_number = c2.citing_app
    JOIN inv b ON b.application_number = c2.cited_app AND a.k = b.k
    UNION
    SELECT DISTINCT c2.citing_app, c2.cited_app FROM c2
    JOIN org a ON a.application_number = c2.citing_app
    JOIN org b ON b.application_number = c2.cited_app AND a.k = b.k""")
    df = con.execute("""
    WITH t AS (
      SELECT c2.* FROM c2 LEFT JOIN selfc s USING (citing_app, cited_app) WHERE s.citing_app IS NULL)
    SELECT f.application_number, f.pub IS NOT NULL AND f.pub <> '' AS published,
           coalesce(count(DISTINCT t.citing_app), 0) AS prior_art_cites,
           coalesce(count(DISTINCT t.citing_app) FILTER (WHERE t.by_examiner), 0) AS prior_art_cites_examiner,
           coalesce(count(DISTINCT t.citing_app) FILTER (WHERE t.is_pub), 0) AS prior_art_cites_pub
    FROM focal f LEFT JOIN t ON t.cited_app = f.application_number
    GROUP BY 1, 2""").df()
    df.to_parquet(P / "prior_art_cites.parquet", index=False)
    summ = con.execute(f"""
    SELECT s.tech_center, s.filing_year, s.granted, count(*) AS n, avg(d.published::DOUBLE) AS published,
           avg(d.prior_art_cites) AS mean_cites, avg((d.prior_art_cites = 0)::DOUBLE) AS share_zero,
           avg(d.prior_art_cites_examiner) AS mean_examiner, avg(d.prior_art_cites_pub) AS mean_pub
    FROM df d JOIN '{P}/pilot_sample.parquet' s USING (application_number)
    GROUP BY ALL ORDER BY ALL""").df()
    summ.to_csv(OUT / "prior_art_cites_summary.csv", index=False)
    print(summ.to_string())
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
