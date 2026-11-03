"""Build the application-level dataset: examiner, first-action outcome, rejection rounds, final disposition.

Sample: utility applications in PatEx filed 2004–2018 (the HUPD years), flagged by whether they are in HUPD
(full text available).

Key variables
- examiner_id: PatEx has no examiner ID; examiners are identified by examiner_full_name (trimmed).
- First action on the merits: the earliest of CTNF (non-final rejection), CTFR (final rejection), MN/=. (notice of
  allowance mailed), or CTEQ (Quayle action: claims are allowable and only formal matters remain). CTRS (restriction
  requirement) is not a merits action and is not counted.
  fa_allow = 1 if the first merits action is a notice of allowance or a Quayle action; when a rejection and an
  allowance fall on the same day, it counts as a rejection.
- n_nonfinal / n_final: counts of CTNF and CTFR (the same code on the same day counts once); n_rejections is their sum.
- granted: has a patent number; abandoned: status description contains "bandon"; all others are pending.

Input: data/processed/patex_{application_data,transactions}.parquet, data/hupd/hupd_metadata_2022-02-22.feather
Output: data/processed/applications_2004_2018.parquet
"""
import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = P / "applications_2004_2018.parquet"


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'")

    hupd = pd.read_feather(ROOT / "data/hupd/hupd_metadata_2022-02-22.feather",
                           columns=["application_number", "decision", "main_cpc_label"])
    con.register("hupd", hupd)

    con.execute(f"""
    CREATE TEMP TABLE apps AS
    SELECT application_number,
           CAST(filing_date AS DATE)                      AS filing_date,
           CAST(substr(filing_date, 1, 4) AS INT)         AS filing_year,
           trim(examiner_full_name)                       AS examiner_id,
           examiner_art_unit                              AS art_unit,
           substr(examiner_art_unit, 1, 2) || '00'        AS tech_center,
           substr(examiner_art_unit, 1, 3)                AS work_group,
           small_entity_indicator,
           aia_first_to_file = 'true'                     AS aia,
           patent_number,
           CAST(patent_issue_date AS DATE)                AS patent_issue_date,
           appl_status_desc
    FROM '{P}/patex_application_data.parquet'
    WHERE application_invention_type = 'Utility'
      AND filing_date BETWEEN '2004-01-01' AND '2018-12-31'
      AND examiner_full_name IS NOT NULL
      AND examiner_art_unit IS NOT NULL
    """)

    con.execute(f"""
    CREATE TEMP TABLE ev AS
    SELECT DISTINCT t.application_number, t.event_code, CAST(t.recorded_date AS DATE) AS d
    FROM '{P}/patex_transactions.parquet' t
    SEMI JOIN apps a ON t.application_number = a.application_number
    WHERE t.event_code IN ('CTNF', 'CTFR', 'MN/=.', 'CTEQ', 'CTRS', 'RCEX', 'AP.B')
    """)

    con.execute("""
    CREATE TEMP TABLE evs AS
    SELECT application_number,
           min(d) FILTER (WHERE event_code IN ('CTNF', 'CTFR'))  AS first_rej_date,
           min(d) FILTER (WHERE event_code IN ('MN/=.', 'CTEQ')) AS first_allow_date,
           min(d) FILTER (WHERE event_code = 'CTRS')             AS first_restriction_date,
           count(*) FILTER (WHERE event_code = 'CTNF')           AS n_nonfinal,
           count(*) FILTER (WHERE event_code = 'CTFR')           AS n_final,
           count(*) FILTER (WHERE event_code = 'RCEX')           AS n_rce,
           count(*) FILTER (WHERE event_code = 'AP.B') > 0       AS appealed
    FROM ev GROUP BY 1
    """)

    con.execute(f"""
    COPY (
      SELECT a.*,
             h.decision                         AS hupd_decision,
             h.main_cpc_label,
             h.application_number IS NOT NULL   AS in_hupd,
             coalesce(h.decision LIKE 'CONT-%', false) AS continuation,
             e.first_rej_date, e.first_allow_date, e.first_restriction_date,
             least(e.first_rej_date, e.first_allow_date) AS first_action_date,
             CASE WHEN e.first_rej_date IS NULL AND e.first_allow_date IS NULL THEN NULL
                  WHEN e.first_rej_date IS NULL THEN 1
                  WHEN e.first_allow_date IS NULL THEN 0
                  WHEN e.first_allow_date < e.first_rej_date THEN 1 ELSE 0 END AS fa_allow,
             coalesce(e.n_nonfinal, 0) AS n_nonfinal,
             coalesce(e.n_final, 0)    AS n_final,
             coalesce(e.n_nonfinal, 0) + coalesce(e.n_final, 0) AS n_rejections,
             coalesce(e.n_rce, 0)      AS n_rce,
             coalesce(e.appealed, false) AS appealed,
             (a.patent_number IS NOT NULL)::INT                                   AS granted,
             (a.patent_number IS NULL AND a.appl_status_desc ILIKE '%bandon%')::INT AS abandoned
      FROM apps a
      LEFT JOIN hupd h USING (application_number)
      LEFT JOIN evs e  USING (application_number)
    ) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
    """)

    print(con.execute(f"""
      SELECT count(*) n_apps, count(DISTINCT examiner_id) n_examiners,
             avg(in_hupd::INT) share_in_hupd,
             avg((fa_allow IS NOT NULL)::INT) share_with_first_action,
             avg(fa_allow) fa_allow_rate, avg(granted) grant_rate, avg(abandoned) aband_rate,
             avg(n_rejections) mean_rejections
      FROM '{OUT}'""").df().T.to_string())


if __name__ == "__main__":
    main()
