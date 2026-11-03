"""Examiner leniency (leave-one-out).

For application i handled by examiner e, leniency = mean outcome of e's other applications: (S_e - y_i) / (n_e - 1).
Computed for two outcomes: first-action allowance (fa_allow) and final grant (granted).
Only examiners with at least MIN_APPS applications in the sample period are kept. All later regressions include
art unit × filing year fixed effects, i.e. they compare only applications in the same unit and year that were
assigned to different examiners.

Input: data/processed/applications_2004_2018.parquet
Output: data/processed/applications_leniency.parquet (application level), data/processed/examiners.parquet
        (examiner level)
"""
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
MIN_APPS = 20


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'")
    con.execute(f"""
    CREATE TEMP TABLE base AS
    SELECT * FROM '{P}/applications_2004_2018.parquet'
    WHERE fa_allow IS NOT NULL
    """)
    con.execute(f"""
    CREATE TEMP TABLE ex AS
    SELECT examiner_id, count(*) AS n_e, sum(fa_allow) AS s_fa, sum(granted) AS s_gr,
           avg(fa_allow) AS fa_rate, avg(granted) AS grant_rate, avg(n_rejections) AS mean_rej,
           mode(tech_center) AS main_tc, min(filing_year) AS first_year, max(filing_year) AS last_year
    FROM base GROUP BY 1
    """)
    con.execute(f"COPY ex TO '{P}/examiners.parquet' (FORMAT parquet)")
    con.execute(f"""
    COPY (
      SELECT b.*, e.n_e AS examiner_n,
             (e.s_fa - b.fa_allow) / (e.n_e - 1) AS lenient_fa,
             (e.s_gr - b.granted)  / (e.n_e - 1) AS lenient_grant,
             b.art_unit || '_' || b.filing_year::VARCHAR AS au_year
      FROM base b JOIN ex e USING (examiner_id)
      WHERE e.n_e >= {MIN_APPS}
    ) TO '{P}/applications_leniency.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(con.execute(f"""
      SELECT count(*) n_apps, count(DISTINCT examiner_id) n_examiners, count(DISTINCT au_year) n_cells,
             median(examiner_n) median_apps_per_examiner
      FROM '{P}/applications_leniency.parquet'""").df().T.to_string())
    print(con.execute(f"""
      SELECT main_tc, count(*) n_examiners, round(median(n_e)) med_apps,
             round(avg(fa_rate), 3) fa_rate, round(stddev(fa_rate), 3) sd_fa_rate,
             round(avg(grant_rate), 3) grant_rate, round(stddev(grant_rate), 3) sd_grant_rate
      FROM ex WHERE n_e >= {MIN_APPS} GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


if __name__ == "__main__":
    main()
