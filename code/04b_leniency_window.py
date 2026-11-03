"""Examiner leniency from the same period (robustness for 04).

04 measures an examiner's leniency on all of the examiner's other applications filed 2004–2018, which assumes the
examiner's bar is stable over fifteen years. Examiners' grant rates rise with tenure (37), so here leniency is the
grant rate on the examiner's other applications filed within two years of the application's filing year (t − 2 to
t + 2, leave-one-out), kept when at least MIN_APPS other applications fall in that window.

Input: data/processed/applications_leniency.parquet (04)
Output: data/processed/leniency_window.parquet (application_number, lenient_win, n_win)
"""
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
MIN_APPS, HALF = 20, 2


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    con.execute(f"""
    CREATE TEMP TABLE ey AS
    SELECT examiner_id, filing_year, count(*) AS n, sum(granted) AS s
    FROM '{P}/applications_leniency.parquet' GROUP BY 1, 2""")
    con.execute(f"""
    COPY (
      SELECT b.application_number, (w.s - b.granted) / (w.n - 1) AS lenient_win, w.n - 1 AS n_win
      FROM '{P}/applications_leniency.parquet' b
      JOIN (SELECT a.examiner_id, a.filing_year, sum(c.n) AS n, sum(c.s) AS s
            FROM (SELECT DISTINCT examiner_id, filing_year FROM ey) a
            JOIN ey c ON c.examiner_id = a.examiner_id AND c.filing_year BETWEEN a.filing_year - {HALF}
                                                                           AND a.filing_year + {HALF}
            GROUP BY 1, 2) w USING (examiner_id, filing_year)
      WHERE w.n - 1 >= {MIN_APPS}
    ) TO '{P}/leniency_window.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    print(con.execute(f"""
      SELECT count(*) n, round(avg(n_win)) mean_n_win,
             round(corr(w.lenient_win, s.lenient_grant), 3) corr_with_04,
             round(avg((w.lenient_win IS NOT NULL)::INT), 4) share_in_main_sample
      FROM '{P}/all_sample.parquet' s LEFT JOIN '{P}/leniency_window.parquet' w USING (application_number)
      """).df().T.to_string())


if __name__ == "__main__":
    main()
