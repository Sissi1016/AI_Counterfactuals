"""Infer inventor gender from first names (WIPO WGND 2.0) and aggregate to the application level.

Rules
- Name normalization: lowercase and drop periods; if the first name is a single-letter initial (e.g. "J."), use the
  middle name instead. Match on the full first name first, then on its first word.
- Look up the name-by-country table first (wgnd_2_0_name-gender-code): assign a gender only if its share wgt >= 0.9;
  otherwise the gender is unknown.
- If the name is not listed for that country, fall back to the global table (wgnd_2_0_name-gender_nocode) and accept
  only M or F.

Input: data/processed/patex_all_inventors.parquet, data/wgnd/*.tab, data/processed/applications_2004_2018.parquet
Output: data/processed/inventor_gender.parquet (inventor level), data/processed/app_gender.parquet (application level)
"""
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
W = ROOT / "data/wgnd"
THRESHOLD = 0.9
NOCODE_BYTES = 56962284  # full size of the global dictionary; skip the fallback match if the download is incomplete


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'")

    con.execute(f"""
    CREATE TEMP TABLE wcode AS
    SELECT name, code,
           max(wgt) FILTER (WHERE gender = 'F') AS p_f,
           max(wgt) FILTER (WHERE gender = 'M') AS p_m
    FROM read_csv('{W}/wgnd_2_0_name-gender-code.tab', delim = '\t', header = true, quote = '"')
    GROUP BY 1, 2
    """)
    nocode = W / "wgnd_2_0_name-gender_nocode.tab"
    if nocode.exists() and nocode.stat().st_size == NOCODE_BYTES:
        con.execute(f"""
        CREATE TEMP TABLE wnocode AS
        SELECT name, any_value(gender) AS gender
        FROM read_csv('{nocode}', delim = '\t', header = true, quote = '"')
        WHERE gender IN ('M', 'F')
        GROUP BY 1
        HAVING count(DISTINCT gender) = 1
        """)
    else:
        print("Warning: global dictionary incomplete; skipping fallback matching. Rerun this script after the download.")
        con.execute("CREATE TEMP TABLE wnocode (name VARCHAR, gender VARCHAR)")

    con.execute(f"""
    CREATE TEMP TABLE inv AS
    WITH base AS (
      SELECT i.application_number,
             TRY_CAST(i.inventor_rank AS INT) AS inventor_rank,
             upper(coalesce(i.inventor_country_code, '')) AS code,
             trim(regexp_replace(lower(coalesce(i.inventor_name_first, '')), '[.]', ' ', 'g')) AS fn,
             trim(regexp_replace(lower(coalesce(i.inventor_name_middle, '')), '[.]', ' ', 'g')) AS mn
      FROM '{P}/patex_all_inventors.parquet' i
      SEMI JOIN '{P}/applications_2004_2018.parquet' a ON i.application_number = a.application_number
    ),
    keyed AS (
      SELECT *,
             CASE WHEN length(split_part(fn, ' ', 1)) <= 1 AND mn <> '' THEN mn ELSE fn END AS name_full
      FROM base
    )
    SELECT *, split_part(name_full, ' ', 1) AS name_first FROM keyed
    """)

    con.execute(f"""
    COPY (
      SELECT i.application_number, i.inventor_rank, i.code, i.name_full,
             coalesce(c1.p_f, c2.p_f) AS p_f, coalesce(c1.p_m, c2.p_m) AS p_m,
             CASE
               WHEN coalesce(c1.p_f, c2.p_f) >= {THRESHOLD} THEN 'F'
               WHEN coalesce(c1.p_m, c2.p_m) >= {THRESHOLD} THEN 'M'
               WHEN c1.name IS NULL AND c2.name IS NULL THEN coalesce(n1.gender, n2.gender)
               ELSE NULL
             END AS gender
      FROM inv i
      LEFT JOIN wcode c1   ON c1.name = i.name_full  AND c1.code = i.code
      LEFT JOIN wcode c2   ON c2.name = i.name_first AND c2.code = i.code AND c1.name IS NULL
      LEFT JOIN wnocode n1 ON n1.name = i.name_full
      LEFT JOIN wnocode n2 ON n2.name = i.name_first AND n1.name IS NULL
    ) TO '{P}/inventor_gender.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)

    con.execute(f"""
    COPY (
      SELECT application_number,
             count(*)                                   AS n_inventors,
             count(gender)                              AS n_gender_known,
             count(*) FILTER (WHERE gender = 'F')       AS n_female,
             any_value(gender) FILTER (WHERE inventor_rank = 1) AS first_inventor_gender,
             any_value(code)   FILTER (WHERE inventor_rank = 1) AS first_inventor_country
      FROM '{P}/inventor_gender.parquet'
      GROUP BY 1
    ) TO '{P}/app_gender.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)

    q = lambda s: print(con.execute(s).df().to_string(index=False), "\n")
    q(f"""SELECT count(*) n_inventors, avg((gender IS NOT NULL)::INT) share_known,
                 avg((gender = 'F')::INT) FILTER (WHERE gender IS NOT NULL) female_share_known
          FROM '{P}/inventor_gender.parquet'""")
    q(f"""SELECT code, count(*) n, round(avg((gender IS NOT NULL)::INT), 3) share_known,
                 round(avg((gender = 'F')::INT) FILTER (WHERE gender IS NOT NULL), 3) female_share
          FROM '{P}/inventor_gender.parquet' GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")


if __name__ == "__main__":
    main()
