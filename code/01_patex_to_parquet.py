"""Convert PatEx 2022 (CSV) to Parquet. All fields are read as strings to keep leading zeros in application numbers.

Input: data/patex/csv/{application_data,transactions,all_inventors,all_applicants,event_codes}.csv
Output: data/processed/patex_*.parquet
"""
import pathlib
import time

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/patex/csv"
OUT = ROOT / "data/processed"
TABLES = ["application_data", "transactions", "all_inventors", "all_applicants", "event_codes"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 64; SET memory_limit = '200GB'; SET preserve_insertion_order = false")
    for t in TABLES:
        t0 = time.time()
        src, dst = RAW / f"{t}.csv", OUT / f"patex_{t}.parquet"
        con.execute(
            f"""COPY (SELECT * FROM read_csv('{src}', header = true, all_varchar = true))
                TO '{dst}' (FORMAT parquet, COMPRESSION zstd)"""
        )
        n = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
        print(f"{t:18s} rows={n:>12,d}  {time.time() - t0:6.1f}s  -> {dst.name}")


if __name__ == "__main__":
    main()
