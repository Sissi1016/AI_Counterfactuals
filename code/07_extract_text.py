"""Extract pilot-sample texts from the HUPD yearly tarballs, parsing each JSON in memory without unpacking to disk.

Fields kept: title, abstract, full claims, background and summary (first 4,000 characters each), and the HUPD
inventor list (as a JSON string). One process per yearly tarball.

Input: data/hupd/data/{year}.tar.gz, data/processed/pilot_sample.parquet
Output: data/processed/pilot_text.parquet
"""
import json
import pathlib
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
HUPD = ROOT / "data/hupd/data"
YEARS = range(2009, 2016)  # pilot filings are 2010–2014; scan one extra year each side in case archive and filing years differ
MAXLEN = 4000


def scan(year: int, wanted: frozenset) -> list[dict]:
    rows = []
    with tarfile.open(HUPD / f"{year}.tar.gz", mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".json"):
                continue
            app = pathlib.PurePosixPath(m.name).stem
            if app not in wanted:
                continue
            d = json.load(tf.extractfile(m))
            rows.append({
                "application_number": app,
                "hupd_year": year,
                "title": d.get("title") or "",
                "abstract": d.get("abstract") or "",
                "claims": d.get("claims") or "",
                "background": (d.get("background") or "")[:MAXLEN],
                "summary": (d.get("summary") or "")[:MAXLEN],
                "inventor_list": json.dumps(d.get("inventor_list") or [], ensure_ascii=False),
            })
    print(f"{year}: {len(rows):,d}", file=sys.stderr, flush=True)
    return rows


def main() -> None:
    wanted = frozenset(pd.read_parquet(P / "pilot_sample.parquet", columns=["application_number"])
                       ["application_number"])
    with ProcessPoolExecutor(max_workers=len(YEARS)) as ex:
        parts = list(ex.map(scan, YEARS, [wanted] * len(YEARS)))
    df = pd.DataFrame([r for part in parts for r in part]).drop_duplicates("application_number")
    df.to_parquet(P / "pilot_text.parquet", index=False)
    print(f"Extracted {len(df):,d} / {len(wanted):,d} applications ({len(df) / len(wanted):.1%})")
    print("By archive year:", df["hupd_year"].value_counts().sort_index().to_dict())
    print("Share with empty abstract:", round((df["abstract"].str.len() == 0).mean(), 4),
          "share with empty claims:", round((df["claims"].str.len() == 0).mean(), 4))


if __name__ == "__main__":
    main()
