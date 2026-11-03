"""Data-construction tasks shared with existing analysis scripts."""
from __future__ import annotations

import importlib.util
import shutil
import sys

from appendix_support import ROOT

sys.path.insert(0, str(ROOT / "code"))


def main():
    if sys.argv[1:] != ["all-center-citations"]:
        raise SystemExit("Usage: python tools/data_tasks.py all-center-citations")
    import duckdb
    path = ROOT / "code/24_human_record.py"
    spec = importlib.util.spec_from_file_location("all_center_citations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with duckdb.connect(config={"threads": 16, "memory_limit": "120GB"}) as connection:
        record = module.build(connection)
        print(f"Built all-center citation outcomes for {len(record):,} applications.")
    shutil.copyfile(ROOT / "resources/us_states_albers_10m.json", ROOT / "data/processed/us_states_albers_10m.json")


if __name__ == "__main__":
    main()
