"""Build the 33 final appendix tables from the recorded source-cell bindings."""
from __future__ import annotations

import argparse
import html

from appendix_support import ROOT, csv_text, html_document, read_json, render_tables, source_tables, table_html, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-sources", action="store_true", help="Parse regenerated LaTeX with Pandoc first")
    parser.add_argument("--check", action="store_true", help="Compare assembled cells with the final appendix snapshot")
    args = parser.parse_args()
    if args.refresh_sources:
        write_json(ROOT / "specification/source_cells.json", source_tables())
    source = read_json(ROOT / "specification/source_cells.json")
    layout = read_json(ROOT / "specification/appendix_layout.json")
    tables = render_tables(layout, source["sources"])
    expected = read_json(ROOT / "reference/appendix_tables.json")
    if args.check:
        assert [(t["id"], t["rows"]) for t in tables] == [(t["id"], t["rows"]) for t in expected], \
            "Assembled results differ from the final appendix; inspect updated estimates and their bindings."
        print(f"All {len(tables)} tables match the final appendix cell for cell.")
        return
    out = ROOT / "results/appendix"
    out.mkdir(parents=True, exist_ok=True)
    for table in tables:
        name = table["id"].replace(" ", "_")
        (out / f"{name}.csv").write_text(csv_text(table["rows"]), encoding="utf-8")
    parts = ["<h1>Final technical appendix: tables and figures</h1>",
        "<p>Tables use the final manuscript's rows, columns, and numbering. "
        "Their numerical cells are assembled from the recorded analysis sources. "
        "Figures below are the exact graphics embedded in the final appendix. "
        "See docs/appendix_map.csv for source scripts and figure-panel selections.</p>"]
    parts.extend(table_html(t) for t in tables)
    for figure in read_json(ROOT / "reference/appendix_figures.json"):
        name = figure["id"].replace(" ", "_") + ".png"
        parts.extend([f"<section id='{name[:-4]}'><h2>{figure['id']}</h2>",
            f"<img src='figures/{name}' alt='{figure['id']}'>", f"<p>{html.escape(figure['caption'])}</p></section>"])
    (out / "index.html").write_text(html_document("\n".join(parts), "Final technical appendix exhibits"), encoding="utf-8")
    write_json(out / "tables.json", tables)
    print(f"Wrote {len(tables)} tables and the appendix exhibit index.")


if __name__ == "__main__":
    main()
