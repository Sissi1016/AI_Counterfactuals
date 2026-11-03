"""Assemble the final proposal's table and exhibit index from supplied results."""
from __future__ import annotations

import argparse
import html

from appendix_support import ROOT, csv_text, html_document, read_json, render_tables, sha256, table_html, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sources = read_json(ROOT / "specification/source_cells.json")["sources"]
    table = render_tables(read_json(ROOT / "specification/proposal_layout.json"), sources)[0]
    expected = read_json(ROOT / "reference/proposal_table.json")
    proposal = read_json(ROOT / "reference/final_proposal.json")
    assert table["rows"] == expected["rows"], "Proposal Table 1 differs from the final source"
    for figure in proposal["figures"]:
        assert sha256(ROOT / figure["file"]) == figure["sha256"], f"Changed proposal graphic: {figure['id']}"
    if args.check:
        print("Final proposal Table 1 and both manuscript figures match the source snapshot.")
        return
    out = ROOT / "results/proposal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "Table_1.csv").write_text(csv_text(table["rows"]), encoding="utf-8")
    write_json(out / "Table_1.json", table)
    pieces = ["<h1>Final proposal: numerical exhibits</h1>",
        "<p>Table 1 is assembled from the same numerical sources as Appendix Table A19. "
        "Both graphics are the exact images embedded in the final proposal. "
        "F − M denotes female minus male. See docs/proposal_map.csv for the source scripts.</p>", table_html(table)]
    for figure in proposal["figures"]:
        filename = figure["id"].replace(" ", "_") + ".png"
        pieces += [f"<h2>{figure['id']}</h2><img src='figures/{filename}' alt='{figure['id']}'>",
                   f"<p>{html.escape(figure['caption'])}</p>"]
    (out / "index.html").write_text(html_document("\n".join(pieces), "Final proposal exhibits"), encoding="utf-8")
    print("Wrote final proposal Table 1 and its two-figure exhibit index.")


if __name__ == "__main__":
    main()
