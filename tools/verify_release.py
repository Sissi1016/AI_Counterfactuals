"""Verify the review snapshot without data, GPU, or third-party Python packages."""
from __future__ import annotations

import ast
import csv
import re

from appendix_support import ROOT, csv_text, read_json, render_tables, sha256


def verify():
    layout = read_json(ROOT / "specification/appendix_layout.json")
    source = read_json(ROOT / "specification/source_cells.json")
    expected = read_json(ROOT / "reference/appendix_tables.json")
    assert [t["id"] for t in expected] == [f"Table A{n}" for n in range(1, 34)]
    assembled = render_tables(layout, source["sources"])
    assert [(t["id"], t["rows"]) for t in assembled] == [(t["id"], t["rows"]) for t in expected]
    for table in assembled:
        path = ROOT / "results/appendix" / (table["id"].replace(" ", "_") + ".csv")
        assert path.read_text(encoding="utf-8") == csv_text(table["rows"]), f"Outdated final table: {path.name}"
    for rel, digest in source["macro_hashes"].items():
        assert sha256(ROOT / rel) == digest, f"Changed macro source: {rel}"
    for rel, item in source["sources"].items():
        if "sha256" in item:
            assert sha256(ROOT / rel) == item["sha256"], f"Changed table source: {rel}"
        for dependency in item.get("inputs", []):
            assert (ROOT / dependency).is_file(), f"Missing derived-cell input: {dependency}"
    figures = read_json(ROOT / "reference/appendix_figures.json")
    assert [f["id"] for f in figures] == [f"Figure A{n}" for n in range(1, 10)]
    for figure in figures:
        assert sha256(ROOT / figure["file"]) == figure["sha256"], f"Changed manuscript graphic: {figure['id']}"
    bindings = sum(isinstance(c, dict) for t in layout["tables"] for row in t["layout"] for c in row)
    assert bindings == read_json(ROOT / "reference/release_origin.json")["bound_numeric_cells"]

    proposal_layout = read_json(ROOT / "specification/proposal_layout.json")
    proposal_table = render_tables(proposal_layout, source["sources"])[0]
    assert proposal_table["rows"] == read_json(ROOT / "reference/proposal_table.json")["rows"]
    assert (ROOT / "results/proposal/Table_1.csv").read_text(encoding="utf-8") == csv_text(proposal_table["rows"])
    proposal = read_json(ROOT / "reference/final_proposal.json")
    assert [f["id"] for f in proposal["figures"]] == ["Figure 1", "Figure 2"]
    for figure in proposal["figures"]:
        assert sha256(ROOT / figure["file"]) == figure["sha256"], f"Changed proposal graphic: {figure['id']}"
    proposal_bindings = sum(isinstance(c, dict) for row in proposal_layout["tables"][0]["layout"] for c in row)

    modules = sorted((ROOT / "code").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py")) + [ROOT / "reproduce.py"]
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.fullmatch(r"\d{2}[A-Za-z0-9_]*\.py", node.value):
                    assert (ROOT / "code" / node.value).exists(), f"Missing imported script: {node.value}"

    prohibited = {"internal", "refs", "models", "data", ".venv", ".venv-llm", ".git", ".cache", "__pycache__"}
    manifest = read_json(ROOT / "MANIFEST.json")
    for rel, digest in manifest["files"].items():
        path = ROOT / rel
        assert not (set(path.relative_to(ROOT).parts) & prohibited), f"Private or external directory in release: {rel}"
        assert path.suffix.lower() not in {".pyc", ".docx", ".doc", ".docm", ".key", ".pem"}, f"Non-code/private file: {rel}"
        assert path.is_file() and not path.is_symlink(), f"Missing file or symlink: {rel}"
        assert sha256(path) == digest, f"Integrity mismatch: {rel}"
    with (ROOT / "docs/appendix_map.csv").open(newline="", encoding="utf-8") as stream:
        mapped = list(csv.DictReader(stream))
    assert len(mapped) == 42
    for entry in mapped:
        for script in entry["Source scripts"].split("; "):
            assert (ROOT / script).is_file(), f"Missing exhibit script: {script}"
    print(f"PASS: final appendix (33 tables, {bindings} numerical result cells, 9 graphics); "
          f"final proposal (1 table, {proposal_bindings} numerical result cells, 2 graphics); "
          f"{len(modules)} Python files; {len(manifest['files'])} file checksums.")


if __name__ == "__main__":
    verify()
