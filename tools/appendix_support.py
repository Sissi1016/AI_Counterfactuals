"""Small, dependency-free utilities for the appendix review package."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"[+\-−]?\d+(?:,\d{3})*(?:\.\d+)?")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_value(text: str) -> Decimal:
    return Decimal(text.replace(",", "").replace("−", "-"))


def numbers(text: str) -> list[str]:
    return NUMBER.findall(text)


def format_like(value: str, example: str) -> str:
    places = len(example.partition(".")[2]) if "." in example else 0
    grouping = "," if "," in example else ""
    result = format(numeric_value(value), f"{grouping}.{places}f")
    if numeric_value(value) == 0 and example.startswith(("−", "-")):
        result = "-" + result.lstrip("-")
    if example.startswith("+") and not result.startswith("-"):
        result = "+" + result
    return result.replace("-", "−") if "−" in example else result


def render_cell(cell, sources: dict) -> str:
    if isinstance(cell, str):
        return cell
    values = []
    for ref, example in zip(cell["refs"], cell["formats"], strict=True):
        source = sources[ref["file"]]
        row = source["rows"][ref["row"]]
        if row[0] != ref["row_label"]:
            raise ValueError(f"Source row changed: {ref['file']} row {ref['row']}")
        token = numbers(row[ref["column"]])[ref["number"]]
        values.append(format_like(token, example))
    return cell["template"].format(*values)


def render_tables(layout: dict, sources: dict) -> list[dict]:
    tables = []
    for table in layout["tables"]:
        rows = [[render_cell(c, sources) for c in row] for row in table["layout"]]
        tables.append({**{k: v for k, v in table.items() if k != "layout"}, "rows": rows})
    return tables


def csv_text(rows: list[list[str]]) -> str:
    import io
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def table_html(table: dict) -> str:
    rows = "\n".join("<tr>" + "".join(f"<td>{html.escape(c).replace(chr(10), '<br>')}</td>" for c in row)
                     + "</tr>" for row in table["rows"])
    return f"<section id='{table['id'].replace(' ', '_')}'><h2>{table['id']}</h2>\n<p>{html.escape(table['caption'])}</p>\n<table>{rows}</table></section>"


def html_document(content: str, title: str) -> str:
    return "<!doctype html><html lang='en'><meta charset='utf-8'>" + \
        f"<title>{html.escape(title)}</title>" + \
        "<style>body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem;color:#222}" \
        "table{border-collapse:collapse;width:100%;font-size:12px;margin:1rem 0 3rem}td{padding:5px 7px;border-bottom:1px solid #ddd;vertical-align:top}" \
        "tr:first-child td{font-weight:bold;border-top:2px solid #222}h2{margin-top:3rem}img{max-width:100%}" \
        "section{overflow:auto}a{color:#185b9b}</style><body>" + content + "</body></html>"


def expand_macros(text: str, macros: dict) -> str:
    pattern = re.compile(r"\\([A-Za-z]+)(?:\{\})?")
    for _ in range(12):
        updated = pattern.sub(lambda m: macros.get(m[1], m[0]), text)
        if updated == text:
            return text
        text = updated
    raise ValueError("Macro expansion did not converge")


def parse_latex(text: str, macros: dict) -> list[list[str]]:
    import subprocess
    text = expand_macros(text, macros)
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(r"\\(?:toprule|midrule|bottomrule|addlinespace)(?:\[[^]]*\])?", "", text)
    text = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", text)
    text = re.sub(r"\\begin\{tabular\}\{[^\n]*\}", "", text)
    text = text.replace(r"\end{tabular}", "")

    def split_top(value, delimiter):
        parts, start, depth, index = [], 0, 0, 0
        while index < len(value):
            ch = value[index]
            if depth == 0 and value.startswith(delimiter, index):
                parts.append(value[start:index])
                index += len(delimiter)
                start = index
                continue
            if ch == "\\" and index + 1 < len(value) and value[index + 1] in "{}&%":
                index += 2
                continue
            depth += (ch == "{") - (ch == "}")
            index += 1
        parts.append(value[start:])
        return parts

    def unwrap(cell):
        for name in (r"\multicolumn", r"\multirow"):
            if not cell.lstrip().startswith(name):
                continue
            index = cell.index(name) + len(name)
            groups = []
            for _ in range(3):
                while cell[index].isspace():
                    index += 1
                assert cell[index] == "{"
                start, depth = index + 1, 1
                index += 1
                while depth:
                    depth += (cell[index] == "{") - (cell[index] == "}")
                    index += 1
                groups.append(cell[start:index - 1])
            return groups[-1] + cell[index:]
        return cell

    raw = [[unwrap(c.strip()) for c in split_top(row, "&")] for row in split_top(text, r"\\") if row.strip()]
    cells = [cell for row in raw for cell in row]
    packed = "\n\n".join(f"REVIEWCELLMARKER{i:05d}\n\n{cell}" for i, cell in enumerate(cells))
    result = subprocess.run(["pandoc", "--from=latex", "--to=plain", "--wrap=none"],
                            input=packed, text=True, capture_output=True, check=True)
    parts = re.split(r"REVIEWCELLMARKER\d{5}", result.stdout)[1:]
    if len(parts) != len(cells):
        raise ValueError("LaTeX cell conversion lost a cell boundary")
    converted = iter(re.sub(r"\s+", " ", part).strip() for part in parts)
    return [[next(converted) for _ in row] for row in raw]


def source_tables() -> dict:
    """Refresh canonical cells from generated LaTeX and supplementary CSV selections."""
    import importlib.util
    definition = read_json(ROOT / "specification/appendix_layout.json")
    macro_files = definition["macro_files"]
    macros = {}
    for rel in macro_files:
        for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", line)
            if match:
                macros[match[1]] = match[2]
    sources = {}
    for rel in definition["latex_sources"]:
        path = ROOT / rel
        sources[rel] = {"rows": parse_latex(path.read_text(encoding="utf-8"), macros),
                        "sha256": sha256(path)}
    spec = importlib.util.spec_from_file_location("supplementary_sources", ROOT / "tools/supplementary_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources.update(module.build(ROOT, macros, sources))
    return {"macro_hashes": {rel: sha256(ROOT / rel) for rel in macro_files}, "sources": sources}
