"""C3 for race and ethnicity: the first inventor's group inferred from the surname, and the citation check by group.

Surnames: the 2010 U.S. Census surname table (all surnames with at least 100 bearers; shares White, Black,
Asian/Pacific Islander, Hispanic, and other), taken from the ethnicolr 2.0.0 package
(data/processed/census_surnames_2010.parquet). The first inventor's last name (PatEx) is matched after keeping letters
only; the inventor is assigned to a group when that group's share is at least 60%, otherwise left unclassified.
Surnames identify Asian and Hispanic inventors well and Black inventors poorly (few surnames are predominantly borne
by Black Americans), so the Black group is small.

For each group against White first inventors: the gap in grants (art unit × year fixed effects), the outcome test
(marginal citations by group, 2SLS as in 21), each evaluator's signed error by group, and the gap in the evaluators'
decisions at the human grant rate. Evaluators: the scores of 21 (grant- and citation-trained, Qwen3-8B zero-shot).

Runs on CPUs only. Input: the record of 21, data/processed/{census_surnames_2010,patex_all_inventors,
citation_scores}.parquet
Output: output/ethnicity.md, output/tables/ethnicity_*.csv
"""
import dataclasses
import importlib.util
import pathlib
import warnings

import duckdb
import pandas as pd

from cfe.check import HUMAN, NO_MODEL, assign_folds, decision_gaps, marginal, ols_coef, recalibrate

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
THRESHOLD = 60.0
GROUPS = {"asian": ("Asian", "pctapi"), "hispanic": ("Hispanic", "pcthispanic"), "black": ("Black", "pctblack"),
          "white": ("White", "pctwhite")}
NAMES = ["Train on grant", "Train on citations (granted only)", "LLM, zero-shot"]


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    warnings.filterwarnings("ignore")
    m21 = load_module("m21", "21_citations.py")
    d = m21.load()
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.register("pilot", d[["application_number"]])
    sn = con.execute(f"""
    SELECT i.application_number, regexp_replace(upper(i.inventor_name_last), '[^A-Z]', '', 'g') AS surname
    FROM '{P}/patex_all_inventors.parquet' i JOIN pilot USING (application_number)
    WHERE i.inventor_rank = '1'""").df().drop_duplicates("application_number")
    cs = pd.read_parquet(P / "census_surnames_2010.parquet")
    cs["surname"] = cs["name"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    cs = cs.drop_duplicates("surname")
    sn = sn.merge(cs[["surname"] + [v for _, v in GROUPS.values()]], on="surname", how="left")
    sn["group"] = "unclassified"
    for key, (_, col) in GROUPS.items():
        sn.loc[sn[col] >= THRESHOLD, "group"] = key
    d = d.merge(sn[["application_number", "group", "pctapi"]], on="application_number", how="left")
    d["group"] = d.group.fillna("unclassified")
    d["matched"] = d.pctapi.notna()
    for key in GROUPS:
        d[key] = (d.group == key).astype(float)

    sc = pd.read_parquet(P / "citation_scores.parquet", columns=["application_number"] + NAMES)
    dd = recalibrate(d.merge(sc, on="application_number", how="left"), NAMES, assign_folds(d, m21.CIT), m21.CIT)

    share = d.groupby("tech_center").agg(applications=("group", "size"), surname_matched=("matched", "mean"),
                                         **{k: (k, "mean") for k in GROUPS}).reset_index()
    rows, est_all, gap_all = [], [], []
    for key in ["asian", "hispanic", "black"]:
        lab = GROUPS[key][0]
        s = dd[dd.group.isin([key, "white"])].copy()
        design = dataclasses.replace(
            m21.CIT, groups=(("All", None), (lab, (key, 1)), ("White", (key, 0))),
            contrasts=((f"{lab} − White", (lab, "White")),), gap_group=key)
        e, c = marginal(s, [NO_MODEL] + NAMES, design=design)
        e["comparison"], c["comparison"] = key, key
        est_all.append((e, c))
        g = decision_gaps(s, NAMES, design)
        g["comparison"] = key
        gap_all.append(g)
        for tc in ["3600", "1600"]:
            x = s[s.tech_center == tc]
            gb, gse = ols_coef(x, "granted", key, fe="au_year", design=design)
            ce = c[(c["sample"] == tc) & (c.series == HUMAN)].iloc[0]
            rows.append(dict(comparison=key, tech_center=tc, n_group=int(x[key].sum()),
                             n_white=int((1 - x[key]).sum()),
                             grant_gap=gb, grant_gap_se=gse, marginal_diff=ce.estimate, marginal_diff_se=ce.se))
    summ = pd.DataFrame(rows)
    E = pd.concat([x[0] for x in est_all])
    C = pd.concat([x[1] for x in est_all])
    G = pd.concat(gap_all)
    for name, t in [("shares", share), ("summary", summ), ("estimates", E), ("contrasts", C), ("gaps", G)]:
        t.to_csv(OUT / f"tables/ethnicity_{name}.csv", index=False)

    def ci(b, se, nd=2, scale=1.0):
        return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"
    L = ["# Race and ethnicity of the first inventor (surname, 2010 Census; share ≥ 60%)", "",
         "| TC | Applications | Surname in Census table | Asian | Hispanic | Black | White |",
         "|---|---|---|---|---|---|---|"]
    for r in share.itertuples():
        L.append(f"| {r.tech_center} | {r.applications:,} | {100 * r.surname_matched:.1f}% | {100 * r.asian:.1f}% | "
                 f"{100 * r.hispanic:.1f}% | {100 * r.black:.1f}% | {100 * r.white:.1f}% |")
    L += ["", "Against White first inventors (grant gap in pp, art unit × year; marginal citations: group − White):",
          "",
          "| Group | TC | Applications in group | Gap in grants (pp) | Difference in marginal citations |",
          "|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {GROUPS[r.comparison][0]} | {r.tech_center} | {r.n_group:,} | "
                 f"{ci(r.grant_gap, r.grant_gap_se, 1, 100)} | {ci(r.marginal_diff, r.marginal_diff_se)} |")
    L += ["", "Evaluators, TC 3600: signed error for the group, the group − White difference in errors, and the gap in "
          "decisions (pp, × human):", "", "| Group | Evaluator | Error, group | Error difference | Gap in decisions | "
          "× human |", "|---|---|---|---|---|---|"]
    for key in ["asian", "hispanic", "black"]:
        lab = GROUPS[key][0]
        e = E[(E.comparison == key) & (E["sample"] == "3600")]
        c = C[(C.comparison == key) & (C["sample"] == "3600")]
        g = G[(G.comparison == key) & (G["sample"] == "3600")]
        for s in [NO_MODEL] + NAMES:
            a = e[(e.series == s) & (e.kind == "error") & (e.group == lab)].iloc[0]
            dif = c[(c.series == s) & (c.kind == "error")].iloc[0]
            gg = g[g.series == s]
            gtxt = f"{ci(gg.iloc[0].gap, gg.iloc[0].se, 1, 100)} | {gg.iloc[0].ratio:.2f}" if len(gg) else "— | —"
            L.append(f"| {lab} | {s} | {ci(a.estimate, a.se)} | {ci(dif.estimate, dif.se)} | {gtxt} |")
        h = g[g.series == "Human decision makers"].iloc[0]
        L.append(f"| {lab} | Human examiners | | | {ci(h.gap, h.se, 1, 100)} | 1.00 |")
    (OUT / "ethnicity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
