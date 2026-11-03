"""Field mix behind two of Figure 2's center rows: TC 1600 and TC 1700.

Two questions about the evaluators' gender gaps (female − male, within art unit × filing year, as in 38/40/44):
1. Why is the zero-shot LLM the most favourable to women in TC 1700 and TC 1600?
2. Why, in TC 1600, does the evaluator trained on citations have a larger gap than the one trained on grants?

Method: the same regressions as 44 (female on the outcome, art unit × year fixed effects, errors clustered by examiner)
repeated with USPC class and USPC subclass fixed effects added, so a gap that comes from which subfields women's
applications are in (field mix) shrinks, and one that holds within subfields does not. Outcomes: the examiners'
grants; each evaluator's decisions when it grants the examiners' share of applications within the center (38's
embedding evaluators, 40's zero-shot Qwen3-8B, scores combined as in 40); each evaluator's standardized score; and the
outcomes the citation-trained evaluator learns from or that relate to it: log(1 + five-year citations) of granted
patents, log(1 + prior-art citations) of published applications (rejected ones included), and renewal at four years of
granted patents (maintenance-fee events, as in 35). Also: how women's and men's applications differ in team size,
entity size, organizational applicant (academic or other), and claim and abstract length, and how each score loads on
these within art unit × year.

Inputs: data/processed/{all_sample,all_scores,llm_scores_allpilot,llm_scores_allcenters,prior_art_cites,
patex_application_data,patex_all_applicants,all_text,applications_2004_2018}.parquet and the maintenance-fee file.
Outputs: output/tables/fx_gaps.csv, fx_features.csv, fx_loadings.csv; output/latex/fx_tab_fieldmix.tex and
fx_numbers.tex (macros \\xFx...); output/field_mix.md.
CPU only, about two minutes: taskset -c 0-15 uv run python code/50_field_mix.py
"""
import re
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyfixest as pf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
P, T, TEX = ROOT / "data/processed", ROOT / "output/tables", ROOT / "output/latex"
TCS = ["1600", "1700"]
WORDS = {"1600": "OneSix", "1700": "OneSeven"}
ACAD = re.compile(r"UNIVERSIT|COLLEGE|REGENTS|INSTITUTE OF TECHNOLOGY|SCHOOL OF MEDICINE|HOSPITAL|MEDICAL CENTER|"
                  r"CLINIC\b|FOUNDATION|RESEARCH INSTITUTE|CANCER CENTER|NATIONAL INSTITUTES|DEPARTMENT OF|SECRETARY|"
                  r"UNITED STATES OF AMERICA|BOARD OF TRUSTEES|TRUSTEES OF|LABORATOR", re.I)
SCORES = {"Train on grant": "grant", "Train on citations (granted only)": "cites", "llm": "llm"}


def load() -> pd.DataFrame:
    d = pd.read_parquet(P / "all_sample.parquet")
    d["tech_center"] = d.tech_center.astype(str)
    d = d[d.tech_center.isin(TCS)]
    d = d.merge(pd.read_parquet(P / "all_scores.parquet"), on="application_number", how="left")
    l0 = pd.read_parquet(P / "llm_scores_allpilot.parquet")
    l = pd.concat([l0[l0.cond == "zero"], pd.read_parquet(P / "llm_scores_allcenters.parquet")])  # as in 40
    l["llm"] = np.log(l.p_yes.clip(1e-12)) - np.log(l.p_no.clip(1e-12))
    d = d.merge(l.drop_duplicates("application_number")[["application_number", "llm"]], on="application_number",
                how="left")
    d = d.merge(pd.read_parquet(P / "prior_art_cites.parquet", columns=["application_number", "published",
                                                                        "prior_art_cites"]),
                on="application_number", how="left")
    ap = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "uspc_class",
                                                                        "uspc_subclass"])
    d = d.merge(ap, on="application_number", how="left")
    d["uspc"] = d.uspc_class.astype(str)
    d["subclass"] = d.uspc + "/" + d.uspc_subclass.astype(str).str.split(".").str[0]
    a = pd.read_parquet(P / "patex_all_applicants.parquet", columns=["application_number", "applicant_organization"])
    a = a[a.application_number.isin(d.application_number) & a.applicant_organization.notna()]
    a["acad"] = a.applicant_organization.str.contains(ACAD)
    o = a.groupby("application_number").acad.max().reset_index()
    d = d.merge(o.assign(has_org=1), on="application_number", how="left")
    d["acad"] = d.acad.fillna(False).astype(int)
    d["org_other"] = ((d.has_org == 1) & (d.acad == 0)).astype(int)
    t = pq.read_table(P / "all_text.parquet", columns=["application_number", "abstract", "claims"],
                      filters=[("application_number", "in", d.application_number.tolist())]).to_pandas()
    t["claims_words"] = t.claims.fillna("").str.split().str.len()
    t["abstract_words"] = t.abstract.fillna("").str.split().str.len()
    d = d.merge(t[["application_number", "claims_words", "abstract_words"]], on="application_number", how="left")
    mf = sorted((ROOT / "data/maintenance_fee/txt").glob("MaintFeeEvents_2*.txt"))[-1]
    con = duckdb.connect()
    con.execute("SET threads = 8")
    con.register("ids", d[["application_number"]])
    r = con.execute(f"""
    WITH m AS (SELECT string_split(trim(regexp_replace(line, '\\s+', ' ', 'g')), ' ') AS f
               FROM read_csv('{mf}', delim = '\x01', header = false, columns = {{'line': 'VARCHAR'}})),
    r AS (SELECT ltrim(f[1], '0') AS pn, max((f[7] IN ('M1551', 'M2551', 'M3551', 'F170'))::INT) AS renew_4y
          FROM m GROUP BY 1)
    SELECT a.application_number, coalesce(r.renew_4y, 0) AS renew_4y
    FROM '{P}/applications_2004_2018.parquet' a JOIN ids USING (application_number)
    LEFT JOIN r ON r.pn = ltrim(a.patent_number, '0')""").df()
    d = d.merge(r, on="application_number", how="left")
    d.loc[d.granted != 1, "renew_4y"] = np.nan
    d["log_cites"] = np.where(d.granted == 1, np.log1p(d.cites_ns), np.nan)
    d["log_pa"] = np.where(d.published == 1, np.log1p(d.prior_art_cites), np.nan)
    for c in ["claims_words", "abstract_words"]:
        d["log_" + c] = np.log1p(d[c])
    for col, k in SCORES.items():
        g = d.groupby("tech_center")[col]
        d["z_" + k] = (d[col] - g.transform("mean")) / g.transform("std")
        d["dec_" + k] = np.nan
        for tc, x in d[d[col].notna()].groupby("tech_center"):   # grant the examiners' share within the center
            thr = x[col].quantile(1 - x.granted.mean())
            d.loc[x.index, "dec_" + k] = (x[col] > thr).astype(float)
    return d


def fe_gap(x: pd.DataFrame, y: str, fe: str) -> tuple:
    x = x[x[y].notna()]
    m = pf.feols(f"{y} ~ female | {fe}", data=x, vcov={"CRV1": "examiner_id"})
    return m.coef()["female"], m.se()["female"], len(x)


ROWS = [("granted", "Examiners: grants"), ("dec_grant", "Evaluator trained on grants: decisions"),
        ("dec_cites", "Evaluator trained on citations: decisions"), ("dec_llm", "Qwen3-8B, zero-shot: decisions"),
        ("z_grant", "Score, trained on grants (SD)"), ("z_cites", "Score, trained on citations (SD)"),
        ("z_llm", "Score, Qwen3-8B zero-shot (SD)"),
        ("log_cites", "Log citations, granted patents"), ("log_pa", "Log prior-art citations, published applications"),
        ("renew_4y", "Renewed at four years, granted patents")]
FES = [("cell", "au_year"), ("class", "au_year + uspc"), ("subclass", "au_year + subclass")]
FEATS = ["n_inventors", "small_entity", "acad", "org_other", "claims_words", "abstract_words"]


def main() -> None:
    d = load()
    rows = []
    for tc in TCS:
        x = d[d.tech_center == tc]
        for y, lab in ROWS:
            for k, fe in FES:
                b, se, n = fe_gap(x, y, fe)
                rows.append(dict(tc=tc, outcome=y, label=lab, fe=k, b=b, se=se, n=n))
    G = pd.DataFrame(rows)
    G.to_csv(T / "fx_gaps.csv", index=False)
    fr, lr = [], []
    for tc in TCS:
        x = d[d.tech_center == tc]
        for f in FEATS:
            b, se, _ = fe_gap(x, f, "au_year")
            fr.append(dict(tc=tc, feature=f, mean=x[f].mean(), female_diff=b, se=se))
        for k in SCORES.values():
            xx = x[x["z_" + k].notna()]
            m = pf.feols(f"z_{k} ~ n_inventors + small_entity + acad + org_other + log_claims_words + "
                         "log_abstract_words | au_year", data=xx, vcov={"CRV1": "examiner_id"})
            for f, b in m.coef().items():
                lr.append(dict(tc=tc, score=k, feature=f, b=b, se=m.se()[f]))
    F, L = pd.DataFrame(fr), pd.DataFrame(lr)
    F.to_csv(T / "fx_features.csv", index=False)
    L.to_csv(T / "fx_loadings.csv", index=False)

    # table: decisions and outcomes, cell / + subclass, both centers (pp for binary outcomes, % for logs)
    def cell(tc, y, fe):
        r = G[(G.tc == tc) & (G.outcome == y) & (G.fe == fe)].iloc[0]
        s = 1 if y.startswith("z_") else 100
        nd = 2 if y.startswith("z_") else 1
        f = lambda v: f"{s * v:.{nd}f}".replace("-", "$-$")   # noqa: E731
        return f"{f(r.b)} {{\\footnotesize [{f(r.b - 1.96 * r.se)}, {f(r.b + 1.96 * r.se)}]}}"
    body = []
    for grp, ys in [("Decisions at the examiners' grant rate (pp)", ["granted", "dec_grant", "dec_cites", "dec_llm"]),
                    ("Outcomes (\\% for logs, pp for renewal)", ["log_cites", "log_pa", "renew_4y"])]:
        body.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{grp}}}}} \\\\")
        for y in ys:
            lab = dict(ROWS)[y]
            body.append(" & ".join([lab] + [cell(tc, y, fe) for tc in TCS for fe in ["cell", "subclass"]]) + r" \\")
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = (r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
           r" & \multicolumn{2}{c}{TC 1600} & \multicolumn{2}{c}{TC 1700} \\" "\n"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
           r"Female $-$ male, within & art unit $\times$ year & + subclass & art unit $\times$ year & + subclass \\"
           "\n" r"\midrule" "\n")
    (TEX / "fx_tab_fieldmix.tex").write_text(hdr + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

    # macros
    N = {}
    def num(v, nd=1):
        return f"{v:.{nd}f}".replace("-", "$-$")
    for tc in TCS:
        w = WORDS[tc]
        for y, _ in ROWS:
            for k, _ in FES:
                r = G[(G.tc == tc) & (G.outcome == y) & (G.fe == k)].iloc[0]
                s, nd = (1, 2) if y.startswith("z_") else (100, 1)
                key = "".join(p.capitalize() for p in y.replace("_", " ").split()) + {"cell": "", "class": "Cls",
                                                                                     "subclass": "Sub"}[k]
                key = re.sub(r"\d", "", key).replace("Renewy", "Renew")
                N[f"xFx{w}{key}"] = num(s * r.b, nd)
                N[f"xFx{w}{key}CI"] = f"[{num(s * (r.b - 1.96 * r.se), nd)}, {num(s * (r.b + 1.96 * r.se), nd)}]"
        for f in FEATS:
            r = F[(F.tc == tc) & (F.feature == f)].iloc[0]
            key = "".join(p.capitalize() for p in f.split("_"))
            nd = 0 if f in ("claims_words", "abstract_words") else 2 if f == "n_inventors" else 1
            sc = 1 if f in ("claims_words", "abstract_words", "n_inventors") else 100
            N[f"xFx{w}Fem{key}"] = num(sc * r.female_diff, nd)
            N[f"xFx{w}Mean{key}"] = num(sc * r["mean"], nd)
    for k in list(N):
        if N[k].startswith("$-$"):
            N[k + "Abs"] = N[k][3:]
    (TEX / "fx_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))

    L_ = ["# Field mix behind the TC 1600 and TC 1700 rows of Figure 2 (script 50)", "",
          "Female − male, art unit × year fixed effects, then + USPC class, then + USPC subclass; clustered by "
          "examiner. Decisions and renewal in pp, logs in %, scores in SD.", "",
          "| TC | Outcome | AU × year | + class | + subclass | n |", "|---|---|---|---|---|---|"]
    for (tc, y), g in G.groupby(["tc", "outcome"], sort=False):
        s = 1 if y.startswith("z_") else 100
        v = {r.fe: f"{s * r.b:+.2f} ({s * r.se:.2f})" for r in g.itertuples()}
        L_.append(f"| {tc} | {dict(ROWS)[y]} | {v['cell']} | {v['class']} | {v['subclass']} | {g.n.iloc[0]:,} |")
    L_ += ["", "Women vs men on application features (within art unit × year)", "", F.round(3).to_markdown(index=False),
           "", "Loadings of each standardized score on the features (within art unit × year)", "",
           L.round(3).to_markdown(index=False)]
    (ROOT / "output/field_mix.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_[:30]))


if __name__ == "__main__":
    main()
