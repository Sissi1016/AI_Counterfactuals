"""Examiners and assignment cells in the main sample (the eight technology centers pooled, filed 2010–2014): the
appendix table of examiners and the permutation test of assignment, replacing the pilot-only versions of 22 and 18.

1. Identification (all applications in PatEx filed 2004–2018, definitions of 22). PatEx has no examiner ID; an
   examiner is the name string examiner_full_name (02). Examiner names in the 2004–2018 base of 02; utility
   applications filed 2004–2018 without an examiner name; applications whose examiner has fewer than 20 applications
   with a first action, the minimum for a leave-one-out leniency (04; examiners.parquet); all applications without a
   leave-one-out leniency (these also include the applications that never received a first action, which 04 leaves
   out; 22 reported this share under the label of the previous one); names that appear in more than one technology
   center (homonyms or transfers).
2. Docketing (main sample: 38's all_sample). PatEx transaction events per application, as in 22: DOCK (docketed to
   an examiner) and DELD (deleted from an examiner's docket). Re-docketing is routine (cases return to the docket
   after replies), so several DOCK events do not by themselves mean a new examiner; an application docketed once has
   had one examiner throughout. PatEx records DELD (and DKST) events only up to 2001, so for applications filed
   2010–2014 "deleted from a docket" is zero by construction; it is kept in the csv and the macros but left out of
   the table (SHOW_DELETED). Writes the flag docketed_once (one DOCK event, no DELD) for the main sample.
3. Assignment cells as C1 and C3 use them: art unit × filing year × USPC class (au_class) with at least five
   examiners, examiners split into leniency quintiles of equal caseload by 35's quintiles(). Examiners per cell,
   applications per examiner within a cell, examiners and applications per cell × quintile, and the quintiles that
   consist of a single examiner. Leniency is the leave-one-out grant rate of 04 (over the examiner's other
   applications filed 2004–2018); its spread within cells, and the mean leniency of strict and lenient examiners
   (below and above the mean of their cell, application-weighted, as in 22), in the cells with at least five
   examiners (the csv also has 22's version over all cells).
4. Permutation test of Kleinberg et al. (2018), as in 18 and 39: within each cell with at least five examiners,
   examiners are split into five leniency groups of equal numbers of examiners; the grant index of 39 (the cross-
   fitted grant score of 38, gender, entity size, number of inventors, projected on the grant decision within each
   center) is regressed on the groups and the F statistic compared with its distribution over 1,000 within-cell
   permutations of the group labels (39's seed and cell order, so the class cells reproduce 39's p-values). Three
   cell definitions: art unit × year × USPC class; art unit × year × USPC subclass (39's au_subclass); and class
   cells with examiners grouped by the residualized leniency of 18 (each 2004–2018 application's grant net of the
   mean grant of its class cell × entity size, leave-one-out mean over the examiner's applications). Under
   as-good-as-random assignment about 5% of cells have p < 0.05. The table shows the eight centers pooled and TC 3600
   (each center in the csv and the markdown), and adds C1's index, the predicted citations, from 35's test in its
   contraction cells (35's pooled numbers; TC 3600 from 46's cell-by-cell recomputation of 35's test).

Examiner-level information appears only aggregated. Runs on CPUs only (DuckDB and BLAS 4 threads each; run under
taskset -c 0-15); about five minutes.
Input: data/processed/{all_sample,all_scores,applications_2004_2018,applications_leniency,examiners,
       patex_application_data,patex_transactions}.parquet; code/35 (quintiles), 39 (load), 18 (f_stat);
       for the table's C1 row: output/tables/c1_kllm_all_permutation.csv (35), tx_c1_permutation.csv (46)
Output: data/processed/docketed_once.parquet; output/latex/design_tab_{examiners,permutation}.tex,
        output/latex/design_numbers.tex; output/tables/design_{examiners,permutation}.csv; output/design_examiners.md
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_v] = "4"

import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
T, TEX = OUT / "tables", OUT / "latex"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
M, Q, MIN_EXAMINERS, MIN_APPS = 1000, 5, 5, 20
SHOW_DELETED = False   # DELD is not recorded after 2001 (see 2 above)
DEFS = {"class": ("au_class", "lenient_grant", "Art unit × year × USPC class"),
        "subclass": ("au_subclass", "lenient_grant", "Art unit × year × USPC subclass"),
        "resid": ("au_class", "lenient_resid", "Art unit × year × USPC class, residualized leniency (18)")}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads = 4; SET memory_limit = '60GB'; SET enable_progress_bar = false")
    return con


# ---------------------------------------------------------------- 1. identification (all of PatEx 2004–2018)

def identification(con) -> dict:
    a = con.execute(f"""
    WITH e AS (SELECT examiner_id, count(DISTINCT tech_center) AS n_tc, count(*) AS n
               FROM '{P}/applications_2004_2018.parquet' WHERE examiner_id IS NOT NULL GROUP BY 1)
    SELECT count(*) AS names, avg((n_tc > 1)::DOUBLE) AS multi_tc,
           sum(n) FILTER (WHERE n_tc > 1) / sum(n) AS multi_tc_apps FROM e""").df().iloc[0]
    b = con.execute(f"""
    SELECT count(*) AS utility_apps,
           avg((examiner_full_name IS NULL OR trim(examiner_full_name) = '')::DOUBLE) AS no_name
    FROM '{P}/patex_application_data.parquet'
    WHERE application_invention_type = 'Utility' AND filing_date BETWEEN '2004-01-01' AND '2018-12-31'
    """).df().iloc[0]
    c = con.execute(f"""
    SELECT count(*) AS base_apps, avg((coalesce(e.n_e, 0) < {MIN_APPS})::DOUBLE) AS few_apps,
           avg((l.lenient_grant IS NULL)::DOUBLE) AS no_leniency, avg((a.fa_allow IS NULL)::DOUBLE) AS no_first_action
    FROM '{P}/applications_2004_2018.parquet' a
    LEFT JOIN '{P}/examiners.parquet' e ON e.examiner_id = a.examiner_id
    LEFT JOIN '{P}/applications_leniency.parquet' l ON l.application_number = a.application_number""").df().iloc[0]
    return dict(names=int(a.names), multi_tc=a.multi_tc, multi_tc_apps=a.multi_tc_apps,
                utility_apps=int(b.utility_apps), no_name=b.no_name, base_apps=int(c.base_apps),
                few_apps=c.few_apps, no_leniency=c.no_leniency, no_first_action=c.no_first_action)


# ---------------------------------------------------------------- 2. docketing (main sample)

def docketing(con) -> tuple[pd.DataFrame, dict]:
    dock = con.execute(f"""
    WITH s AS (SELECT application_number FROM '{P}/all_sample.parquet'),
         t AS (SELECT application_number, count(*) FILTER (WHERE event_code = 'DOCK') AS n_dock,
                      count(*) FILTER (WHERE event_code = 'DELD') AS n_deld
               FROM '{P}/patex_transactions.parquet' WHERE application_number IN (SELECT * FROM s) GROUP BY 1)
    SELECT s.application_number, coalesce(n_dock, 0)::INT AS n_dock, coalesce(n_deld, 0)::INT AS n_deld
    FROM s LEFT JOIN t USING (application_number) ORDER BY 1""").df()
    dock["docketed_once"] = (dock.n_dock == 1) & (dock.n_deld == 0)
    info = con.execute(f"""
    SELECT max(recorded_date) AS last_deld, count(*) AS n_deld_events
    FROM '{P}/patex_transactions.parquet' WHERE event_code = 'DELD'""").df().iloc[0]
    return dock, dict(last_deld=str(info.last_deld), n_deld_events=int(info.n_deld_events))


# ---------------------------------------------------------------- data for 3 and 4

def residualized_leniency(con, keys: pd.DataFrame) -> pd.DataFrame:
    """18's residualized_leniency (4 DuckDB threads), for the main sample."""
    con.register("keys", keys)
    return con.execute(f"""
    WITH b AS (
      SELECT l.application_number, l.examiner_id, l.granted,
             l.art_unit || '_' || l.filing_year::VARCHAR || '_' || coalesce(u.uspc_class, 'none') AS cell,
             l.small_entity_indicator IN ('SMALL', 'MICRO') AS small
      FROM '{P}/applications_leniency.parquet' l
      LEFT JOIN '{P}/patex_application_data.parquet' u USING (application_number)
    ), r AS (
      SELECT application_number, examiner_id, granted - avg(granted) OVER (PARTITION BY cell, small) AS resid FROM b
    ), e AS (SELECT examiner_id, sum(resid) AS s, count(*) AS n FROM r GROUP BY 1)
    SELECT r.application_number, (e.s - r.resid) / (e.n - 1) AS lenient_resid
    FROM r JOIN e USING (examiner_id) JOIN keys USING (application_number)""").df()


def load(con, m39) -> pd.DataFrame:
    d = m39.load()   # all_sample with 39's grant index, in 39's row order
    n = len(d)
    sub = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "uspc_subclass"])
    d = d.merge(sub, on="application_number", how="left")
    d["au_subclass"] = d.au_class + "/" + d.uspc_subclass.fillna("none")
    d = d.merge(residualized_leniency(con, d[["application_number"]]), on="application_number", how="left")
    assert len(d) == n
    return d


# ---------------------------------------------------------------- 3. cells and quintiles

def cell_stats(x: pd.DataFrame, n_all: int) -> dict:
    """x: main-sample applications (with leniency) and their 35-quintile q (NaN outside cells with ≥5 examiners)."""
    c = x[x.q.notna()]
    ec = c.groupby(["au_class", "examiner_id"]).size()
    cq = c.groupby(["au_class", "q"]).agg(examiners=("examiner_id", "nunique"), apps=("application_number", "size"))
    single = c.join(cq.examiners.rename("cq_examiners"), on=["au_class", "q"]).cq_examiners == 1
    r = dict(applications=n_all, examiners=x.examiner_id.nunique(), cells=c.au_class.nunique(),
             cells_apps=len(c), cells_share=len(c) / n_all,
             ex_per_cell=c.groupby("au_class").examiner_id.nunique().median(),
             ex_per_cell_mean=c.groupby("au_class").examiner_id.nunique().mean(),
             apps_per_ex=ec.median(), ex_per_quint=cq.examiners.median(), apps_per_quint=cq.apps.median(),
             single_quint=(cq.examiners == 1).mean(), single_apps=single.mean(),
             all_cells=x.au_class.nunique(),
             ex_per_cell_all_mean=x.groupby("au_class").examiner_id.nunique().mean())
    for tag, s in [("", c), ("_all", x)]:   # cells with ≥5 examiners; all cells (22's version)
        z = s.lenient_grant - s.groupby("au_class").lenient_grant.transform("mean")
        r["sd_len" + tag] = z.std()
        r["strict" + tag] = s.lenient_grant[z <= 0].mean()
        r["lenient" + tag] = s.lenient_grant[z > 0].mean()
    return r


# ---------------------------------------------------------------- 4. permutation test

def permutation(d: pd.DataFrame, cell: str, z: str, f_stat, seed: int) -> pd.DataFrame:
    """39's permutation_p for any cell definition and leniency (same rng use, so au_class × lenient_grant
    reproduces 39)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (tc, c), x in d[d[z].notna()].groupby(["tech_center", cell], sort=True):
        ex = x.groupby("examiner_id")[z].mean().sort_values(kind="mergesort")
        if len(ex) < MIN_EXAMINERS:
            continue
        g = x.examiner_id.map(pd.Series(np.arange(len(ex)) * Q // len(ex), index=ex.index)).to_numpy()
        y = x.index_grant.to_numpy(float)
        f0 = f_stat(y, g, Q)[0]
        fp = f_stat(y, g[np.argsort(rng.random((M, len(g))), axis=1)], Q)
        rows.append(dict(tech_center=tc, cell=c, n=len(x), examiners=len(ex),
                         p=(1 + (fp >= f0 - 1e-12).sum()) / (1 + M)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- formatting

def num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def med(x: float) -> str:
    return f"{int(x):,}" if float(x).is_integer() else f"{x:.1f}"


def tex_table(path: pathlib.Path, colspec: str, header: list[str], rows: list, ncol: int) -> None:
    """booktabs tabular body; rows: list of cells, or ("panel", title) / ("space",)."""
    L = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header + ([r"\midrule"] if header else [])
    for r in rows:
        if r[0] == "panel":
            L.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{r[1]}}}}} \\")
        elif r[0] == "space":
            L.append(r"\addlinespace")
        else:
            L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    con = connect()
    with threadpool_limits(4):
        m35 = load_module("m35", "35_c1_kllm.py")
        m39 = load_module("m39", "39_c3_checks.py")
        f_stat = load_module("m18", "18_assignment_checks.py").f_stat
        seed = m39.SEED

        ident = identification(con)
        dock, dinfo = docketing(con)
        dock[["application_number", "n_dock", "n_deld", "docketed_once"]].to_parquet(
            P / "docketed_once.parquet", index=False)
        d = load(con, m39)
        d = d.merge(dock, on="application_number", how="left")
        print(f"data loaded: {time.time() - t0:.0f}s")

        # 3. cells and quintiles (35's quintiles, cells with ≥5 examiners), pooled and by center
        e = d[d.lenient_grant.notna()].copy()
        e["q"] = m35.quintiles(e)
        rows = []
        for smp, units in [("All centers", TCS)] + [(tc, [tc]) for tc in TCS]:
            x = e[e.tech_center.isin(units)]
            n_all = int(d.tech_center.isin(units).sum())
            r = dict(sample=smp, **cell_stats(x, n_all))
            dd = d[d.tech_center.isin(units)]
            r.update(docketed_once=dd.docketed_once.mean(), docketed_once_n=int(dd.docketed_once.sum()),
                     deleted=(dd.n_deld > 0).mean(), docketed_more=(dd.n_dock > 1).mean(),
                     mean_dock=dd.n_dock.mean())
            rows.append(r)
        cells = pd.DataFrame(rows).set_index("sample")
        print(f"cells: {time.time() - t0:.0f}s")

        # 4. permutation test
        perm, summ = {}, []
        for key, (cell, z, lab) in DEFS.items():
            tp = time.time()
            p = permutation(d, cell, z, f_stat, seed)
            perm[key] = p
            for smp, units in [("All centers", TCS)] + [(tc, [tc]) for tc in TCS]:
                x = p[p.tech_center.isin(units)]
                summ.append(dict(definition=key, cells=lab, sample=smp, cells_tested=len(x),
                                 apps_tested=int(x.n.sum()),
                                 apps_covered=x.n.sum() / d.tech_center.isin(units).sum(),
                                 mean_examiners=x.examiners.mean(), p_below_05=(x.p < 0.05).mean(),
                                 p_below_10=(x.p < 0.10).mean(), mean_p=x.p.mean()))
            print(f"permutation {key}: {len(p):,} cells, {100 * (p.p < 0.05).mean():.1f}% p<0.05, "
                  f"{time.time() - tp:.0f}s")
        summ = pd.DataFrame(summ)

    # reproduce 39's class-cell p-values
    old = T / "c3_checks_permutation.csv"
    check = "c3_checks_permutation.csv not found"
    if old.exists():
        o = pd.read_csv(old, dtype={"tech_center": str})
        mm = perm["class"].merge(o.rename(columns={"au_class": "cell"}), on=["tech_center", "cell"], how="outer",
                                 suffixes=("", "_39"), indicator=True)
        both = mm[mm._merge == "both"]
        check = (f"39's file: {len(o):,} cells, {100 * (o.p < 0.05).mean():.1f}% p<0.05; matched cells "
                 f"{len(both):,}, identical p-values {int((both.p == both.p_39).sum()):,}, max |diff| "
                 f"{(both.p - both.p_39).abs().max():.4f}")
    print(check)

    # ---------------------------------------------------------------- outputs
    A = cells.loc["All centers"]
    S = summ.set_index(["definition", "sample"])
    rows_a = [("names", "Examiner names", f"{ident['names']:,}"),
              ("no_name", "Applications without an examiner name (%)", num(100 * ident["no_name"], 2)),
              ("few_apps", f"Applications whose examiner has <{MIN_APPS} applications (%)",
               num(100 * ident["few_apps"], 2)),
              ("multi_tc", "Names in more than one technology center (% of names)", num(100 * ident["multi_tc"], 1))]
    rows_b = [("applications", "Applications", lambda r: f"{int(r.applications):,}"),
              ("examiners", "Examiners", lambda r: f"{int(r.examiners):,}"),
              ("docketed_once", "Docketed to an examiner only once (%)", lambda r: num(100 * r.docketed_once, 1)),
              ("deleted", "Deleted from an examiner's docket (%)", lambda r: num(100 * r.deleted, 1)),
              ("cells", "Cells with >=5 examiners", lambda r: f"{int(r.cells):,}"),
              ("cells_share", "Applications in these cells (%)", lambda r: num(100 * r.cells_share, 1)),
              ("ex_per_cell", "Examiners per cell (median)", lambda r: med(r.ex_per_cell)),
              ("apps_per_ex", "Applications per examiner within a cell (median)", lambda r: med(r.apps_per_ex)),
              ("ex_per_quint", "Examiners per cell x quintile (median)", lambda r: med(r.ex_per_quint)),
              ("apps_per_quint", "Applications per cell x quintile (median)", lambda r: med(r.apps_per_quint)),
              ("single_quint", "Cell x quintiles with a single examiner (%)", lambda r: num(100 * r.single_quint, 1)),
              ("single_apps", "Applications in single-examiner quintiles (%)", lambda r: num(100 * r.single_apps, 1)),
              ("sd_len", "SD of leniency within cells", lambda r: num(r.sd_len, 3)),
              ("strict", "Leniency of strict examiners (mean, %)", lambda r: num(100 * r.strict, 1)),
              ("lenient", "Leniency of lenient examiners (mean, %)", lambda r: num(100 * r.lenient, 1)),
              ("sd_len_all", "SD of leniency within cells (all cells, as in 22)", lambda r: num(r.sd_len_all, 3)),
              ("strict_all", "Leniency of strict examiners (all cells, as in 22; %)", lambda r: num(100 * r.strict_all, 1)),
              ("lenient_all", "Leniency of lenient examiners (all cells, as in 22; %)",
               lambda r: num(100 * r.lenient_all, 1)),
              ("ex_per_cell_all_mean", "Examiners per cell (mean, all cells, as in 22)",
               lambda r: num(r.ex_per_cell_all_mean, 1))]

    # csv: panel A (global) and panel B pooled and by center (raw values)
    raw_a = dict(names=ident["names"], no_name=ident["no_name"], few_apps=ident["few_apps"],
                 multi_tc=ident["multi_tc"])
    csv = [dict(panel="A", key=k, label=lab, **{"All centers": raw_a[k]}) for k, lab, _ in rows_a]
    csv += [dict(panel="A", key="no_leniency", label="Applications without a leave-one-out leniency (%)",
                 **{"All centers": ident["no_leniency"]}),
            dict(panel="A", key="no_first_action", label="Applications without a first action (%)",
                 **{"All centers": ident["no_first_action"]}),
            dict(panel="A", key="multi_tc_apps", label="Applications of names in more than one center (%)",
                 **{"All centers": ident["multi_tc_apps"]})]
    extra_b = [("docketed_once_n", "Applications docketed only once"), ("docketed_more", "Docketed more than once (%)"),
               ("mean_dock", "Docketing events per application (mean)"), ("cells_apps", "Applications in these cells"),
               ("ex_per_cell_mean", "Examiners per cell (mean, cells with >=5 examiners)"),
               ("all_cells", "Cells (all)")]
    for k, lab, *_ in rows_b + extra_b:
        csv.append(dict(panel="B", key=k, label=lab, **{s: cells.loc[s, k] for s in cells.index}))
    pd.DataFrame(csv).to_csv(T / "design_examiners.csv", index=False)
    summ.to_csv(T / "design_permutation.csv", index=False)

    # Table A2
    tl = {"names": "Examiner names", "no_name": r"Applications without an examiner name (\%)",
          "few_apps": rf"Applications whose examiner has $<${MIN_APPS} applications (\%)",
          "multi_tc": r"Names in more than one technology center (\% of names)",
          "applications": "Applications", "examiners": "Examiners",
          "docketed_once": r"Docketed to an examiner only once (\%)",
          "deleted": r"Deleted from an examiner's docket (\%)",
          "cells": r"Cells (art unit $\times$ year $\times$ class) with $\geq$5 examiners",
          "cells_share": r"\quad Applications in these cells (\%)",
          "ex_per_cell": r"\quad Examiners per cell (median)",
          "apps_per_ex": r"\quad Applications per examiner in a cell (median)",
          "ex_per_quint": r"\quad Examiners per cell $\times$ leniency quintile (median)",
          "single_quint": r"\quad Cell $\times$ quintiles with a single examiner (\%)",
          "single_apps": r"\quad Applications in single-examiner quintiles (\%)",
          "sd_len": r"\quad SD of leniency within cells",
          "strict": r"\quad Leniency of strict examiners (mean, \%)",
          "lenient": r"\quad Leniency of lenient examiners (mean, \%)"}
    fmt_b = {k: f for k, _, f in rows_b}
    tex_b = {k: fmt_b[k](A) for k in fmt_b}
    tex_b["sd_len"] = num(A.sd_len, 2)
    tex_b["strict"], tex_b["lenient"] = num(100 * A.strict, 0), num(100 * A.lenient, 0)
    keys_b = ["applications", "examiners", "docketed_once"] + (["deleted"] if SHOW_DELETED else []) + [
        "cells", "cells_share", "ex_per_cell", "apps_per_ex", "ex_per_quint", "single_quint", "single_apps",
        "sd_len", "strict", "lenient"]
    body = [("panel", "A. All applications in PatEx, 2004--2018")]
    body += [[tl[k], v] for k, _, v in rows_a]
    body += [("space",), ("panel", "B. Main sample (eight centers, filed 2010--2014)")]
    body += [[tl[k], tex_b[k]] for k in keys_b]
    tex_table(TEX / "design_tab_examiners.tex", "lr", [], body, 2)

    # Table A3: cell definitions and indexes as rows; all centers and TC 3600 as columns (by center: csv, macros).
    # C1's index (predicted citations; 35's Figure III test in its contraction cells): pooled numbers from 35's
    # c1_kllm_all_permutation.csv, by center from 46's cell-level recomputation (tx_c1_permutation.csv), if present
    order = [("class", "Class"), ("subclass", "Subclass"), ("resid", "Class, residualized leniency")]

    def perm_cells(r) -> list:
        return [f"{int(r.cells_tested):,}", num(100 * r.apps_covered, 1), num(100 * r.p_below_05, 1)]
    hdr = [r" & \multicolumn{3}{c}{All centers} & \multicolumn{3}{c}{TC 3600} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Cells: art unit $\times$ year $\times$ & Cells & Covered (\%) & $p<0.05$ (\%) & Cells & Covered (\%) & "
           r"$p<0.05$ (\%) \\"]
    body = [("panel", "Index: predicted grant (text score, gender, entity size, inventors)")]
    body += [[rf"\quad USPC {lab[0].lower() + lab[1:]}"] + perm_cells(S.loc[(k, "All centers")]) + perm_cells(S.loc[(k, "3600")])
             for k, lab in order]
    # C1's index under both losses: Poisson (the main specification) and squared error (C1's first version)
    c1_rows = []
    for sfx, lab in [("_poisson", "Poisson model (main)"), ("", "squared-error model (first version)")]:
        c1_all, c1_tc = T / f"c1_kllm_all{sfx}_permutation.csv", T / f"tx_c1_permutation{sfx}.csv"
        if not c1_all.exists():
            continue
        o = pd.read_csv(c1_all).iloc[0]
        row = [rf"\quad USPC class, {lab}", f"{int(o.cells):,}", "", num(100 * o.share_p_below_05, 1)]
        if c1_tc.exists():
            b = pd.read_csv(c1_tc, dtype={"sample": str}).set_index("sample")
            assert int(b.loc["All centers", "cells_tested"]) == int(o.cells) and np.isclose(
                b.loc["All centers", "p_below_05"], o.share_p_below_05), "46's C1 permutation does not match 35's"
            row = [row[0]] + perm_cells(b.loc["All centers"]) + perm_cells(b.loc["3600"])
        else:
            print(f"{c1_tc.name} not found (run code/46_tests_extra.py --c1perm): C1's row without TC 3600")
            row += ["--", "--", "--"]
        c1_rows.append(row)
    if c1_rows:
        body += [("space",), ("panel", r"Index: predicted citations (C1; test and imputation sets)")] + c1_rows
    tex_table(TEX / "design_tab_permutation.tex", "lrrrrrr", hdr, body, 7)

    # macros
    Nm = {"Examiners": f"{ident['names']:,}", "NoName": num(100 * ident["no_name"], 2),
          "FewApps": num(100 * ident["few_apps"], 2), "NoLeniency": num(100 * ident["no_leniency"], 1),
          "NoFirstAction": num(100 * ident["no_first_action"], 1), "MultiTC": num(100 * ident["multi_tc"], 1),
          "MultiTCApps": num(100 * ident["multi_tc_apps"], 1), "UtilityApps": f"{ident['utility_apps']:,}",
          "Apps": f"{int(A.applications):,}", "SampleExaminers": f"{int(A.examiners):,}",
          "DockOnce": num(100 * A.docketed_once, 1), "DockOnceN": f"{int(A.docketed_once_n):,}",
          "Deleted": num(100 * A.deleted, 1), "DeldLastYear": dinfo["last_deld"][:4],
          "Cells": f"{int(A.cells):,}", "CellsApps": f"{int(A.cells_apps):,}", "CellsShare": num(100 * A.cells_share, 1),
          "ExPerCell": med(A.ex_per_cell), "AppsPerEx": med(A.apps_per_ex), "ExPerQuint": med(A.ex_per_quint),
          "AppsPerQuint": med(A.apps_per_quint), "SingleQuint": num(100 * A.single_quint, 0),
          "SingleApps": num(100 * A.single_apps, 0), "SDLen": num(A.sd_len, 2),
          "Strict": num(100 * A.strict, 0), "Lenient": num(100 * A.lenient, 0)}
    bc = cells.drop(index="All centers")
    lo, hi = bc.docketed_once.idxmin(), bc.docketed_once.idxmax()
    Nm.update(DockOnceMin=num(100 * bc.loc[lo, "docketed_once"], 1), DockOnceMinTC=lo,
              DockOnceMax=num(100 * bc.loc[hi, "docketed_once"], 0), DockOnceMaxTC=hi)
    for k, tag in [("class", "Class"), ("subclass", "Sub"), ("resid", "Resid")]:
        a, pc = S.loc[(k, "All centers")], S.loc[k].drop(index="All centers").p_below_05
        Nm[f"Perm{tag}"] = num(100 * a.p_below_05, 1)
        Nm[f"Perm{tag}Cells"] = f"{int(a.cells_tested):,}"
        Nm[f"Perm{tag}Cover"] = num(100 * a.apps_covered, 0)
        Nm[f"Perm{tag}Ex"] = num(a.mean_examiners, 1)
        Nm[f"Perm{tag}Max"], Nm[f"Perm{tag}MaxTC"] = num(100 * pc.max(), 1), pc.idxmax()
        Nm[f"Perm{tag}Min"], Nm[f"Perm{tag}MinTC"] = num(100 * pc.min(), 1), pc.idxmin()
    for k, tag in [("class", "Class"), ("subclass", "Sub"), ("resid", "Resid")]:   # TC 3600, the table's 2nd column
        b = S.loc[(k, "3600")]
        Nm[f"Perm{tag}ThreeSix"] = num(100 * b.p_below_05, 1)
        Nm[f"Perm{tag}ThreeSixCells"] = f"{int(b.cells_tested):,}"
        Nm[f"Perm{tag}ThreeSixCover"] = num(100 * b.apps_covered, 0)
    (TEX / "design_numbers.tex").write_text("".join(rf"\newcommand{{\xDe{k}}}{{{v}}}" + "\n" for k, v in Nm.items()))

    # markdown
    L = ["# Examiners and assignment cells: eight technology centers, applications filed 2010–2014", "",
         "## A. All applications in PatEx, 2004–2018 (examiners are identified by name)", "",
         f"- Examiner names: {ident['names']:,}; {100 * ident['multi_tc']:.1f}% appear in more than one technology "
         f"center (homonyms or transfers), covering {100 * ident['multi_tc_apps']:.1f}% of applications.",
         f"- Utility applications without an examiner name: {100 * ident['no_name']:.2f}% of "
         f"{ident['utility_apps']:,}.",
         f"- Applications whose examiner has fewer than {MIN_APPS} applications with a first action (no leave-one-out "
         f"leniency): {100 * ident['few_apps']:.2f}%. All applications without a leniency: "
         f"{100 * ident['no_leniency']:.2f}%, most of them because they never received a first action "
         f"({100 * ident['no_first_action']:.2f}%); 22's table reported the latter share under the former label.", "",
         "## B. Main sample: docketing, cells, leniency", "",
         "Cells: art unit × year × USPC class with at least five examiners; quintiles of equal caseload (35). "
         f"DELD (deleted from a docket) is recorded in PatEx only up to {dinfo['last_deld'][:4]} "
         f"({dinfo['n_deld_events']:,} events in all), so its zero share carries no information for this sample.", "",
         "| Row | All centers | " + " | ".join(TCS) + " |", "|---|" + "---|" * (len(TCS) + 1)]
    for k, lab, f in rows_b:
        L.append(f"| {lab} | " + " | ".join(f(cells.loc[s]) for s in cells.index) + " |")
    L += ["", "## C. Permutation test (grant index on leniency groups, 1,000 within-cell permutations)", "",
          "| Cells | Center | Cells tested | Applications covered | Examiners per cell | p < 0.05 | p < 0.10 | "
          "Mean p |", "|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.cells} | {r.sample} | {r.cells_tested:,} | {100 * r.apps_covered:.1f}% | "
                 f"{r.mean_examiners:.1f} | {100 * r.p_below_05:.1f}% | {100 * r.p_below_10:.1f}% | {r.mean_p:.2f} |")
    L += ["", f"Check against 39 (class cells): {check}.", "", f"Runtime: {time.time() - t0:.0f} s."]
    (OUT / "design_examiners.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
