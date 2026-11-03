"""Prior-art citations (06c): what is counted, unpublished applications, truncation, and the two sets of means in use.

06c counts, for every application, the later-filed applications whose examination cites its pre-grant publication or,
once granted, its patent, excluding self-citations, from the USPTO Office Action Research Dataset (OARD). This script
answers four questions about that outcome on the main sample (38's all_sample: eight technology centers, filed
2010–2014) and proposes one definition for the appendix:
(a) Unpublished applications: how many there are, by decision and center; how 06c, 35, and 34's Table A1 treat them
    (the lines of code are quoted from the scripts); whether HUPD, the source of the texts and of the sample, contains
    unpublished applications at all; and, for context, how many applications of the same centers and years are not in
    HUPD and why.
(b) Sources: OARD's citations.csv flags each cited reference as listed on Form PTO-892 (the examiner's notice of
    references cited), on Form PTO-1449 (the applicant's information disclosure statement, initialed by the examiner),
    or in the action sentence of an office action (citation_in_oa: the reference is applied in a rejection). For every
    counted citing–cited pair, the flags of all its rows are combined; the shares of pairs by source and the means of
    the counts by decision with all sources, examiner sources only (06c's prior_art_cites_examiner: 892 or in the
    action), applicant sources only, and the publication only (06c's prior_art_cites_pub).
(c) Truncation: OARD ends in mid-2017. Means by filing year and decision, the years from the focal publication (and,
    for granted applications, from the grant) to 30 June 2017, and the ratio of the rejected to the granted mean within
    each filing year and within filing year × center cells.
(d) Reconciliation: Table A1's total row (34's sample_all) and the A5 text (macros \\xKPaGr, \\xKPaRej, written by 34
    from 35's output/tables/c1_kllm_all_cf_valid.csv), recomputed from the data, with the step from one to the other.
(e) One definition for Table A1, the A5 text, and the proposal, and a proposed appendix text.

06c's citing–cited pairs are rebuilt here for the main sample with the same rules (keys, later filing, the
self-citation rules of 06b with 06c's organization normalization, imported from 06c), keeping each pair's source
flags; the resulting counts are compared with data/processed/prior_art_cites.parquet. 35's lock box is rebuilt with
35's own partition function and seed (35's load_all row order); if 35 has written data/processed/c1_preds_all_all.parquet,
the rebuilt partition is checked against it.

Runs on CPUs only (DuckDB CFE_THREADS threads, default 4) in a few minutes. citations.csv and office_actions.csv are
read from CFE_OA_DIR if set, else from data/processed/_oa_tmp (06c's temporary directory) if they are there, else
extracted from the zip to data/processed/_oa_tmp_06d, which is removed at the end.
Input: data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip (citations.csv, office_actions.csv),
       data/hupd/hupd_metadata_2022-02-22.feather,
       data/processed/{all_sample,all_text,prior_art_cites,applications_2004_2018,patex_application_data,
       patex_all_inventors,patex_all_applicants}.parquet, output/tables/c1_kllm_all_cf_valid.csv,
       output/latex/ext_numbers.tex (for the values now in use)
Output: output/prior_art_diagnostics.md, output/tables/prior_art_diagnostics_{unpublished,universe,sources,means,
        centers,cohorts,ratios,reconcile}.csv
"""
import importlib.util
import os
import pathlib
import re
import shutil
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
T = ROOT / "output/tables"
OUT = ROOT / "output"
ZIP = ROOT / "data/office_actions/OfficeActions_2017_PTOFFACT_csv.zip"
HUPD = ROOT / "data/hupd/hupd_metadata_2022-02-22.feather"
THREADS = int(os.environ.get("CFE_THREADS", "4"))
END = pd.Timestamp("2017-06-30")   # OARD's end, as the appendix states it
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
DEC = {1: "granted", 0: "rejected"}
COUNTS = {   # column: label
    "n_all": "All sources (06c's prior_art_cites)",
    "n_examiner": "Examiner: Form 892 or in the action (06c's prior_art_cites_examiner)",
    "n_in_oa": "Examiner: applied in the action sentence (a rejection)",
    "n_applicant_any": "Applicant: on Form 1449 (with or without the examiner)",
    "n_applicant_only": "Applicant only: Form 1449, not on 892 nor in the action",
    "n_pub": "Publication only (06c's prior_art_cites_pub)",
    "n_pub_examiner": "Publication only, examiner sources only",
}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quote(file: str, needle: str) -> str:
    """`file:line`: the first line of code/<file> that contains needle, for the report."""
    lines = (ROOT / "code" / file).read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if needle in line:
            return f"`code/{file}:{i}`: `{line.strip()}`"
    return f"`code/{file}`: (line containing `{needle}` not found)"


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> list:
    fmt = fmt or {}

    def cell(c, v):
        if c in fmt:
            return fmt[c](v)
        if isinstance(v, (bool, np.bool_)) or c == "filing_year":
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, r)) + " |" for r in df.itertuples(index=False)]
    return out


def oa_dir() -> tuple[pathlib.Path, bool]:
    """Directory with citations.csv and office_actions.csv, and whether to remove it at the end."""
    cands = ([pathlib.Path(os.environ["CFE_OA_DIR"])] if os.environ.get("CFE_OA_DIR") else []) + [P / "_oa_tmp"]
    for c in cands:
        if (c / "citations.csv").exists() and (c / "office_actions.csv").exists():
            return c, False
    tmp = P / "_oa_tmp_06d"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for f in ["citations.csv", "office_actions.csv"]:
            z.extract(f, tmp)
    return tmp, True


# ---------------------------------------------------------------- 06c's pairs, with their sources

def pairs(con: duckdb.DuckDBPyConnection, oa: pathlib.Path) -> pd.DataFrame:
    """06c's citing–cited pairs for the main sample, one row per pair with its combined source flags, and the
    per-application counts. Returns the counts for every main-sample application."""
    org = load_module("m06c", "06c_prior_art_citations.py").ORG   # the same organization normalization as 06c
    h = feather.read_table(HUPD, columns=["application_number", "publication_number",
                                          "date_application_published"]).to_pandas()
    s = pd.read_parquet(P / "all_sample.parquet", columns=["application_number"])
    con.register("hupd", h[h.application_number.isin(s.application_number)])
    con.execute(f"""
    CREATE TEMP TABLE focal AS
    SELECT s.application_number, s.tech_center, s.filing_year, s.granted, CAST(a.filing_date AS DATE) AS filing_date,
           regexp_extract(a.earliest_pgpub_number, '([0-9]{{11}})', 1) AS pub,
           CAST(a.earliest_pgpub_date AS DATE) AS pub_date_patex, ltrim(a.patent_number, '0') AS pat,
           CAST(a.patent_issue_date AS DATE) AS issue_date, h.publication_number AS hupd_pub,
           CAST(h.date_application_published AS DATE) AS pub_date_hupd
    FROM '{P}/all_sample.parquet' s
    JOIN '{P}/patex_application_data.parquet' a USING (application_number)
    LEFT JOIN hupd h USING (application_number)""")
    con.execute("""
    CREATE TEMP TABLE keys AS
    SELECT application_number, pub AS k, true AS is_pub FROM focal WHERE pub IS NOT NULL AND pub <> ''
    UNION ALL
    SELECT application_number, pat AS k, false AS is_pub FROM focal WHERE pat IS NOT NULL AND pat <> ''""")
    con.execute(f"""
    CREATE TEMP TABLE c AS
    SELECT lpad(CAST(c.app_id AS VARCHAR), 8, '0') AS citing_app, k.application_number AS cited_app,
           bool_or(k.is_pub) AS via_pub, bool_or(NOT k.is_pub) AS via_pat,
           bool_or(c.form892 = '1') AS f892, bool_or(c.form1449 = '1') AS f1449,
           bool_or(c.citation_in_oa = '1') AS in_oa,
           bool_or(k.is_pub AND (c.form892 = '1' OR c.citation_in_oa = '1')) AS pub_examiner,
           count(*) AS n_rows
    FROM read_csv('{oa}/citations.csv', header = true, all_varchar = true) c
    JOIN keys k ON k.k = ltrim(c.parsed, '0')
    WHERE lpad(CAST(c.app_id AS VARCHAR), 8, '0') <> k.application_number
    GROUP BY 1, 2""")
    con.execute(f"""
    CREATE TEMP TABLE c2 AS
    SELECT c.* FROM c
    JOIN '{P}/patex_application_data.parquet' a ON a.application_number = c.citing_app
    JOIN focal f ON f.application_number = c.cited_app
    WHERE CAST(a.filing_date AS DATE) > f.filing_date""")
    con.execute("""
    CREATE TEMP TABLE apps AS
    SELECT application_number FROM focal UNION SELECT DISTINCT citing_app FROM c2""")
    con.execute(f"""
    CREATE TEMP TABLE inv AS
    SELECT DISTINCT i.application_number,
           regexp_replace(upper(inventor_name_last), '[^A-Z]', '', 'g') || '|' ||
           coalesce(regexp_extract(upper(inventor_name_first), '[A-Z]+'), '') AS k
    FROM '{P}/patex_all_inventors.parquet' i JOIN apps USING (application_number)
    WHERE inventor_name_last IS NOT NULL""")
    con.execute(f"""
    CREATE TEMP TABLE org AS
    SELECT * FROM (
      SELECT DISTINCT o.application_number, {org} AS k
      FROM '{P}/patex_all_applicants.parquet' o JOIN apps USING (application_number)
      WHERE applicant_organization IS NOT NULL) WHERE length(k) > 1""")
    con.execute("""
    CREATE TEMP TABLE selfc AS
    SELECT DISTINCT c2.citing_app, c2.cited_app FROM c2
    JOIN inv a ON a.application_number = c2.citing_app
    JOIN inv b ON b.application_number = c2.cited_app AND a.k = b.k
    UNION
    SELECT DISTINCT c2.citing_app, c2.cited_app FROM c2
    JOIN org a ON a.application_number = c2.citing_app
    JOIN org b ON b.application_number = c2.cited_app AND a.k = b.k""")
    con.execute("""
    CREATE TEMP TABLE t AS
    SELECT c2.*, (f892 OR in_oa) AS examiner FROM c2
    LEFT JOIN selfc s USING (citing_app, cited_app) WHERE s.citing_app IS NULL""")
    return con.execute("""
    SELECT f.application_number, f.tech_center, f.filing_year, f.granted, f.filing_date,
           f.pub IS NOT NULL AND f.pub <> '' AS published, f.hupd_pub IS NOT NULL AND f.hupd_pub <> '' AS in_hupd_pub,
           coalesce(f.pub_date_hupd, f.pub_date_patex) AS pub_date, f.issue_date,
           count(t.citing_app) AS n_all,
           count(t.citing_app) FILTER (WHERE t.examiner) AS n_examiner,
           count(t.citing_app) FILTER (WHERE t.in_oa) AS n_in_oa,
           count(t.citing_app) FILTER (WHERE t.f1449) AS n_applicant_any,
           count(t.citing_app) FILTER (WHERE t.f1449 AND NOT t.examiner) AS n_applicant_only,
           count(t.citing_app) FILTER (WHERE t.via_pub) AS n_pub,
           count(t.citing_app) FILTER (WHERE t.pub_examiner) AS n_pub_examiner,
           count(t.citing_app) FILTER (WHERE t.f892 AND NOT t.in_oa AND NOT t.f1449) AS k_892_only,
           count(t.citing_app) FILTER (WHERE t.in_oa AND NOT t.f1449) AS k_oa_not1449,
           count(t.citing_app) FILTER (WHERE t.examiner AND t.f1449) AS k_both,
           count(t.citing_app) FILTER (WHERE t.in_oa AND t.f1449) AS k_oa_and1449
    FROM focal f LEFT JOIN t ON t.cited_app = f.application_number
    GROUP BY ALL ORDER BY 1""").df()


def oard_facts(con: duckdb.DuckDBPyConnection, oa: pathlib.Path) -> dict:
    """Row-level source flags of the whole citations.csv and the mail dates of office_actions.csv."""
    r = con.execute(f"""
    SELECT count(*) AS rows, avg((form892 = '1')::DOUBLE) AS f892, avg((form1449 = '1')::DOUBLE) AS f1449,
           avg((citation_in_oa = '1')::DOUBLE) AS in_oa,
           avg((form892 <> '1' AND form1449 <> '1' AND citation_in_oa <> '1')::DOUBLE) AS none,
           count(DISTINCT app_id) AS apps
    FROM read_csv('{oa}/citations.csv', header = true, all_varchar = true)""").df().iloc[0].to_dict()
    m = con.execute(f"""
    SELECT left(mail_dt, 7) AS month, count(*) AS n
    FROM read_csv('{oa}/office_actions.csv', header = true, all_varchar = true) GROUP BY 1 ORDER BY 1""").df()
    r["first_month"] = m[m.n >= 1000].month.min()
    r["last_date"] = con.execute(f"""SELECT max(mail_dt) FROM read_csv('{oa}/office_actions.csv', header = true,
                                     all_varchar = true)""").fetchone()[0]
    r["months_2017"] = ", ".join(f"{x.month[5:]}: {x.n:,}" for x in m[m.month >= "2017"].itertuples())
    return r


# ---------------------------------------------------------------- 35's lock box

def lockbox(d: pd.DataFrame) -> pd.Series:
    """35's partition of the main sample (CFE_SAMPLE=all), with 35's function, seed, and row order."""
    m35 = load_module("m35", "35_c1_kllm.py")
    keys = pd.read_parquet(P / "all_text.parquet", columns=["application_number"])
    x = keys.merge(pd.read_parquet(P / "all_sample.parquet"), on="application_number")
    x = x.sort_values("application_number").reset_index(drop=True)
    a = pd.read_parquet(P / "applications_2004_2018.parquet", columns=["application_number", "filing_date"])
    x = x.merge(a, on="application_number", how="left")
    x["filing_date"] = pd.to_datetime(x.filing_date)
    x["part"] = m35.partition(x, np.random.default_rng(m35.SEED + 9999))   # run_tc's first draw for tc == "all"
    f = P / "c1_preds_all_all.parquet"
    if f.exists():
        chk = x[["application_number", "part"]].merge(pd.read_parquet(f, columns=["application_number", "part"]),
                                                      on="application_number")
        assert (chk.part_x == chk.part_y).all(), "rebuilt partition differs from 35's c1_preds_all_all.parquet"
        print("partition checked against", f.name)
    # which rule put an application in the lock box (35 applies them in this order)
    late = x.filing_date >= m35.LOCK_START
    size = x.groupby("examiner_id").size().sort_index().sort_values(ascending=False, kind="mergesort")
    busy = x.examiner_id.isin(size.head(m35.N_BUSY).index)
    why = np.select([x.part != "lockbox", late, busy], ["not in lock box", f"filed on or after {m35.LOCK_START}",
                                                         "examiners with the largest caseloads"], "random 10%")
    out = pd.DataFrame(dict(application_number=x.application_number, part=x.part, lock_rule=why))
    return d.merge(out, on="application_number", how="left")


# ---------------------------------------------------------------- tables

def means(x: pd.DataFrame, cols: list) -> dict:
    g = x.granted == 1
    r = dict(n_granted=int(g.sum()), n_rejected=int((~g).sum()))
    for c in cols:
        r[f"{c}_granted"], r[f"{c}_rejected"] = x.loc[g, c].mean(), x.loc[~g, c].mean()
        r[f"{c}_ratio"] = r[f"{c}_rejected"] / r[f"{c}_granted"]
    return r


def within_ratio(x: pd.DataFrame, col: str, by: list) -> float:
    """Rejected over granted mean within cells, pooled with the rejected applications' weights (as 35's alpha_table)."""
    m = x.groupby(by + ["granted"])[col].agg(["mean", "size"]).unstack("granted").dropna()
    return (m[("size", 0)] * m[("mean", 0)]).sum() / (m[("size", 0)] * m[("mean", 1)]).sum()


def main() -> None:
    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}; SET memory_limit = '40GB'")
    oa, cleanup = oa_dir()
    print("OARD files:", oa)
    d = pairs(con, oa)
    facts = oard_facts(con, oa)
    if cleanup:
        shutil.rmtree(oa)
    print("pairs done:", len(d))

    # agreement with 06c's output
    pa = pd.read_parquet(P / "prior_art_cites.parquet")
    chk = d.merge(pa, on="application_number", how="left")
    mism = {c: int((chk[a] != chk[c]).sum()) for a, c in [("n_all", "prior_art_cites"),
                                                         ("n_examiner", "prior_art_cites_examiner"),
                                                         ("n_pub", "prior_art_cites_pub")]}
    mism["published"] = int((chk.published_x != chk.published_y).sum())
    mism["missing in 06c"] = int(chk.prior_art_cites.isna().sum())
    print("mismatches with prior_art_cites.parquet:", mism)

    for c in ["filing_date", "pub_date", "issue_date"]:
        d[c] = pd.to_datetime(d[c])
    d = lockbox(d)
    d["dec"] = d.granted.map(DEC)
    pub = d[d.published]
    cols = list(COUNTS)

    # (a) unpublished applications
    rows = []
    for tc, x in list(d.groupby("tech_center")) + [("All centers", d)]:
        for gv in [1, 0, None]:
            y = x if gv is None else x[x.granted == gv]
            u = y[~y.published]
            rows.append(dict(center=tc, decision="all" if gv is None else DEC[gv], n=len(y),
                             unpublished_patex=len(u), share_unpublished=len(u) / len(y),
                             unpublished_hupd=int((~y.in_hupd_pub).sum()),
                             mean_prior_art_cites_unpublished=u.n_all.mean() if len(u) else np.nan))
    unpub = pd.DataFrame(rows)
    unpub.to_csv(T / "prior_art_diagnostics_unpublished.csv", index=False)
    uni = con.execute(f"""
    SELECT a.granted, a.in_hupd, count(*) AS n,
           count(*) FILTER (WHERE p.earliest_pgpub_number IS NOT NULL AND p.earliest_pgpub_number <> '') AS published
    FROM '{P}/applications_2004_2018.parquet' a
    JOIN (SELECT application_number, earliest_pgpub_number FROM '{P}/patex_application_data.parquet') p
        USING (application_number)
    WHERE a.tech_center IN ({', '.join(f"'{t}'" for t in TCS)}) AND a.filing_year BETWEEN 2010 AND 2014
    GROUP BY ALL ORDER BY ALL""").df()
    uni["decision"] = uni.granted.map(DEC)
    uni["unpublished"] = uni.n - uni.published
    uni = uni[["decision", "in_hupd", "n", "published", "unpublished"]]
    uni.to_csv(T / "prior_art_diagnostics_universe.csv", index=False)
    h = feather.read_table(HUPD, columns=["publication_number"]).column(0).to_pandas()
    hupd_n, hupd_nopub = len(h), int((h.isna() | (h == "")).sum())

    # (b) sources: shares of counted pairs, and means of the counts
    src = []
    for lab, x in [("granted", pub[pub.granted == 1]), ("rejected", pub[pub.granted == 0]), ("all", pub)]:
        n = x.n_all.sum()
        src.append(dict(decision=lab, pairs=int(n),
                        examiner_only=(x.n_examiner.sum() - x.k_both.sum()) / n,
                        examiner_892_only=x.k_892_only.sum() / n, examiner_in_action_not_1449=x.k_oa_not1449.sum() / n,
                        examiner_and_applicant=x.k_both.sum() / n, applicant_only=x.n_applicant_only.sum() / n,
                        any_examiner=x.n_examiner.sum() / n, any_in_action=x.n_in_oa.sum() / n,
                        any_applicant_1449=x.n_applicant_any.sum() / n,
                        via_publication=x.n_pub.sum() / n, via_patent_only=1 - x.n_pub.sum() / n))
    src = pd.DataFrame(src)
    src.to_csv(T / "prior_art_diagnostics_sources.csv", index=False)
    mrow = []
    for c in cols:
        r = means(pub, [c])
        mrow.append(dict(count=c, definition=COUNTS[c], granted=r[f"{c}_granted"], rejected=r[f"{c}_rejected"],
                         ratio_rejected_granted=r[f"{c}_ratio"], within_year_center_ratio=within_ratio(
                             pub, c, ["filing_year", "tech_center"]),
                         share_zero_granted=(pub[pub.granted == 1][c] == 0).mean(),
                         share_zero_rejected=(pub[pub.granted == 0][c] == 0).mean()))
    mtab = pd.DataFrame(mrow)
    mtab.to_csv(T / "prior_art_diagnostics_means.csv", index=False)
    crow = []
    for tc, x in list(pub.groupby("tech_center")) + [("All centers", pub)]:
        r = means(x, ["n_all", "n_examiner", "n_pub"])
        crow.append(dict(center=tc, **r))
    ctab = pd.DataFrame(crow)
    ctab.to_csv(T / "prior_art_diagnostics_centers.csv", index=False)

    # (c) truncation by filing year
    yrow = []
    for (yr, gv), x in pub.groupby(["filing_year", "granted"]):
        issued = x.issue_date.notna() & (x.issue_date <= END)
        yrow.append(dict(filing_year=yr, decision=DEC[gv], n=len(x), mean_all=x.n_all.mean(),
                         mean_examiner=x.n_examiner.mean(), mean_pub=x.n_pub.mean(),
                         share_zero=(x.n_all == 0).mean(),
                         median_years_pub_to_end=((END - x.pub_date).dt.days / 365.25).median(),
                         median_months_filing_to_pub=((x.pub_date - x.filing_date).dt.days
                                                      / 30.44).median(),
                         share_issued_by_end=issued.mean() if gv == 1 else np.nan,
                         median_years_issue_to_end=((END - x.issue_date[issued]).dt.days / 365.25).median()
                         if gv == 1 else np.nan))
    ytab = pd.DataFrame(yrow)
    ytab.to_csv(T / "prior_art_diagnostics_cohorts.csv", index=False)
    rrow = []
    for yr, x in list(pub.groupby("filing_year")) + [("2010–2014, within filing year", pub)]:
        r = dict(filing_year=yr)
        for c in ["n_all", "n_examiner", "n_pub", "n_pub_examiner"]:
            r[f"{c}_ratio"] = (within_ratio(x, c, ["filing_year"]) if isinstance(yr, str)
                               else x[x.granted == 0][c].mean() / x[x.granted == 1][c].mean())
            r[f"{c}_ratio_within_center"] = within_ratio(x, c, ["filing_year", "tech_center"])
        rrow.append(r)
    r = dict(filing_year="2010–2014, pooled (no cells)")
    for c in ["n_all", "n_examiner", "n_pub", "n_pub_examiner"]:
        r[f"{c}_ratio"] = pub[pub.granted == 0][c].mean() / pub[pub.granted == 1][c].mean()
    rrow.append(r)
    rtab = pd.DataFrame(rrow)
    rtab.to_csv(T / "prior_art_diagnostics_ratios.csv", index=False)

    # (d) reconciliation
    used = {}
    tex = OUT / "latex/ext_numbers.tex"
    if tex.exists():
        for k in ["xKPaGr", "xKPaRej"]:
            m = re.search(rf"\\newcommand{{\\{k}}}{{([^}}]*)}}", tex.read_text())
            used[k] = m.group(1) if m else ""
    cfv = T / "c1_kllm_all_cf_valid.csv"
    cf = pd.read_csv(cfv).iloc[-1] if cfv.exists() else None
    specs = [
        ("Table A1 (34 sample_all)", "main sample; unpublished at 06c's value (0 if rejected, patent route if granted)",
         d, "n_all"),
        ("Recommended", "main sample, published applications only; all sources", pub, "n_all"),
        ("A5 text (35 cf_valid → 34 \\xKPaGr, \\xKPaRej)",
         "main sample with text (all of it), outside 35's lock box, published only; all sources",
         pub[pub.part != "lockbox"], "n_all"),
        ("35's lock box only", "published; all sources", pub[pub.part == "lockbox"], "n_all"),
    ] + [(f"  lock box: {w}", "published; all sources", pub[pub.lock_rule == w], "n_all")
         for w in sorted(pub.lock_rule.unique()) if w != "not in lock box"] + [
        ("Recommended, examiner sources only", "main sample, published; Form 892 or in the action", pub,
         "n_examiner"),
        ("A5 sample, examiner sources only", "outside 35's lock box, published; Form 892 or in the action",
         pub[pub.part != "lockbox"], "n_examiner"),
        ("Recommended, filed 2010–2013", "main sample, published, filed before 2014", pub[pub.filing_year <= 2013],
         "n_all"),
        ("Recommended, filed before 2014-07-01", "main sample, published, filed before 35's lock-box date",
         pub[pub.filing_date < pd.Timestamp("2014-07-01")], "n_all")]
    rec = []
    for name, defn, x, c in specs:
        r = means(x, [c])
        rec.append(dict(row=name, definition=defn, n_granted=r["n_granted"], n_rejected=r["n_rejected"],
                        granted=r[f"{c}_granted"], rejected=r[f"{c}_rejected"], ratio=r[f"{c}_ratio"],
                        mean_filing_year=x.filing_year.mean()))
    rec = pd.DataFrame(rec)
    rec.to_csv(T / "prior_art_diagnostics_reconcile.csv", index=False)

    # ---------------------------------------------------------------- report
    write_report(d, pub, mism, facts, unpub, uni, hupd_n, hupd_nopub, src, mtab, ctab, ytab, rtab, rec, used, cf)


# the lines of code the report quotes: (key, file, a string that identifies the line)
QUOTES = [("38_hupd", "38_all_centers.py", "AND a.in_hupd AND NOT a.continuation"),
          ("06c_pub", "06c_prior_art_citations.py", "SELECT application_number, pub AS k, true AS is_pub"),
          ("06c_pat", "06c_prior_art_citations.py", "SELECT application_number, pat AS k, false AS is_pub"),
          ("06c_zero", "06c_prior_art_citations.py", "coalesce(count(DISTINCT t.citing_app), 0) AS prior_art_cites,"),
          ("06c_left", "06c_prior_art_citations.py", "FROM focal f LEFT JOIN t ON t.cited_app = f.application_number"),
          ("06c_flag", "06c_prior_art_citations.py", "AS published,"),
          ("06c_ex", "06c_prior_art_citations.py", "(c.form892 = '1' OR c.citation_in_oa = '1') AS by_examiner"),
          ("06c_expub", "06c_prior_art_citations.py", "FILTER (WHERE t.is_pub)"),
          ("35_nan", "35_c1_kllm.py", 'd["prior_art"] = np.where(d.published.fillna(False), d.prior_art_cites, np.nan)'),
          ("35_mean", "35_c1_kllm.py", "mean_prior_art_granted=gx.prior_art.mean()"),
          ("35_lock", "35_c1_kllm.py", 'gx, rx = x[g & (x.part != "lockbox")]'),
          ("35_zero1", "35_c1_kllm.py", "pa = np.nan_to_num(x.prior_art.to_numpy(float))"),
          ("35_zero2", "35_c1_kllm.py", "z = pd.DataFrame(dict(y=np.nan_to_num(t.prior_art.to_numpy(float))"),
          ("34_merge", "34_extension_tables.py", 'pd.read_parquet(ROOT / "data/processed/prior_art_cites.parquet"), on='),
          ("34_mean", "34_extension_tables.py", "num(x.prior_art_cites[g].mean()), num(x.prior_art_cites[~g].mean())"),
          ("34_macro", "34_extension_tables.py", 'N["xKPaRej"], N["xKPaGr"]')]


def write_report(d, pub, mism, facts, unpub, uni, hupd_n, hupd_nopub, src, mtab, ctab, ytab, rtab, rec, used,
                 cf) -> None:
    Q = {k: quote(f, n) for k, f, n in QUOTES}
    R = rec.set_index("row")
    rc, a5 = R.loc["Recommended"], R.loc["A5 text (35 cf_valid → 34 \\xKPaGr, \\xKPaRej)"]
    ta1 = R.loc["Table A1 (34 sample_all)"]
    M = mtab.set_index("count")
    S = src.set_index("decision").loc["all"]
    Y = ytab.set_index(["filing_year", "decision"])
    RT = rtab.set_index("filing_year")
    yrs = sorted(pub.filing_year.unique())
    ua = unpub[(unpub.center == "All centers")].set_index("decision")
    n_unpub = int(ua.loc["all", "unpublished_patex"])
    nh = uni[~uni.in_hupd].set_index("decision")
    a1 = OUT / "latex/ext_tab_sample_all.tex"
    a1_total = next((ln.strip() for ln in a1.read_text().splitlines() if ln.startswith("All centers")), "") \
        if a1.exists() else "(not found)"
    L = ["# Prior-art citations (06c): unpublished applications, sources, truncation, and the means in use", "",
         f"Main sample: 38's all_sample, {len(d):,} applications in eight technology centers filed 2010–2014 "
         f"({int((d.granted == 1).sum()):,} granted, {int((d.granted == 0).sum()):,} not granted, called rejected "
         "below). Counts rebuilt from OARD's citations.csv with 06c's rules; applications whose rebuilt value differs "
         f"from data/processed/prior_art_cites.parquet: {mism}.", "",
         "## (a) Unpublished applications", "",
         "- **The main sample contains only published applications, by construction.** 38 keeps applications in "
         f"HUPD ({Q['38_hupd']}), and HUPD is a corpus of published applications: all {hupd_n:,} records of its "
         f"metadata carry a publication number ({hupd_nopub} without). Applications filed with a non-publication "
         "request, or abandoned before publication, are not in HUPD and therefore not in the sample.",
         f"- In PatEx, {n_unpub} of the {len(d):,} sample applications ({100 * n_unpub / len(d):.4f}%) have no "
         f"pre-grant publication number ({int(ua.loc['granted', 'unpublished_patex'])} granted, "
         f"{int(ua.loc['rejected', 'unpublished_patex'])} rejected); all {n_unpub} have one in HUPD "
         f"(unpublished per HUPD: {int(ua.loc['all', 'unpublished_hupd'])}), so this is a gap in PatEx's "
         "earliest_pgpub_number, not a missing publication. 06c's `published` flag is False for them.",
         "- Context: all applications of the eight centers filed 2010–2014 in PatEx (applications_2004_2018, before "
         "the sample's other restrictions). An application that is neither published nor patented is confidential "
         "(35 U.S.C. 122(a)); Public PAIR, PatEx's source, shows it only if it was later opened to the public (for "
         "example, because a published application or patent claims its benefit), so most rejected applications "
         "abandoned before publication are not observed at all. Of the unpublished applications PatEx does list, "
         f"{int(nh.loc['granted', 'unpublished']):,} are granted (mostly non-publication requests that became "
         f"patents) and {int(nh.loc['rejected', 'unpublished']):,} are not; none is in HUPD, so none is in the "
         "sample.", ""]
    L += md_table(uni.assign(share_unpublished=uni.unpublished / uni.n))
    L += ["", "**Main-sample applications without a PatEx publication number, by center and decision** "
          "(prior_art_diagnostics_unpublished.csv):", ""]
    L += md_table(unpub[unpub.decision != "all"].drop(columns=["mean_prior_art_cites_unpublished"]),
                  {"share_unpublished": lambda v: f"{100 * v:.4f}%"})
    L += ["", "**How each script treats them:**", "",
          f"- 06c gives them a count, not a missing value. `published` is whether the publication number exists "
          f"({Q['06c_flag']}); only the patent is a key ({Q['06c_pub']}; {Q['06c_pat']}); every focal application is "
          f"kept by the left join with a zero default ({Q['06c_left']}; {Q['06c_zero']}). So an unpublished rejected "
          "application counts 0, and an unpublished granted one counts the citations of its patent.",
          f"- 35 excludes them (missing value): {Q['35_nan']} (the same line in load() and load_all()). Means skip "
          f"them, so cf_valid and hence \\xKPaGr, \\xKPaRej do: {Q['35_mean']}. Some later steps fill them back "
          f"with zeros, e.g. {Q['35_zero1']} and {Q['35_zero2']} (with {n_unpub} such applications this changes "
          "nothing).",
          f"- 34's Table A1 keeps 06c's value (zero for the rejected): {Q['34_merge']}; {Q['34_mean']}.",
          f"- Effect on the All-centers means: granted {ta1.granted:.4f} (Table A1's rule) vs {rc.granted:.4f} "
          f"(published only), rejected {ta1.rejected:.4f} vs {rc.rejected:.4f}. Unpublished applications explain "
          "neither the gap between granted and rejected nor the gap between Table A1 and the A5 text.", "",
          "## (b) Sources of the counted citations", "",
          "OARD's citations.csv (Variable tables v20171120) has, for each application–reference pair (and, when the "
          "reference is applied in an action, for each office action and action type), `form892` (listed on Form "
          "PTO-892, the examiner's Notice of References Cited), `form1449` (listed on Form PTO-1449/SB08, the "
          "applicant's information disclosure statement, which the examiner initials as considered), and "
          "`citation_in_oa` (referenced in the action sentence of the office action, i.e. applied in a rejection). "
          "The dataset documentation: \"The citations file is derived from citations referenced on the Form PTO-892, "
          f"Form PTO-1449, and text of Office actions.\" In the whole file ({facts['rows']:,.0f} rows, "
          f"{facts['apps']:,.0f} citing applications) {100 * facts['f892']:.1f}% of rows are on Form 892, "
          f"{100 * facts['f1449']:.1f}% on Form 1449, {100 * facts['in_oa']:.1f}% in the action text; rows with no "
          f"flag: {100 * facts['none']:.2f}%. Office actions run from {facts['first_month']} (first month with at "
          f"least 1,000 actions) to {facts['last_date']}; actions by month in 2017: {facts['months_2017']}.", "",
          f"06c: {Q['06c_ex']}. prior_art_cites counts distinct citing applications from all three sources; "
          "prior_art_cites_examiner those with at least one examiner row (Form 892 or in the action), whether or not "
          "the applicant also listed the reference; prior_art_cites_pub those citing the 11-digit publication number "
          f"({Q['06c_expub']}), i.e. it leaves out citing applications that cite only the patent. For a rejected "
          "application prior_art_cites = prior_art_cites_pub (it has no patent).", "",
          "**Counted citing–cited pairs by source** (published main-sample applications; flags combined over all rows "
          "of a pair; examiner_only + examiner_and_applicant + applicant_only = 100%; prior_art_diagnostics_sources.csv):",
          ""]
    L += md_table(src, {c: (lambda v: f"{100 * v:.1f}%") for c in src.columns if c not in ["decision", "pairs"]})
    L += ["", "**Means by decision** (published main-sample applications; ratio = rejected / granted; the next "
          "column pools it within filing year × center cells with the rejected applications' weights; "
          "prior_art_diagnostics_means.csv):", ""]
    L += md_table(mtab.drop(columns=["count"]))
    Sg = src.set_index("decision").loc["granted"]
    L += ["", f"**The patent route.** {100 * Sg.via_patent_only:.1f}% of the citing applications counted for granted "
          "applications cite only the patent, a route rejected applications do not have. Counting citations of the "
          f"publication only, the means are {M.loc['n_pub', 'granted']:.2f} (granted) and "
          f"{M.loc['n_pub', 'rejected']:.2f} (rejected), ratio {M.loc['n_pub', 'ratio_rejected_granted']:.2f}, and "
          f"{M.loc['n_pub', 'within_year_center_ratio']:.2f} within filing year × center cells. Most of the gap "
          "between granted and rejected applications in prior_art_cites is thus the grant's own effect (the patent "
          "is cited), not a difference in how often the publications are cited.", "",
          "## (c) Truncation", "",
          "Citations are observed only in office actions mailed by mid-2017, so the window shrinks with the filing "
          "year. Years from the focal publication date (HUPD, else PatEx) to 30 June 2017, and for granted "
          "applications from the grant (those granted by then); prior_art_diagnostics_cohorts.csv:", ""]
    L += md_table(ytab)
    L += ["", "**Rejected / granted ratio by filing year** (within center: pooled over centers with the rejected "
          "applications' weights; prior_art_diagnostics_ratios.csv):", ""]
    L += md_table(rtab)
    L += ["", "## (d) The two sets of means", "",
          f"- Table A1's total row now: `{a1_total}`. Built by 34's sample_all on all {len(d):,} main-sample "
          "applications with 06c's prior_art_cites (unpublished at 06c's value; all filing years).",
          f"- A5 text and proposal: \\xKPaGr = {used.get('xKPaGr', '?')}, \\xKPaRej = {used.get('xKPaRej', '?')} "
          f"({Q['34_macro']}), from 35's c1_kllm_all_cf_valid.csv"
          + (f" (mean_prior_art_granted = {cf.mean_prior_art_granted:.4f}, mean_prior_art_rejected = "
             f"{cf.mean_prior_art_rejected:.4f})" if cf is not None else "")
          + (f"; recomputed here: {a5.granted:.4f} and {a5.rejected:.4f}, "
             f"{'identical' if abs(a5.granted - cf.mean_prior_art_granted) + abs(a5.rejected - cf.mean_prior_art_rejected) < 1e-9 else 'DIFFERENT'}"
             if cf is not None else "")
          + f"; computed on {Q['35_lock']}: the main sample (35's load_all keeps the applications with text, which "
          "is all of them), outside 35's lock box, published applications only. The lock box (35's partition: all "
          "filings from 2014-07-01, then all applications of examiners drawn from the 25 with the largest caseloads "
          f"until 10%, then a random 10%) holds {int((d.part == 'lockbox').sum()):,} applications.",
          "- Recomputed (prior_art_diagnostics_reconcile.csv):", ""]
    L += md_table(rec)
    L += ["", f"So the difference is the sample, not the definition: excluding the lock box raises both means "
          f"({rc.granted:.2f} → {a5.granted:.2f} granted, {rc.rejected:.2f} → {a5.rejected:.2f} rejected) because "
          "it drops the late filings, which have had little time to be cited; the rejected / granted ratio barely "
          f"moves ({rc.ratio:.3f} vs {a5.ratio:.3f}). The treatment of unpublished applications changes the means in "
          "the fourth decimal.", "",
          "## (e) Recommendation", "",
          "Use one definition everywhere (Table A1, the A5 text, the proposal): 06c's prior_art_cites (distinct "
          "later-filed applications whose office actions cite the focal application's pre-grant publication or its "
          "patent, from Form 892, Form 1449, or the action text, excluding self-citations) on the published "
          f"applications of the whole main sample ({len(pub):,}; the {n_unpub} without a PatEx publication number "
          "excluded, as in 35), all filing years, no lock-box exclusion. This is the variable 35's analyses use. "
          f"All centers: **granted {rc.granted:.2f}, rejected {rc.rejected:.2f}** (ratio {rc.ratio:.2f}); examiner "
          f"sources only {M.loc['n_examiner', 'granted']:.2f} and {M.loc['n_examiner', 'rejected']:.2f}; publication "
          f"only {M.loc['n_pub', 'granted']:.2f} and {M.loc['n_pub', 'rejected']:.2f}. Table A1's numbers do not "
          f"change at two decimals ({ta1.granted:.2f}, {ta1.rejected:.2f}); the A5 text and the proposal should "
          f"replace {a5.rejected:.2f} / {a5.granted:.2f} with {rc.rejected:.2f} / {rc.granted:.2f} (in 34, take "
          "\\xKPaGr and \\xKPaRej from sample_all's published applications instead of from 35's cf_valid). The "
          "analyses of 35 (α, reranking, contraction) keep their own samples and compare applications within "
          "filing-year cells. Where the text compares granted with rejected applications, it should say that granted "
          "applications are also cited through their patent and give the publication-only means.", "",
          "By center under this definition (prior_art_diagnostics_centers.csv):", ""]
    L += md_table(ctab[["center", "n_granted", "n_rejected", "n_all_granted", "n_all_rejected", "n_all_ratio",
                        "n_examiner_granted", "n_examiner_rejected", "n_pub_granted", "n_pub_rejected"]])
    y0, y1 = yrs[0], yrs[-1]
    gap = max(abs(Y.loc[(y, 'rejected'), 'median_years_pub_to_end'] - Y.loc[(y, 'granted'), 'median_years_pub_to_end'])
              for y in yrs)
    gap = np.ceil(10 * gap) / 10
    text = (
        "For every application we count the later-filed applications whose office actions cite its pre-grant "
        "publication or, once granted, its patent, excluding self-citations (script 06c; USPTO Office Action Research "
        "Dataset, office actions mailed from 2008 to mid-2017). The dataset records the references the examiner "
        "cites (Form PTO-892, or applied in a rejection) and those the applicant discloses in an information "
        "disclosure statement that the examiner considers (Form PTO-1449); "
        f"{100 * S.any_examiner:.0f}% of the counted citing applications cite through the examiner, and with "
        f"examiner citations alone the means are {M.loc['n_examiner', 'granted']:.2f} (granted) and "
        f"{M.loc['n_examiner', 'rejected']:.2f} (rejected) instead of {rc.granted:.2f} and {rc.rejected:.2f}. "
        "Every application in the sample was published, because HUPD contains only published applications "
        "(applications with a non-publication request or abandoned before publication are not in the sample), so no "
        f"count is zero for lack of a publication ({n_unpub} applications whose publication number is missing in "
        "PatEx are excluded). Because the data end in mid-2017, the median window after publication shrinks from "
        f"{Y.loc[(y0, 'granted'), 'median_years_pub_to_end']:.1f} years for {y0} filings to "
        f"{Y.loc[(y1, 'granted'), 'median_years_pub_to_end']:.1f} for {y1} filings and the means fall (granted "
        f"{Y.loc[(y0, 'granted'), 'mean_all']:.2f} to {Y.loc[(y1, 'granted'), 'mean_all']:.2f}, rejected "
        f"{Y.loc[(y0, 'rejected'), 'mean_all']:.2f} to {Y.loc[(y1, 'rejected'), 'mean_all']:.2f}), but granted and "
        f"rejected applications filed in the same year are published at about the same time (medians within "
        f"{gap:.1f} years), so comparisons within filing-year cells are not affected: there, citations of the "
        "publication give a rejected-to-granted ratio of "
        f"{min(RT.loc[yrs, 'n_pub_ratio']):.2f}–{max(RT.loc[yrs, 'n_pub_ratio']):.2f} in every cohort, and "
        f"{min(RT.loc[yrs, 'n_all_ratio']):.2f}–{max(RT.loc[yrs, 'n_all_ratio']):.2f} with citations of the "
        "patent, which accrue only after grant and so are cut shorter in later cohorts.")
    L += ["", "**Proposed appendix text:**", "", "> " + text, ""]
    (OUT / "prior_art_diagnostics.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
