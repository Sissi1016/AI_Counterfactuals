"""C1 (35) follow-ups: examples by technology center, a family check of the rejected applications, contraction with
transformed citation outcomes, the sign of Table V's binary outcome, and the main-text figure.

All from the cached predictions of 35's main run (data/processed/c1_preds_all_all.parquet), without refitting.
1. Examples (Table A18), by a rule fixed before looking at outcomes (see examples()): in each center, one rejected
   application without a granted relative drawn at random (seed 20261101) from those whose counterfactual score is
   in the top 1% of granted applications' scores in the same USPC class × filing year; with representativeness
   statistics of the pools (--examples runs this part only).
2. Family check: a rejected application (no patent) whose invention may have been patented through a related
   application. PatEx continuity data (2022 release, continuity_parents.csv and continuity_children.csv): (i) a granted
   child (continuation, continuation-in-part, or divisional); (ii) a granted member of the continuity family (all U.S.
   utility applications connected by such links: children, grandchildren, siblings through a common nonprovisional
   parent); (iii) (ii) or another granted U.S. application claiming the same provisional or PCT application.
   Overall, by center, by decile of the counterfactual score, among the top 1% and the top 10 by score; granted
   applications for comparison.
3. Contraction (35's contraction(), same ranking, same quintiles and cells, B = 500 bootstrap draws of art units, own
   random stream) with transformed citation outcomes: log(1 + citations), and citations top-coded at the 99th
   percentile of granted applications of the same center and filing year. Rejected applications have zero.
4. Table V's binary outcome ('uncited', a granted patent no one else cites within five years) is a bad outcome: a rule
   does better the more of it it removes. 35 reports rule − examiners in the relative change of the outcome per
   application, so a negative value there means the rule removes more uncited patents than the examiners. Restated
   here with the sign 'positive = better'.
5. Main-text figure (journal style): (a) where the strictest quintile's extra rejections and the algorithm's fall
   across 20 bins of predicted citations (35's fig6); (b) prior-art citations by decile of the counterfactual score,
   rejected against granted applications (35's cf_deciles).

6. Tail diagnostics: Table III with top-coded citations in the main and each robustness design of 35's robust(),
   the main design without art unit 3721 or TC 3700, the split of algorithm − examiners into the selection gain
   (algorithm − random, the random rule's expected curve) and random − examiners, and the estimator's weight on the
   surgical-stapler cells (art unit 3721 × USPC 227).
7. --lite: only the citation algorithm refitted (main run's hyperparameters): trained on 2010–2012 filings and
   evaluated on 2013–2014 ones; without entity size and number of inventors (outputs c1x_lite_*).
8. --poisson: 35's Poisson variant (CFE_LOSS=poisson, outputs *_all_poisson) against the main run (c1x_poisson*).

Runs on CPUs (CFE_THREADS, default 4) in a few minutes (--lite about five); --figure redraws the figure only; --tail,
--lite and --poisson run only that part and merge their macros into c1x_numbers.tex. With CFE_LOSS=poisson,
everything is computed from 35's Poisson variant and written with the prefix c1x_poisson (c1x_poisson_numbers.tex has
the same macro names); --poisson writes its comparison (c1x_poisson_compare.*) and macros to both macro files.
Input: 35's outputs and cached predictions; data/patex/PatEx_2022_ECOPAIR_csv.zip; data/processed/
       {all_sample,applications_2004_2018,patex_application_data}.parquet
Output: output/tables/c1x_*.csv, output/latex/c1x_tab_examples.tex, output/latex/c1x_numbers.tex (macros \\xCe...),
        output/figures/journal/fig_main_c1.{pdf,png}, output/c1x{,_tail,_lite}.md, output/c1x_poisson_compare.md
"""
import importlib.util
import os
import pathlib
import warnings
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
T = ROOT / "output/tables"
TEX = ROOT / "output/latex"
OUT = ROOT / "output"
PATEX = ROOT / "data/patex/PatEx_2022_ECOPAIR_csv.zip"
# CFE_LOSS=poisson: everything from 35's Poisson variant (outputs *_all_poisson); this script's outputs then carry the
# prefix c1x_poisson (same macro names, in c1x_poisson_numbers.tex)
LOSS = os.environ.get("CFE_LOSS", "poisson")
PSFX = "_poisson" if LOSS == "poisson" else ""
S35, PRE = f"_all{PSFX}", f"c1x{PSFX}"
CMP_CSV = "c1x_poisson_compare.csv"
UTIL = "^(0[7-9]|1[0-9])[0-9]{6}$"   # U.S. utility nonprovisional application series (07–19)
TC_NAMES = {"1600": "Biotechnology, organic chemistry", "1700": "Chemical and materials engineering",
            "2100": "Computer architecture, software, information security",
            "2400": "Networking, multiplexing, cable, security", "2600": "Communications",
            "2800": "Semiconductors, electrical and optical systems",
            "3600": "Transportation, construction, e-commerce, business methods",
            "3700": "Mechanical engineering, manufacturing, medical devices"}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace("-", "$-$")


def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def tabular(path: pathlib.Path, colspec: str, header: list, rows: list, groups: dict | None = None) -> None:
    """A booktabs tabular; groups: {row index: panel title}."""
    L = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if groups and i in groups:
            if i:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{len(header)}}}{{@{{}}l}}{{\textit{{{groups[i]}}}}} \\")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- 2. family check

def family_flags(apps: pd.Series) -> pd.DataFrame:
    """For each application in apps: whether a related U.S. application was granted (see the docstring), and the type
    of the granted children."""
    import duckdb
    import pyarrow as pa
    import pyarrow.csv as pacsv
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = zipfile.ZipFile(PATEX)

    def read(name, cols):
        with z.open(name) as h:
            return pacsv.read_csv(h, convert_options=pacsv.ConvertOptions(
                column_types={c: pa.string() for c in cols}, include_columns=cols))
    par = read("continuity_parents.csv", ["application_number", "parent_application_number", "continuation_type"])
    chi = read("continuity_children.csv", ["application_number", "child_application_number"])
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('CFE_THREADS', '4'))}")
    con.register("par", par)
    con.register("chi", chi)
    con.register("apps", pd.DataFrame(dict(application_number=apps.astype(str).to_numpy())))
    con.execute(f"""CREATE TABLE pat AS SELECT application_number, patent_number IS NOT NULL AS g
                    FROM '{P}/patex_application_data.parquet'""")
    # parent → child links between U.S. utility applications: typed in the parents table, also those only in the
    # children table (type unknown)
    con.execute(f"""CREATE TABLE e AS
      WITH a AS (SELECT parent_application_number p, application_number c, any_value(continuation_type) t FROM par
                 WHERE continuation_type IN ('CON', 'CIP', 'DIV')
                   AND regexp_full_match(parent_application_number, '{UTIL}')
                   AND regexp_full_match(application_number, '{UTIL}') GROUP BY 1, 2),
           b AS (SELECT DISTINCT application_number p, child_application_number c FROM chi
                 WHERE regexp_full_match(application_number, '{UTIL}')
                   AND regexp_full_match(child_application_number, '{UTIL}'))
      SELECT coalesce(a.p, b.p) p, coalesce(a.c, b.c) c, coalesce(a.t, 'other') t
      FROM a FULL JOIN b ON a.p = b.p AND a.c = b.c""")
    e = con.execute("""SELECT e.p, e.c, e.t, coalesce(gp.g, false) gp, coalesce(gc.g, false) gc FROM e
                       LEFT JOIN pat gp ON gp.application_number = e.p
                       LEFT JOIN pat gc ON gc.application_number = e.c""").df()
    # children granted, by type
    kids = e[e.gc & e.p.isin(apps)].groupby("p").t.agg(lambda s: set(s))
    out = pd.DataFrame(dict(application_number=apps.astype(str).to_numpy()))
    out["child"] = out.application_number.isin(kids.index)
    for t in ["CON", "CIP", "DIV"]:
        out[f"child_{t.lower()}"] = out.application_number.map(lambda a, t=t: t in kids.get(a, ()))
    out["parent"] = out.application_number.isin(e.c[e.gp])
    # continuity families: connected components of the links
    nodes = pd.Index(pd.unique(np.concatenate([e.p.to_numpy(), e.c.to_numpy()])))
    i, j = nodes.get_indexer(e.p), nodes.get_indexer(e.c)
    _, comp = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(nodes), len(nodes))),
                                   directed=False)
    g_node = pd.Series(False, index=nodes)
    g_node.loc[e.p[e.gp]] = True
    g_node.loc[e.c[e.gc]] = True
    comp_g = np.bincount(comp, weights=g_node.to_numpy(float))
    k = nodes.get_indexer(out.application_number)
    own = out.application_number.map(con.execute("SELECT * FROM pat WHERE application_number IN "
                                                 "(SELECT application_number FROM apps)").df().set_index(
        "application_number").g).fillna(False).to_numpy(bool)
    ck = comp[np.maximum(k, 0)]   # each application's family (k < 0: no continuity links)
    out["family"] = np.where(k >= 0, comp_g[ck] - own > 0, False)
    out["family_size"] = np.where(k >= 0, np.bincount(comp)[ck], 1)
    # applications claiming the same provisional or PCT application
    sib = con.execute(f"""
      WITH pr AS (SELECT DISTINCT application_number a, parent_application_number p FROM par
                  WHERE continuation_type IN ('PRO', 'NST', 'CON', 'CIP', 'DIV')
                    AND NOT regexp_full_match(parent_application_number, '{UTIL}')
                    AND regexp_full_match(application_number, '{UTIL}')),
           prg AS (SELECT pr.*, coalesce(pat.g, false)::INT g FROM pr LEFT JOIN pat ON pat.application_number = pr.a),
           ng AS (SELECT p, sum(g) ng FROM prg GROUP BY p)
      SELECT prg.a application_number, max(ng.ng - prg.g) > 0 sib FROM prg JOIN ng USING (p)
      WHERE prg.a IN (SELECT application_number FROM apps) GROUP BY 1""").df().set_index("application_number").sib
    out["priority_sibling"] = out.application_number.map(sib).fillna(False).astype(bool)
    out["any"] = out.family | out.priority_sibling | out.child | out.parent
    return out


def family(x: pd.DataFrame) -> tuple:
    """Shares of applications with a granted relative: rejected overall, by center, by score decile, top 1% / top 10
    by score (test and imputation sets); granted applications for comparison."""
    st = pd.read_parquet(P / "applications_2004_2018.parquet", columns=["application_number", "abandoned"])
    f = family_flags(x.application_number)
    z = x[["application_number", "tech_center", "part", "granted", "cf", "prior_art"]].merge(f, on="application_number")
    z = z.merge(st, on="application_number", how="left")
    cols = ["child", "child_con", "child_cip", "child_div", "family", "any"]

    def row(lab, t):
        r = dict(group=lab, n=len(t), abandoned=t.abandoned.mean())
        r.update({c: t[c].mean() for c in cols})
        r["mean_prior_art"] = t.prior_art.mean()
        return r
    rej, gr = z[z.granted == 0], z[z.granted == 1]
    ev = rej[rej.part.isin(["test", "impute"])].sort_values("cf", ascending=False)
    rows = [row("Rejected, all", rej), row("Granted, all (comparison)", gr)]
    rows += [row(f"Rejected, TC {tc}", t) for tc, t in rej.groupby(rej.tech_center.astype(str))]
    rows.append(row("Rejected, test + imputation", ev))
    dec = pd.qcut(ev.cf.rank(method="first"), 10, labels=False) + 1
    rows += [row(f"Rejected, test + imputation, score decile {k}", t) for k, t in ev.groupby(dec)]
    rows.append(row("Rejected, test + imputation, top 1% by score", ev.head(int(np.ceil(0.01 * len(ev))))))
    rows.append(row("Rejected, test + imputation, top 10 by score (Table A17)", ev.head(10)))
    tab = pd.DataFrame(rows)
    # does the score still rank the rejected applications without a granted relative?
    from scipy.stats import spearmanr
    rx = z[(z.granted == 0) & (z.part != "lockbox")]
    sp = []
    for lab, t in [("all rejected", rx), ("no granted relative", rx[~rx["any"]]), ("granted relative", rx[rx["any"]])]:
        sp.append(dict(group=lab, n=len(t), spearman_prior_art=spearmanr(t.cf, t.prior_art.fillna(0))[0],
                       mean_score=t.cf.mean(), mean_prior_art=t.prior_art.mean()))
    return tab, pd.DataFrame(sp), z[["application_number", "child", "family", "any"]]


# ---------------------------------------------------------------- 1. examples

EX_SEED, EX_TOP, EX_MIN_GRANTED = 20261101, 99.0, 20


def field_percentile(x: pd.DataFrame) -> tuple:
    """Percentile of each application's counterfactual score (cf) among the scores of granted applications of the
    same USPC class × filing year (share of them at or below it, as 35's percentile_among_granted); where a class ×
    year has fewer than EX_MIN_GRANTED granted applications, among those of the same center × filing year (flagged)."""
    cls = x.au_class.astype(str).str.split("_").str[2].to_numpy()
    key = pd.Series(cls + "_" + x.filing_year.astype(str).to_numpy())
    v, g = x.cf.to_numpy(), (x.granted == 1).to_numpy()
    n_g = pd.Series(g.astype(int)).groupby(key.to_numpy()).sum()
    pct = np.full(len(x), np.nan)
    fallback = key.map(n_g).to_numpy() < EX_MIN_GRANTED
    kk = key.to_numpy()
    for k in n_g.index[n_g >= EX_MIN_GRANTED]:
        s_ = kk == k
        ref = np.sort(v[s_ & g])
        pct[s_] = 100 * np.searchsorted(ref, v[s_], side="right") / len(ref)
    cell = (x.tech_center.astype(str) + "_" + x.filing_year.astype(str)).to_numpy()
    for c_ in np.unique(cell[fallback]):
        s_ = cell == c_
        ref = np.sort(v[s_ & g])
        pct[s_ & fallback] = 100 * np.searchsorted(ref, v[s_ & fallback], side="right") / len(ref)
    return pct, fallback, cls


def examples(x: pd.DataFrame, M, fam: pd.DataFrame, N: dict) -> tuple:
    """Table A18, a rule fixed before looking at outcomes: (1) rejected applications of the test and imputation sets
    without a granted relative (family check: continuation, CIP, divisional, the same provisional or PCT
    application); (2) their counterfactual score's percentile among granted applications of the same USPC class ×
    filing year (field_percentile); (3) within each center, those at or above the 99th percentile form the pool, and
    one is drawn at random (seed EX_SEED, centers in ascending order, pool in application-number order). Shown with
    their prior-art citations as a percentile among the rejected applications of the same center × filing year
    (mid-ranks, so the mean over all rejected applications is 50; raw counts are truncated for late filings).
    Representativeness: for each center's pool and pooled, and for comparison all rejected applications and the old
    rule (two highest raw scores per center), the mean prior-art percentile and the shares above 50 and 75."""
    pct, fb, cls = field_percentile(x)
    x = x.assign(field_pct=pct, fallback=fb, uspc=cls)
    rej = x[x.granted == 0].copy()
    pa = rej.prior_art.fillna(0)
    grp = (rej.tech_center.astype(str) + "_" + rej.filing_year.astype(str)).to_numpy()
    rej["pa_pct"] = 100 * (pa.groupby(grp).rank(method="average") - 0.5) / pa.groupby(grp).transform("size")
    ev = rej[rej.part.isin(["test", "impute"])].merge(fam[["application_number", "any"]], on="application_number",
                                                      how="left")
    ev["any"] = ev["any"].fillna(False).astype(bool)
    ev["tc"] = ev.tech_center.astype(str)
    clean = ev[~ev["any"]]
    pool = clean[clean.field_pct >= EX_TOP].sort_values(["tc", "application_number"])
    rng = np.random.default_rng(EX_SEED)
    drawn = pd.concat([t.iloc[[int(rng.integers(len(t)))]] for _, t in pool.groupby("tc", sort=True)])
    old = pd.concat([t.sort_values("cf", ascending=False).head(2) for _, t in ev.groupby("tc", sort=True)])
    titles = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "invention_title"])
    drawn = drawn.merge(titles, on="application_number", how="left")

    def stats(t, lab, tc="All"):
        return dict(group=lab, tech_center=tc, n=len(t), mean_pa_pct=t.pa_pct.mean(),
                    share_above_50=float((t.pa_pct > 50).mean()), share_above_75=float((t.pa_pct > 75).mean()),
                    share_fallback=float(t.fallback.mean()), share_granted_relative=float(t["any"].mean()),
                    mean_score=t.cf.mean())
    rep = [stats(t, "Pool: top 1% in class × year, no granted relative", tc) for tc, t in pool.groupby("tc")]
    rep += [stats(pool, "Pool: top 1% in class × year, no granted relative"),
            stats(ev[ev.field_pct >= EX_TOP], "Top 1% in class × year, granted relatives included"),
            stats(clean, "All rejected (test + imputation), no granted relative"),
            stats(ev, "All rejected (test + imputation)"),
            stats(old, "Old rule: two highest raw scores per center"),
            stats(drawn, "Drawn examples")]
    rep = pd.DataFrame(rep)
    # table
    def short(t):   # sentence case, cut at a word boundary
        t = str(t).strip().capitalize()
        return esc(t if len(t) <= 52 else t[:49].rsplit(" ", 1)[0] + "...")
    tex = []
    for r in drawn.itertuples():
        ti = short(r.invention_title)
        tex.append([r.tc, str(r.application_number).zfill(8), str(r.filing_year), r.uspc, ti, num(r.cf, 1),
                    num(r.field_pct, 1) + ("$^\\dagger$" if r.fallback else ""), num(r.pa_pct, 0)])
    tabular(TEX / f"{PRE}_tab_examples.tex", "llllp{5.0cm}rrr",
            ["TC", "Application", "Filed", "USPC", "Title", "Predicted citations", "Percentile, class $\\times$ year",
             "Prior-art percentile"], tex)
    # macros
    P_, A_, O_, D_ = (rep[rep.group == g_].query("tech_center == 'All'").iloc[0] for g_ in
                      ["Pool: top 1% in class × year, no granted relative",
                       "All rejected (test + imputation), no granted relative",
                       "Old rule: two highest raw scores per center", "Drawn examples"])
    for m, r in [("Pool", P_), ("All", A_), ("Old", O_), ("Draw", D_)]:
        N[f"xCeEx{m}PaMean"] = num(r.mean_pa_pct, 0)
        N[f"xCeEx{m}PaAboveFifty"] = num(100 * r.share_above_50, 0)
        N[f"xCeEx{m}PaAboveSeventyFive"] = num(100 * r.share_above_75, 0)
    N["xCeExPoolN"] = f"{len(pool):,}"
    pn = pool.groupby("tc").size()
    N["xCeExPoolMinN"], N["xCeExPoolMaxN"] = str(int(pn.min())), str(int(pn.max()))
    N["xCeExPoolFallback"] = num(100 * pool.fallback.mean(), 1)
    N["xCeExDrawFallback"] = str(int(drawn.fallback.sum()))
    N["xCeExDrawAboveFiftyN"] = str(int((drawn.pa_pct > 50).sum()))
    N["xCeExDrawAboveSeventyFiveN"] = str(int((drawn.pa_pct > 75).sum()))
    N["xCeExN"], N["xCeExNTopRel"] = str(len(drawn)), str(int(drawn["any"].sum()))
    N["xCeExOldN"], N["xCeExOldRel"] = str(len(old)), str(int(old["any"].sum()))
    N["xCeExSeed"] = str(EX_SEED)
    ts = drawn[drawn.tc == "3600"].iloc[0]   # the example kept in the text (TC 3600's draw)
    N["xCeExTsApp"], N["xCeExTsTitle"] = str(ts.application_number).zfill(8), short(ts.invention_title)
    N["xCeExTsYear"], N["xCeExTsUspc"] = str(ts.filing_year), str(ts.uspc)
    N["xCeExTsScore"], N["xCeExTsFieldPct"] = num(ts.cf, 1), num(ts.field_pct, 1)
    N["xCeExTsPaPct"], N["xCeExTsPaRaw"] = num(ts.pa_pct, 0), num(ts.prior_art, 0)
    cols = ["tc", "application_number", "filing_year", "uspc", "invention_title", "cf", "field_pct", "fallback",
            "pa_pct", "prior_art", "any"]
    return drawn[cols], rep, pool[["tc", "application_number", "filing_year", "uspc", "cf", "field_pct", "fallback",
                                   "pa_pct"]]


def examples_report(drawn: pd.DataFrame, rep: pd.DataFrame) -> None:
    L = [f"# C1 examples (Table A18; {'Poisson' if PSFX else 'squared-error'} scores; 43_c1_extra.py --examples)", "",
         "Rule fixed before looking at outcomes: rejected applications of the test and imputation sets without a "
         "granted relative (continuation, CIP, divisional, same provisional or PCT application); the counterfactual "
         "score's percentile among granted applications of the same USPC class × filing year (center × year where the "
         f"class × year has fewer than {EX_MIN_GRANTED} granted applications, flagged †); within each center, the "
         f"pool is those at or above the {EX_TOP:.0f}th percentile, and one is drawn at random (seed {EX_SEED}). "
         "Prior-art citations are shown as a percentile among the rejected applications of the same center × filing "
         "year (mid-ranks; 50 on average).", "",
         "| TC | Application | Filed | USPC | Title | Predicted citations | Percentile, class × year | "
         "Prior-art percentile | Prior-art citations |", "|---|---|---|---|---|---|---|---|---|"]
    for r in drawn.itertuples():
        L.append(f"| {r.tc} | {r.application_number} | {r.filing_year} | {r.uspc} | "
                 f"{str(r.invention_title).capitalize()[:70]} | {r.cf:.1f} | {r.field_pct:.1f}"
                 f"{' †' if r.fallback else ''} | {r.pa_pct:.0f} | {r.prior_art:.0f} |")
    L += ["", "Representativeness (prior-art percentile among rejected applications of the same center × year):", "",
          "| Group | TC | N | Mean prior-art percentile | Above 50th (%) | Above 75th (%) | Fallback to center (%) | "
          "With a granted relative (%) | Mean predicted citations |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rep.itertuples():
        L.append(f"| {r.group} | {r.tech_center} | {r.n:,} | {r.mean_pa_pct:.1f} | {100 * r.share_above_50:.0f} | "
                 f"{100 * r.share_above_75:.0f} | {100 * r.share_fallback:.0f} | "
                 f"{100 * r.share_granted_relative:.0f} | {r.mean_score:.2f} |")
    (OUT / f"{PRE}_examples.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- 3. transformed outcomes

def evaluated(x: pd.DataFrame, M) -> pd.DataFrame:
    """Table A10's evaluation sample: test and imputation sets, quintiles within cells as in 35."""
    e = x[x.part.isin(["test", "impute"])].copy()
    e["q"] = M.quintiles(e)
    return e[e.q.notna()]


def contraction_draws(M, e: pd.DataFrame, o: str, bad: dict, rng: np.random.Generator) -> tuple:
    """35's contraction(), also returning each rule's bootstrap draws of its change at the examiners' extra rejections
    (mean over quintiles 2–5), recorded from contraction()'s percentile calls and checked against the rows it returns;
    so rule − random has a paired interval."""
    rec, pc0 = [], M.pc

    def pc_rec(v, lo):
        rec.append(np.asarray(v, float).copy())
        return pc0(v, lo)
    M.pc = pc_rec
    try:
        a, b, _ = M.contraction(e, o, bad, rng)
    finally:
        M.pc = pc0
    draws = {}
    for i, r in enumerate(list(bad) + [M.RANDOM]):
        k = 30 + 30 * i + 24   # 30 calls for the quintile rows, then 6 per quintile row and rule; the mean row last
        row = next(x for x in b if x["rule"] == r and x["quintile"] == "2–5 (mean)")
        assert np.isclose(pc0(rec[k], 1), row["lo"]) and np.isclose(pc0(rec[k + 2], 1), row["diff_lo"])
        draws[r] = (row["change_at_examiner_rejections"], rec[k])
    return a, b, draws


def over_random(o: str, draws: dict, M, sgn: int = 1) -> list:
    """Rule − random in the change at the examiners' extra rejections (sgn = −1 for a bad outcome: positive = the
    rule removes more of it), with paired bootstrap intervals."""
    r0, rd = draws[M.RANDOM]
    rows = []
    for r, (v, dv) in draws.items():
        if r != M.RANDOM:
            d = sgn * (dv - rd)
            rows.append(dict(outcome=o, rule=r, over_random=sgn * (v - r0), over_random_lo=M.pc(d, 1),
                             over_random_hi=M.pc(d, 0)))
    return rows


def transformed(x: pd.DataFrame, M) -> tuple:
    g = (x.granted == 1).to_numpy()
    q99 = x[g].groupby(["tech_center", "filing_year"]).cites.quantile(0.99).rename("q99")
    x = x.join(q99, on=["tech_center", "filing_year"])
    x["log_cites"] = np.where(g, np.log1p(x.cites), np.nan)
    x["trim_cites"] = np.where(g, np.minimum(x.cites, x.q99), np.nan)
    info = dict(share_top_coded=float((x.cites[g] > x.q99[g]).mean()), mean_cites=float(x.cites[g].mean()),
                mean_trim=float(x.trim_cites[g].mean()), mean_log=float(x.log_cites[g].mean()),
                q99_median=float(q99.median()), q99_min=float(q99.min()), q99_max=float(q99.max()))
    e = evaluated(x, M)
    rng = np.random.default_rng(M.SEED + 43)
    qs, cs, orr = [], [], []
    for o in ["cites", "log_cites", "trim_cites"]:
        bad = {M.ALGO["cites"]: -e.ml_cites.to_numpy(), M.JUDGE: -e.ml_judge.to_numpy()}
        a, b, dr = contraction_draws(M, e, o, bad, rng)
        qs += a
        cs += b
        orr += over_random(o, dr, M)
        print(o, flush=True)
    return pd.DataFrame(qs), pd.DataFrame(cs), pd.DataFrame(orr), info


def table_v_random(x: pd.DataFrame, M) -> tuple:
    """Table V's rules against random rejection (paired intervals; own random stream), and how much of the
    examiners' change differs from random because of where (in which cells) they reject more: the examiners' change
    if, within each cell, their extra rejections were drawn at random from the most lenient quintile's grants."""
    e = evaluated(x, M)
    rng = np.random.default_rng(M.SEED + 44)
    orr, comp = [], []
    for o in ["cites", "uncited", "renew", "top10", "pa_granted"]:
        if o == "uncited":
            bad = {M.ALGO["uncited"]: e.ml_uncited.to_numpy(), M.JUDGE: -e.ml_judge.to_numpy()}
        else:
            bad = {M.ALGO["cites"]: -e.ml_cites.to_numpy(), M.JUDGE: -e.ml_judge.to_numpy()}
        if o != "cites":   # citations: transformed() has it
            _, _, dr = contraction_draws(M, e, o, bad, rng)
            orr += over_random(o, dr, M, -1 if o == "uncited" else 1)
        r = []
        for _, c in e.groupby(M.CELL, sort=True):
            y, g, q = np.nan_to_num(c[o].to_numpy(float)), c.granted.to_numpy(float), c.q.to_numpy(int)
            rel = np.array([g[q == k].mean() for k in range(1, 6)])
            out = np.array([y[q == k].mean() for k in range(1, 6)])
            five = (q == 5) & (g == 1)
            r.append((len(c), ((1 - rel[[3, 2, 1, 0]]) - (1 - rel[4])).mean(), (out[[3, 2, 1, 0]] - out[4]).mean(),
                      out[4], y[five].mean() if five.any() else np.nan))
        r = pd.DataFrame(r, columns=["n", "dd", "do", "base", "yg5"])
        ok = r.yg5.notna()
        comp.append(dict(outcome=o, examiners=(r.n * r.do).sum() / (r.n * r.base).sum(),
                         random_within_cells=-(r.n * r.dd * r.yg5)[ok].sum() / (r.n * r.base)[ok].sum(),
                         random_same_share_every_cell=-(r.n * r.dd).sum() / r.n.sum() * (r.n * r.yg5)[ok].sum()
                         / (r.n * r.base)[ok].sum()))
        print(o, flush=True)
    return pd.DataFrame(orr), pd.DataFrame(comp)


# ---------------------------------------------------------------- 6. tail diagnostics

DESIGNS = {"main": "Main design", "ten": "Cells with at least ten examiners",
           "fifteen": "Cells with at least fifteen examiners",
           "window": "Leniency from the same period (filing years t - 2 to t + 2)",
           "dock": "Applications docketed to an examiner only once"}


def with_top_coded(x: pd.DataFrame) -> pd.DataFrame:
    """trim_cites: citations top-coded at the 99th percentile of granted applications of the same center and filing
    year (as in transformed())."""
    g = (x.granted == 1).to_numpy()
    q99 = x[g].groupby(["tech_center", "filing_year"]).cites.quantile(0.99).rename("q99")
    x = x.join(q99, on=["tech_center", "filing_year"])
    x["trim_cites"] = np.where(g, np.minimum(x.cites, x.q99), np.nan)
    return x


def designs(x: pd.DataFrame, M) -> dict:
    """Table III's evaluation sample (test and imputation sets) in the main design and in each robustness design of
    35's robust(), with the quintiles as there."""
    ev = x[x.part.isin(["test", "impute"])]
    out = {}
    for key in DESIGNS:
        if key == "dock":
            e = ev[ev.docketed_once.fillna(False).astype(bool)].copy()
        else:
            e = ev.copy()
        if key == "window":
            e["lenient_grant"] = e.lenient_win
        e["q"] = M.quintiles(e, {"ten": 10, "fifteen": 15}.get(key, M.MIN_EXAMINERS))
        out[key] = e[e.q.notna()]
    return out


def contraction_paired(e: pd.DataFrame, outcome: str, score: np.ndarray, rng: np.random.Generator, M) -> dict:
    """Table III's mean over quintiles 2–5 for one rule (score: rejected first = largest), computed as in 35's
    contraction(), against the examiners and against random rejection. The random rule is its expected curve (each
    extra rejection in a cell removes the mean outcome of the most lenient quintile's grants there), so that
    rule − examiners = (rule − random: the selection gain) + (random − examiners) is free of one random ordering's
    noise. Paired bootstrap of art units (B draws). Returns {quantity: (estimate, lo, hi)}."""
    PP = M.PP
    cells = list(e.groupby(M.CELL, sort=True))
    nc = len(cells)
    n_cell = np.array([len(c) for _, c in cells], float)
    au = np.array([c.art_unit.iloc[0] for _, c in cells])
    dd, do, base = np.full((nc, 4), np.nan), np.full((nc, 4), np.nan), np.zeros(nc)
    red_a, red_r = np.full((nc, len(PP)), np.nan), np.full((nc, len(PP)), np.nan)
    tb = rng.random(len(e))
    pos = pd.Series(np.arange(len(e)), index=e.index)
    for i, (_, c) in enumerate(cells):
        y = np.nan_to_num(c[outcome].to_numpy(float))
        g = c.granted.to_numpy(float)
        q = c.q.to_numpy(int)
        rel = np.array([g[q == k].mean() for k in range(1, 6)])
        out = np.array([y[q == k].mean() for k in range(1, 6)])
        dd[i] = (1 - rel[[3, 2, 1, 0]]) - (1 - rel[4])
        do[i] = out[[3, 2, 1, 0]] - out[4]
        base[i] = out[4]
        five = q == 5
        k = five.sum() / 100 * PP
        gi = np.where(five & (g == 1))[0]
        p = pos[c.index].to_numpy()
        ii = gi[np.lexsort((tb[p][gi], -score[p][gi]))]
        ok = k <= len(ii) + 1e-9
        red_a[i] = np.where(ok, np.interp(k, np.arange(len(ii) + 1), np.concatenate([[0], np.cumsum(y[ii])])),
                            np.nan) / five.sum()
        red_r[i] = np.where(ok, k * y[gi].sum() / max(len(gi), 1), np.nan) / five.sum()
    u, inv = np.unique(au, return_inverse=True)
    mult = np.ones((M.B + 1, nc))
    for b in range(1, M.B + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
    W = mult * n_cell

    def wmean(v, w):
        ok = ~np.isnan(v)
        return (np.where(ok, v, 0) * w).sum(0) / np.maximum((ok * w).sum(0), 1e-12)
    X = np.array([wmean(dd, w[:, None]) for w in W])
    Y = np.array([wmean(do, w[:, None]) / (w @ base) * w.sum() for w in W])

    def at(red):
        cv = np.array([-(np.nan_to_num(red) * w[:, None]).sum(0) /
                       np.maximum(((~np.isnan(red)) * (w * base)[:, None]).sum(0), 1e-12) for w in W])
        a = np.full((M.B + 1, 4), np.nan)
        for b in range(M.B + 1):
            for j in range(4):
                if X[b, j] <= PP[-1] / 100:
                    a[b, j] = np.interp(X[b, j], PP / 100, cv[b])
        return a.mean(1)
    A, R, E = at(red_a), at(red_r), Y.mean(1)
    res = dict(examiners=E, rule=A, random=R, diff=A - E, selection=A - R, random_gap=R - E,
               extra_rejections=X.mean(1))
    return {k: (v[0], M.pc(v[1:], 1), M.pc(v[1:], 0)) for k, v in res.items()}


def staple_cells(e: pd.DataFrame, M) -> dict:
    """Art unit 3721 × USPC class 227 (surgical staplers) in the main design: cells, application share, and their
    share of the estimator's weights (the examiners' relative change is sum_c n_c ΔY_c / sum_c n_c Y5_c, cells
    weighted by size n_c; Y5_c the most lenient quintile's citations per application)."""
    rows = []
    for cell, c in e.groupby(M.CELL, sort=True):
        y = np.nan_to_num(c.cites.to_numpy(float))
        q = c.q.to_numpy(int)
        out = np.array([y[q == k].mean() for k in range(1, 6)])
        rows.append(dict(cell=cell, au=str(c.art_unit.iloc[0]), n=len(c), base=out[4],
                         do=(out[[3, 2, 1, 0]] - out[4]).mean()))
    t = pd.DataFrame(rows)
    s = (t.au == "3721") & t.cell.str.endswith("_227")
    return dict(cells=int(s.sum()), applications=int(t.n[s].sum()), application_share=t.n[s].sum() / t.n.sum(),
                weight_share=(t.n * t.base)[s].sum() / (t.n * t.base).sum(),
                change_share=(t.n * t.do)[s].sum() / (t.n * t.do).sum(),
                cites_lenient_staple=(t.n * t.base)[s].sum() / t.n[s].sum(),
                cites_lenient_other=(t.n * t.base)[~s].sum() / t.n[~s].sum())


def tail(x: pd.DataFrame, M, N: dict) -> tuple:
    """Where Table III's differences between designs come from: algorithm − examiners with citations top-coded, in
    the main and each robustness design; the main design without art unit 3721 and without TC 3700; the split into
    selection gain and random − examiners; the weight of the surgical-stapler cells. Own random stream."""
    x = with_top_coded(x)
    rng = np.random.default_rng(M.SEED + 45)
    D = designs(x, M)
    D["no_au"] = D["main"][D["main"].art_unit.astype(str) != "3721"]
    D["no_tc"] = D["main"][D["main"].tech_center.astype(str) != "3700"]
    lab = dict(DESIGNS, no_au="Main design without art unit 3721", no_tc="Main design without TC 3700")
    rows = []
    for key, e in D.items():
        for o in ["cites", "trim_cites"]:
            r = contraction_paired(e, o, -e.ml_cites.to_numpy(), rng, M)
            row = dict(design=key, label=lab[key], outcome=o, n=len(e), cells=e[M.CELL].nunique())
            for k, (v, lo, hi) in r.items():
                row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
            rows.append(row)
            print(key, o, f"diff {100 * r['diff'][0]:.1f}", flush=True)
    t = pd.DataFrame(rows)
    st = staple_cells(D["main"], M)
    mac = dict(main="Main", ten="Ten", fifteen="Fifteen", window="Win", dock="Dock", no_au="NoAu", no_tc="NoTc")
    ti = t.set_index(["design", "outcome"])

    def ci(r, k):
        return f"[{num(100 * r[f'{k}_lo'])}, {num(100 * r[f'{k}_hi'])}]"
    for key, m in mac.items():
        raw, tr = ti.loc[(key, "cites")], ti.loc[(key, "trim_cites")]
        if key in ("no_au", "no_tc"):
            N[f"xCe{m}Diff"], N[f"xCe{m}DiffCI"] = num(100 * raw["diff"]), ci(raw, "diff")
            N[f"xCe{m}Ex"] = num(-100 * raw.examiners)
            N[f"xCe{m}Sel"], N[f"xCe{m}SelCI"] = num(100 * raw.selection), ci(raw, "selection")
            N[f"xCe{m}TrimDiff"], N[f"xCe{m}TrimDiffCI"] = num(100 * tr["diff"]), ci(tr, "diff")
            continue
        N[f"xCeTrim{m}"], N[f"xCeTrim{m}CI"] = num(100 * tr["diff"]), ci(tr, "diff")
        N[f"xCeDiff{m}"], N[f"xCeDiff{m}CI"] = num(100 * raw["diff"]), ci(raw, "diff")
        N[f"xCeSel{m}"], N[f"xCeSel{m}CI"] = num(100 * raw.selection), ci(raw, "selection")
        N[f"xCeRndGap{m}"], N[f"xCeRndGap{m}CI"] = num(100 * raw.random_gap), ci(raw, "random_gap")
        N[f"xCeSelTrim{m}"], N[f"xCeSelTrim{m}CI"] = num(100 * tr.selection), ci(tr, "selection")
        N[f"xCeRndGapTrim{m}"], N[f"xCeRndGapTrim{m}CI"] = num(100 * tr.random_gap), ci(tr, "random_gap")
    N["xCeStapleCells"] = str(st["cells"])
    N["xCeStapleApps"] = num(100 * st["application_share"], 2)
    N["xCeStapleWeight"] = num(100 * st["weight_share"], 1)
    N["xCeStapleCitesLenient"] = num(st["cites_lenient_staple"], 1)
    N["xCeOtherCitesLenient"] = num(st["cites_lenient_other"], 2)
    return t, pd.DataFrame([st])


def write_macros(N: dict, f: pathlib.Path | None = None) -> None:
    """Merge N into output/latex/c1x_numbers.tex, or c1x_poisson_numbers.tex with CFE_LOSS=poisson (existing macros
    keep their place; new ones are appended)."""
    import re
    f = f or TEX / f"{PRE}_numbers.tex"
    old = {}
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.match(r"\\newcommand\{\\(xCe[A-Za-z]+)\}\{(.*)\}$", line)
            if m:
                old[m.group(1)] = m.group(2)
    old.update(N)
    f.write_text("% C1 follow-ups (43_c1_extra.py)\n" +
                 "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in old.items()) + "\n")


def tail_report(t: pd.DataFrame, st: pd.DataFrame) -> None:
    def c(r, k):
        return f"{100 * r[k]:.1f} [{100 * r[k + '_lo']:.1f}, {100 * r[k + '_hi']:.1f}]".replace("-", "−")
    s = st.iloc[0]
    L = ["# C1 tail diagnostics (43_c1_extra.py --tail)", "",
         "Table III (algorithm trained on citations against the examiners; test and imputation sets; mean over the "
         "stricter quintiles 2–5), for raw and top-coded citations (99th percentile of granted applications within "
         "center × filing year), in the main design and each robustness design of 35's robust(). The random rule is "
         "its expected curve, so rule − examiners = selection gain (rule − random) + (random − examiners) exactly. "
         "Changes are falls in citations per application (%), differences in pp; 95% intervals from 500 paired "
         "bootstrap draws of art units (own random stream).", "",
         "| Design | Outcome | Applications | Cells | Examiners' change (%) | Algorithm − examiners (pp) | "
         "Algorithm − random (pp) | Random − examiners (pp) |", "|---|---|---|---|---|---|---|---|"]
    for r in t.to_dict("records"):
        L.append(f"| {r['label']} | {'top-coded' if r['outcome'] == 'trim_cites' else 'raw'} | {r['n']:,} | "
                 f"{r['cells']:,} | {c(r, 'examiners')} | {c(r, 'diff')} | {c(r, 'selection')} | "
                 f"{c(r, 'random_gap')} |")
    L += ["", f"Surgical-stapler cells (art unit 3721 × USPC 227) in the main design: {int(s.cells)} cells, "
              f"{int(s.applications)} applications ({100 * s.application_share:.2f}% of the evaluated applications), "
              f"{100 * s.weight_share:.1f}% of the estimator's weight (the denominator of the examiners' relative "
              f"change: the most lenient quintile's citations per application, cell-size weighted; "
              f"{s.cites_lenient_staple:.1f} citations per application there against {s.cites_lenient_other:.2f} "
              f"elsewhere). Their share of the numerator (the summed change) is {100 * s.change_share:.1f}%: there the "
              "stricter quintiles' applications are the more cited ones, which pulls the examiners' fall down.", ""]
    (OUT / f"{PRE}_tail.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- 7. lite robustness (citation algorithm refitted)

TIME, NOENT = ("Algorithm trained on 2010–2012 filings", "Algorithm without entity size and number of inventors")
REFIT = "Algorithm, refitted (same pipeline, all inputs)"


def lite(M, N: dict) -> tuple:
    """Refit only the citation algorithm (training model, the main run's hyperparameters) and rerun Table III:
    (i) trained on training-set applications filed 2010–2012 (principal components also fitted on them), evaluated on
    the 2013–2014 applications of the test and imputation sets, against the main model on the same applications;
    (ii) without entity size and number of inventors (text, center, filing year), in the main design. Same cells and
    quintiles as Table III; 500 bootstrap draws of art units, own random stream. The main model refitted through the
    same pipeline is a third rule, a check: with CFE_THREADS=16 and 16 BLAS threads, as in the main run, it reproduces
    the cached predictions exactly; with other thread counts the principal components and the trees differ slightly
    (correlation 0.98 with 4 threads), and this rule shows how much that alone moves the results."""
    from sklearn.decomposition import PCA
    x = with_top_coded(M.cached(M.load_all(), "all"))
    emb = np.load(P / "emb/all_s0.npy", mmap_mode="r")
    E = np.asarray(emb, dtype=np.float32)
    assert len(E) == len(x)
    prm = pd.read_csv(T / f"c1_kllm{S35}_models.csv").set_index("model").loc["training: cites"]
    prm = dict(depth=int(prm.depth), rate=float(prm.rate), trees=int(prm.trees))
    tr = (x.part == "train").to_numpy()
    g = (x.granted == 1).to_numpy()
    yr = x.filing_year.to_numpy()
    y = np.nan_to_num(x.cites.to_numpy(float))
    tcd = pd.get_dummies(x.tech_center).to_numpy(np.float32)
    other = x[["small_entity", "n_inventors"]].to_numpy(np.float32)

    def design(Z, keep):   # as in 35's run_tc: components, entity size, inventors, center dummies, filing year
        parts = [Z] + ([other] if keep else []) + [tcd, x[["filing_year"]].to_numpy(np.float32)]
        return np.hstack(parts)
    pca = PCA(M.N_PC, random_state=M.SEED).fit(E[tr])
    Z = pca.transform(E)
    X = design(Z, True)
    f, _, _ = M.fit_gbm(X[tr & g], y[tr & g], "gaussian", M.SEED, prm)   # the main model, refitted: a check
    refit = f(X)
    x["ml_refit"] = refit
    fit = [dict(check="refitted main model vs cached predictions (all applications)",
                corr=float(np.corrcoef(refit, x.ml_cites)[0, 1]), max_abs_diff=float(np.abs(refit - x.ml_cites).max()))]
    print(fit[0], flush=True)
    m12 = tr & (yr <= 2012)
    pca12 = PCA(M.N_PC, random_state=M.SEED).fit(E[m12])
    X12 = design(pca12.transform(E), True)
    f, _, _ = M.fit_gbm(X12[m12 & g], y[m12 & g], "gaussian", M.SEED, prm)
    x["ml_time"] = f(X12)
    f, _, _ = M.fit_gbm(design(Z, False)[tr & g], y[tr & g], "gaussian", M.SEED, prm)
    x["ml_noent"] = f(design(Z, False))
    del E, Z
    te = x[(x.part == "test") & g]
    for col, lab, sub in [("ml_cites", "main model", te.filing_year >= 2013), ("ml_time", TIME, te.filing_year >= 2013),
                          ("ml_cites", "main model", te.filing_year > 0), ("ml_noent", NOENT, te.filing_year > 0)]:
        t = te[sub]
        yrs = "2013–2014" if sub.mean() < 1 else "2010–2014"
        fit.append(dict(check=f"{lab}: granted test applications filed {yrs}",
                        corr=float(np.corrcoef(t[col], t.cites)[0, 1]),
                        spearman=float(t[[col, "cites"]].corr("spearman").iloc[0, 1]), n=len(t)))
    e = evaluated(x, M)
    rng = np.random.default_rng(M.SEED + 46)
    rows_q, rows_c = [], []
    for key, sub, rule in [("time", e[e.filing_year >= 2013], TIME), ("noent", e, NOENT)]:
        col = "ml_time" if key == "time" else "ml_noent"
        for o in ["cites", "trim_cites"]:
            bad = {rule: -sub[col].to_numpy(), M.ALGO["cites"]: -sub.ml_cites.to_numpy(),
                   REFIT: -sub.ml_refit.to_numpy()}
            a, b, _ = M.contraction(sub, o, bad, rng)
            rows_q += [dict(r, design=key, n=len(sub), cells=sub[M.CELL].nunique()) for r in a]
            rows_c += [dict(r, design=key) for r in b]
            print(key, o, flush=True)
    q, c = pd.DataFrame(rows_q), pd.DataFrame(rows_c)
    cm = c[c.quintile.astype(str) == "2–5 (mean)"].set_index(["design", "outcome", "rule"])
    qm = q[q.quintile.astype(str) == "2–5 (mean)"].set_index(["design", "outcome"])
    for key, mac, rule in [("time", "Time", TIME), ("noent", "NoEntity", NOENT)]:
        for o, oo in [("cites", ""), ("trim_cites", "Trim")]:
            for r_, rm in [(rule, ""), (M.ALGO["cites"], "Main"), (REFIT, "Refit")]:
                z = cm.loc[(key, o, r_)]
                N[f"xCe{mac}{oo}{rm}Diff"] = num(100 * z["diff"])
                N[f"xCe{mac}{oo}{rm}DiffCI"] = f"[{num(100 * z.diff_lo)}, {num(100 * z.diff_hi)}]"
            N[f"xCe{mac}{oo}Ex"] = num(-100 * qm.loc[(key, o), "change"])
    ft = pd.DataFrame(fit)
    N["xCeTimeCorr"], N["xCeTimeMainCorr"] = num(ft["corr"].iloc[2], 2), num(ft["corr"].iloc[1], 2)
    N["xCeNoEntityCorr"], N["xCeNoEntityMainCorr"] = num(ft["corr"].iloc[4], 2), num(ft["corr"].iloc[3], 2)
    N["xCeTimeN"] = f"{int(qm.loc[('time', 'cites'), 'n']):,}"
    return q, c, ft


def lite_report(q: pd.DataFrame, c: pd.DataFrame, ft: pd.DataFrame) -> None:
    def ci(v, lo, hi):
        return f"{100 * v:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]".replace("-", "−")
    L = ["# C1 lite robustness (43_c1_extra.py --lite)", "",
         "Only the citation algorithm (training model) is refitted, with the main run's hyperparameters; Table III "
         "(test and imputation sets, same cells and quintiles; mean over the stricter quintiles 2–5; 500 bootstrap "
         "draws of art units, own random stream). (i) Temporal: trained on training-set applications filed 2010–2012 "
         "(principal components fitted on them too), evaluated on the 2013–2014 applications, with the main model on "
         "the same applications. (ii) Without entity size and number of inventors (text, center, filing year).", "",
         "| Design | Outcome | Applications | Examiners' change (%) | Rule | Rule's change (%) | "
         "Rule − examiners (pp) |",
         "|---|---|---|---|---|---|---|"]
    qm = q[q.quintile.astype(str) == "2–5 (mean)"].set_index(["design", "outcome"])
    for r in c[c.quintile.astype(str) == "2–5 (mean)"].itertuples():
        z = qm.loc[(r.design, r.outcome)]
        L.append(f"| {'2013–2014 filings' if r.design == 'time' else 'Main design'} | "
                 f"{'top-coded' if r.outcome == 'trim_cites' else 'raw'} | {int(z.n):,} | "
                 f"{ci(z.change, z.change_lo, z.change_hi)} | {r.rule} | "
                 f"{ci(r.change_at_examiner_rejections, r.lo, r.hi)} | {ci(r.diff, r.diff_lo, r.diff_hi)} |")
    L += ["", "Fit (Pearson and Spearman correlation of predicted and observed citations):", "",
          "| Check | Pearson | Spearman | N | Max abs. difference |", "|---|---|---|---|---|"]
    for r in ft.to_dict("records"):
        n = "" if pd.isna(r.get("n", np.nan)) else f"{int(r['n']):,}"
        sp = "" if pd.isna(r.get("spearman", np.nan)) else f"{r['spearman']:.3f}"
        md = "" if pd.isna(r.get("max_abs_diff", np.nan)) else f"{r['max_abs_diff']:.4f}"
        L.append(f"| {r['check']} | {r['corr']:.3f} | {sp} | {n} | {md} |")
    (OUT / f"{PRE}_lite.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- 8. Poisson variant against the main run

def get_(w: pd.DataFrame, q: str, v: str = "value", ver: str = "Poisson") -> float:
    return float(w.loc[q, (v, ver)])


def poisson_compare(M, N: dict) -> pd.DataFrame:
    """35's main numbers from the main run (squared-error loss) and from its Poisson variant (CFE_LOSS=poisson,
    outputs *_all_poisson), side by side; and the share of negative predictions."""
    ALG = M.ALGO["cites"]
    rows = []

    def add(q, ver, v, lo=np.nan, hi=np.nan):
        rows.append(dict(quantity=q, version=ver, value=v, lo=lo, hi=hi))
    for sfx, ver in [("_all", "squared error"), ("_all_poisson", "Poisson")]:
        rd = lambda k: pd.read_csv(T / f"c1_kllm{sfx}_{k}.csv")  # noqa: E731
        c = rd("contraction")
        c = c[c.quintile.astype(str) == "2–5 (mean)"]
        q = rd("quintiles")
        q = q[q.quintile.astype(str) == "2–5 (mean)"]
        add("Table III: examiners' change, citations (%)", ver,
            *(100 * q[(q["sample"] == "test + imputation") & (q.outcome == "cites")][["change", "change_lo",
                                                                                     "change_hi"]].iloc[0]))
        for smp in ["test + imputation", "lock box", "balanced cells"]:
            for rule in [ALG, M.LENIENT, M.PA_ALL, M.JUDGE, M.RANDOM]:
                z = c[(c["sample"] == smp) & (c.outcome == "cites") & (c.rule == rule)]
                if len(z):
                    z = z.iloc[0]
                    add(f"Table III ({smp}): {rule} − examiners (pp)", ver, 100 * z["diff"], 100 * z.diff_lo,
                        100 * z.diff_hi)
        for o in ["renew", "top10", "pa_granted"]:
            z = c[(c["sample"] == "test + imputation") & (c.outcome == o) & (c.rule == ALG)].iloc[0]
            add(f"Table V, {o}: algorithm − examiners (pp)", ver, 100 * z["diff"], 100 * z.diff_lo, 100 * z.diff_hi)
        t4 = rd("table4")
        t4["a"] = t4.alpha.astype(str)
        for o, a in [("cites", "1.0"), ("cites", "1.5"), ("cites", "2.0"), ("cites", "inf"), ("pa_obs", "1.0"),
                     ("pa_obs", "observed")]:
            z = t4[(t4.outcome == o) & (t4.rule == ALG) & (t4.a == a)].iloc[0]
            add(f"Table IV, {o}, α = {a}: algorithm's change (%)", ver, 100 * z.outcome_change,
                100 * z.outcome_change_lo, 100 * z.outcome_change_hi)
        lv = rd("levels")
        lv = lv[lv.quintile.astype(str) == "2–5 (pooled)"].set_index("series")
        for sname in ["observed citations", M.CF, f"{M.CF} − observed citations", M.CF_PA,
                      f"{M.CF_PA} − observed citations"]:
            z = lv.loc[sname]
            add(f"Levels at the margin (pooled): {sname}", ver, z.value, z.lo, z.hi)
        v = rd("cf_valid").set_index("score")
        for sc in [M.CF, M.CF_PA]:
            for k in ["corr_cites_granted", "spearman_cites_granted", "spearman_prior_art_granted",
                      "spearman_prior_art_rejected", "mean_granted", "mean_rejected",
                      "share_rejected_above_granted_median", "share_rejected_top_quarter"]:
                add(f"Score validation, {sc}: {k}", ver, v.loc[sc, k])
        al = rd("alpha")
        al = al[al.decile.astype(str) == "all"].set_index("prediction")
        add("α on prior-art citations, counterfactual score", ver, al.loc["counterfactual citation score", "alpha"])
        ft = rd("fit").iloc[0]
        for k in ["corr_cites", "spearman_cites", "auc_grant_by_predicted_cites", "lowest1_grant_rate",
                  "lowest1_cites"]:
            add(f"Figure II / fit: {k}", ver, ft[k])
        f2 = rd("fig2")
        add("Figure II: predicted citations, lowest percentile bin", ver, f2.predicted.min())
        add("Figure II: percentile bins with negative mean prediction", ver, float((f2.predicted < 0).sum()))
        pr = pd.read_parquet(P / f"c1_preds{sfx}_all.parquet", columns=["part", "ml_cites", "imp_cites"])
        cf = np.where(pr.part == "impute", pr.ml_cites, pr.imp_cites)
        add("Share of negative predictions: training model, all applications", ver, float((pr.ml_cites < 0).mean()))
        add("Share of negative predictions: imputation model, all applications", ver,
            float((pr.imp_cites < 0).mean()))
        add("Share of negative predictions: counterfactual score, all applications", ver, float((cf < 0).mean()))
        sc_ = pd.read_csv(T / f"c1_counterfactual_scores{sfx}.csv", usecols=["cf_cites"])
        add("Share of negative predictions: counterfactual score, rejected applications", ver,
            float((sc_.cf_cites < 0).mean()))
    # Table III with the random rule's expected curve (selection gain) and with top-coded citations (as in tail())
    d = with_top_coded(M.load_all())
    rng = np.random.default_rng(M.SEED + 47)
    for sfx, ver in [("_all", "squared error"), ("_all_poisson", "Poisson")]:
        pr = pd.read_parquet(P / f"c1_preds{sfx}_all.parquet", columns=["application_number", "part", "ml_cites"])
        e = evaluated(d.merge(pr, on="application_number"), M)
        for o, lab in [("cites", "raw"), ("trim_cites", "top-coded")]:
            r = contraction_paired(e, o, -e.ml_cites.to_numpy(), rng, M)
            add(f"Table III, {lab} citations: algorithm − examiners (pp)", ver, *(100 * np.array(r["diff"])))
            add(f"Table III, {lab} citations: algorithm − random, expected curve (pp)", ver,
                *(100 * np.array(r["selection"])))
    t = pd.DataFrame(rows)
    w = t.pivot_table(index="quantity", columns="version", values=["value", "lo", "hi"], sort=False)
    sq, po = "squared error", "Poisson"
    for lab, mac in [("top-coded citations: algorithm − examiners", "PoisTrimDiff"),
                     ("raw citations: algorithm − random, expected curve", "PoisSel"),
                     ("top-coded citations: algorithm − random, expected curve", "PoisTrimSel")]:
        q_ = f"Table III, {lab} (pp)"
        N[f"xCe{mac}"], N[f"xCe{mac}CI"] = num(get_(w, q_)), f"[{num(get_(w, q_, 'lo'))}, {num(get_(w, q_, 'hi'))}]"

    def get(q, v="value", ver=po):
        return float(w.loc[q, (v, ver)])
    tab3 = f"Table III (test + imputation): {ALG} − examiners (pp)"
    N["xCePoisDiff"], N["xCePoisDiffCI"] = num(get(tab3)), f"[{num(get(tab3, 'lo'))}, {num(get(tab3, 'hi'))}]"
    r1 = f"Table IV, cites, α = 1.0: algorithm's change (%)"
    N["xCePoisRerank"] = num(get(r1))
    N["xCePoisRerankCI"] = f"[{num(get(r1, 'lo'))}, {num(get(r1, 'hi'))}]"
    lvq = f"Levels at the margin (pooled): {M.CF}"
    N["xCePoisLevelScore"], N["xCePoisLevelObs"] = num(get(lvq), 2), num(
        get("Levels at the margin (pooled): observed citations"), 2)
    N["xCePoisCorr"] = num(get(f"Score validation, {M.CF}: corr_cites_granted"), 2)
    N["xCePoisSpRej"] = num(get(f"Score validation, {M.CF}: spearman_prior_art_rejected"), 2)
    N["xCePoisSpGr"] = num(get(f"Score validation, {M.CF}: spearman_prior_art_granted"), 2)
    N["xCeNegTrain"] = num(100 * get("Share of negative predictions: training model, all applications", ver=sq), 1)
    N["xCeNegScoreRej"] = num(100 * get("Share of negative predictions: counterfactual score, rejected applications",
                                        ver=sq), 1)
    return t


# ---------------------------------------------------------------- 9. C1's permutation test (46), as macros

def perm_macros(N: dict) -> None:
    """46's C1 permutation test by center (tx_c1_permutation.csv; with CFE_LOSS=poisson the Poisson index,
    tx_c1_permutation_poisson.csv) as macros with the same names for both losses."""
    f = T / f"tx_c1_permutation{PSFX}.csv"
    if not f.exists():
        return
    s = pd.read_csv(f, dtype={"sample": str}).set_index("sample")
    a, bc = s.loc["All centers"], s.drop(index="All centers")
    N["xCePermShare"], N["xCePermShareTen"] = num(100 * a.p_below_05), num(100 * a.p_below_10)
    N["xCePermCells"] = f"{int(a.cells_tested):,}"
    N["xCePermThreeSix"] = num(100 * s.loc["3600"].p_below_05)
    N["xCePermMax"], N["xCePermMaxTC"] = num(100 * bc.p_below_05.max()), str(bc.p_below_05.idxmax())
    N["xCePermMin"], N["xCePermMinTC"] = num(100 * bc.p_below_05.min()), str(bc.p_below_05.idxmin())


# ---------------------------------------------------------------- 10. for whom (Kleinberg et al.'s Tables VI–VII)

def fw_rules(M) -> dict:
    return {"Examiners": "Examiners", M.ALGO["cites"]: "Algorithm (trained on citations)",
            M.JUDGE: "Predicted examiner (trained on grants)",
            M.EQUAL: "Algorithm, equal grant rates for women and men"}


def rerank_fw(te: pd.DataFrame, M, rng: np.random.Generator) -> tuple:
    """Table VII on the test set, with the predicted examiner added: all applications reranked at the examiners'
    grant rate (rejected applications' citations: the imputation model's prediction, α = 1). For each rule: the
    change in citations per application against the examiners and the female − male grant rate (pooled), with
    bootstrap draws of art units (as 35's rerank_gender); and the decisions at the estimate."""
    g = te.granted.to_numpy(float)
    f = te.female.to_numpy(float) == 1
    y = np.where(g == 1, np.nan_to_num(te.cites.to_numpy(float)), te.imp_cites.to_numpy())
    mult = M.boot_mult(te.art_unit.to_numpy(), rng)
    tb = rng.random(len(te))
    oa, oj = np.lexsort((tb, -te.ml_cites.to_numpy())), np.lexsort((tb, -te.ml_judge.to_numpy()))
    ones = np.ones(len(te), bool)

    def top(order, m_, mask, n):
        o = order[mask[order]]
        a = np.zeros(len(te))
        a[o[np.cumsum(m_[o]) <= n + 1e-9]] = 1
        return a
    rules = fw_rules(M)
    res = {r: [] for r in rules}
    dec = {}
    for b, m_ in enumerate(mult):
        N, Nf, Nm = m_.sum(), m_[f].sum(), m_[~f].sum()
        G = (m_ * g).sum()
        Yj = (m_ * g * y).sum() / N
        sel = {"Examiners": g, M.ALGO["cites"]: top(oa, m_, ones, G), M.JUDGE: top(oj, m_, ones, G),
               M.EQUAL: top(oa, m_, f, G / N * Nf) + top(oa, m_, ~f, G / N * Nm)}
        for r, a in sel.items():
            res[r].append(((m_ * a * y).sum() / N / Yj - 1, (m_ * a * f).sum() / Nf - (m_ * a * ~f).sum() / Nm))
            if b == 0:
                dec[r] = a
    out = []
    for r, v in res.items():
        v = np.array(v)
        out.append(dict(rule=r, rerank_change=v[0, 0], rerank_change_lo=M.pc(v[1:, 0], 1),
                        rerank_change_hi=M.pc(v[1:, 0], 0), gap_pooled=v[0, 1], gap_pooled_lo=M.pc(v[1:, 1], 1),
                        gap_pooled_hi=M.pc(v[1:, 1], 0)))
    return pd.DataFrame(out), dec


def gap_within(te: pd.DataFrame, dec: dict) -> pd.DataFrame:
    """Female − male grant rate within art unit × filing year (decision ~ female | au_year), standard errors clustered
    by examiner (as C3)."""
    import pyfixest as pf
    rows = []
    for r, a in dec.items():
        z = pd.DataFrame(dict(dec=a, female=te.female.to_numpy(float), au_year=te.au_year.to_numpy(),
                              examiner_id=te.examiner_id.astype(str).to_numpy()))
        m = pf.feols("dec ~ female | au_year", data=z, vcov={"CRV1": "examiner_id"})
        b, se = float(m.coef()["female"]), float(m.se()["female"])
        rows.append(dict(rule=r, gap_within=b, gap_within_se=se, gap_within_lo=b - 1.96 * se,
                         gap_within_hi=b + 1.96 * se, n=len(z)))
    return pd.DataFrame(rows)


def outcome_test(e: pd.DataFrame, M, rng: np.random.Generator, rate: float, band: float = 0.05) -> pd.DataFrame:
    """Each AI rule's marginal rejections: within each cell, the most lenient quintile's grants the rule rejects
    between (rate − band) and rate of that quintile's caseload (rate: the examiners' mean extra rejections; fractional
    applications weighted as in 35's contraction); band = rate: all of them. Women's against men's citations among
    them (raw and top-coded), pooled and within center × filing year (strata with both, weighted by women's mass).
    The examiners' (from quintile differences, cell-size weighted, mean over quintiles 2–5): the extra rejections'
    citations by gender. Bootstrap of art units; positive = the rule's marginal rejected women are cited more than its
    marginal rejected men (a higher bar for women)."""
    rules = [M.ALGO["cites"], M.JUDGE, M.EQUAL]
    cells = list(e.groupby(M.CELL, sort=True))
    nc = len(cells)
    tb = rng.random(len(e))
    pos = pd.Series(np.arange(len(e)), index=e.index)
    sa, sj = -e.ml_cites.to_numpy(), -e.ml_judge.to_numpy()
    # per cell: [women's mass, men's mass, women's citations, men's citations, women's top-coded, men's top-coded]
    A = {r: np.zeros((nc, 6)) for r in rules + ["Examiners"]}
    strata = np.empty(nc, dtype=object)
    au = np.array([c.art_unit.iloc[0] for _, c in cells])

    def weights(nrank, lo, hi):
        r = np.arange(1, nrank + 1)
        return np.clip(np.minimum(r, hi) - np.maximum(r - 1, lo), 0, 1)
    for i, (_, c) in enumerate(cells):
        strata[i] = f"{c.tech_center.iloc[0]}_{c.filing_year.iloc[0]}"
        y, yt = np.nan_to_num(c.cites.to_numpy(float)), np.nan_to_num(c.trim_cites.to_numpy(float))
        g, f, q = c.granted.to_numpy(float), c.female.to_numpy(float), c.q.to_numpy(int)
        five = q == 5
        hi, lo = five.sum() * rate, five.sum() * max(rate - band, 0)
        gi = np.where(five & (g == 1))[0]
        p = pos[c.index].to_numpy()
        for r, sc in [(M.ALGO["cites"], sa), (M.JUDGE, sj)]:
            ii = gi[np.lexsort((tb[p][gi], -sc[p][gi]))]
            w = weights(len(ii), lo, hi)
            fi = f[ii]
            A[r][i] = [(w * fi).sum(), (w * (1 - fi)).sum(), (w * fi * y[ii]).sum(), (w * (1 - fi) * y[ii]).sum(),
                       (w * fi * yt[ii]).sum(), (w * (1 - fi) * yt[ii]).sum()]
        ii = gi[np.lexsort((tb[p][gi], -sa[p][gi]))]   # equal rates: women and men each in the algorithm's order
        sh = f[gi].mean() if len(gi) else 0.0
        iw, im = ii[f[ii] == 1], ii[f[ii] != 1]
        ww, wm = weights(len(iw), lo * sh, hi * sh), weights(len(im), lo * (1 - sh), hi * (1 - sh))
        A[M.EQUAL][i] = [ww.sum(), wm.sum(), (ww * y[iw]).sum(), (wm * y[im]).sum(), (ww * yt[iw]).sum(),
                         (wm * yt[im]).sum()]
        v = np.zeros(6)   # the examiners: quintile differences per application, summed over quintiles 2–5
        for k in [4, 3, 2, 1]:
            b_ = q == k
            for j, (wt, yy) in enumerate([(f, 1), (1 - f, 1), (f, y), (1 - f, y), (f, yt), (1 - f, yt)]):
                v[j] += (wt * g * yy)[five].mean() - (wt * g * yy)[b_].mean()
        A["Examiners"][i] = v * len(c)
    mult = M.boot_mult(au, rng)
    rows = []
    su, sinv = np.unique(strata, return_inverse=True)
    for r, a in A.items():
        for o, (jf, jm) in [("cites", (2, 3)), ("top-coded cites", (4, 5))]:
            fm = (mult @ a[:, jf]) / (mult @ a[:, 0])
            mm = (mult @ a[:, jm]) / (mult @ a[:, 1])
            d = fm - mm
            bl = "last 5 pp" if band < rate else "all extra rejections"
            row = dict(rule=r, outcome=o, band=bl + (" (quintile differences)" if r == "Examiners" else ""),
                       female=fm[0], male=mm[0], diff=d[0], diff_lo=M.pc(d[1:], 1), diff_hi=M.pc(d[1:], 0),
                       female_mass=a[:, 0].sum(), male_mass=a[:, 1].sum())
            if r != "Examiners":   # within center × filing year
                dw = []
                for m_ in mult:
                    S = np.zeros((len(su), 6))
                    np.add.at(S, sinv, a * m_[:, None])
                    ok = (S[:, 0] > 1e-9) & (S[:, 1] > 1e-9)
                    dd = S[ok, jf] / S[ok, 0] - S[ok, jm] / S[ok, 1]
                    dw.append((dd * S[ok, 0]).sum() / S[ok, 0].sum())
                dw = np.array(dw)
                row.update(diff_within=dw[0], diff_within_lo=M.pc(dw[1:], 1), diff_within_hi=M.pc(dw[1:], 0))
            rows.append(row)
    return pd.DataFrame(rows)


def forwhom(x: pd.DataFrame, M, N: dict) -> tuple:
    """C1 'for whom' for the algorithm trained on citations, the predicted examiner trained on grants, and the
    algorithm constrained to equal grant rates, against the examiners: Table III's citations, Table VI's female share
    of the extra rejections, Table VII's gender gap after reranking (within art unit × filing year), and the outcome
    test. Own random stream."""
    rng = np.random.default_rng(M.SEED + 49)
    x = with_top_coded(x)
    rules = fw_rules(M)
    c = pd.read_csv(T / f"c1_kllm{S35}_contraction.csv")
    q = pd.read_csv(T / f"c1_kllm{S35}_quintiles.csv")
    c = c[(c["sample"] == "test + imputation") & (c.outcome == "cites") & (c.quintile.astype(str) == "2–5 (mean)")]
    q = q[(q["sample"] == "test + imputation") & (q.outcome == "cites") & (q.quintile.astype(str) == "2–5 (mean)")]
    q = q.iloc[0]
    c = c.set_index("rule")
    con = [dict(rule="Examiners", lost=-q.change, lost_lo=-q.change_hi, lost_hi=-q.change_lo, better=np.nan,
                better_lo=np.nan, better_hi=np.nan, female_share=q.female_share, female_share_lo=q.female_share_lo,
                female_share_hi=q.female_share_hi)]
    for r in list(rules)[1:]:
        z = c.loc[r]
        con.append(dict(rule=r, lost=-z.change_at_examiner_rejections, lost_lo=-z.hi, lost_hi=-z.lo, better=z["diff"],
                        better_lo=z.diff_lo, better_hi=z.diff_hi, female_share=z.female_share,
                        female_share_lo=z.female_share_lo, female_share_hi=z.female_share_hi))
    con = pd.DataFrame(con)
    te = x[x.part == "test"]
    rr, dec = rerank_fw(te, M, rng)
    gw = gap_within(te, dec)
    e = evaluated(x, M)
    ot5 = outcome_test(e, M, rng, float(q.extra_rejections), 0.05)
    ota = outcome_test(e, M, rng, float(q.extra_rejections), float(q.extra_rejections))
    ot5 = ot5[ot5.rule != "Examiners"]   # the examiners' marginal rejections are not ordered: quintile differences
    ot = pd.concat([ot5, ota])
    tab = con.merge(rr, on="rule").merge(gw, on="rule")
    o5 = pd.concat([ot5, ota[ota.rule == "Examiners"]]).set_index(["rule", "outcome"])
    for col in ["female", "male", "diff", "diff_lo", "diff_hi", "diff_within", "diff_within_lo", "diff_within_hi"]:
        for o, sfx in [("cites", ""), ("top-coded cites", "_trim")]:
            tab[f"ot_{col}{sfx}"] = tab.rule.map(lambda r, o=o, col=col: o5.loc[(r, o), col] if (r, o) in o5.index
                                                 else np.nan)
    tab["label"] = tab.rule.map(rules)
    # macros
    mac = {"Examiners": "Ex", M.ALGO["cites"]: "Algo", M.JUDGE: "Judge", M.EQUAL: "Eq"}

    def ci(lo, hi, s=100, nd=1):
        return f"[{num(s * lo, nd)}, {num(s * hi, nd)}]"
    for r_ in tab.itertuples():
        m = mac[r_.rule]
        N[f"xCeFw{m}Lost"] = num(100 * r_.lost)
        if m != "Ex":
            N[f"xCeFw{m}Better"], N[f"xCeFw{m}BetterCI"] = num(100 * r_.better), ci(r_.better_lo, r_.better_hi)
            N[f"xCeFw{m}Rerank"], N[f"xCeFw{m}RerankCI"] = num(100 * r_.rerank_change), ci(r_.rerank_change_lo,
                                                                                          r_.rerank_change_hi)
        N[f"xCeFw{m}Fem"], N[f"xCeFw{m}FemCI"] = num(100 * r_.female_share), ci(r_.female_share_lo,
                                                                              r_.female_share_hi)
        N[f"xCeFw{m}Gap"], N[f"xCeFw{m}GapCI"] = num(100 * r_.gap_within), ci(r_.gap_within_lo, r_.gap_within_hi)
        N[f"xCeFw{m}GapPooled"] = num(100 * r_.gap_pooled)
        for sfx, mm in [("", ""), ("_trim", "Trim")]:
            d, lo, hi = getattr(r_, f"ot_diff{sfx}"), getattr(r_, f"ot_diff_lo{sfx}"), getattr(r_, f"ot_diff_hi{sfx}")
            N[f"xCeFw{m}Ot{mm}"], N[f"xCeFw{m}Ot{mm}CI"] = num(d, 2), ci(lo, hi, 1, 2)
            N[f"xCeFw{m}Ot{mm}Fem"] = num(getattr(r_, f"ot_female{sfx}"), 2)
            N[f"xCeFw{m}Ot{mm}Male"] = num(getattr(r_, f"ot_male{sfx}"), 2)
            if m != "Ex":
                dw, lw, hw = (getattr(r_, f"ot_diff_within{sfx}"), getattr(r_, f"ot_diff_within_lo{sfx}"),
                              getattr(r_, f"ot_diff_within_hi{sfx}"))
                N[f"xCeFw{m}Ot{mm}Within"], N[f"xCeFw{m}Ot{mm}WithinCI"] = num(dw, 2), ci(lw, hw, 1, 2)
    N["xCeFwRate"] = num(100 * float(q.extra_rejections))
    # LaTeX table
    rows = []
    for r_ in tab.itertuples():
        rows.append([esc(r_.label), num(100 * r_.lost),
                     "" if pd.isna(r_.better) else f"{num(100 * r_.better)} {ci(r_.better_lo, r_.better_hi)}",
                     f"{num(100 * r_.female_share)} {ci(r_.female_share_lo, r_.female_share_hi)}",
                     f"{num(100 * r_.gap_within)} {ci(r_.gap_within_lo, r_.gap_within_hi)}",
                     f"{num(r_.ot_diff, 2)} {ci(r_.ot_diff_lo, r_.ot_diff_hi, 1, 2)}"])
    tabular(TEX / f"{PRE}_tab_forwhom.tex", "lrllll",
            ["", "Citations lost (\\%)", "Rule $-$ examiners (pp)", "Women among extra rejections (\\%)",
             "Female $-$ male grant rate, reranked (pp)", "Outcome test: female $-$ male citations"], rows)
    return tab, rr, gw, ot


def forwhom_report(tab: pd.DataFrame, ot: pd.DataFrame) -> None:
    def c(v, lo, hi, s=100, nd=1):
        return f"{s * v:.{nd}f} [{s * lo:.{nd}f}, {s * hi:.{nd}f}]".replace("-", "−")
    L = [f"# C1 for whom ({'Poisson loss' if PSFX else 'squared-error loss'}; 43_c1_extra.py --forwhom)", "",
         "Kleinberg et al.'s Tables VI and VII for the AI rules: the algorithm trained on outcomes (citations), the "
         "predicted examiner trained on decisions (grants), and the algorithm constrained to equal grant rates for "
         "women and men. Contraction (test and imputation sets, mean over quintiles 2–5): citations per application "
         "lost at the examiners' extra rejections and the female share of the extra rejections (35's contraction "
         "output). Reranking (test set, all applications at the examiners' grant rate, α = 1): change in citations "
         "per application and the female − male grant rate within art unit × filing year (decision ~ female | "
         "art unit × year, standard errors clustered by examiner). Outcome test: among each rule's marginal "
         "rejections (the most lenient quintile's grants it rejects in the last 5 pp before the examiners' "
         "extra-rejection rate), women's minus men's citations; for the examiners, the extra rejections' citations "
         "by gender from quintile differences. Positive = the marginal rejected women are cited more, a higher bar "
         "for women. 95% intervals: bootstrap of art units (500 draws), clustered standard errors for the gap.", "",
         "| Rule | Citations lost (%) | Rule − examiners (pp) | Women among extra rejections (%) | Reranked: "
         "citations per application vs examiners (%) | Female − male grant rate within AU × year (pp) | Same, "
         "pooled (pp) | Outcome test, female − male citations (raw) | Same, top-coded | Same, within center × year "
         "(raw) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.label} | {c(r.lost, r.lost_lo, r.lost_hi)} | "
                 f"{'' if pd.isna(r.better) else c(r.better, r.better_lo, r.better_hi)} | "
                 f"{c(r.female_share, r.female_share_lo, r.female_share_hi)} | "
                 f"{c(r.rerank_change, r.rerank_change_lo, r.rerank_change_hi)} | "
                 f"{c(r.gap_within, r.gap_within_lo, r.gap_within_hi)} | {100 * r.gap_pooled:.1f} | "
                 f"{c(r.ot_diff, r.ot_diff_lo, r.ot_diff_hi, 1, 2)} | "
                 f"{c(r.ot_diff_trim, r.ot_diff_lo_trim, r.ot_diff_hi_trim, 1, 2)} | "
                 f"{'' if pd.isna(r.ot_diff_within) else c(r.ot_diff_within, r.ot_diff_within_lo,
                                                            r.ot_diff_within_hi, 1, 2)} |")
    L += ["", "Outcome test in detail (citations of the marginal rejections; masses are applications):", "",
          "| Rule | Outcome | Band | Women | Men | Female − male | Within center × year | Women's mass | Men's mass |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ot.itertuples():
        w = "" if pd.isna(getattr(r, "diff_within", np.nan)) else c(r.diff_within, r.diff_within_lo,
                                                                     r.diff_within_hi, 1, 2)
        L.append(f"| {r.rule} | {r.outcome} | {r.band} | {r.female:.2f} | {r.male:.2f} | "
                 f"{c(r.diff, r.diff_lo, r.diff_hi, 1, 2)} | {w} | {r.female_mass:.1f} | {r.male_mass:.1f} |")
    (OUT / f"{PRE}_forwhom.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- 11. the score's validation within groups

WR_GROUPS = {"Pooled": None, "TcYear": ["tech_center", "filing_year"], "AuYear": ["art_unit", "filing_year"],
             "ClsYear": ["uspc", "filing_year"]}
WR_B = 200


def within_spearman(z: pd.DataFrame, a: str, b: str, by: list | None) -> float:
    """Spearman correlation of a and b pooled within groups: percentile ranks within each group (ties averaged),
    demeaned by group, then their Pearson correlation; with by = None, the ordinary Spearman correlation."""
    if by is None:
        ra, rb = z[a].rank(pct=True), z[b].rank(pct=True)
    else:
        gr = z.groupby(by, sort=False)
        ra, rb = gr[a].rank(pct=True), gr[b].rank(pct=True)
        ra = ra - ra.groupby([z[c] for c in by]).transform("mean")
        rb = rb - rb.groupby([z[c] for c in by]).transform("mean")
    ra, rb = ra.to_numpy(), rb.to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def within_validation(x: pd.DataFrame, M) -> pd.DataFrame:
    """The counterfactual score against prior-art citations (unpublished = 0, as 35's cf_valid) for rejected and for
    granted applications (lock box excluded, as cf_valid), and against citations for granted ones: pooled, as cf_valid,
    and within center × filing year, art unit × filing year, and USPC class × filing year, so that differences in
    citation levels and truncation between centers and years do not drive it. 95% intervals: WR_B bootstrap draws of
    art units (own random stream), the same draws for rejected and granted, so their difference has an interval."""
    z = x[x.part != "lockbox"].copy()
    z["uspc"] = z.au_class.astype(str).str.split("_").str[2]
    z["pa"] = z.prior_art.fillna(0)
    z["art_unit"] = z.art_unit.astype(str)
    g = (z.granted == 1).to_numpy()
    rej, gr = z[~g].reset_index(drop=True), z[g].reset_index(drop=True)

    def stats(r_, g_):
        out = {}
        for k, by in WR_GROUPS.items():
            out[f"rej_{k}"] = within_spearman(r_, "cf", "pa", by)
            out[f"gr_{k}"] = within_spearman(g_, "cf", "pa", by)
            out[f"citesgr_{k}"] = within_spearman(g_, "cf", "cites", by)
        return out
    est = stats(rej, gr)
    aus = np.unique(z.art_unit.to_numpy())
    pos_r = rej.groupby("art_unit").indices
    pos_g = gr.groupby("art_unit").indices
    rng = np.random.default_rng(M.SEED + 51)
    draws = []
    for _ in range(WR_B):
        pick = aus[rng.integers(0, len(aus), len(aus))]
        ir = np.concatenate([pos_r[a] for a in pick if a in pos_r])
        ig = np.concatenate([pos_g[a] for a in pick if a in pos_g])
        draws.append(stats(rej.iloc[ir].reset_index(drop=True), gr.iloc[ig].reset_index(drop=True)))
    D = pd.DataFrame(draws)
    rows = []
    for k in WR_GROUPS:
        for q, lab in [("rej", "prior art, rejected"), ("gr", "prior art, granted"),
                       ("citesgr", "citations, granted"), ("diff", "prior art, rejected − granted")]:
            v = est[f"rej_{k}"] - est[f"gr_{k}"] if q == "diff" else est[f"{q}_{k}"]
            dv = D[f"rej_{k}"] - D[f"gr_{k}"] if q == "diff" else D[f"{q}_{k}"]
            rows.append(dict(grouping=k, measure=lab, key=q, value=v, lo=float(np.percentile(dv, 2.5)),
                             hi=float(np.percentile(dv, 97.5)), n_rejected=len(rej), n_granted=len(gr)))
    return pd.DataFrame(rows)


def within_report(t: pd.DataFrame, N: dict) -> None:
    lab = {"Pooled": "Pooled (as cf_valid)", "TcYear": "Within center × filing year",
           "AuYear": "Within art unit × filing year", "ClsYear": "Within USPC class × filing year"}
    mac = {"rej": "Rej", "gr": "Gr", "citesgr": "CitesGr", "diff": "Diff"}
    for r in t.itertuples():
        m = f"xCeWr{mac[r.key]}{r.grouping}"
        N[m], N[m + "CI"] = num(r.value, 3), f"[{num(r.lo, 3)}, {num(r.hi, 3)}]"
    L = [f"# The counterfactual score's validation within groups ({'Poisson' if PSFX else 'squared-error'} scores; "
         "43_c1_extra.py --within)", "",
         "Spearman correlation of the counterfactual score with prior-art citations (unpublished applications: 0, as "
         "35's cf_valid) among rejected and among granted applications (lock box excluded), and with citations among "
         "granted applications; pooled (35's cf_valid) and pooled within groups (percentile ranks within each group, "
         f"demeaned by group, Pearson). 95% intervals: {WR_B} bootstrap draws of art units, the same draws for "
         "rejected and granted applications.", "",
         "| Grouping | Prior art, rejected | Prior art, granted | Rejected − granted | Citations, granted |",
         "|---|---|---|---|---|"]
    ti = t.set_index(["grouping", "key"])

    def c(k, q):
        r = ti.loc[(k, q)]
        return f"{r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}]".replace("-", "−")
    for k in WR_GROUPS:
        L.append(f"| {lab[k]} | {c(k, 'rej')} | {c(k, 'gr')} | {c(k, 'diff')} | {c(k, 'citesgr')} |")
    L += ["", f"Rejected: {int(t.n_rejected.iloc[0]):,} applications; granted: {int(t.n_granted.iloc[0]):,}."]
    (OUT / f"{PRE}_within.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def level_macros(M, N: dict) -> None:
    """Levels at the margin (35's extension C) with citations and the score top-coded at the same caps
    (c1_kllm*_levels_topcoded.csv): per stricter quintile and pooled, and the top-coded mean of granted applications."""
    f = T / f"c1_kllm{S35}_levels_topcoded.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t["quintile"] = t.quintile.astype(str)
    names = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "2–5 (pooled)": "Pooled"}
    for ser, m in [("observed citations", "Obs"), (M.CF, "Score")]:
        for q_, qn in names.items():
            r = t[(t.series == ser) & (t.quintile == q_)].iloc[0]
            N[f"xCeLv{m}Trim{qn}"], N[f"xCeLv{m}Trim{qn}CI"] = num(r.value, 2), f"[{num(r.lo, 2)}, {num(r.hi, 2)}]"
    for lab, m in [("top-coded", "Trim"), ("raw", "Raw")]:   # the contraction sample's granted applications
        g = t[t.series == f"mean citations of granted applications, {lab}"]
        if len(g):
            N[f"xCeLvGranted{m}"] = num(float(g.value.iloc[0]), 2)


def capped_cells(x: pd.DataFrame, M, N: dict) -> None:
    """Table III's rules reject, in each cell, k = n_5 × rate of the most lenient quintile's grants; where k exceeds
    them, contraction() leaves the cell out of that rule's comparison at that rate (contraction_plot.R's na.rm). The
    number and share of such cells at the strictest quintile's extra-rejection rate, and the algorithm's loss of
    citations (point estimates, evaluated at each quintile's rate) with those cells left out, as Table III, or capped
    (all their grants rejected) instead."""
    e = evaluated(x, M)
    q = pd.read_csv(T / f"c1_kllm{S35}_quintiles.csv")
    q = q[(q["sample"] == "test + imputation") & (q.outcome == "cites") & (q.quintile.astype(str) != "2–5 (mean)")]
    X = q.sort_values("quintile").extra_rejections.to_numpy()   # quintiles 2, 3, 4, 5
    rows = []
    for _, c in e.groupby(M.CELL, sort=True):
        y, g, qq = np.nan_to_num(c.cites.to_numpy(float)), c.granted.to_numpy(float), c.q.to_numpy(int)
        five = qq == 5
        gi = np.where(five & (g == 1))[0]
        ii = gi[np.argsort(c.ml_cites.to_numpy()[gi], kind="stable")]
        cum = np.r_[0, np.cumsum(y[ii])]
        n5, base = five.sum(), y[five].mean()
        for j, xj in enumerate(X):
            k = n5 * xj
            rows.append(dict(j=j, n=len(c), base=base, short=k > len(ii) + 1e-9,
                             r_drop=np.interp(k, np.arange(len(ii) + 1), cum) / n5 if k <= len(ii) + 1e-9 else np.nan,
                             r_cap=np.interp(min(k, len(ii)), np.arange(len(ii) + 1), cum) / n5))
    t = pd.DataFrame(rows)
    out = {}
    for j, z in t.groupby("j"):
        ok = ~z.short
        out[j] = (-(z.n * z.r_drop)[ok].sum() / (z.n * z.base)[ok].sum(),
                  -(z.n * z.r_cap).sum() / (z.n * z.base).sum())
    z5 = t[t.j == 3]
    N["xCeCapCells"], N["xCeCapCellsAll"] = f"{int(z5.short.sum()):,}", f"{len(z5):,}"
    N["xCeCapCellsShare"] = num(100 * z5.short.mean(), 1)
    N["xCeCapAppsShare"] = num(100 * z5.n[z5.short].sum() / z5.n.sum(), 1)
    # citations lost (%, positive), as Table III's "Citations lost"
    N["xCeCapAlgoDrop"] = num(-100 * np.mean([out[j][0] for j in out]), 1)
    N["xCeCapAlgoCap"] = num(-100 * np.mean([out[j][1] for j in out]), 1)
    N["xCeCapAlgoDropFive"], N["xCeCapAlgoCapFive"] = num(-100 * out[3][0], 1), num(-100 * out[3][1], 1)


def grid_macros(M, N: dict) -> None:
    """How many of the boosted models chosen by cross-validation picked the grid's largest number of trees."""
    mo = pd.read_csv(T / f"c1_kllm{S35}_models.csv")
    mo = mo[mo.cv.notna() & ~mo.model.str.startswith("linear")]
    top = int(M.TREES.max())
    N["xCeGridMaxTrees"], N["xCeGridModels"] = f"{top:,}", str(len(mo))
    N["xCeGridAtMax"] = str(int((mo.trees == top).sum()))


# ---------------------------------------------------------------- 5. figure

def figure() -> None:
    J = load_module("jf20", "20_journal_figures.py")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5})
    BLACK, BLUE, VERM, GREY = "#000000", "#0072B2", "#D55E00", "#7F7F7F"   # Okabe–Ito
    f6 = pd.read_csv(T / f"c1_kllm{S35}_fig6.csv")
    f6 = f6[pd.to_numeric(f6.quintile, errors="coerce") == 5].sort_values("bin")
    dc = pd.read_csv(T / f"c1_kllm{S35}_cf_deciles.csv")

    def letter(fig, x_mm, y_mm, s, title):
        W, H = fig.get_size_inches() * 25.4
        fig.text(x_mm / W, y_mm / H, s, fontsize=8, fontweight="bold", va="top", ha="left")
        fig.text((x_mm + 3.4) / W, y_mm / H, title, fontsize=7, va="top", ha="left")
    fig = J.figure(183, 64)
    # a: where the extra rejections fall
    ax = J.axes(fig, 13, 11, 74, 47)
    # the rules work within each cell to match the strictest quintile's extra rejections there (35's misranking)
    ser = [("examiners", GREY, "Strictest examiners' extra rejections"),
           ("algorithm", BLUE, "Algorithm (trained on citations)")]
    if "judge" in f6 and f6.judge.notna().all():
        ser.append(("judge", VERM, "Predicted examiner (trained on grants)"))
    w = 0.84 / len(ser)
    for j_, (col_, colr, lab) in enumerate(ser):
        ax.bar(f6.bin + (j_ - (len(ser) - 1) / 2) * w, 100 * f6[col_], width=w, color=colr, label=lab)
    ax.axhline(5, color=BLACK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    top_ = 100 * float(f6[[c_ for c_, _, _ in ser]].max().max())
    ax.text(12.0, 6.3, "Random: about 5% per bin", ha="center", va="bottom", fontsize=6.5, color=BLACK)
    ax.set_xlim(0.3, 20.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, max(10.0, top_ * 1.5))
    ax.set_xlabel("Predicted citations, bin of 5% of applications (1 = lowest)")
    ax.set_ylabel("Share of extra rejections (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    J.hgrid(ax, "y")
    letter(fig, 1.5, 63.0, "a", "Where the extra rejections fall")
    # b: prior-art citations by score decile
    ax = J.axes(fig, 106, 11, 74, 47)
    for gv, col, mk, lab in [(1, BLUE, "o", "Granted applications"), (0, VERM, "s", "Rejected applications")]:
        t = dc[dc.granted == gv].sort_values("dec")
        ax.plot(t.dec, t.prior_art, color=col, marker=mk, ms=3.2, lw=1.0, label=lab, clip_on=False, zorder=3)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0, max(8.5, 1.08 * float(dc.prior_art.max())))
    ax.set_xlabel("Score decile, within center × year (1 = lowest)")
    ax.set_ylabel("Prior-art citations per application")
    ax.legend(loc="upper left")
    J.hgrid(ax, "y")
    letter(fig, 94.5, 63.0, "b", "Prior-art citations by counterfactual score")
    J.save(fig, f"fig_main_c1{PSFX}")


# ---------------------------------------------------------------- report

def main() -> None:
    warnings.filterwarnings("ignore")
    if "--figure" in os.sys.argv:
        figure()
        return
    M = load_module("m35", "35_c1_kllm.py")
    if "--poisson" in os.sys.argv:   # compare 35's Poisson variant with the main run; macros merged
        N = {}
        t = poisson_compare(M, N)
        t.to_csv(T / CMP_CSV, index=False)
        for f in ["c1x_numbers.tex", "c1x_poisson_numbers.tex"]:   # a comparison: the same in both macro files
            write_macros(N, TEX / f)
        w = t.pivot_table(index="quantity", columns="version", values="value", sort=False)
        L = ["# C1: Poisson loss (35 with CFE_LOSS=poisson, outputs *_all_poisson) against the main run", "",
             "Same partition, inputs, grid and random streams; the count models (citations, prior-art citations) "
             "with a Poisson loss and the mean Poisson deviance as the cross-validation criterion.", "",
             "| Quantity | Squared error (main) | Poisson |", "|---|---|---|"]
        for qq, r in w.iterrows():
            def f(ver):
                z = t[(t.quantity == qq) & (t.version == ver)].iloc[0]
                v = f"{z.value:.3f}" if abs(z.value) < 1 else f"{z.value:.2f}"
                v = v if pd.isna(z.lo) else f"{v} [{z.lo:.2f}, {z.hi:.2f}]"
                return v.replace("-", "−")
            L.append(f"| {qq} | {f('squared error')} | {f('Poisson')} |")
        (OUT / "c1x_poisson_compare.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    if "--within" in os.sys.argv:   # the score's validation within groups; macros merged into the macro file
        N = {}
        t = within_validation(M.cached(M.load_all(), "all"), M)
        t.to_csv(T / f"{PRE}_within.csv", index=False)
        within_report(t, N)
        write_macros(N)
        return
    if "--examples" in os.sys.argv:   # Table A18 only; macros merged into the macro file
        N = {}
        x = M.cached(M.load_all(), "all")
        rj = x[(x.granted == 0) & x.part.isin(["test", "impute"])].application_number
        ex, rep, pool = examples(x, M, family_flags(rj), N)
        ex.to_csv(T / f"{PRE}_examples.csv", index=False)
        rep.to_csv(T / f"{PRE}_examples_repr.csv", index=False)
        pool.to_csv(T / f"{PRE}_examples_pool.csv", index=False)
        write_macros(N)
        examples_report(ex, rep)
        return
    if "--forwhom" in os.sys.argv:   # C1 for whom; macros merged into the macro file
        N = {}
        tab, rr, gw, ot = forwhom(M.cached(M.load_all(), "all"), M, N)
        tab.to_csv(T / f"{PRE}_forwhom.csv", index=False)
        ot.to_csv(T / f"{PRE}_forwhom_outcome_test.csv", index=False)
        write_macros(N)
        forwhom_report(tab, ot)
        return
    if "--perm" in os.sys.argv:   # 46's C1 permutation test as macros
        N = {}
        perm_macros(N)
        write_macros(N)
        print(N)
        return
    if "--lite" in os.sys.argv:   # lite robustness only; macros merged into c1x_numbers.tex
        N = {}
        q, c, ft = lite(M, N)
        q.to_csv(T / f"{PRE}_lite_quintiles.csv", index=False)
        c.to_csv(T / f"{PRE}_lite_contraction.csv", index=False)
        ft.to_csv(T / f"{PRE}_lite_fit.csv", index=False)
        write_macros(N)
        lite_report(q, c, ft)
        return
    if "--tail" in os.sys.argv:   # tail diagnostics only; macros merged into c1x_numbers.tex
        N = {}
        t, st = tail(M.cached(M.load_all(), "all"), M, N)
        t.to_csv(T / f"{PRE}_tail_designs.csv", index=False)
        st.to_csv(T / f"{PRE}_tail_staple.csv", index=False)
        write_macros(N)
        tail_report(t, st)
        return
    N, L = {}, []
    x = M.cached(M.load_all(), "all")

    # 2. family check
    fam_tab, fam_sp, fam = family(x)
    fam_tab.to_csv(T / f"{PRE}_family.csv", index=False)
    fam_sp.to_csv(T / f"{PRE}_family_spearman.csv", index=False)
    ft = fam_tab.set_index("group")

    def fpct(lab, c="any", nd=1):
        return num(100 * ft.loc[lab, c], nd)
    for lab, mac in [("Rejected, all", "Rej"), ("Granted, all (comparison)", "Gr"),
                     ("Rejected, test + imputation, top 1% by score", "TopOne"),
                     ("Rejected, test + imputation, top 10 by score (Table A17)", "TopTen")]:
        N[f"xCeFam{mac}Any"] = fpct(lab, "any")
        N[f"xCeFam{mac}Family"] = fpct(lab, "family")
        N[f"xCeFam{mac}Child"] = fpct(lab, "child")
    N["xCeFamTopTenN"] = str(int(round(10 * ft.loc["Rejected, test + imputation, top 10 by score (Table A17)", "any"])))
    tcs = ft[ft.index.str.startswith("Rejected, TC")]
    N["xCeFamTcMin"], N["xCeFamTcMax"] = num(100 * tcs["any"].min(), 1), num(100 * tcs["any"].max(), 1)
    N["xCeFamDecLow"] = fpct("Rejected, test + imputation, score decile 1")
    N["xCeFamDecHigh"] = fpct("Rejected, test + imputation, score decile 10")
    sp = fam_sp.set_index("group")
    N["xCeFamSpAll"], N["xCeFamSpClean"] = (num(sp.loc["all rejected", "spearman_prior_art"], 2),
                                            num(sp.loc["no granted relative", "spearman_prior_art"], 2))

    # 1. examples
    ex, ex_rep, ex_pool = examples(x, M, fam, N)
    ex.to_csv(T / f"{PRE}_examples.csv", index=False)
    ex_rep.to_csv(T / f"{PRE}_examples_repr.csv", index=False)
    ex_pool.to_csv(T / f"{PRE}_examples_pool.csv", index=False)
    examples_report(ex, ex_rep)

    # 4. Table V's binary outcome and the predicted examiner (from 35's tables)
    c = pd.read_csv(T / f"c1_kllm{S35}_contraction.csv")
    q = pd.read_csv(T / f"c1_kllm{S35}_quintiles.csv")
    c = c[(c["sample"] == "test + imputation") & (c.quintile.astype(str) == "2–5 (mean)")]
    c = c.set_index(["outcome", "rule"])
    q = q[(q["sample"] == "test + imputation") & (q.quintile.astype(str) == "2–5 (mean)")].set_index("outcome")
    unc = c.loc["uncited"]
    a_u = unc.loc["Algorithm (trained on no citations)"]
    N["xCeUncEx"] = num(-100 * q.loc["uncited", "change"])                  # fewer uncited patents per application, %
    N["xCeUncAlgo"] = num(-100 * a_u.change_at_examiner_rejections)
    N["xCeUncRnd"] = num(-100 * unc.loc[M.RANDOM].change_at_examiner_rejections)
    N["xCeUncJudge"] = num(-100 * unc.loc[M.JUDGE].change_at_examiner_rejections)
    N["xCeUncAlgoMore"] = num(-100 * a_u["diff"])                            # positive = algorithm removes more
    N["xCeUncAlgoMoreCI"] = f"[{num(-100 * a_u.diff_hi)}, {num(-100 * a_u.diff_lo)}]"
    j_u = unc.loc[M.JUDGE]
    N["xCeUncJudgeMore"] = num(-100 * j_u["diff"])
    N["xCeUncJudgeMoreCI"] = f"[{num(-100 * j_u.diff_hi)}, {num(-100 * j_u.diff_lo)}]"
    v_rows = []
    for o, mac in [("cites", "Cites"), ("uncited", "Unc"), ("renew", "Renew"), ("top10", "Top"),
                   ("pa_granted", "PaG")]:
        z = c.loc[o]
        sgn = -1 if o == "uncited" else 1      # positive = better than the examiners
        for rule, rm in [(M.ALGO["uncited" if o == "uncited" else "cites"], "Algo"), (M.JUDGE, "Judge"),
                         (M.RANDOM, "Rnd")]:
            r = z.loc[rule]
            lo, hi = sorted([sgn * r.diff_lo, sgn * r.diff_hi])
            v_rows.append(dict(outcome=o, rule=rule, examiners_change=q.loc[o, "change"],
                               rule_change=r.change_at_examiner_rejections, better_than_examiners=sgn * r["diff"],
                               better_lo=lo, better_hi=hi))
            if o != "uncited":
                N[f"xCeV{mac}{rm}"] = num(100 * sgn * r["diff"])
                N[f"xCeV{mac}{rm}CI"] = f"[{num(100 * lo)}, {num(100 * hi)}]"
    vt = pd.DataFrame(v_rows)
    vt.to_csv(T / f"{PRE}_table_v_signed.csv", index=False)

    # 5. misranking shares (35's fig6, strictest quintile)
    f6 = pd.read_csv(T / f"c1_kllm{S35}_fig6.csv")
    f6 = f6[pd.to_numeric(f6.quintile, errors="coerce") == 5]
    N["xCeSixExLow"] = num(100 * f6.examiners[f6.bin <= 5].sum())
    N["xCeSixExHigh"] = num(100 * f6.examiners[f6.bin > 15].sum())
    N["xCeSixAlgoLow"] = num(100 * f6.algorithm[f6.bin <= 5].sum())
    N["xCeSixAlgoHigh"] = num(100 * f6.algorithm[f6.bin > 15].sum())
    N["xCeSixAlgoBins"] = str(int((f6.algorithm > 0).sum()))
    if "judge" in f6:
        N["xCeSixJudgeLow"] = num(100 * f6.judge[f6.bin <= 5].sum())
        N["xCeSixJudgeHigh"] = num(100 * f6.judge[f6.bin > 15].sum())
    N["xCeSixAlgoPooledLow"] = num(100 * f6.algorithm_pooled[f6.bin <= 5].sum()) if "algorithm_pooled" in f6 else "--"
    N["xCeSixExMin"], N["xCeSixExMax"] = num(100 * f6.examiners.min()), num(100 * f6.examiners.max())
    level_macros(M, N)
    capped_cells(x, M, N)
    grid_macros(M, N)
    cv = pd.read_csv(T / f"c1_kllm{S35}_cf_valid.csv").set_index("score")
    N["xCeMeanCitesGranted"] = num(cv.mean_cites_granted.iloc[0], 2)
    N["xCeCfPaAbove"] = num(100 * cv.loc[M.CF_PA, "share_rejected_above_granted_median"], 0)
    N["xCeCfPaTopQ"] = num(100 * cv.loc[M.CF_PA, "share_rejected_top_quarter"], 0)
    N["xCeCfAbove"] = num(100 * cv.loc[M.CF, "share_rejected_above_granted_median"], 0)
    N["xCeCfTopQ"] = num(100 * cv.loc[M.CF, "share_rejected_top_quarter"], 0)

    # 3. transformed outcomes
    tq, tcn, torr, info = transformed(x, M)
    tq.to_csv(T / f"{PRE}_transformed_quintiles.csv", index=False)
    tcn.to_csv(T / f"{PRE}_transformed.csv", index=False)
    vorr, comp = table_v_random(x, M)
    orr = pd.concat([torr, vorr])
    orr.to_csv(T / f"{PRE}_over_random.csv", index=False)
    comp.to_csv(T / f"{PRE}_table_v_composition.csv", index=False)
    om = orr.set_index(["outcome", "rule"])
    for o, mac in [("cites", "Raw"), ("log_cites", "Log"), ("trim_cites", "Trim"), ("uncited", "VUnc"),
                   ("renew", "VRenew"), ("top10", "VTop"), ("pa_granted", "VPaG")]:
        for rule, rm in [(M.ALGO["uncited" if o == "uncited" else "cites"], "Algo"), (M.JUDGE, "Judge")]:
            r = om.loc[(o, rule)]
            N[f"xCe{mac}{rm}OverRnd"] = num(100 * r.over_random)
            N[f"xCe{mac}{rm}OverRndCI"] = f"[{num(100 * r.over_random_lo)}, {num(100 * r.over_random_hi)}]"
    pd.DataFrame([info]).to_csv(T / f"{PRE}_transformed_info.csv", index=False)
    tqm = tq[tq.quintile.astype(str) == "2–5 (mean)"].set_index("outcome")
    tcm = tcn[tcn.quintile.astype(str) == "2–5 (mean)"].set_index(["outcome", "rule"])
    for o, mac in [("cites", "Raw"), ("log_cites", "Log"), ("trim_cites", "Trim")]:
        a_, r_, j_ = (tcm.loc[(o, M.ALGO["cites"])], tcm.loc[(o, M.RANDOM)], tcm.loc[(o, M.JUDGE)])
        N[f"xCe{mac}Ex"] = num(-100 * tqm.loc[o, "change"])                 # outcome lost per application, %
        N[f"xCe{mac}ExCI"] = f"[{num(-100 * tqm.loc[o, 'change_hi'])}, {num(-100 * tqm.loc[o, 'change_lo'])}]"
        N[f"xCe{mac}Algo"] = num(-100 * a_.change_at_examiner_rejections)
        N[f"xCe{mac}Rnd"] = num(-100 * r_.change_at_examiner_rejections)
        N[f"xCe{mac}Judge"] = num(-100 * j_.change_at_examiner_rejections)
        N[f"xCe{mac}Diff"] = num(100 * a_["diff"])
        N[f"xCe{mac}DiffCI"] = f"[{num(100 * a_.diff_lo)}, {num(100 * a_.diff_hi)}]"
        N[f"xCe{mac}RndDiff"] = num(100 * r_["diff"])
        N[f"xCe{mac}JudgeDiff"] = num(100 * j_["diff"])
        N[f"xCe{mac}JudgeDiffCI"] = f"[{num(100 * j_.diff_lo)}, {num(100 * j_.diff_hi)}]"
    N["xCeTrimShare"] = num(100 * info["share_top_coded"], 2)
    N["xCeTrimMean"], N["xCeRawMean"] = num(info["mean_trim"], 2), num(info["mean_cites"], 2)

    perm_macros(N)
    wt = within_validation(x, M)
    wt.to_csv(T / f"{PRE}_within.csv", index=False)
    within_report(wt, N)

    # 6. tail diagnostics
    t, st = tail(x, M, N)
    t.to_csv(T / f"{PRE}_tail_designs.csv", index=False)
    st.to_csv(T / f"{PRE}_tail_staple.csv", index=False)
    tail_report(t, st)

    write_macros(N)
    figure()
    report(N, fam_tab, fam_sp, ex, vt, tq, tcn, info, orr, comp)


def report(N, fam_tab, fam_sp, ex, vt, tq, tcn, info, orr, comp) -> None:
    def pc_(v):
        t = f"{100 * v:.1f}"
        return (t.lstrip("-") if float(t) == 0 else t).replace("-", "−")
    L = ["# C1 follow-ups (43_c1_extra.py)", "",
         f"From 35's cached predictions{' (Poisson loss)' if PSFX else ''} (no refitting). Macros in "
         f"output/latex/{PRE}_numbers.tex.", ""]
    L += ["## Counterfactual-score validation (Table A16), corrected", "",
          "Each score's share of rejected applications above the granted median / in the granted top quarter is now "
          "computed from that score's own percentile among the granted applications of the same center and filing "
          "year (35 had used the text score's percentile for both rows).", "",
          f"- Text score: {N['xCeCfAbove']}% above the granted median, {N['xCeCfTopQ']}% in the top quarter "
          "(unchanged).",
          f"- Text + prior-art citations: {N['xCeCfPaAbove']}% above the granted median, {N['xCeCfPaTopQ']}% in the "
          "top quarter (was 51% and 25%).", ""]
    L += ["## Table V's binary outcome ('uncited') and the predicted examiner", "",
          "'Uncited' (granted, no citations by others within five years) is a bad outcome. 35 reports, for every "
          "outcome, rule − examiners in the relative change of the outcome per application at the examiners' extra "
          "rejections. For 'uncited' the changes are negative (fewer uncited patents per application), so a negative "
          "difference means the rule removes MORE uncited patents than the examiners: better. The rule in this row is "
          "the algorithm trained on 'uncited' (rejects the highest predicted risk of no citations first).", "",
          f"- Uncited patents per application fall by {N['xCeUncEx']}% under the examiners' extra rejections, by "
          f"{N['xCeUncAlgo']}% under the algorithm, by {N['xCeUncRnd']}% under random rejections, by "
          f"{N['xCeUncJudge']}% under the predicted examiner.",
          f"- The algorithm removes {N['xCeUncAlgoMore']} pp more uncited patents than the examiners "
          f"{N['xCeUncAlgoMoreCI']} (the table's −2.6 with the sign 'positive = better'); the predicted examiner "
          f"{N['xCeUncJudgeMore']} {N['xCeUncJudgeMoreCI']}.", "",
          "All of Table V with the sign 'positive = better than the examiners' (pp; 95% CI):", "",
          "| Outcome | Rule | Examiners' change (%) | Rule's change (%) | Better than the examiners (pp) |",
          "|---|---|---|---|---|"]
    for r in vt.itertuples():
        L.append(f"| {r.outcome} | {r.rule} | {pc_(r.examiners_change)} | {pc_(r.rule_change)} | "
                 f"{pc_(r.better_than_examiners)} [{pc_(r.better_lo)}, {pc_(r.better_hi)}] |")
    L += ["", "Random rejection also does better than the examiners on renewals, top-decile citations and prior-art "
              "citations, so random rather than zero is the benchmark for a rule's gain. Rules against random "
              "(positive = better; paired 95% intervals from 500 bootstrap draws of art units, own random stream; "
              "citations from the transformed-outcome run below):", "",
          "| Outcome | Rule | Rule − random (pp) |", "|---|---|---|"]
    for r in orr.itertuples():
        L.append(f"| {r.outcome} | {r.rule} | {pc_(r.over_random)} "
                 f"[{pc_(r.over_random_lo)}, {pc_(r.over_random_hi)}] |")
    L += ["", "Why random beats the examiners: the examiners' relative change per application, against random "
              "rejection within each cell among the most lenient quintile's grants (same cell-specific extra "
              "rejections) and random rejection of the same share in every cell (what the random rule does):", "",
          "| Outcome | Examiners (%) | Random within cells (%) | Random, same share in every cell (%) |",
          "|---|---|---|---|"]
    for r in comp.itertuples():
        L.append(f"| {r.outcome} | {pc_(r.examiners)} | {pc_(r.random_within_cells)} | "
                 f"{pc_(r.random_same_share_every_cell)} |")
    L += ["", "For renewals, top-decile and prior-art citations most of the gap is within cells: the stricter "
              "quintiles' outcomes per application fall more than random rejections within the same cells would make "
              "them fall. Either their extra rejections are somewhat better-than-average applications on these "
              "outcomes, or stricter examination also lowers the outcomes of the patents they grant (e.g. narrower "
              "claims), which the contraction comparison attributes to the rejections. (Citations: the examiners lose "
              "slightly fewer than random within cells would; 'uncited': they remove slightly fewer.)", ""]
    L += ["## Where the strictest quintile's extra rejections fall (Figure A4a; main-text Figure 1a)", "",
          f"Strictest against most lenient quintile, 20 bins of predicted citations: the examiners' extra rejections "
          f"are {N['xCeSixExLow']}% in the lowest quarter (bins 1–5) and {N['xCeSixExHigh']}% in the highest (16–20); "
          f"per bin between {N['xCeSixExMin']}% and {N['xCeSixExMax']}% (random: 5%). The rules work within each "
          f"cell to match the strictest quintile's extra rejections there (Table III's rule): the algorithm's are "
          f"{N['xCeSixAlgoLow']}% in the lowest quarter and {N['xCeSixAlgoHigh']}% in the highest; the predicted "
          f"examiner's {N.get('xCeSixJudgeLow', '--')}% and {N.get('xCeSixJudgeHigh', '--')}%. (Taking the "
          f"lowest-predicted applications across the pooled sample instead, as the earlier version did, the "
          f"algorithm's would be {N['xCeSixAlgoPooledLow']}% in the lowest quarter.)", ""]
    L += ["## Family check of the rejected applications (PatEx continuity data)", "",
          "Shares (%) of applications with a granted related U.S. application. child: a granted continuation, "
          "continuation-in-part or divisional (by type: CON, CIP, DIV); family: a granted member of the continuity "
          "family (connected through such links); any: family, or another granted U.S. application claiming the same "
          "provisional or PCT application. Rejected = no patent (abandoned for all but a few hundred pending).", "",
          "| Group | N | Abandoned | Child | CON | CIP | DIV | Family | Any | Prior-art cites |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in fam_tab.itertuples():
        L.append(f"| {r.group} | {r.n:,} | {pc_(r.abandoned)} | {pc_(r.child)} | {pc_(r.child_con)} | "
                 f"{pc_(r.child_cip)} | {pc_(r.child_div)} | {pc_(r.family)} | {pc_(r.any)} | "
                 f"{r.mean_prior_art:.2f} |")
    L += ["", "Spearman correlation of the text score with prior-art citations among rejected applications (lock box "
              "excluded):", "", "| Rejected | N | Spearman | Mean score | Mean prior-art cites |",
          "|---|---|---|---|---|"]
    for r in fam_sp.itertuples():
        L.append(f"| {r.group} | {r.n:,} | {r.spearman_prior_art:.3f} | {r.mean_score:.2f} | {r.mean_prior_art:.2f} |")
    L += ["", f"## Examples of rejected applications (output/latex/{PRE}_tab_examples.tex; rule and representativeness "
              f"in output/{PRE}_examples.md)", "",
          "| TC | Application | Filed | USPC | Title | Predicted citations | Percentile, class × year | "
          "Prior-art percentile |", "|---|---|---|---|---|---|---|---|"]
    for r in ex.itertuples():
        L.append(f"| {r.tc} | {r.application_number} | {r.filing_year} | {r.uspc} | "
                 f"{str(r.invention_title).capitalize()[:70]} | {r.cf:.1f} | {r.field_pct:.1f} | {r.pa_pct:.0f} |")
    L += ["", "## Contraction with transformed citation outcomes (Table A10, robustness)", "",
          "Same ranking (the citation algorithm's predictions), quintiles and cells as Table A10 (test and imputation "
          "sets); 500 bootstrap draws of art units (own random stream). Changes are the relative fall in the outcome "
          f"per application (rejected = 0). Top-coding: {N['xCeTrimShare']}% of granted applications are above the "
          f"99th percentile of their center and filing year (median cap {info['q99_median']:.0f} citations, range "
          f"{info['q99_min']:.0f}–{info['q99_max']:.0f}); mean citations of granted applications {N['xCeRawMean']} "
          f"raw, {N['xCeTrimMean']} top-coded.", "",
          "| Outcome | Examiners: fall (%) | Algorithm: fall (%) | Random: fall (%) | Predicted examiner: fall (%) | "
          "Algorithm − examiners (pp) | Predicted examiner − examiners (pp) | Algorithm − random (pp) |",
          "|---|---|---|---|---|---|---|---|"]
    for o, mac, lab in [("cites", "Raw", "Citations (as in Table A10)"), ("log_cites", "Log", "log(1 + citations)"),
                        ("trim_cites", "Trim", "Citations top-coded at the 99th percentile")]:
        L.append(f"| {lab} | {N[f'xCe{mac}Ex']} {N[f'xCe{mac}ExCI']} | {N[f'xCe{mac}Algo']} | {N[f'xCe{mac}Rnd']} | "
                 f"{N[f'xCe{mac}Judge']} | {N[f'xCe{mac}Diff']} {N[f'xCe{mac}DiffCI']} | "
                 f"{N[f'xCe{mac}JudgeDiff']} {N[f'xCe{mac}JudgeDiffCI']} | "
                 f"{N[f'xCe{mac}AlgoOverRnd']} {N[f'xCe{mac}AlgoOverRndCI']} |")
    L += ["", "By stricter quintile (examiners' relative change, %, with 95% CI):", "",
          "| Outcome | Quintile | Extra rejections (pp) | Change (%) |", "|---|---|---|---|"]
    for r in tq.itertuples():
        L.append(f"| {r.outcome} | {r.quintile} | {100 * r.extra_rejections:.1f} | {100 * r.change:.1f} "
                 f"[{100 * r.change_lo:.1f}, {100 * r.change_hi:.1f}] |")
    L += ["", f"## Main-text Figure 1 (output/figures/journal/fig_main_c1{PSFX}.{{pdf,png}})", "",
          CAPTION.format(**N), "", f"## Macros (output/latex/{PRE}_numbers.tex)", ""]
    L += [f"- `\\{k}` = {v.replace('$-$', '−')}" for k, v in N.items()]
    L = [line.replace("$-$", "−") for line in L]
    (OUT / f"{PRE}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


CAPTION = ("**Figure 1. Whom stricter examiners reject, and whether the counterfactual score ranks rejected "
           "applications.** (a) Applications in the test and imputation sets are divided into 20 equal-sized bins of "
           "predicted citations. Grey: the share of the strictest examiner quintile's extra rejections (relative to "
           "the most lenient quintile in the same cell) that falls in each bin; blue and orange: where the "
           "algorithm's and the predicted examiner's would fall if, within every cell, they rejected as many of the "
           "most lenient quintile's grants as the strictest quintile does, lowest score first; dashed line: random "
           "rejection (5% per bin). The examiners' extra rejections are spread evenly ({xCeSixExLow}% in the lowest "
           "quarter, {xCeSixExHigh}% in the highest), as are the predicted examiner's ({xCeSixJudgeLow}% and "
           "{xCeSixJudgeHigh}%); the algorithm's are concentrated among low predicted citations ({xCeSixAlgoLow}% "
           "and {xCeSixAlgoHigh}%). (b) Prior-art citations (citations by later applications' examiners and "
           "applicants, observed whether or not an application was granted) by decile of the out-of-sample "
           "counterfactual citation score within technology center and filing year, for rejected and granted "
           "applications filed 2010–2012. The score ranks rejected applications as it ranks granted ones; at every "
           "decile granted applications receive more prior-art citations, the visibility a grant itself adds.")


if __name__ == "__main__":
    main()
