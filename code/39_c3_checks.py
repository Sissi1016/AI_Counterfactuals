"""Checks of C3 and of the design on the main sample: the eight technology centers, applications filed 2010–2014.

1. The outcome test of Arnold, Dobbie, and Yang (2018), computed with the comparison of Kleinberg et al. (2018,
   Table III) rather than two-stage least squares: within each art unit × year × USPC class cell with at least five
   examiners, examiners are split into leniency quintiles (35's quintiles: leave-one-out leniency, equal caseloads);
   the extra grants of the most lenient quintile over each stricter one are the applications the stricter examiners
   reject, and their mean citations are the difference in citations per application over the difference in grant
   rates, summed over the four stricter quintiles and over cells (cells weighted by the group's applications). The
   same is done separately for female and male first inventors, with every examiner in the quintile set by all of the
   cell's applications; a group whose extra grants are cited more faced a higher bar. 95% intervals from 500
   bootstrap draws of art units (female − male from the same draws). Pooled over the eight centers and by center;
   by center and cohort (2004–2009, 2010–2014, the sample rules of 24); and under alternative designs (cells that
   pass the permutation test below, subclass cells, without the classes most affected by Alice, Mayo, and Myriad).
2. The permutation balance test of Kleinberg et al. (2018): within each cell with at
   least five examiners, an index of characteristics fixed at filing (the cross-fitted grant score of 38, gender,
   entity size, number of inventors, projected on the grant decision within each center) is regressed on examiner
   leniency quintiles and the F statistic compared with 1,000 within-cell permutations; cells with p < 0.05 are
   dropped for the balanced design of 1 (cells with fewer than five examiners are not used by 1 anyway).
3. A version of the test of Frandsen, Lefgren, and Leslie (2023): under exclusion and monotonicity, an examiner's
   mean outcome is a smooth function of the examiner's grant rate, with a non-negative slope for a non-negative
   outcome. Within each art unit, examiner means of grant and of granted × citations (both net of cell means) are
   fitted by a cubic; the misfit is measured against sampling noise (the noise of the examiner's grant rate enters
   through the slope) and calibrated by permutation (citations shuffled among the granted applications of each cell,
   49 times), and the slope is checked over the observed range. Examiners with fewer than 20 applications are left
   out. The chi-square approximation alone over-rejects here: with shuffled citations, where the null holds, the
   statistic is several times its degrees of freedom.
4. The name test of Jensen, Kovács, and Sorenson (2018): among applications whose first inventor is a woman (gender
   from the first name, 03), those with a common first name, whose gender an examiner reads at a glance, against
   those with a rare one; common = at or above the median frequency of the name among U.S. inventors in PatEx
   2004–2018 (set before estimation). Grant on indicators for the two groups (men are the reference), art unit ×
   year fixed effects, errors clustered by examiner.
5. Diagnostics of the 2004–2009 cohort in TC 3600 (why the women's outcome-test result of 2010–2014 is not repeated):
   coverage of gender inference, first stage by gender, citation means, self-citation shares.

6. Robustness of 1 (single-examiner quintiles are common in small cells): cells with at least ten or fifteen
   examiners; terciles and halves instead of quintiles (the same comparison, most lenient group against each
   stricter one); leniency from the same period (04b: the examiner's other applications filed within two years);
   applications docketed to an examiner only once (41). Also the median examiners per group, the share of
   single-examiner groups, and the cells in which women's applications reach both the most lenient and another group
   (only those enter the women's estimate). `--robust` runs only this part.

Runs on CPUs only (DuckDB limited to 16 threads).
Input: data/processed/{all_sample,all_scores,inventor_gender,app_gender,applications_leniency,cites_noself_all,
       patex_application_data,leniency_window,docketed_once}.parquet
Output: output/{c3_checks,c3_robust}.md, output/tables/c3_checks_*.csv
"""
import importlib.util
import pathlib
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from sklearn.linear_model import LogisticRegression


ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
T = OUT / "tables"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
SEED, M, Q, MIN_EXAMINERS, MIN_FLL, B_FLL, B_BOOT = 20261031, 1000, 5, 5, 20, 49, 500


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load() -> pd.DataFrame:
    d = pd.read_parquet(P / "all_sample.parquet")
    s = pd.read_parquet(P / "all_scores.parquet", columns=["application_number", "Train on grant"])
    d = d.merge(s, on="application_number", how="left").reset_index(drop=True)
    for tc in TCS:
        m = (d.tech_center == tc).to_numpy()
        X = d.loc[m, ["Train on grant", "female", "small_entity", "n_inventors"]].to_numpy(float)
        d.loc[m, "index_grant"] = LogisticRegression(max_iter=2000).fit(X, d.loc[m, "granted"]).predict_proba(X)[:, 1]
    return d


def leniency_groups(e: pd.DataFrame, G: int, min_ex: int) -> pd.Series:
    """35's quintiles() with G groups of equal caseload instead of five (same ordering and hole filling)."""
    e = e[e.lenient_grant.notna()]
    ex = e.groupby(["au_class", "examiner_id"]).agg(z=("lenient_grant", "first"), cases=("granted", "size")).reset_index()
    ex["n_ex"] = ex.groupby("au_class").z.transform("size")
    ex = ex[ex.n_ex >= min_ex].sort_values(["au_class", "z", "examiner_id"], kind="mergesort")
    parts = []
    for _, t in ex.groupby("au_class", sort=False):
        cum = t.cases.cumsum().to_numpy() / t.cases.sum()
        q = np.clip(np.ceil(cum * G - 1e-9).astype(int), 1, G)
        n = len(q)
        q[0] = 1
        for i in range(1, n):
            q[i] = max(min(max(q[i], q[i - 1]), q[i - 1] + 1), G - (n - 1 - i))
        parts.append(t.assign(q=q))
    ex = pd.concat(parts)
    return e[["au_class", "examiner_id"]].merge(ex[["au_class", "examiner_id", "q"]], how="left").set_index(e.index).q


def quintile_outcome(x: pd.DataFrame, rng: np.random.Generator, cell: str = "au_class", G: int = 5,
                     min_ex: int = MIN_EXAMINERS) -> dict:
    """The outcome test by Kleinberg et al.'s quintile comparison (see 1 above), for all applications and by gender.
    G < 5: the same comparison with G leniency groups (the most lenient against each stricter one); min_ex: fewest
    examiners in a cell."""
    m35 = load_module("m35", "35_c1_kllm.py")
    x = x[x.lenient_grant.notna()].copy()
    x["au_class"] = x[cell]   # 35's quintiles() works on the column au_class
    x["q"] = m35.quintiles(x, min_ex) if G == 5 else leniency_groups(x, G, min_ex)
    x = x[x.q.notna()]
    ne = x.groupby(["au_class", "q"]).examiner_id.nunique()
    fq = x[x.female == 1].groupby("au_class").q.agg(lambda s: (G in set(s)) and len(set(s) - {G}) > 0)
    diag = dict(examiners_per_group=float(ne.median()), single_examiner_groups=float((ne == 1).mean()),
                cells_female=int(fq.sum()), female_apps_in_cells=float(x[x.female == 1].au_class.isin(
                    fq.index[fq]).mean()))
    x["y"] = np.where(x.granted == 1, x.cites_ns.fillna(0), 0.0)
    cells = pd.Index(sorted(x.au_class.unique()))
    au = x.groupby("au_class").art_unit.first().reindex(cells).to_numpy()
    parts = {}
    for gname, sel in [("All", None), ("Female", 1), ("Male", 0)]:
        z = x if sel is None else x[x.female == sel]
        m = z.groupby(["au_class", "q"]).agg(g=("granted", "mean"), y=("y", "mean")).unstack("q").reindex(cells)
        n = z.groupby("au_class").size().reindex(cells).fillna(0).to_numpy(float)
        D, V = np.zeros(len(cells)), np.zeros(len(cells))
        g5 = m[("g", G)].to_numpy() if ("g", G) in m else np.full(len(cells), np.nan)
        y5 = m[("y", G)].to_numpy() if ("y", G) in m else np.full(len(cells), np.nan)
        for k in range(1, G):
            if ("g", k) not in m:
                continue
            gk, yk = m[("g", k)].to_numpy(), m[("y", k)].to_numpy()
            ok = ~np.isnan(g5) & ~np.isnan(gk)
            D += np.where(ok, g5 - gk, 0.0)
            V += np.where(ok, y5 - yk, 0.0)
        parts[gname] = (n, D, V)
    u, inv = np.unique(au, return_inverse=True)
    mult = np.ones((B_BOOT + 1, len(cells)))
    for b in range(1, B_BOOT + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
    est = {}
    for gname, (n, D, V) in parts.items():
        W = mult * n
        est[gname] = (W @ V) / (W @ D)
        est[gname + "_extra"] = (W @ D) / ((G - 1) * W.sum(1))
    est["fm"] = est["Female"] - est["Male"]
    r = dict(n=len(x), cells=len(cells), extra_grants=est["All_extra"][0], **diag)
    for k, lab in [("All", "marginal"), ("Female", "female"), ("Male", "male"), ("fm", "f_minus_m")]:
        v = est[k]
        r[lab], r[lab + "_lo"], r[lab + "_hi"] = v[0], np.nanpercentile(v[1:], 2.5), np.nanpercentile(v[1:], 97.5)
    return r


def outcome_tests(d: pd.DataFrame, bad: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The main sample by center and pooled, and the pooled sample under alternative designs."""
    rows = [dict(design="all cells", sample=tc, **quintile_outcome(d[d.tech_center == tc], rng)) for tc in TCS]
    rows.append(dict(design="all cells", sample="All centers", **quintile_outcome(d, rng)))
    keep = ~(d.tech_center + "|" + d.au_class).isin(bad.tech_center + "|" + bad.au_class)
    sub = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "uspc_subclass"])
    ds = d.merge(sub, on="application_number", how="left")
    ds["au_subclass"] = ds.au_class + "/" + ds.uspc_subclass.fillna("none")
    cls = d.au_class.str.split("_").str[-1].str.lstrip("0")
    for lab, x, cell in [("cells that pass the permutation test", d[keep], "au_class"),
                         ("subclass cells", ds, "au_subclass"),
                         ("without classes 705, 435, 436, 536", d[~cls.isin(["705", "435", "436", "536"])], "au_class")]:
        for smp, units in [("All centers", TCS), ("3600", ["3600"])]:
            rows.append(dict(design=lab, sample=smp, **quintile_outcome(x[x.tech_center.isin(units)], rng, cell)))
    return pd.DataFrame(rows)


ROBUST = [("ten or more examiners, quintiles", dict(min_ex=10)),   # the main design is section 1
          ("fifteen or more examiners, quintiles", dict(min_ex=15)),
          ("terciles", dict(G=3)), ("halves", dict(G=2)),
          ("leniency from the same period (t - 2 to t + 2)", dict(window=True)),
          ("docketed to an examiner only once", dict(docket=True)),
          # citations are heavy-tailed: a few cells of extremely cited patents carry much of the weight
          ("citations top-coded at the 99th percentile", dict(trim=True))]


def robust_tests(d: pd.DataFrame) -> pd.DataFrame:
    """The outcome test with fewer single-examiner groups (cells with at least ten or fifteen examiners; terciles and
    halves instead of quintiles), with leniency from the same period (04b), and on the applications docketed to an
    examiner only once (41); pooled and by center. Own random stream, so section 1 does not depend on it."""
    rng = np.random.default_rng(SEED + 5)
    win = pd.read_parquet(P / "leniency_window.parquet", columns=["application_number", "lenient_win"])
    d = d.merge(win, on="application_number", how="left")
    dock = P / "docketed_once.parquet"
    if dock.exists():
        d = d.merge(pd.read_parquet(dock, columns=["application_number", "docketed_once"]), on="application_number",
                    how="left")
    rows = []
    for lab, kw in ROBUST:
        x = d
        if kw.get("window"):
            x = d.assign(lenient_grant=d.lenient_win)
        if kw.get("docket"):
            if "docketed_once" not in d:
                continue
            x = d[d.docketed_once.fillna(False).astype(bool)]
        if kw.get("trim"):   # cap at the 99th percentile of granted applications' citations, center × filing year
            cap = d[d.granted == 1].groupby(["tech_center", "filing_year"]).cites_ns.quantile(0.99).rename("cap")
            x = d.join(cap, on=["tech_center", "filing_year"])
            x["cites_ns"] = x[["cites_ns", "cap"]].min(axis=1)
        opts = {k: v for k, v in kw.items() if k in ("G", "min_ex")}
        for smp in ["All centers"] + TCS:
            z = x if smp == "All centers" else x[x.tech_center == smp]
            rows.append(dict(design=lab, sample=smp, **quintile_outcome(z, rng, **opts)))
        print(lab, flush=True)
    return pd.DataFrame(rows)


def robust_report(rb: pd.DataFrame) -> None:
    def q95(r, k):
        return f"{getattr(r, k):.2f} [{getattr(r, k + '_lo'):.2f}, {getattr(r, k + '_hi'):.2f}]"
    L = ["# Robustness of the outcome test (C3): fewer single-examiner groups, same-period leniency, docketed once",
         "", "| Design | Sample | Applications | Cells | Examiners per group (median) | Single-examiner groups | "
         "Cells with women in the most lenient and another group | Women's applications in them | Extra grants (pp) | "
         "All | Female | Male | Female − male |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rb.itertuples():
        L.append(f"| {r.design} | {r.sample} | {r.n:,} | {r.cells:,} | {r.examiners_per_group:.0f} | "
                 f"{100 * r.single_examiner_groups:.0f}% | {r.cells_female:,} | {100 * r.female_apps_in_cells:.0f}% | "
                 f"{100 * r.extra_grants:.1f} | {q95(r, 'marginal')} | {q95(r, 'female')} | {q95(r, 'male')} | "
                 f"{q95(r, 'f_minus_m')} |")
    (OUT / "c3_robust.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cohorts(rng: np.random.Generator) -> pd.DataFrame:
    """By center and cohort, with 24's sample rules: the grant gap and the outcome test."""
    m24 = load_module("m24", "24_human_record.py")
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    h = m24.build(con)
    # a fixed row order: quintiles() takes an examiner's leniency from the first of the examiner's cases in the cell,
    # and leave-one-out leniency differs slightly between them
    h = h.sort_values("application_number", kind="mergesort").reset_index(drop=True)
    rows = []
    for (tc, coh), x in list(h.groupby(["tech_center", "cohort"])) + [(("All centers", c), z)
                                                                   for c, z in h.groupby("cohort")]:
        f = pf.feols("granted ~ female | au_year", data=x, vcov={"CRV1": "examiner_id"})
        rows.append(dict(tech_center=tc, cohort=coh, grant_gap=f.coef()["female"], grant_gap_se=f.se()["female"],
                         **quintile_outcome(x, rng)))
    return pd.DataFrame(rows)


def permutation_p(d: pd.DataFrame) -> pd.DataFrame:
    f_stat = load_module("m18", "18_assignment_checks.py").f_stat
    rng = np.random.default_rng(SEED)
    rows = []
    for (tc, c), x in d[d.lenient_grant.notna()].groupby(["tech_center", "au_class"], sort=True):
        ex = x.groupby("examiner_id").lenient_grant.mean().sort_values(kind="mergesort")
        if len(ex) < MIN_EXAMINERS:
            continue
        g = x.examiner_id.map(pd.Series(np.arange(len(ex)) * Q // len(ex), index=ex.index)).to_numpy()
        y = x.index_grant.to_numpy(float)
        f0 = f_stat(y, g, Q)[0]
        fp = f_stat(y, g[np.argsort(rng.random((M, len(g))), axis=1)], Q)
        rows.append(dict(tech_center=tc, au_class=c, n=len(x), p=(1 + (fp >= f0 - 1e-12).sum()) / (1 + M)))
    return pd.DataFrame(rows)


def fll_stat(x: pd.DataFrame) -> tuple[float, int, float, int]:
    """Within each art unit: examiner means of y and of the grant (net of cell means) fitted by a cubic in the grant
    rate; chi-square of the misfit against sampling noise, and the smallest slope over the 5th–95th percentiles."""
    x = x.assign(dd=x.granted.astype(float))
    for v in ["y", "dd"]:
        x[v + "_r"] = x[v] - x.groupby("au_class")[v].transform("mean")
    stat, df, slope_min, n_ex = 0.0, 0, np.inf, 0
    for _, z in x.groupby("art_unit"):
        e = z.groupby("examiner_id").agg(n=("y_r", "size"), p=("dd_r", "mean"), y=("y_r", "mean"))
        e = e[e.n >= MIN_FLL]
        if len(e) < 8:
            continue
        V = np.vander(e.p.to_numpy(), 4, increasing=True)
        w = e.n.to_numpy(float)
        beta = np.linalg.lstsq(V * np.sqrt(w)[:, None], e.y.to_numpy() * np.sqrt(w), rcond=None)[0]
        slope = beta[1] + 2 * beta[2] * e.p.to_numpy() + 3 * beta[3] * e.p.to_numpy() ** 2
        zz = z[z.examiner_id.isin(e.index)]
        u = zz.y_r - zz.examiner_id.map(pd.Series(slope, index=e.index)) * zz.dd_r
        var = u.groupby(zz.examiner_id).var(ddof=1).reindex(e.index).to_numpy() / e.n.to_numpy()
        stat += float(((e.y.to_numpy() - V @ beta) ** 2 / var).sum())
        df += len(e) - 4
        n_ex += len(e)
        grid = np.linspace(np.percentile(e.p, 5), np.percentile(e.p, 95), 25)
        slope_min = min(slope_min, float((beta[1] + 2 * beta[2] * grid + 3 * beta[3] * grid ** 2).min()))
    return stat, df, slope_min, n_ex


def fll(d: pd.DataFrame) -> pd.DataFrame:
    """A version of Frandsen, Lefgren, and Leslie (2023), calibrated by permutation: the statistic is compared with
    its distribution when citations are shuffled among the granted applications of each cell, which makes exclusion
    and monotonicity hold by construction (the chi-square approximation over-rejects with these heavy-tailed
    outcomes and small examiner samples)."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for gname, sel in [("All", None), ("Female", 1), ("Male", 0)]:
        x = d[d.lenient_grant.notna()] if sel is None else d[d.lenient_grant.notna() & (d.female == sel)]
        x = x.reset_index(drop=True)
        g = (x.granted == 1).to_numpy()
        cites = x.cites_ns.fillna(0).to_numpy(float)
        stat, df, slope_min, n_ex = fll_stat(x.assign(y=np.where(g, cites, 0.0)))
        cell = x.au_class.to_numpy()
        idx = np.where(g)[0]
        order = idx[np.argsort(cell[idx], kind="mergesort")]
        bounds = np.flatnonzero(np.r_[True, cell[order][1:] != cell[order][:-1], True])
        null = []
        for _ in range(B_FLL):
            y = np.zeros(len(x))
            perm = order.copy()
            for a_, b_ in zip(bounds[:-1], bounds[1:]):
                perm[a_:b_] = rng.permutation(order[a_:b_])
            y[order] = cites[perm]
            null.append(fll_stat(x.assign(y=y))[0])
        null = np.array(null)
        rows.append(dict(group=gname, examiners=n_ex, chi2=stat, df=df, ratio=stat / df,
                         null_ratio=float(np.mean(null) / df), p=(1 + (null >= stat).sum()) / (1 + B_FLL),
                         min_slope=slope_min))
    return pd.DataFrame(rows)


def names(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    con.execute("SET threads = 16")
    freq = con.execute(f"""SELECT name_full, count(*) AS freq FROM '{P}/inventor_gender.parquet'
                           WHERE code = 'US' AND name_full IS NOT NULL GROUP BY 1""").df()
    first = con.execute(f"""SELECT application_number, name_full, p_f FROM '{P}/inventor_gender.parquet'
                            WHERE inventor_rank = 1""").df()
    x = d.merge(first, on="application_number", how="left").merge(freq, on="name_full", how="left")
    cut = x.loc[x.female == 1, "freq"].median()
    x["fem_common"] = ((x.female == 1) & (x.freq >= cut)).astype(int)
    x["fem_rare"] = ((x.female == 1) & (x.freq < cut)).astype(int)
    rows = []
    for smp, units in [("All centers", TCS), ("3600", ["3600"]), ("1600", ["1600"])]:
        s = x[x.tech_center.isin(units)]
        m = pf.feols("granted ~ fem_common + fem_rare | au_year", data=s, vcov={"CRV1": "examiner_id"})
        b, se = m.coef(), m.se()
        m2 = pf.feols("granted ~ female + fem_common | au_year", data=s, vcov={"CRV1": "examiner_id"})
        rows.append(dict(sample=smp, n=len(s), n_common=int(s.fem_common.sum()), n_rare=int(s.fem_rare.sum()),
                         gap_common=b["fem_common"], gap_common_se=se["fem_common"], gap_rare=b["fem_rare"],
                         gap_rare_se=se["fem_rare"], diff=m2.coef()["fem_common"], diff_se=m2.se()["fem_common"]))
    info = dict(cut=float(cut), common_example=", ".join(x[x.fem_common == 1].name_full.value_counts().head(5).index),
                rare_example=", ".join(x[x.fem_rare == 1].name_full.value_counts().head(5).index))
    return pd.DataFrame(rows), info


def cohort_diagnostics() -> pd.DataFrame:
    m24 = load_module("m24", "24_human_record.py")
    con = duckdb.connect()
    con.execute("SET threads = 16; SET memory_limit = '100GB'")
    d = m24.build(con)
    d = d[d.tech_center == "3600"]
    g = con.execute(f"""SELECT a.filing_year, avg((g.first_inventor_gender IN ('F', 'M'))::DOUBLE) AS gender_known,
                               avg((g.first_inventor_country = 'US')::DOUBLE) AS us_first
                        FROM '{P}/applications_leniency.parquet' a JOIN '{P}/app_gender.parquet' g
                        USING (application_number)
                        WHERE a.tech_center = '3600' AND a.in_hupd AND NOT a.continuation
                          AND a.filing_year BETWEEN 2004 AND 2014 GROUP BY 1""").df()
    g["cohort"] = np.where(g.filing_year <= 2009, "2004–2009", "2010–2014")
    rows = []
    for coh, s in d.groupby("cohort"):
        gr = s[s.granted == 1]
        fs = {}
        for sex, v in [("female", 1), ("male", 0)]:
            z = s[s.female == v]
            zz = z.assign(dr=z.granted - z.groupby("au_class").granted.transform("mean"),
                          zr=z.lenient_grant - z.groupby("au_class").lenient_grant.transform("mean"))
            fs[sex] = float((zz.dr * zz.zr).sum() / (zz.zr ** 2).sum())
        gg = g[g.cohort == coh]
        rows.append(dict(cohort=coh, applications=len(s), female_share=s.female.mean(), grant_rate=s.granted.mean(),
                         grant_rate_female=s[s.female == 1].granted.mean(), grant_rate_male=s[s.female == 0].granted.mean(),
                         first_stage_female=fs["female"], first_stage_male=fs["male"],
                         cites_female=gr[gr.female == 1].cites_ns.mean(), cites_male=gr[gr.female == 0].cites_ns.mean(),
                         gender_known=gg.gender_known.mean(), us_first=gg.us_first.mean()))
    return pd.DataFrame(rows)


def ci(b, se, nd=2, s=1.0):
    return f"{s * b:.{nd}f} [{s * (b - 1.96 * se):.{nd}f}, {s * (b + 1.96 * se):.{nd}f}]"


def main() -> None:
    warnings.filterwarnings("ignore")
    d = load()
    if "--robust" in __import__("sys").argv:   # only the robustness of the outcome test
        rb = robust_tests(d)
        rb.to_csv(T / "c3_checks_robust.csv", index=False)
        robust_report(rb)
        return
    perm = permutation_p(d)
    perm.to_csv(T / "c3_checks_permutation.csv", index=False)
    bad = perm[perm.p < 0.05]
    keep = ~(d.tech_center + "|" + d.au_class).isin(bad.tech_center + "|" + bad.au_class)
    rng = np.random.default_rng(SEED + 2)
    ot = outcome_tests(d, bad, rng)
    ot.to_csv(T / "c3_checks_outcome_test.csv", index=False)
    oc = cohorts(rng)
    oc.to_csv(T / "c3_checks_outcome_cohorts.csv", index=False)
    fl = fll(d)
    fl.to_csv(T / "c3_checks_fll.csv", index=False)
    nm, info = names(d)
    nm.to_csv(T / "c3_checks_names.csv", index=False)
    cd = cohort_diagnostics()
    cd.to_csv(T / "c3_checks_cohorts.csv", index=False)

    L = ["# C3 and design checks: eight technology centers, applications filed 2010–2014", "",
         f"Permutation test (grant index, art unit × year × class cells with at least five examiners): {len(perm):,} "
         f"cells tested, {100 * (perm.p < 0.05).mean():.1f}% with p < 0.05 (5% expected); dropped for the balanced "
         f"version: {len(bad):,} cells, {int((~keep).sum()):,} applications.", "",
         "## 1. Outcome test (Arnold, Dobbie, and Yang 2018) by the quintile comparison of Kleinberg et al. (2018): "
         "citations of the extra grants of the most lenient quintile, by gender of the first inventor", "",
         "| Design | Sample | Applications | Cells | Extra grants (pp) | All | Female | Male | Female − male |",
         "|---|---|---|---|---|---|---|---|---|"]

    def q95(r, k):
        return f"{getattr(r, k):.2f} [{getattr(r, k + '_lo'):.2f}, {getattr(r, k + '_hi'):.2f}]"
    for r in ot.itertuples():
        L.append(f"| {r.design} | {r.sample} | {r.n:,} | {r.cells:,} | {100 * r.extra_grants:.1f} | {q95(r, 'marginal')} | "
                 f"{q95(r, 'female')} | {q95(r, 'male')} | {q95(r, 'f_minus_m')} |")
    L += ["", "By center and cohort (sample rules of 24): grant gap (female − male, pp, art unit × year fixed effects) and "
              "the outcome test", "", "| Center | Cohort | Applications | Grant gap | All | Female − male |",
          "|---|---|---|---|---|---|"]
    for r in oc.itertuples():
        L.append(f"| {r.tech_center} | {r.cohort} | {r.n:,} | {ci(r.grant_gap, r.grant_gap_se, 1, 100)} | "
                 f"{q95(r, 'marginal')} | {q95(r, 'f_minus_m')} |")
    L += ["", "## 3. Frandsen, Lefgren, and Leslie (2023), calibrated by permutation: examiner means against a cubic in "
              "the examiner's grant rate, within art unit", "",
          f"p: share of {B_FLL} within-cell shufflings of citations among granted applications (the null holds by "
          "construction) with a statistic at least as large.", "",
          "| Group | Examiners | Chi-square / df | Same under the null (mean) | p |", "|---|---|---|---|---|"]
    for r in fl.itertuples():
        L.append(f"| {r.group} | {r.examiners:,} | {r.ratio:.2f} | {r.null_ratio:.2f} | {r.p:.2f} |")
    L += ["", "## 4. Jensen, Kovács, and Sorenson (2018): women with common and with rare first names", "",
          f"Common: the name's frequency among U.S. inventors in PatEx 2004–2018 at or above the median for female "
          f"first inventors ({info['cut']:,.0f}); most frequent common names: {info['common_example']}; rare: "
          f"{info['rare_example']}. Grant gap against men (pp), art unit × year fixed effects, clustered by examiner.",
          "", "| Sample | Applications | Women, common | Women, rare | Gap, common | Gap, rare | Common − rare |",
          "|---|---|---|---|---|---|---|"]
    for r in nm.itertuples():
        L.append(f"| {r.sample} | {r.n:,} | {r.n_common:,} | {r.n_rare:,} | {ci(r.gap_common, r.gap_common_se, 1, 100)} | "
                 f"{ci(r.gap_rare, r.gap_rare_se, 1, 100)} | {ci(r.diff, r.diff_se, 1, 100)} |")
    L += ["", "## 5. TC 3600 by cohort (diagnostics)", "",
          "| Cohort | Applications | Female share | Grant rate F / M | First stage F / M | Citations F / M (granted) | "
          "Gender inferred | U.S. first inventor |", "|---|---|---|---|---|---|---|---|"]
    for r in cd.itertuples():
        L.append(f"| {r.cohort} | {r.applications:,} | {100 * r.female_share:.1f}% | {100 * r.grant_rate_female:.1f}% / "
                 f"{100 * r.grant_rate_male:.1f}% | {r.first_stage_female:.2f} / {r.first_stage_male:.2f} | "
                 f"{r.cites_female:.2f} / {r.cites_male:.2f} | {100 * r.gender_known:.1f}% | {100 * r.us_first:.1f}% |")
    (OUT / "c3_checks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    rb = robust_tests(d)
    rb.to_csv(T / "c3_checks_robust.csv", index=False)
    robust_report(rb)


if __name__ == "__main__":
    main()
