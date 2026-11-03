"""C3 on the main sample (eight technology centers, applications filed 2010–2014): tables, checks, and the main-text
figure of the gender gap in grant decisions by evaluator.

Evaluators and decisions are those of 38 (cross-fitted text models in all eight centers) and 40 (Qwen3-8B zero-shot in
all eight centers): each evaluator grants the examiners' share of applications within each center; a gap is the
female − male (or group − White) difference in decisions within art unit × filing-year cells, errors clustered by
examiner (cfe.estimation, which matches pyfixest's CRV1).

1. Gender gap by center (Table A20 rebuilt; Figure 2 of the main text): read from 38's all_centers_summary.csv and
   40's llm_all_centers.csv, not re-estimated. The decisions are rebuilt here (38: score at or above the center's
   (1 − grant rate) quantile, as cfe.check; 40: top share by score, ties broken by 40's random draw) and the gaps
   recomputed only to check that they reproduce those files.
2. All evaluators in TC 3600 and pooled, from that one run, plus the evaluators that exist only in TC 3600 and 1600,
   scored on the same TC 3600 applications: Qwen3-8B with lenient / strict examiners' examples (14's prompts;
   decisions by 40's rule) and Qwen3-8B fine-tuned with LoRA (30's ft_preds; decisions by cfe.check's rule, as 30).
3. Race and ethnicity of the first inventor, by 33's rule (surname in the 2010 Census table, group share >= 60%),
   pooled and in TC 3600: each group against White first inventors, the gap in grants and in each evaluator's
   decisions (decisions made on all of the center's applications, as for gender; gaps estimated on the group's and
   White inventors' applications); the gap on the probability scale (each evaluator's score mapped to a grant
   probability within center by a one-variable logistic fit, so no threshold); and 39's outcome test by group
   (Kleinberg et al.'s quintile comparison: mean citations of the most lenient quintile's extra grants; quintiles set
   by all of the cell's applications; group − White, 95% interval from 500 bootstrap draws of art units).
   The same probability-scale gaps for gender.
4. Why the citation-trained evaluator keeps a gender gap in TC 1600 and 3700 but not in 3600: within art unit × year
   cells, the female − male difference in the citations of granted patents (log(1 + citations), the evaluator's
   training target), in prior-art citations of all published applications (06c; rejected ones included), and in the
   evaluator's predicted log citations; the same within art unit × year × USPC class cells (field mix within the art
   unit); the gap in actual citations at the same predicted citations; and the names version of the evaluator.
5. Figure: output/figures/journal/fig_main_c3.{pdf,png} (style helpers of 20, Okabe–Ito colours). a: the female −
   male gap in decisions by center (1); b: group − White, all eight centers (3). Hollow markers: the gap on the
   probability scale (3) for the grant-trained text model and the zero-shot LLM, joined to the decision gap.
6. Robustness (`--lite`): (i) a temporal split: within each center, the grant-, first-action- and citation-trained
   evaluators of 38 (same inputs, models, and penalty search, through 38's logit_task / ridge_task) are trained on
   applications filed 2010–2012 and score those filed 2013–2014; each grants the examiners' 2013–2014 share within
   each center, and the gender gap on 2013–2014 is set next to the cross-fitted evaluators' gap (38's scores, the same
   decision rule on the same applications) and the examiners'. (ii) Which inputs 38's evaluators use: the md quotes
   38's code lines; no applicant field (entity size, number of inventors) is an input of the text-only evaluators.
   Writes output/c3x_lite.md, output/tables/c3x_lite_temporal.csv, output/latex/c3x_lite_tab_temporal.tex, the
   scores data/processed/c3x_lite_temporal_scores.parquet, and adds \\xCtLite... macros to c3x_numbers.tex
   (about ten minutes on four threads).

Runs on CPUs (four threads) in a few minutes.
Run: uv run python code/44_c3_extra.py            (1–5)
     uv run python code/44_c3_extra.py --figure   (only the figure, from the saved CSVs)
     uv run python code/44_c3_extra.py --lite     (6; the macros of 1–5 are kept)
Input: data/processed/{all_sample,all_scores,llm_scores_allpilot,llm_scores_allcenters,ft_preds,
       patex_all_inventors,census_surnames_2010,prior_art_cites}.parquet, output/tables/{all_centers_summary,
       llm_all_centers}.csv
Output: output/c3x.md, output/tables/c3x_*.csv, output/latex/c3x_tab_*.tex, output/latex/c3x_numbers.tex (\\xCt...),
        output/figures/journal/fig_main_c3.{pdf,png}
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from cfe.estimation import cluster_se, tsls  # noqa: E402

P = ROOT / "data/processed"
OUT = ROOT / "output"
T = OUT / "tables"
TEX = OUT / "latex"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
ORDER = ["All centers", "3700", "3600", "2800", "2600", "2400", "2100", "1700", "1600"]  # rows of tables and figure
SEED_LLM = 20261028   # 10's SEED: 40's tie-break draw
SEED = 20261101       # this script's bootstrap
B_BOOT, MIN_EX, THRESHOLD = 500, 5, 60.0   # 39's draws and cells; 33's surname threshold
G, GN, FA, C, CN = ("Train on grant", "Train on grant + names", "Train on first action",
                    "Train on citations (granted only)", "Train on citations + names")
LLM0, LLML, LLMS = "LLM, zero-shot", "LLM, lenient examiners' examples", "LLM, strict examiners' examples"
FTG, FTC = "LLM fine-tuned on grant", "LLM fine-tuned on citations"
EMB = [G, GN, FA, C, CN]
LABEL = {"Examiners": "Examiners", G: "Trained on grant", GN: "Trained on grant + names",
         FA: "Trained on first action", C: "Trained on citations", CN: "Trained on citations + names",
         LLM0: "LLM, zero-shot", LLML: "LLM, lenient examiners' examples", LLMS: "LLM, strict examiners' examples",
         FTG: "LLM fine-tuned on grant", FTC: "LLM fine-tuned on citations"}
ETH = {"hispanic": ("Hispanic", "pcthispanic", "Hisp"), "asian": ("Asian", "pctapi", "Asian"),
       "black": ("Black", "pctblack", "Black"), "white": ("White", "pctwhite", "White")}
SMP = {"All centers": ("All", TCS), "3600": ("ThreeSix", ["3600"])}
TC_SHORT = {"1600": "Biotechnology, organic chemistry", "1700": "Chemical and materials engineering",
            "2100": "Computer architecture, software", "2400": "Networking, cable, security",
            "2600": "Communications", "2800": "Semiconductors, electrical systems",
            "3600": "Transportation, e-commerce, business methods",
            "3700": "Mechanical engineering, medical devices"}
CW = {"1600": "OneSix", "1700": "OneSeven", "2100": "TwoOne", "2400": "TwoFour", "2600": "TwoSix",
      "2800": "TwoEight", "3600": "ThreeSix", "3700": "ThreeSeven", "All centers": "All"}


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv", dtype={"sample": str, "tech_center": str})


# ---------------------------------------------------------------- data and decisions

def load() -> pd.DataFrame:
    """The main sample (38's all_sample, application_number order) with every evaluator's score and decision."""
    d = pd.read_parquet(P / "all_sample.parquet")
    d = d.merge(pd.read_parquet(P / "all_scores.parquet"), on="application_number", how="left", validate="1:1")
    floor = 1e-12   # as in 10, 14, 40
    pilot = pd.read_parquet(P / "llm_scores_allpilot.parquet")
    zero = pd.concat([pilot[pilot.cond == "zero"], pd.read_parquet(P / "llm_scores_allcenters.parquet")])
    for x in [zero, pilot]:
        x["score"] = np.log(x.p_yes.clip(lower=floor)) - np.log(x.p_no.clip(lower=floor))
    d = d.merge(zero[["application_number", "score"]].rename(columns={"score": LLM0}), on="application_number",
                how="left", validate="1:1")
    few = pilot[pilot.cond != "zero"].pivot(index="application_number", columns="cond", values="score")
    few = few.rename(columns={"few_lenient": LLML, "few_strict": LLMS})[[LLML, LLMS]].reset_index()
    d = d.merge(few, on="application_number", how="left", validate="1:1")
    ft = pd.read_parquet(P / "ft_preds.parquet", columns=["application_number", "grant", "cites"])
    d = d.merge(ft.rename(columns={"grant": FTG, "cites": FTC}), on="application_number", how="left", validate="1:1")
    assert d[EMB + [LLM0]].notna().all().all(), "missing scores"
    assert d.application_number.is_monotonic_increasing
    tb = np.random.default_rng(SEED_LLM).random(len(d))   # 40 draws it over the same rows in the same order
    for tc in TCS:
        m = (d.tech_center == tc).to_numpy()
        r = d.loc[m, "granted"].mean()
        for c in EMB + [FTG, FTC]:   # cfe.check's decision_gaps (38, 30)
            v = d.loc[m, c]
            if v.notna().all():
                d.loc[m, f"dec::{c}"] = (v >= v.quantile(1 - r)).astype(float)
        for c in [LLM0, LLML, LLMS]:   # 40's rule
            v = d.loc[m, c].to_numpy()
            if np.isnan(v).any():
                continue
            order = np.lexsort((tb[m], -v))
            dec = np.zeros(m.sum())
            dec[order[: int(round(r * m.sum()))]] = 1
            d.loc[m, f"dec::{c}"] = dec
    return d


def calibrate(d: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Probability scale: each score mapped to a grant probability within its center by a logistic fit of the grant on
    the score (two parameters per center and score, so no threshold and no amplification)."""
    from sklearn.linear_model import LogisticRegression
    for c in cols:
        for tc in TCS:
            m = (d.tech_center == tc).to_numpy()
            x = d.loc[m, c].to_numpy(float)
            if np.isnan(x).any():
                continue
            mu, sd = x.mean(), x.std()
            z = ((x - mu) / sd)[:, None]
            d.loc[m, f"p::{c}"] = LogisticRegression(C=1e6, max_iter=1000).fit(
                z, d.loc[m, "granted"]).predict_proba(z)[:, 1]
    return d


def gaps(s: pd.DataFrame, cols: list, grp: str = "female", cell: str = "au_year", W: list | None = None) -> pd.DataFrame:
    """Coefficient on the group indicator for each column, absorbing cell fixed effects, errors clustered by examiner."""
    Wm = s[W].to_numpy(float) if W else None
    b, psi = tsls(s[cols].to_numpy(float), s[grp].to_numpy(float), s[grp].to_numpy(float), s[cell], Wm)
    se = cluster_se(psi, s.examiner_id)
    return pd.DataFrame({"series": cols, "gap": b, "se": se})


def check_reproduction(d: pd.DataFrame) -> pd.DataFrame:
    """The decisions rebuilt here against 38's and 40's saved gaps (should agree to rounding error)."""
    S = rd("all_centers_summary").set_index("sample")
    L40 = rd("llm_all_centers").set_index("sample")
    rows = []
    for smp in ORDER:
        s = d if smp == "All centers" else d[d.tech_center == smp]
        g = gaps(s, ["granted"] + [f"dec::{c}" for c in EMB + [LLM0]]).set_index("series")
        rows.append(dict(sample=smp, human=g.loc["granted", "gap"] - S.loc[smp, "human_gap"],
                         llm=g.loc[f"dec::{LLM0}", "gap"] - L40.loc[smp, "llm_gap"],
                         llm_se=g.loc[f"dec::{LLM0}", "se"] - L40.loc[smp, "llm_gap_se"],
                         **{c: g.loc[f"dec::{c}", "gap"] - S.loc[smp, f"gap::{c}"] for c in EMB},
                         **{c + " (se)": g.loc[f"dec::{c}", "se"] - S.loc[smp, f"gap_se::{c}"] for c in EMB}))
    r = pd.DataFrame(rows)
    worst = r.drop(columns="sample").abs().max().max()
    print(f"reproduction of 38 and 40: largest absolute difference {worst:.2e}")
    assert worst < 1e-6, r
    return r


# ---------------------------------------------------------------- 1–2. gender

def gender_tables(d: pd.DataFrame) -> pd.DataFrame:
    """All evaluators, pooled and TC 3600 (and every center for the CSV): gap in decisions and on the probability
    scale. The 38/40 evaluators' decision gaps are taken from their files; the others are computed here."""
    S = rd("all_centers_summary").set_index("sample")
    L40 = rd("llm_all_centers").set_index("sample")
    rows = []
    for smp in ORDER:
        s = d if smp == "All centers" else d[d.tech_center == smp]
        pcols = [f"p::{c}" for c in [G, FA, C, LLM0] if s[f"p::{c}"].notna().all()]
        pg = gaps(s, pcols).set_index("series")
        h = (S.loc[smp, "human_gap"], S.loc[smp, "human_gap_se"])
        rows.append(dict(sample=smp, series="Examiners", gap=h[0], se=h[1], ratio=1.0, source="38",
                         p_gap=h[0], p_se=h[1]))
        for c in EMB:
            rows.append(dict(sample=smp, series=c, gap=S.loc[smp, f"gap::{c}"], se=S.loc[smp, f"gap_se::{c}"],
                             ratio=S.loc[smp, f"gap::{c}"] / h[0], source="38",
                             p_gap=pg.gap.get(f"p::{c}", np.nan), p_se=pg.se.get(f"p::{c}", np.nan)))
        rows.append(dict(sample=smp, series=LLM0, gap=L40.loc[smp, "llm_gap"], se=L40.loc[smp, "llm_gap_se"],
                         ratio=L40.loc[smp, "llm_gap"] / h[0], source="40",
                         p_gap=pg.gap.get(f"p::{LLM0}", np.nan), p_se=pg.se.get(f"p::{LLM0}", np.nan)))
        extra = [c for c in [LLML, LLMS, FTG, FTC] if smp != "All centers" and s[f"dec::{c}"].notna().all()]
        if extra:
            g = gaps(s, [f"dec::{c}" for c in extra]).set_index("series")
            for c in extra:
                rows.append(dict(sample=smp, series=c, gap=g.loc[f"dec::{c}", "gap"], se=g.loc[f"dec::{c}", "se"],
                                 ratio=g.loc[f"dec::{c}", "gap"] / h[0], source="44 (pilot-only evaluator)",
                                 p_gap=np.nan, p_se=np.nan))
    r = pd.DataFrame(rows)
    r.to_csv(T / "c3x_gaps.csv", index=False)
    return r


# ---------------------------------------------------------------- 3. race and ethnicity

def ethnicity_groups(d: pd.DataFrame) -> pd.Series:
    """33's rule: the first inventor's surname (letters only) in the 2010 Census table, a group's share >= 60%."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.register("s", d[["application_number"]])
    sn = con.execute(f"""
    SELECT i.application_number, regexp_replace(upper(i.inventor_name_last), '[^A-Z]', '', 'g') AS surname
    FROM '{P}/patex_all_inventors.parquet' i JOIN s USING (application_number)
    WHERE i.inventor_rank = '1'""").df().drop_duplicates("application_number")
    cs = pd.read_parquet(P / "census_surnames_2010.parquet")
    cs["surname"] = cs["name"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    cs = cs.drop_duplicates("surname")
    sn = sn.merge(cs[["surname"] + [v[1] for v in ETH.values()]], on="surname", how="left")
    sn["group"] = np.where(sn.pctwhite.notna() | sn.pctapi.notna(), "unclassified", "unmatched")
    for key in ["asian", "hispanic", "black", "white"]:   # 33's order; at most one share can reach 60%
        sn.loc[sn[ETH[key][1]] >= THRESHOLD, "group"] = key
    return d[["application_number"]].merge(sn[["application_number", "group"]], how="left").group.fillna(
        "unmatched").set_axis(d.index)


def outcome_by_group(x: pd.DataFrame, groups: dict, pairs: list, rng: np.random.Generator) -> dict:
    """39's quintile_outcome() for any set of groups: x must carry the quintile q (set by all of the cell's
    applications). groups: {name: boolean mask over x}; pairs: [(group, reference)] differences, same draws."""
    x = x[x.q.notna()]
    y = np.where(x.granted == 1, x.cites_ns.fillna(0), 0.0)
    x = x.assign(y=y)
    cells = pd.Index(sorted(x.au_class.unique()))
    au = x.groupby("au_class").art_unit.first().reindex(cells).to_numpy()
    u, inv = np.unique(au, return_inverse=True)
    mult = np.ones((B_BOOT + 1, len(cells)))
    for b in range(1, B_BOOT + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
    est, info = {}, {}
    for gname, mask in groups.items():
        z = x[mask[x.index]] if mask is not None else x
        m = z.groupby(["au_class", "q"]).agg(g=("granted", "mean"), y=("y", "mean")).unstack("q").reindex(cells)
        n = z.groupby("au_class").size().reindex(cells).fillna(0).to_numpy(float)
        D, V = np.zeros(len(cells)), np.zeros(len(cells))
        g5 = m[("g", 5)].to_numpy() if ("g", 5) in m else np.full(len(cells), np.nan)
        y5 = m[("y", 5)].to_numpy() if ("y", 5) in m else np.full(len(cells), np.nan)
        for k in range(1, 5):
            if ("g", k) not in m:
                continue
            gk, yk = m[("g", k)].to_numpy(), m[("y", k)].to_numpy()
            ok = ~np.isnan(g5) & ~np.isnan(gk)
            D += np.where(ok, g5 - gk, 0.0)
            V += np.where(ok, y5 - yk, 0.0)
        Wt = mult * n
        est[gname] = (Wt @ V) / (Wt @ D)
        both = (~np.isnan(g5)) & (D != 0)
        info[gname] = dict(n=len(z), cells_used=int(((n > 0) & both).sum()),
                           apps_in_cells=float(n[both].sum() / max(n.sum(), 1)))
    for a, b in pairs:
        est[f"{a} − {b}"] = est[a] - est[b]
    out = {}
    for k, v in est.items():
        out[k] = dict(estimate=v[0], lo=np.nanpercentile(v[1:], 2.5), hi=np.nanpercentile(v[1:], 97.5),
                      **info.get(k, {}))
    return out


def ethnicity(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m35 = load_module("m35", "35_c1_kllm.py")
    d = d.copy()
    d["group"] = ethnicity_groups(d)
    share = (d.groupby("tech_center").group.value_counts(normalize=True).unstack().reindex(columns=[
        "white", "asian", "hispanic", "black", "unclassified", "unmatched"]).fillna(0))
    share.loc["All centers"] = d.group.value_counts(normalize=True).reindex(share.columns).fillna(0)
    share["applications"] = d.groupby("tech_center").size().reindex(share.index).fillna(len(d)).astype(int)
    share.reset_index().rename(columns={"index": "sample"}).to_csv(T / "c3x_ethnicity_shares.csv", index=False)

    # decision gaps and probability-scale gaps against White
    ev = [G, C, LLM0]
    rows = []
    for smp, (_, units) in SMP.items():
        for key in ["hispanic", "asian", "black"]:
            s = d[d.tech_center.isin(units) & d.group.isin([key, "white"])].assign(grp=lambda x: (x.group == key) * 1.0)
            g = gaps(s, ["granted"] + [f"dec::{c}" for c in ev] + [f"p::{c}" for c in ev], grp="grp").set_index(
                "series")
            r = dict(sample=smp, group=ETH[key][0], n_group=int(s.grp.sum()), n_white=int((1 - s.grp).sum()),
                     examiners_group=int(s[s.grp == 1].examiner_id.nunique()),
                     exam_gap=g.loc["granted", "gap"], exam_se=g.loc["granted", "se"])
            for c in ev:
                r[f"gap::{c}"], r[f"se::{c}"] = g.loc[f"dec::{c}", "gap"], g.loc[f"dec::{c}", "se"]
                r[f"pgap::{c}"], r[f"pse::{c}"] = g.loc[f"p::{c}", "gap"], g.loc[f"p::{c}", "se"]
            rows.append(r)
    eg = pd.DataFrame(rows)

    # outcome test by group (quintiles set by all of the cell's applications, as in 39)
    x = d[d.lenient_grant.notna()].copy()
    x["q"] = m35.quintiles(x, MIN_EX)
    rng = np.random.default_rng(SEED)
    orows = []
    for smp, (_, units) in SMP.items():
        z = x[x.tech_center.isin(units)]
        grp = {"All": None, "Female": z.female == 1, "Male": z.female == 0,
               **{ETH[k][0]: z.group == k for k in ETH}}
        res = outcome_by_group(z, grp, [("Female", "Male")] + [(ETH[k][0], "White") for k in
                                                              ["hispanic", "asian", "black"]], rng)
        for k, v in res.items():
            orows.append(dict(sample=smp, group=k, **v))
    ot = pd.DataFrame(orows)
    # 39's point estimates for gender, as a check of the reimplementation
    o39 = rd("c3_checks_outcome_test")
    o39 = o39[o39.design == "all cells"].set_index("sample")
    for smp in SMP:
        for k, col in [("All", "marginal"), ("Female", "female"), ("Male", "male")]:
            mine = ot[(ot["sample"] == smp) & (ot.group == k)].estimate.iloc[0]
            assert abs(mine - o39.loc[smp, col]) < 1e-9, (smp, k, mine, o39.loc[smp, col])
    print("outcome test: 39's point estimates for gender reproduced")
    eg.to_csv(T / "c3x_ethnicity_gaps.csv", index=False)
    ot.to_csv(T / "c3x_ethnicity_outcome.csv", index=False)
    return share, eg, ot


# ---------------------------------------------------------------- 4. the citation-trained evaluator by center

def why_citations(d: pd.DataFrame) -> pd.DataFrame:
    pa = pd.read_parquet(P / "prior_art_cites.parquet", columns=["application_number", "published", "prior_art_cites"])
    d = d.merge(pa, on="application_number", how="left")
    d["log_cites"] = np.log1p(d.cites_ns)
    d["log_pa"] = np.log1p(d.prior_art_cites)
    # decisions at each art unit × year's grant rate (no threshold across art units): the top share within the cell,
    # ties broken at random
    tb = np.random.default_rng(SEED + 1).random(len(d))
    for c in [C, G]:
        key = pd.DataFrame({"cell": d.au_year, "s": -d[c].to_numpy(), "tb": tb}).sort_values(["cell", "s", "tb"])
        rank = key.groupby("cell").cumcount().reindex(d.index).to_numpy()
        k = np.round(d.groupby("au_year").granted.transform("mean") * d.groupby("au_year").granted.transform("size"))
        d[f"decau::{c}"] = (rank < k.to_numpy()).astype(float)
    rows = []
    for smp in ORDER:
        s = d if smp == "All centers" else d[d.tech_center == smp]
        g = s[s.granted == 1]
        pub = s[s.published.fillna(False).astype(bool)]
        sd = s.groupby("tech_center")[C].transform("std")
        s = s.assign(z_cites=s[C] / sd, z_grant=s[G] / s.groupby("tech_center")[G].transform("std"))
        r = dict(sample=smp, applications=len(s), female=s.female.mean(), granted=len(g),
                 female_granted=g.female.mean(), mean_cites=g.cites_ns.mean())
        for lab, x, cols, cell in [
                ("granted", g, ["log_cites", "cites_ns", C, CN], "au_year"),
                ("granted_class", g, ["log_cites", "cites_ns", C], "au_class"),
                ("published", pub, ["log_pa", "prior_art_cites"], "au_year"),
                ("published_class", pub, ["log_pa"], "au_class"),
                ("all", s, [C, CN, "z_cites", "z_grant", f"dec::{C}", f"dec::{G}", f"decau::{C}", f"decau::{G}",
                            "granted"], "au_year"),
                ("all_class", s, ["z_cites", f"dec::{C}", f"dec::{G}", "granted"], "au_class")]:
            e = gaps(x, cols, cell=cell).set_index("series")
            for c in cols:
                r[f"{lab}::{c}"], r[f"{lab}_se::{c}"] = e.loc[c, "gap"], e.loc[c, "se"]
        # prior-art citations net of the grant itself (a grant makes an application more visible)
        e = gaps(pub, ["log_pa"], W=["granted"]).set_index("series")
        r["published_netgrant::log_pa"], r["published_netgrant_se::log_pa"] = e.loc["log_pa", "gap"], e.loc[
            "log_pa", "se"]
        # actual citations at the same predicted citations (text-only evaluator), among granted
        e = gaps(g, ["log_cites"], W=[C]).set_index("series")
        r["resid::log_cites"], r["resid_se::log_cites"] = e.loc["log_cites", "gap"], e.loc["log_cites", "se"]
        # calibration slope of the evaluator on granted applications (within cells)
        gg = g.copy()
        for v in ["log_cites", C]:
            gg[v + "_dm"] = gg[v] - gg.groupby("au_year")[v].transform("mean")
        r["slope"] = (gg[C + "_dm"] * gg["log_cites_dm"]).sum() / (gg[C + "_dm"] ** 2).sum()
        rows.append(r)
    w = pd.DataFrame(rows)
    w.to_csv(T / "c3x_why_citations.csv", index=False)

    # the USPC classes behind the female share, TC 1600 and 3700 (granted applications)
    cls = []
    for tc in ["1600", "3700", "3600"]:
        g = d[(d.tech_center == tc) & (d.granted == 1)].copy()
        g["uspc"] = g.au_class.str.split("_").str[-1]
        t = g.groupby("uspc").agg(granted=("female", "size"), female=("female", "mean"),
                                  mean_cites=("cites_ns", "mean")).reset_index()
        t = t[t.granted >= 300].sort_values("granted", ascending=False)
        t.insert(0, "tech_center", tc)
        cls.append(t)
    pd.concat(cls).to_csv(T / "c3x_why_citations_classes.csv", index=False)
    return w


def a21_changes(gp: pd.DataFrame) -> pd.DataFrame:
    """Every cell of the old Table A21 (29's cit_tab_c3_gaps.tex, from 21's two-center run) against the unified
    numbers: TC 3600 from this table, TC 1600 from Table A20 (38, 40). Compared at the displayed precision."""
    old = rd("citations_gaps")
    gx = gp.set_index(["sample", "series"])
    rows = []
    for k in ["Examiners", G, GN, FA, C, CN, LLM0, LLML, LLMS]:
        ser = "Human decision makers" if k == "Examiners" else k
        for smp in ["3600", "1600"]:
            o = old[(old["sample"] == smp) & (old.series == ser)].iloc[0]
            n = gx.loc[(smp, k)] if (smp, k) in gx.index else None
            fo = f"{100 * o.gap:.1f} [{100 * (o.gap - 1.96 * o.se):.1f}, {100 * (o.gap + 1.96 * o.se):.1f}]"
            fn = "dropped" if n is None else \
                f"{100 * n.gap:.1f} [{100 * (n.gap - 1.96 * n.se):.1f}, {100 * (n.gap + 1.96 * n.se):.1f}]"
            r = dict(sample=smp, evaluator=LABEL[k], old=fo, new=fn, changed=fo != fn)
            if smp == "3600":
                r.update(old_ratio=f"{o.ratio:.2f}", new_ratio=f"{n.ratio:.2f}")
                r["changed"] = r["changed"] or r["old_ratio"] != r["new_ratio"]
            rows.append(r)
    x = pd.DataFrame(rows)
    x.to_csv(T / "c3x_a21_changes.csv", index=False)
    return x


# ---------------------------------------------------------------- 5. figure

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7"}
FIG_SERIES = [("Examiners", "Examiners", OI["black"], "o"),
              (G, "Text model trained on grants", OI["vermillion"], "s"),
              (C, "Text model trained on citations", OI["blue"], "^"),
              (LLM0, "LLM, zero-shot", OI["purple"], "D")]
# Hollow markers (gap in the predicted grant probability, before the cutoff) only for the evaluators whose score is a
# grant probability; the citation-trained score is not (section 3), and the examiners have no score.
FIG_HOLLOW = [G, LLM0]
FIG_ETH = ["Hispanic", "Black", "Asian"]   # rows of panel b
FIG_XLIM = (-19.5, 6.0)                    # both panels: same scale, zero lines aligned


def figure() -> None:
    """Figure 2 of the main text. a: female − male gap in grant decisions by technology center (38's and 40's files),
    with the gap in predicted grant probability (c3x_gaps.csv, p_gap) as hollow markers joined to the decision gap;
    b: race and ethnicity of the first inventor against White, all eight centers (c3x_ethnicity_gaps.csv). Reads only
    output/tables; prints every value it draws."""
    J = load_module("jf20", "20_journal_figures.py")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    S = rd("all_centers_summary").set_index("sample")
    L40 = rd("llm_all_centers").set_index("sample")
    GP = rd("c3x_gaps").set_index(["sample", "series"])
    EG = rd("c3x_ethnicity_gaps")
    EG = EG[EG["sample"] == "All centers"].set_index("group")

    def val(smp, key):
        if key == "Examiners":
            return S.loc[smp, "human_gap"], S.loc[smp, "human_gap_se"]
        if key == LLM0:
            return L40.loc[smp, "llm_gap"], L40.loc[smp, "llm_gap_se"]
        return S.loc[smp, f"gap::{key}"], S.loc[smp, f"gap_se::{key}"]

    def gender_row(smp):
        out = []
        for key, *_ in FIG_SERIES:
            b, se = val(smp, key)
            assert abs(b - GP.loc[(smp, key), "gap"]) < 1e-12, (smp, key)   # c3x_gaps.csv is from the same run
            out.append((b, se, GP.loc[(smp, key), "p_gap"] if key in FIG_HOLLOW else None))
        return out

    def eth_row(grp):
        r = EG.loc[grp]
        out = [(r.exam_gap, r.exam_se, None)]
        for key, *_ in FIG_SERIES[1:]:
            out.append((r[f"gap::{key}"], r[f"se::{key}"], r[f"pgap::{key}"] if key in FIG_HOLLOW else None))
        return out

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5})
    W_MM, H_MM = 140, 141
    LEFT, WIDTH = 47, 89
    fig = J.figure(W_MM, H_MM)
    W, H = fig.get_size_inches() / J.MM
    DY = [0.30, 0.10, -0.10, -0.30]          # series within a center's row
    DY_WIDE = [0.42, 0.14, -0.14, -0.42]     # rows with value labels: room for the labels

    def lab(ax, x, y, b, col, side, bg, fs=6.0):
        ax.text(x + (0.4 if side > 0 else -0.4), y, J.pp(b), ha="left" if side > 0 else "right", va="center",
                fontsize=fs, color=col, zorder=6,   # bg masks the zero line under a label that crosses it
                bbox=dict(boxstyle="square,pad=0.05", fc=bg, ec="none") if bg else None)

    def draw(ax, ys, rows, labels, printed):
        """labels: {row: (label the hollow markers too?, background colour behind the labels or None)}."""
        for yy, (name, row) in zip(ys, rows):
            for (key, slab, col, mk), dy, (b, se, p) in zip(FIG_SERIES, DY_WIDE if name in labels else DY, row):
                y = yy + dy
                lo, hi = 100 * (b - 1.96 * se), 100 * (b + 1.96 * se)
                ms = 3.0 if mk != "D" else 2.6
                ax.plot([lo, hi], [y] * 2, color=col, lw=0.8, solid_capstyle="butt", zorder=3)
                if p is not None:
                    ax.plot([100 * p, 100 * b], [y] * 2, color=col, lw=0.5, ls=(0, (1.2, 1.0)), zorder=3)
                    ax.plot([100 * p], [y], marker=mk, ms=ms, mfc="white", mec=col, mew=0.6, ls="none", zorder=4)
                ax.plot([100 * b], [y], marker=mk, ms=ms, color=col, mec="white", mew=0.3, ls="none", zorder=5)
                if name in labels:
                    hollow, bg = labels[name]
                    away = (p - b) if p is not None else -b   # the decision gap's label on the side away from the
                    if away > 0:                              # hollow marker (or from zero), the hollow one beyond it
                        lab(ax, lo, y, b, col, -1, bg)
                    else:
                        lab(ax, hi, y, b, col, +1, bg)
                    if p is not None and hollow:
                        lab(ax, 100 * p + (0.25 if p > b else -0.25), y, p, col, 1 if p > b else -1, bg, fs=5.6)
                printed.append((name, slab, b, se, p))

    def frame(ax, ys, xlabel):
        ax.axvline(0, color=J.INK2, lw=0.6, zorder=2)
        ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
        ax.set_xlim(*FIG_XLIM)
        ticks = np.arange(-15, 6, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:d}".replace("-", "−") for v in ticks])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=J.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)

    def ylabel(ax, y, top, sub, bold=False):
        tr = ax.get_yaxis_transform()
        ax.text(-0.02, y + 0.10, top, transform=tr, ha="right", va="center", fontsize=7, color=J.INK,
                fontweight="bold" if bold else "normal")
        ax.text(-0.02, y - 0.23, sub, transform=tr, ha="right", va="center", fontsize=5.7, color=J.INK2)

    def letter(x_mm, y_mm, s, title):   # 43's panel letters
        fig.text(x_mm / W, y_mm / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=J.INK)
        fig.text((x_mm + 3.4) / W, y_mm / H, title, fontsize=7, va="top", ha="left", color=J.INK)

    printed = []
    # a: gender, by technology center
    BOT_A, H_A = 52, 68
    ax = J.axes(fig, LEFT, BOT_A, WIDTH, H_A)
    ya = np.arange(len(ORDER))[::-1].astype(float)
    ya[1:] -= 0.5   # air between the pooled row and the centers
    ax.axhspan(ya[0] - 0.6, ya[0] + 0.6, color=J.BAND, lw=0, zorder=0)
    draw(ax, ya, [(smp, gender_row(smp)) for smp in ORDER], {"All centers": (True, J.BAND)}, printed)
    frame(ax, ya, "Female − male difference in grant decisions (percentage points)")
    for smp, yy in zip(ORDER, ya):
        if smp == "All centers":
            ylabel(ax, yy, "All eight centers", f"{int(S.loc[smp, 'applications']):,} applications", bold=True)
        else:
            ylabel(ax, yy, f"TC {smp}", TC_SHORT[smp])
    letter(2, BOT_A + H_A + 3.5, "a", "Gender of the first inventor, by technology center")
    # b: race and ethnicity, all eight centers
    BOT_B, H_B = 11, 26
    ax = J.axes(fig, LEFT, BOT_B, WIDTH, H_B)
    yb = 1.25 * np.arange(len(FIG_ETH))[::-1]
    draw(ax, yb, [(g, eth_row(g)) for g in FIG_ETH], {g: (False, None) for g in FIG_ETH}, printed)
    frame(ax, yb, "Group − White difference in grant decisions (percentage points)")
    for g, yy in zip(FIG_ETH, yb):
        ylabel(ax, yy, f"{g} − White", f"{int(EG.loc[g, 'n_group']):,} applications")
    letter(2, BOT_B + H_B + 4.5, "b", "Race and ethnicity of the first inventor (from surnames), all eight centers")

    # legend: the four series, then the hollow markers
    handles = [Line2D([], [], marker=mk, color=col, mec="white", mew=0.3, ms=3.4 if mk != "D" else 3.0, lw=0.8,
                      label=slab) for _, slab, col, mk in FIG_SERIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(2 / W, (H_MM - 7.5) / H), ncol=4,
               frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0)
    yk = H_MM - 13.0
    x0, x1 = 3.2, 7.6
    fig.add_artist(Line2D([x0 / W, x1 / W], [yk / H] * 2, color=J.INK2, lw=0.5, ls=(0, (1.2, 1.0)),
                          transform=fig.transFigure))
    fig.add_artist(Line2D([x0 / W], [yk / H], marker="o", ms=3.4, mfc="white", mec=J.INK2, mew=0.6, ls="none",
                          transform=fig.transFigure))
    fig.add_artist(Line2D([x1 / W], [yk / H], marker="o", ms=3.4, color=J.INK2, mec="white", mew=0.3, ls="none",
                          transform=fig.transFigure))
    fig.text((x1 + 2.0) / W, yk / H, "Hollow: gap in predicted grant probability, before the cutoff (text model "
             "trained on grants, LLM)", fontsize=6.5, va="center", ha="left", color=J.INK)
    fig.text(2 / W, (H_MM - 2.2) / H, "AI evaluators and group gaps in patent grants",
             fontsize=7.5, fontweight="bold", va="top", ha="left", color=J.INK)
    J.save(fig, "fig_main_c3")
    plt.close(fig)
    print(f"saved {J.OUT / 'fig_main_c3.pdf'} ({W_MM} x {H_MM} mm); values drawn (percentage points):")
    for name, slab, b, se, p in printed:
        hol = f"   hollow {100 * p:6.2f}" if p is not None else ""
        print(f"  {name:12s} {slab:32s} {100 * b:6.2f} [{100 * (b - 1.96 * se):6.2f}, {100 * (b + 1.96 * se):6.2f}]"
              f"{hol}")


# ---------------------------------------------------------------- tables, macros, report

def tables(gp: pd.DataFrame, share: pd.DataFrame, eg: pd.DataFrame, ot: pd.DataFrame, why: pd.DataFrame) -> dict:
    M29 = load_module("m29", "29_citation_tables.py")
    tabular, num = M29.tabular, M29.num
    N: dict = {}
    pc = lambda x, nd=1: num(100 * x, nd)  # noqa: E731

    def ci(b, se, nd=1, sc=100):
        return f"[{num(sc * (b - 1.96 * se), nd)}, {num(sc * (b + 1.96 * se), nd)}]"

    S = rd("all_centers_summary").set_index("sample")
    L40 = rd("llm_all_centers").set_index("sample")

    # Table A20 rebuilt: two lines per center (gap; 95% interval)
    ser = [("Examiners", None), (G, None), (C, None), (LLM0, None), (FA, None)]

    def gse(smp, key):
        if key == "Examiners":
            return S.loc[smp, "human_gap"], S.loc[smp, "human_gap_se"]
        if key == LLM0:
            return L40.loc[smp, "llm_gap"], L40.loc[smp, "llm_gap_se"]
        return S.loc[smp, f"gap::{key}"], S.loc[smp, f"gap_se::{key}"]
    rows = []
    for smp in ORDER:
        lab = "All centers" if smp == "All centers" else f"TC {smp}"
        top = [lab, f"{int(S.loc[smp, 'applications']):,}", pc(S.loc[smp, "female"])]
        bot = ["", "", ""]
        for k, _ in ser:
            b, se = gse(smp, k)
            top.append(pc(b))
            bot.append(r"{\footnotesize " + ci(b, se) + "}")
        rows += [top, bot]
    tabular(TEX / "c3x_tab_allcenters.tex", "lrrccccc",
            ["Center", "Applications", "Female (\\%)", "Examiners", "Trained on grant", "Trained on citations",
             "LLM, zero-shot", "Trained on first action"], rows)

    # macros: gender, pooled and every center (decision gaps of Table A20)
    short = {"Examiners": "Exam", G: "Grant", C: "Cites", LLM0: "LLM", FA: "First", GN: "GrantNames",
             CN: "CitesNames", LLML: "LLMLenient", LLMS: "LLMStrict", FTG: "FtGrant", FTC: "FtCites"}
    for smp in ORDER:
        for k, _ in ser:
            b, se = gse(smp, k)
            N[f"xCt{CW[smp]}{short[k]}"] = pc(b)
            N[f"xCt{CW[smp]}{short[k]}CI"] = ci(b, se)
            if k != "Examiners":
                N[f"xCt{CW[smp]}{short[k]}Ratio"] = num(b / S.loc[smp, "human_gap"], 2)
    N["xCtAllN"] = f"{int(S.loc['All centers', 'applications']):,}"

    # Table A21 unified: all evaluators, pooled and TC 3600
    ev = ["Examiners", G, GN, FA, C, CN, LLM0, LLML, LLMS, FTG, FTC]
    gx = gp.set_index(["sample", "series"])
    rows, groups = [], {0: "Examiners and text-embedding evaluators", 6: "Qwen3-8B"}
    for k in ev:
        row = [LABEL[k]]
        for smp in ["All centers", "3600"]:
            if (smp, k) in gx.index:
                r = gx.loc[(smp, k)]
                row += [f"{pc(r.gap)} {ci(r.gap, r.se)}", "1.00" if k == "Examiners" else num(r.ratio, 2)]
            else:
                row += ["--", "--"]
        rows.append(row)
        if (("3600", k) in gx.index) and k not in ser:
            r = gx.loc[("3600", k)]
            N[f"xCtThreeSix{short[k]}"], N[f"xCtThreeSix{short[k]}CI"] = pc(r.gap), ci(r.gap, r.se)
            N[f"xCtThreeSix{short[k]}Ratio"] = num(r.ratio, 2)
        if (("All centers", k) in gx.index) and k in (GN, CN):
            r = gx.loc[("All centers", k)]
            N[f"xCtAll{short[k]}"], N[f"xCtAll{short[k]}CI"] = pc(r.gap), ci(r.gap, r.se)
            N[f"xCtAll{short[k]}Ratio"] = num(r.ratio, 2)
    tabular(TEX / "c3x_tab_gaps.tex", "lrrrr",
            ["Evaluator", "All centers (pp)", "$\\times$ examiners", "TC 3600 (pp)", "$\\times$ examiners"],
            rows, groups)
    # probability scale, gender (macros only; the table below has it with the ethnic groups)
    for smp in ["All centers", "3600"]:
        for k in [G, LLM0, FA]:
            r = gx.loc[(smp, k)]
            N[f"xCt{CW[smp]}P{short[k]}"], N[f"xCt{CW[smp]}P{short[k]}CI"] = pc(r.p_gap), ci(r.p_gap, r.p_se)

    # race and ethnicity: decision gaps with intervals and the outcome test
    E = eg.set_index(["sample", "group"])
    O = ot.set_index(["sample", "group"])
    rows, groups = [], {}
    for smp in ["All centers", "3600"]:
        groups[len(rows)] = "All eight centers" if smp == "All centers" else "TC 3600"
        for key in ["hispanic", "asian", "black"]:
            lab = ETH[key][0]
            r, o = E.loc[(smp, lab)], O.loc[(smp, f"{lab} − White")]
            top = [lab, f"{int(r.n_group):,}", pc(r.exam_gap)] + [pc(r[f"gap::{c}"]) for c in [G, C, LLM0]] + \
                  [num(o.estimate, 2)]
            bot = ["", "", r"{\footnotesize " + ci(r.exam_gap, r.exam_se) + "}"] + \
                  [r"{\footnotesize " + ci(r[f"gap::{c}"], r[f"se::{c}"]) + "}" for c in [G, C, LLM0]] + \
                  [r"{\footnotesize [" + num(o.lo, 2) + ", " + num(o.hi, 2) + "]}"]
            rows += [top, bot]
            m = f"xCtEth{ETH[key][2]}{SMP[smp][0]}"
            N[m + "N"] = f"{int(r.n_group):,}"
            N[m + "Exam"], N[m + "ExamCI"] = pc(r.exam_gap), ci(r.exam_gap, r.exam_se)
            for c in [G, C, LLM0]:
                N[m + short[c]], N[m + short[c] + "CI"] = pc(r[f"gap::{c}"]), ci(r[f"gap::{c}"], r[f"se::{c}"])
                if c != C:
                    N[m + "P" + short[c]] = pc(r[f"pgap::{c}"])
                    N[m + "P" + short[c] + "CI"] = ci(r[f"pgap::{c}"], r[f"pse::{c}"])
            N[m + "Out"] = num(o.estimate, 2)
            N[m + "OutCI"] = f"[{num(o.lo, 2)}, {num(o.hi, 2)}]"
            og = O.loc[(smp, lab)]
            N[m + "OutLevel"] = num(og.estimate, 2)
        w = O.loc[(smp, "White")]
        rows.append(["White (reference)", f"{int(E.loc[(smp, 'Hispanic')].n_white):,}", "", "", "", "",
                     num(w.estimate, 2)])
        rows.append(["", "", "", "", "", "", r"{\footnotesize [" + num(w.lo, 2) + ", " + num(w.hi, 2) + "]}"])
        N[f"xCtEthWhite{SMP[smp][0]}N"] = f"{int(E.loc[(smp, 'Hispanic')].n_white):,}"
        N[f"xCtEthWhite{SMP[smp][0]}OutLevel"] = num(w.estimate, 2)
    tabular(TEX / "c3x_tab_ethnicity.tex", "lrccccc",
            ["Group", "Applications", "Examiners (pp)", "Trained on grant", "Trained on citations",
             "LLM, zero-shot", "Outcome test (citations)"], rows, groups)
    sh = share
    for smp in ["All centers", "3600"]:
        for key in ["hispanic", "asian", "black", "white"]:
            N[f"xCtEthShare{ETH[key][2]}{SMP[smp][0]}"] = pc(sh.loc[smp, key])
        N[f"xCtEthMatched{SMP[smp][0]}"] = pc(1 - sh.loc[smp, "unmatched"], 0)

    # probability scale: gender and the ethnic groups (grant-predicting evaluators only: the citation-trained score
    # has no grant probability of its own; mapped to one, it barely varies, so its gap there says little)
    PE = [G, LLM0]
    rows, groups = [], {}
    for smp in ["All centers", "3600"]:
        groups[len(rows)] = "All eight centers" if smp == "All centers" else "TC 3600"
        items = [("Female $-$ male", (gx.loc[(smp, "Examiners")].gap, gx.loc[(smp, "Examiners")].se),
                  [v for c in PE for v in [(gx.loc[(smp, c)].gap, gx.loc[(smp, c)].se),
                                           (gx.loc[(smp, c)].p_gap, gx.loc[(smp, c)].p_se)]])]
        for key in ["hispanic", "asian", "black"]:
            r = E.loc[(smp, ETH[key][0])]
            items.append((f"{ETH[key][0]} $-$ White", (r.exam_gap, r.exam_se),
                          [v for c in PE for v in [(r[f"gap::{c}"], r[f"se::{c}"]), (r[f"pgap::{c}"], r[f"pse::{c}"])]]))
        for lab, ex, vals in items:
            vals = [ex] + vals
            rows.append([lab] + [pc(b) for b, _ in vals])
            rows.append([""] + [r"{\footnotesize " + ci(b, se) + "}" for b, se in vals])
    tabular(TEX / "c3x_tab_prob.tex", "lccccc",
            ["Comparison", "Examiners", "Grant-trained: decisions", "Grant-trained: probability",
             "LLM: decisions", "LLM: probability"], rows, groups)

    # the citation-trained evaluator by center (investigation)
    Wh = why.set_index("sample")
    rows = []
    for smp in ORDER:
        r = Wh.loc[smp]
        lab = "All centers" if smp == "All centers" else f"TC {smp}"
        cells = [(r["granted::log_cites"], r["granted_se::log_cites"]),
                 (r["published_netgrant::log_pa"], r["published_netgrant_se::log_pa"]),
                 (r[f"all::{C}"], r[f"all_se::{C}"]),
                 (r["resid::log_cites"], r["resid_se::log_cites"]),
                 (r[f"all::dec::{C}"], r[f"all_se::dec::{C}"]),
                 (r[f"all_class::dec::{C}"], r[f"all_class_se::dec::{C}"]),
                 (r[f"all::decau::{C}"], r[f"all_se::decau::{C}"])]
        rows.append([lab, pc(r.female)] + [pc(b) for b, _ in cells])
        rows.append(["", ""] + [r"{\footnotesize " + ci(b, se) + "}" for b, se in cells])
        m = f"xCtWhy{CW[smp]}"
        for (b, se), k in zip(cells, ["Cites", "PriorArt", "Pred", "Resid", "Dec", "DecClass", "DecAu"]):
            N[m + k], N[m + k + "CI"] = pc(b), ci(b, se)
        N[m + "PredNames"] = pc(r[f"all::{CN}"])
        N[m + "PredSD"] = num(r["all::z_cites"], 2)
        N[m + "Female"] = pc(r.female)
    tabular(TEX / "c3x_tab_why.tex", "lrccccccc",
            ["Center", "Female (\\%)", "Citations of granted", "Prior-art cit., net of grant", "Predicted citations",
             "Citations at same prediction", "Decisions", "Decisions, class cells", "Decisions, art-unit rate"], rows)
    lite_macros(N)   # from --lite, if it has run
    (TEX / "c3x_numbers.tex").write_text(
        "% Generated by code/44_c3_extra.py; do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    print(f"{len(N)} macros")
    return N


# ---------------------------------------------------------------- 6. robustness (--lite)

TEMP_TRAIN, TEMP_TEST = (2010, 2012), (2013, 2014)
LITE = [(G, "logit", "granted"), (FA, "logit", "fa_allow"), (C, "ridge", None)]
LITE_SHORT = {"Examiners": "Exam", G: "Grant", FA: "First", C: "Cites"}


def temporal_scores(d: pd.DataFrame) -> pd.DataFrame:
    """38's evaluators trained on 2010–2012 applications of each center, scoring its 2013–2014 applications."""
    path = P / "c3x_lite_temporal_scores.parquet"
    if path.exists():
        sc = pd.read_parquet(path)
        if len(sc) == (d.filing_year >= TEMP_TEST[0]).sum():
            print("temporal scores: reusing", path.name)
            return sc
    import time
    m38 = load_module("m38", "38_all_centers.py")   # THREADS = 4 inside each fit; jobs run one at a time
    X = np.load(P / "emb/all_s0.npy").astype(np.float32)
    assert len(X) == len(d), "the vectors must align with all_sample (application_number order)"
    ex = pd.factorize(d.examiner_id)[0]
    ylog = np.log1p(d.cites_ns.to_numpy(float))
    g = d.granted.to_numpy() == 1
    te_all = np.where(d.filing_year.between(*TEMP_TEST).to_numpy())[0]
    sc = pd.DataFrame({"application_number": d.application_number.to_numpy()[te_all]})
    pos = pd.Series(np.arange(len(te_all)), index=te_all)
    for i, tc in enumerate(TCS):
        rows = np.where((d.tech_center == tc).to_numpy())[0]
        yr = d.filing_year.to_numpy()[rows]
        tr, te = rows[(yr >= TEMP_TRAIN[0]) & (yr <= TEMP_TRAIN[1])], rows[(yr >= TEMP_TEST[0]) & (yr <= TEMP_TEST[1])]
        for j, (name, kind, y) in enumerate(LITE):
            t0 = time.time()
            if kind == "logit":
                out = m38.logit_task(X, d[y].to_numpy(float), ex, tr, te, SEED + 100 * i + j)
            else:
                out = m38.ridge_task(X, ylog, ex, tr[g[tr]], te, SEED + 100 * i + j)
            sc.loc[pos[te].to_numpy(), name] = out
            print(f"temporal: TC {tc}, {name}: train {len(tr):,}, test {len(te):,} ({time.time() - t0:.0f} s)",
                  flush=True)
    sc.to_parquet(path, index=False)
    return sc


def temporal(d: pd.DataFrame) -> pd.DataFrame:
    """Gender gaps on 2013–2014: examiners, cross-fitted (38) and temporal evaluators, the same decision rule."""
    from sklearn.metrics import roc_auc_score
    sc = temporal_scores(d)
    t = d[d.filing_year.between(*TEMP_TEST)][["application_number", "examiner_id", "au_year", "tech_center", "female",
                                              "granted", "fa_allow", "cites_ns"] + [n for n, _, _ in LITE]]
    t = t.merge(sc.rename(columns={n: f"tmp::{n}" for n, _, _ in LITE}), on="application_number", validate="1:1")
    for tc in TCS:   # cfe.check's rule: the examiners' 2013–2014 share of each center
        m = (t.tech_center == tc).to_numpy()
        r = t.loc[m, "granted"].mean()
        for n, _, _ in LITE:
            for c in [n, f"tmp::{n}"]:
                v = t.loc[m, c]
                t.loc[m, f"dec::{c}"] = (v >= v.quantile(1 - r)).astype(float)
    rows = []
    for smp in ORDER:
        s = t if smp == "All centers" else t[t.tech_center == smp]
        cols = ["granted"] + [f"dec::{c}" for n, _, _ in LITE for c in [n, f"tmp::{n}"]]
        e = gaps(s, cols).set_index("series")
        h = e.loc["granted"]
        rows.append(dict(sample=smp, series="Examiners", split="actual decisions", applications=len(s),
                         gap=h.gap, se=h.se, ratio=1.0))
        for n, _, y in LITE:
            for c, split in [(n, "cross-fitted by examiner (38)"), (f"tmp::{n}", "trained on 2010–2012")]:
                gr = s[s.granted == 1]
                fit = dict(auc_grant=roc_auc_score(s.granted, s[c]),
                           corr_log_cites=np.corrcoef(gr[c], np.log1p(gr.cites_ns))[0, 1])
                if y == "fa_allow":
                    ok = s.fa_allow.notna()
                    fit["auc_first_action"] = roc_auc_score(s.fa_allow[ok], s[c][ok])
                rows.append(dict(sample=smp, series=n, split=split, applications=len(s),
                                 gap=e.loc[f"dec::{c}", "gap"], se=e.loc[f"dec::{c}", "se"],
                                 ratio=e.loc[f"dec::{c}", "gap"] / h.gap, **fit))
    r = pd.DataFrame(rows)
    r.to_csv(T / "c3x_lite_temporal.csv", index=False)
    return r


def lite_macros(N: dict) -> None:
    """\\xCtLite... from c3x_lite_temporal.csv (if --lite has run): sample (All, ThreeSix) + Exam / Grant / First /
    Cites + Cross (cross-fitted) / Temp (2010–2012 → 2013–2014) + CI / AUC."""
    f = T / "c3x_lite_temporal.csv"
    if not f.exists():
        return
    M29 = load_module("m29", "29_citation_tables.py")
    num = M29.num
    r = pd.read_csv(f, dtype={"sample": str}).set_index(["sample", "series", "split"])
    for smp in ["All centers", "3600"]:
        h = r.loc[(smp, "Examiners", "actual decisions")]
        m = f"xCtLite{CW[smp]}"
        N[m + "Exam"] = num(100 * h.gap, 1)
        N[m + "ExamCI"] = f"[{num(100 * (h.gap - 1.96 * h.se), 1)}, {num(100 * (h.gap + 1.96 * h.se), 1)}]"
        N[m + "N"] = f"{int(h.applications):,}"
        for n, _, _ in LITE:
            for split, k in [("cross-fitted by examiner (38)", "Cross"), ("trained on 2010–2012", "Temp")]:
                x = r.loc[(smp, n, split)]
                key = m + LITE_SHORT[n] + k
                N[key] = num(100 * x.gap, 1)
                N[key + "CI"] = f"[{num(100 * (x.gap - 1.96 * x.se), 1)}, {num(100 * (x.gap + 1.96 * x.se), 1)}]"
                N[key + "Ratio"] = num(x.ratio, 2)
                N[key + "AUC"] = num(x.auc_grant, 3)


def lite_inputs_lines() -> list:
    """38's lines that define the evaluators' inputs, quoted with their line numbers."""
    src = (ROOT / "code/38_all_centers.py").read_text().splitlines()
    keys = ["LOGIT = [", '("Train on first action", "s0"', "RIDGE = [", 'd["s0"] = (', '+ d["claims"]).str[:',
            'd["s1"] = d["inventor_list"]', "calls.append(delayed(logit_task)(X[x]",
            "calls.append(delayed(ridge_task)(X[x]", "m = LogisticRegression(C=best",
            "return Ridge(alpha=best).fit("]
    out = []
    for i, line in enumerate(src, 1):
        if any(k in line for k in keys):
            out.append(f"    {i:>4}  {line.strip()}")
    return out


def lite() -> None:
    d = load()
    r = temporal(d)
    rx = r.set_index(["sample", "series", "split"])
    M29 = load_module("m29", "29_citation_tables.py")
    tabular, num = M29.tabular, M29.num

    def ci(b, se):
        return r"{\footnotesize [" + num(100 * (b - 1.96 * se), 1) + ", " + num(100 * (b + 1.96 * se), 1) + "]}"
    rows = []
    lab = {G: "Trained on grant", FA: "Trained on first action", C: "Trained on citations"}
    top, bot = ["Examiners (actual decisions)"], [""]
    for smp in ["All centers", "3600"]:   # the same applications for both splits: one value across both columns
        h = rx.loc[(smp, "Examiners", "actual decisions")]
        top.append(r"\multicolumn{2}{c}{" + num(100 * h.gap, 1) + "}")
        bot.append(r"\multicolumn{2}{c}{" + ci(h.gap, h.se) + "}")
    rows += [top, bot]
    for n, _, _ in LITE:
        top, bot = [lab[n]], [""]
        for smp in ["All centers", "3600"]:
            for split in ["cross-fitted by examiner (38)", "trained on 2010–2012"]:
                x = rx.loc[(smp, n, split)]
                top.append(num(100 * x.gap, 1))
                bot.append(ci(x.gap, x.se))
        rows += [top, bot]
    tabular(TEX / "c3x_lite_tab_temporal.tex", "lcccc",
            ["", "All centers: cross-fitted", "All centers: 2010--12 $\\to$ 2013--14",
             "TC 3600: cross-fitted", "TC 3600: 2010--12 $\\to$ 2013--14"], rows)
    # macros: keep every existing one, add / replace the \\xCtLite ones
    N: dict = {}
    lite_macros(N)
    f = TEX / "c3x_numbers.tex"
    old = f.read_text().splitlines() if f.exists() else ["% Generated by code/44_c3_extra.py; do not edit by hand."]
    keep = [x for x in old if not x.startswith("\\newcommand{\\xCtLite")]
    f.write_text("\n".join(keep) + "\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    print(f"{len(N)} lite macros added to {f.name} ({len(keep) - 1} kept)")

    def fm(x):
        return f"{100 * x.gap:.1f} [{100 * (x.gap - 1.96 * x.se):.1f}, {100 * (x.gap + 1.96 * x.se):.1f}]"
    L = ["# C3 robustness: a temporal split and the evaluators' inputs (code/44_c3_extra.py --lite)", "",
         "## (i) Temporal split: trained on 2010–2012, deciding 2013–2014", "",
         "Within each center, 38's grant-, first-action- and citation-trained evaluators (Qwen3-Embedding-0.6B vectors "
         "of title, abstract, and claims; L2 logistic regression / ridge on log(1 + citations) of granted applications; "
         "penalty by inner cross-validation grouped by examiner, through 38's logit_task and ridge_task) are trained on "
         "applications filed 2010–2012 and score those filed 2013–2014. Each grants the examiners' 2013–2014 share of "
         "applications within each center (cfe.check's rule); the cross-fitted evaluators of 38 (five folds grouped by "
         "examiner, all years) decide the same applications by the same rule. Gaps: female − male within art unit × "
         "year, errors clustered by examiner, pp. The temporal split keeps examiners on both sides (an examiner's 2013–"
         "2014 applications are scored by a model that saw that examiner's 2010–2012 decisions); the cross-fitted "
         "evaluators never score an examiner they were trained on.", "",
         "| Sample | Applications 2013–14 | Evaluator | Cross-fitted (38) | × examiners | 2010–12 → 2013–14 | × examiners | "
         "AUC grant: cross / temporal |", "|---|---|---|---|---|---|---|---|"]
    for smp in ORDER:
        h = rx.loc[(smp, "Examiners", "actual decisions")]
        L.append(f"| {smp} | {int(h.applications):,} | Examiners | {fm(h)} | 1.00 | | | |")
        for n, _, _ in LITE:
            x, y = rx.loc[(smp, n, "cross-fitted by examiner (38)")], rx.loc[(smp, n, "trained on 2010–2012")]
            L.append(f"| {smp} | | {lab[n]} | {fm(x)} | {x.ratio:.2f} | {fm(y)} | {y.ratio:.2f} | "
                     f"{x.auc_grant:.3f} / {y.auc_grant:.3f} |")
    L += ["", "Correlation with log(1 + citations) among granted 2013–2014 applications (cross / temporal): " + "; ".join(
        f"{smp}: " + ", ".join(f"{lab[n]} {rx.loc[(smp, n, 'cross-fitted by examiner (38)')].corr_log_cites:.3f} / "
                               f"{rx.loc[(smp, n, 'trained on 2010–2012')].corr_log_cites:.3f}" for n, _, _ in LITE)
        for smp in ["All centers", "3600"]) + ".", ""]
    L += ["## (ii) Do applicant fields enter the evaluators?", "",
          "No. 38's text evaluators (grant-, first-action- and citation-trained) take one input: the "
          "Qwen3-Embedding-0.6B vector of 's0' = title + abstract + claims (first 6,000 characters). Entity size "
          "(small_entity), the number of inventors (n_inventors), and the inventors' locations are not inputs; the "
          "'+ names' variants (input 's1') add the inventor list as 'Inventors: First Last; ...' (names only), so they "
          "can read gender and the number of inventors from it, and they are reported as separate rows. The zero-shot "
          "LLM's prompt (10's app_block) is title, abstract, and first claim. All first inventors in the sample are in "
          "the U.S. (38's prepare: first_inventor_country = 'US'). Nothing to refit. The lines of "
          "code/38_all_centers.py:", "", "```"] + lite_inputs_lines() + ["```", "",
          "small_entity and n_inventors enter only cfe.check's marginal-application regressions as controls and 39's "
          "balance index, not the evaluators' scores or decisions.", "",
          "## Macros (added to output/latex/c3x_numbers.tex)", "",
          "`\\xCtLite` + sample (All, ThreeSix) + Exam / Grant / First / Cites + Cross (cross-fitted, 38) / Temp "
          "(2010–2012 → 2013–2014) + (none) / CI / Ratio / AUC; `\\xCtLite{All,ThreeSix}N` = 2013–2014 applications.", ""]
    L += [f"- `\\{k}` = {v}" for k, v in sorted(N.items())]
    (OUT / "c3x_lite.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def main() -> None:
    warnings.filterwarnings("ignore")
    if "--figure" in sys.argv:
        figure()
        return
    if "--lite" in sys.argv:
        lite()
        return
    d = load()
    rep = check_reproduction(d)
    rep.to_csv(T / "c3x_reproduction.csv", index=False)
    d = calibrate(d, [G, FA, C, LLM0])
    gp = gender_tables(d)
    share, eg, ot = ethnicity(d)
    why = why_citations(d)
    ch = a21_changes(gp)
    figure()
    N = tables(gp, share, eg, ot, why)
    write_report(gp, share, eg, ot, why, N, ch)


def narrative(gp, eg, ot, why, ch) -> list:
    """The prose of output/c3x.md: figure caption, table notes, the old Table A21 versus the unified one, and the
    answer on the citation-trained evaluator in TC 1600 and 3700. Every number is read from the results."""
    gx = gp.set_index(["sample", "series"])
    W = why.set_index("sample")
    E = eg.set_index(["sample", "group"])
    O = ot.set_index(["sample", "group"])

    def v(x, nd=1):
        return f"{100 * x:.{nd}f}".replace("-", "−")

    def u(x, nd=2):   # unscaled, with a minus sign
        return f"{x:.{nd}f}".replace("-", "−")

    def cv(b, se, nd=1):
        return f"{v(b, nd)} [{v(b - 1.96 * se, nd)}, {v(b + 1.96 * se, nd)}]"

    def g(smp, k):
        r = gx.loc[(smp, k)]
        return r.gap, r.se

    def w(smp, k):
        r = W.loc[smp]
        return r[k], r[k.replace("::", "_se::", 1)]
    a = "All centers"
    L = ["## Figure 2 (main text): caption", "",
         "**Figure 2. Gender gap in grant decisions made by examiners and by three AI evaluators, by technology "
         "center.** Each evaluator scores every application filed 2010–2014 in the eight utility technology centers "
         f"({int(W.loc[a].applications):,} applications) and grants the share that examiners grant in that center. "
         "Dots: difference between applications whose first inventor is a woman and those whose first inventor is a "
         "man in the share granted, within art unit × filing-year cells (percentage points); lines: 95% confidence "
         "intervals, standard errors clustered by examiner. Black: examiners' actual decisions. Vermillion: text model "
         "(Qwen3-Embedding-0.6B representation of title, abstract, and claims; logistic regression) trained on grant "
         "decisions. Blue: the same text model trained on citations received within five years of grant, excluding "
         "self-citations (granted applications). Purple: Qwen3-8B asked zero-shot whether the application will be "
         "granted. Text models are cross-fitted in five folds grouped by examiner within each center. All eight "
         f"centers: examiners {cv(*g(a, 'Examiners'))}, grant-trained {cv(*g(a, G))}, citation-trained "
         f"{cv(*g(a, C))}, zero-shot LLM {cv(*g(a, LLM0))}. Table A20 reports every value.", "",
         "Shorter caption: **Figure 2. Gender gap in grant decisions by evaluator and technology center.** Female − "
         "male difference in the share granted within art unit × filing-year cells (percentage points), 95% "
         "confidence intervals clustered by examiner; each evaluator grants the share of applications that examiners "
         "grant in the center. Applications filed 2010–2014 in eight technology centers.", "",
         "In-figure title: 'AI evaluators and the gender gap in patent grants, by technology center' (delete it in "
         "the figure code if the caption title suffices).", "",
         "## Tables (output/latex)", "",
         "- `c3x_tab_allcenters.tex` (Table A20 rebuilt): rows All centers, TC 3700, 3600, 2800, 2600, 2400, 2100, "
         "1700, 1600; columns Applications, Female (%), and the gap of examiners, trained on grant, trained on "
         "citations, LLM zero-shot, trained on first action; each gap on one line and its 95% interval on the line "
         "below. Numbers are 38's all_centers_summary.csv and 40's llm_all_centers.csv, unchanged (the decisions are "
         "rebuilt here and reproduce both files to 1e-6; c3x_reproduction.csv).",
         "- `c3x_tab_gaps.tex` (Table A21 unified): every evaluator, All centers and TC 3600, gap [95% CI] and × "
         "examiners. The 38/40 evaluators come from the same files as Table A20, so every TC 3600 number equals Table "
         "A20's. Rows that exist only in the two pilot centers (LLM with lenient / strict examiners' examples; "
         "fine-tuned LLM on grant / on citations) are computed here on the same TC 3600 applications with the same "
         "decision rule (40's rule for prompted LLMs, cfe.check's for the fine-tuned scores, as 30); All centers is "
         "'--' for them. The TC 1600 column is dropped (TC 1600 is in Table A20 and Figure 2).",
         "- `c3x_tab_ethnicity.tex` (Table A29 rebuilt): All eight centers and TC 3600; Hispanic, Asian, Black "
         "against White first inventors: applications, the examiners' grant gap, the grant-trained, "
         "citation-trained and zero-shot-LLM decision gaps (each with its 95% interval below), and the outcome test "
         "(group − White difference in mean five-year citations of the most lenient quintile's extra grants, "
         "500 art-unit bootstrap draws); a White reference row gives the White applications and their outcome-test "
         "level.",
         "- `c3x_tab_prob.tex`: gender and the ethnic groups, decisions vs probability scale for the grant-trained "
         "evaluator and the zero-shot LLM (optional; for the threshold point of comments 31 and 39).",
         "- `c3x_tab_why.tex`: the citation-trained evaluator by center (the investigation below).", ""]

    L += ["## Old Table A21 versus the unified table", "",
          "Old A21 = 29's cit_tab_c3_gaps.tex (21's two-center run). New: TC 3600 from c3x_tab_gaps.tex; the old TC "
          "1600 column is dropped and its new values (Table A20) are listed for reference.", "",
          "| TC | Evaluator | Old | New | Old × | New × | Changed |", "|---|---|---|---|---|---|---|"]
    for r in ch.itertuples():
        L.append(f"| {r.sample} | {r.evaluator} | {r.old} | {r.new} | {getattr(r, 'old_ratio', '') if r.sample == '3600' else ''} | "
                 f"{getattr(r, 'new_ratio', '') if r.sample == '3600' else ''} | {'yes' if r.changed else ''} |")
    L += ["", "New rows: All centers column for the seven evaluators run in all eight centers; LLM fine-tuned on grant "
          f"({cv(*g('3600', FTG))}, {u(gx.loc[('3600', FTG)].ratio)}×) and on citations ({cv(*g('3600', FTC))}, "
          f"{u(gx.loc[('3600', FTC)].ratio)}×) in TC 3600, identical to 30's finetune_gaps.csv (Table A13).", ""]

    o1, o7, o6 = W.loc["1600"], W.loc["3700"], W.loc["3600"]
    others = W.drop(index=["All centers", "1600", "3700"])
    big1 = W.drop(index="All centers")["granted::log_cites"].idxmin() == "1600"
    o39 = rd("c3_checks_outcome_test")
    o39 = o39[(o39.design == "all cells") & (o39["sample"] == "1600")].iloc[0]
    L += ["## Why the citation-trained evaluator keeps a gender gap in TC 1600 and 3700 (comment 35)", "",
          "Short answer: it reproduces an outcome gap, not the examiners' decisions. In these two centers women's "
          "inventions are cited less than men's in the same art unit and year, and the text the model reads carries "
          "part of that difference; in TC 3600 the text does not.", "",
          f"1. **Women's patents are cited less within cells in 1600 and 3700.** Among granted patents, log(1 + "
          f"citations) is {v(-o1['granted::log_cites'])}% lower for female first inventors in TC 1600 "
          f"({cv(*w('1600', 'granted::log_cites'))}){', the largest shortfall of the eight centers' if big1 else ''}, "
          f"and {v(-o7['granted::log_cites'])}% lower in TC 3700 ({cv(*w('3700', 'granted::log_cites'))}). The shortfall "
          "is not produced by which applications examiners grant: prior-art citations of all published applications, "
          f"rejected ones included, are lower too ({v(o1['published::log_pa'])}% and {v(o7['published::log_pa'])}%; "
          f"{v(o1['published_netgrant::log_pa'])}% and {v(o7['published_netgrant::log_pa'])}% net of the grant). "
          f"39's outcome test points the same way in TC 1600: women's marginal grants are cited {u(o39.female)} times "
          f"on average, men's {u(o39.male)} (difference {u(o39.f_minus_m)} [{u(o39.f_minus_m_lo)}, "
          f"{u(o39.f_minus_m_hi)}]).",
          f"2. **The text predicts it there, much more than elsewhere.** The evaluator's predicted log citations are "
          f"{v(-o1[f'all::{C}'])}% lower for women's applications in TC 1600 ({cv(*w('1600', f'all::{C}'))}) and "
          f"{v(-o7[f'all::{C}'])}% lower in TC 3700 ({cv(*w('3700', f'all::{C}'))}) within art unit × year: "
          f"{u(o1['all::z_cites'])} and {u(o7['all::z_cites'])} standard deviations of the score, the two largest "
          f"negative gaps of the eight centers (the others range from {u(others['all::z_cites'].min())} to "
          f"+{u(others['all::z_cites'].max())} SD). Granting the top share by predicted citations turns these into "
          f"{v(o1[f'all::dec::{C}'])} and {v(o7[f'all::dec::{C}'])} pp. In TC 3600 women's patents are also cited "
          f"somewhat less ({cv(*w('3600', 'granted::log_cites'))} among granted; prior-art "
          f"{v(o6['published::log_pa'])}%), but the text "
          f"predicts slightly more for women (+{v(o6[f'all::{C}'])}%): at the same predicted citations women's "
          f"patents in 3600 receive {v(-o6['resid::log_cites'])}% fewer ({cv(*w('3600', 'resid::log_cites'))}). "
          f"So in 3600 the evaluator shows no gap (+{v(o6[f'all::dec::{C}'])} pp) while its target does.",
          f"3. **Not field mix.** Within art unit × year × USPC class the decision gaps are "
          f"{v(o1[f'all_class::dec::{C}'])} (1600) and {v(o7[f'all_class::dec::{C}'])} (3700); when each art unit × "
          "year grants its own share instead of the center's (no threshold across art units), "
          f"{v(o1[f'all::decau::{C}'])} and {v(o7[f'all::decau::{C}'])}. The gap is between women's and men's "
          "applications in the same narrow field.",
          f"4. **Not miscalibration, and names add to it.** Within cells the slope of actual on predicted log "
          f"citations among granted patents is {o1.slope:.2f} (1600) and {o7.slope:.2f} (3700). The version that also "
          f"reads the inventors' names predicts {v(o1[f'all::{CN}'])}% for women in 1600 (vs {v(o1[f'all::{C}'])}% "
          "without names) and gives a larger decision gap (Table A20 source: "
          f"{v(rd('all_centers_summary').set_index('sample').loc['1600', f'gap::{CN}'])} pp), i.e. names carry "
          "gender information that predicts fewer citations.",
          f"5. **Why it exceeds the examiners' gap in 1600.** The examiners' gap there is small "
          f"({cv(*g('1600', 'Examiners'))}) and the text predicts it poorly (grant-trained {v(g('1600', G)[0])}), "
          "while the citation shortfall of women's patents is the largest of all centers. In TC 1600 women are also "
          f"{100 * o1.female:.1f}% of applications, {o1.female / W.loc[a].female:.1f} times the eight-center share "
          f"({100 * W.loc[a].female:.1f}%).", "",
          "Reading for the text: a citation-trained evaluator removes the gap only where women's applications do not "
          "look less cited from their text. Where the outcome itself differs by gender (TC 1600, 3700), it inherits "
          "that difference. Whether the citation difference reflects the inventions or how a field cites women's work "
          "cannot be separated with these data; the label is not neutral just because it is an outcome.", ""]

    def n(smp, lab):
        return f"{int(E.loc[(smp, lab)].n_group):,}"

    def oc(smp, k):
        o = O.loc[(smp, k)]
        return f"{u(o.estimate)} [{u(o.lo)}, {u(o.hi)}]"
    t39 = rd("c3_checks_outcome_test")
    t39 = t39[t39.design == "all cells"].set_index("sample")

    def fm39(smp):
        r = t39.loc[smp]
        return f"{u(r.f_minus_m)} [{u(r.f_minus_m_lo)}, {u(r.f_minus_m_hi)}]"
    pc_max = gp.loc[gp.series == C, "p_gap"].abs().max()
    pe_max = eg[f"pgap::{C}"].abs().max()
    diffs = ot[ot.group.str.endswith("− White")]
    rej = diffs[(diffs.lo > 0) | (diffs.hi < 0)]
    reject_txt = ("none of the group − White differences rejects equal bars (every 95% interval includes zero)."
                  if rej.empty else "intervals that exclude zero: " + "; ".join(
                      f"{r.group} in {r['sample']}" for _, r in rej.iterrows()) + ".")
    L += ["## Notes on race and ethnicity", "",
          f"- Group sizes (applications): All centers Hispanic {n(a, 'Hispanic')}, Asian {n(a, 'Asian')}, Black "
          f"{n(a, 'Black')}, White {int(E.loc[(a, 'Hispanic')].n_white):,}; TC 3600 Hispanic {n('3600', 'Hispanic')}, "
          f"Asian {n('3600', 'Asian')}, Black {n('3600', 'Black')}. Surname-based classification identifies Black "
          "inventors poorly (few surnames are borne mostly by Black Americans), so the Black group is small and its "
          f"estimates are imprecise; the TC 3600 Black row (n = {n('3600', 'Black')}) should not be quoted in the "
          "main text. Pooling multiplies the Hispanic count by "
          f"{E.loc[(a, 'Hispanic')].n_group / E.loc[('3600', 'Hispanic')].n_group:.1f} and the Black count by "
          f"{E.loc[(a, 'Black')].n_group / E.loc[('3600', 'Black')].n_group:.1f} relative to TC 3600.",
          "- Decisions are made on all of each center's applications (as for gender), and the gap is estimated on the "
          "group's and White inventors' applications. 33 (old Table A29) re-ranked within the group + White subsample, "
          "so TC 3600 evaluator numbers differ slightly from the old table: Hispanic grant-trained "
          f"{v(E.loc[('3600', 'Hispanic')][f'gap::{G}'])} (old −6.2), citation-trained "
          f"{v(E.loc[('3600', 'Hispanic')][f'gap::{C}'])} (old 0.8), LLM {v(E.loc[('3600', 'Hispanic')][f'gap::{LLM0}'])}"
          " (old −6.5); Black grant-trained "
          f"{v(E.loc[('3600', 'Black')][f'gap::{G}'])} (old −7.7), citation-trained "
          f"{v(E.loc[('3600', 'Black')][f'gap::{C}'])} (old 1.4), LLM {v(E.loc[('3600', 'Black')][f'gap::{LLM0}'])} "
          "(old −13.3); examiners' gaps are identical (same applications).",
          f"- Outcome test: {reject_txt} "
          f"Hispanic − White: {oc(a, 'Hispanic − White')} pooled, {oc('3600', 'Hispanic − White')} in TC 3600; "
          f"Asian − White {oc(a, 'Asian − White')} pooled, {oc('3600', 'Asian − White')} in TC 3600; Black − White "
          f"{oc(a, 'Black − White')} pooled. The test is weak for small groups: a cell contributes only if the group "
          "has applications at both the most lenient and a stricter quintile, which holds for "
          f"{100 * O.loc[(a, 'Hispanic')].apps_in_cells:.0f}% of Hispanic and "
          f"{100 * O.loc[(a, 'Black')].apps_in_cells:.0f}% of Black applications pooled (White "
          f"{100 * O.loc[(a, 'White')].apps_in_cells:.0f}%); the Black interval is uninformative. For comparison, "
          f"female − male (39's c3_checks_outcome_test.csv): {fm39('All centers')} pooled, {fm39('3600')} in TC 3600.",
          f"- Probability scale: the threshold amplifies group differences. Pooled, the grant-trained evaluator's "
          f"probability-scale gap for women is {v(gx.loc[(a, G)].p_gap)} pp against {v(gx.loc[(a, G)].gap)} in "
          f"decisions (TC 3600: {v(gx.loc[('3600', G)].p_gap)} vs {v(gx.loc[('3600', G)].gap)}); the zero-shot LLM's "
          f"{v(gx.loc[(a, LLM0)].p_gap)} vs {v(gx.loc[(a, LLM0)].gap)}; examiners {v(gx.loc[(a, 'Examiners')].gap)}. "
          f"Hispanic − White pooled: grant-trained {v(E.loc[(a, 'Hispanic')][f'pgap::{G}'])} vs "
          f"{v(E.loc[(a, 'Hispanic')][f'gap::{G}'])}; Black − White {v(E.loc[(a, 'Black')][f'pgap::{G}'])} vs "
          f"{v(E.loc[(a, 'Black')][f'gap::{G}'])}. The probability is the evaluator's score mapped to a grant "
          "probability within center by a logistic fit (monotone, two parameters). Not reported for the "
          "citation-trained evaluator: its score has no grant probability of its own, and mapped to one it barely "
          f"varies (its probability-scale gaps are within ±{100 * max(pc_max, pe_max):.1f} pp in every center and "
          "group; c3x_gaps.csv, c3x_ethnicity_gaps.csv).", ""]
    return L


def write_report(gp, share, eg, ot, why, N, ch) -> None:
    def f(b, se=None, sc=100, nd=1):
        s = f"{sc * b:.{nd}f}"
        return s if se is None else f"{s} [{sc * (b - 1.96 * se):.{nd}f}, {sc * (b + 1.96 * se):.{nd}f}]"
    L = ["# C3 on the main sample: gender and ethnic gaps by evaluator (code/44_c3_extra.py)", "",
         "Eight technology centers pooled, applications filed 2010–2014 (38's all_sample). Each evaluator grants the "
         "examiners' share of applications within each center; gaps are group differences in decisions within art "
         "unit × filing-year cells, errors clustered by examiner; 95% intervals in brackets; percentage points.", ""]
    L += narrative(gp, eg, ot, why, ch)
    L += ["# Result tables", ""]
    L += ["## Gender gap by center (Table A20 rebuilt; Figure 2)", "",
          "| Center | Examiners | Trained on grant | Trained on citations | LLM, zero-shot | Trained on first action |",
          "|---|---|---|---|---|---|"]
    gx = gp.set_index(["sample", "series"])
    for smp in ORDER:
        L.append(f"| {smp} | " + " | ".join(f(gx.loc[(smp, k)].gap, gx.loc[(smp, k)].se)
                                         for k in ["Examiners", G, C, LLM0, FA]) + " |")
    L += ["", "## All evaluators, pooled and TC 3600", "", "| Evaluator | All centers | × | TC 3600 | × |",
          "|---|---|---|---|---|"]
    for k in ["Examiners", G, GN, FA, C, CN, LLM0, LLML, LLMS, FTG, FTC]:
        cells = []
        for smp in ["All centers", "3600"]:
            if (smp, k) in gx.index:
                r = gx.loc[(smp, k)]
                cells += [f(r.gap, r.se), f"{r.ratio:.2f}"]
            else:
                cells += ["—", "—"]
        L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    L += ["", "Probability scale (gender): mean difference in the grant probability implied by each score "
          "(logistic fit of the grant on the score within center), same cells and clustering.", "",
          "| Sample | Examiners (grants) | Grant-trained | First action | LLM, zero-shot |",
          "|---|---|---|---|---|"]
    for smp in ORDER:
        L.append(f"| {smp} | {f(gx.loc[(smp, 'Examiners')].gap, gx.loc[(smp, 'Examiners')].se)} | " + " | ".join(
            f(gx.loc[(smp, k)].p_gap, gx.loc[(smp, k)].p_se) for k in [G, FA, LLM0]) + " |")
    L += ["", "## Race and ethnicity (first inventor's surname, 2010 Census, share ≥ 60%)", "",
          "| Sample | White | Asian | Hispanic | Black | Unclassified | Not in Census table |", "|---|---|---|---|---|---|---|"]
    for smp in ORDER:
        r = share.loc[smp]
        L.append(f"| {smp} | " + " | ".join(f"{100 * r[k]:.1f}%" for k in
                                         ["white", "asian", "hispanic", "black", "unclassified", "unmatched"]) + " |")
    L += ["", "| Sample | Group | Applications (group / White) | Examiners | Grant-trained | Citation-trained | "
          "LLM, zero-shot | Prob. scale: grant-trained | Prob. scale: LLM | Outcome test: group − White (citations) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    O = ot.set_index(["sample", "group"])
    for _, r in eg.iterrows():
        o = O.loc[(r["sample"], f"{r.group} − White")]
        L.append(f"| {r['sample']} | {r.group} | {r.n_group:,} / {r.n_white:,} | {f(r.exam_gap, r.exam_se)} | "
                 + " | ".join(f(r[f'gap::{c}'], r[f'se::{c}']) for c in [G, C, LLM0]) + " | "
                 + " | ".join(f(r[f'pgap::{c}'], r[f'pse::{c}']) for c in [G, LLM0])
                 + f" | {o.estimate:.2f} [{o.lo:.2f}, {o.hi:.2f}] |")
    L += ["", "Outcome test levels (mean citations of the most lenient quintile's extra grants):", "",
          "| Sample | Group | Estimate [95%] | Applications | Cells used | Share of the group's applications in "
          "cells used |", "|---|---|---|---|---|---|"]
    for _, r in ot.iterrows():
        extra = "" if pd.isna(r.get("n")) else f"{int(r.n):,} | {int(r.cells_used):,} | {100 * r.apps_in_cells:.0f}%"
        L.append(f"| {r['sample']} | {r.group} | {r.estimate:.2f} [{r.lo:.2f}, {r.hi:.2f}] | {extra or '— | — | —'} |")
    L += ["", "## The citation-trained evaluator by center", "",
          "Female − male within art unit × year cells (class: art unit × year × USPC class). Citations: "
          "log(1 + five-year citations excl. self-citations) of granted patents (×100 ≈ %); prior-art: log(1 + "
          "prior-art citations), published applications incl. rejected; predicted: the evaluator's predicted log "
          "citations, all applications; at same prediction: citations of granted patents controlling for the "
          "prediction; decisions: the evaluator's decision gap (pp); class cells: the same decisions, gap within art "
          "unit × year × USPC class; art-unit rate: the evaluator grants each art unit × year's grant share instead of "
          "the center's.", "",
          "| Center | Female share | Cit., granted | Cit., class cells | Prior-art cit. | Prior-art, net of grant | "
          "Predicted | Predicted (SD) | Predicted, + names | At same prediction | Decisions | Decisions, class cells | "
          "Decisions, art-unit rate | Grant-trained decisions | Calibration slope |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    Wh = why.set_index("sample")
    for smp in ORDER:
        r = Wh.loc[smp]
        L.append(f"| {smp} | {100 * r.female:.1f}% | {f(r['granted::log_cites'], r['granted_se::log_cites'])} | "
                 f"{f(r['granted_class::log_cites'], r['granted_class_se::log_cites'])} | "
                 f"{f(r['published::log_pa'], r['published_se::log_pa'])} | "
                 f"{f(r['published_netgrant::log_pa'], r['published_netgrant_se::log_pa'])} | "
                 f"{f(r[f'all::{C}'], r[f'all_se::{C}'])} | {r['all::z_cites']:.3f} | "
                 f"{f(r[f'all::{CN}'], r[f'all_se::{CN}'])} | {f(r['resid::log_cites'], r['resid_se::log_cites'])} | "
                 f"{f(r[f'all::dec::{C}'], r[f'all_se::dec::{C}'])} | "
                 f"{f(r[f'all_class::dec::{C}'], r[f'all_class_se::dec::{C}'])} | "
                 f"{f(r[f'all::decau::{C}'], r[f'all_se::decau::{C}'])} | "
                 f"{f(r[f'all::dec::{G}'], r[f'all_se::dec::{G}'])} | {r.slope:.2f} |")
    L += ["", "## Macros (output/latex/c3x_numbers.tex)", "",
          "Names: `\\xCt` + sample (All = eight centers pooled; OneSix … ThreeSeven = TC 1600 … 3700) + evaluator "
          "(Exam, Grant, GrantNames, First, Cites, CitesNames, LLM, LLMLenient, LLMStrict, FtGrant, FtCites) + "
          "CI / Ratio (× examiners); `P` before the evaluator = probability scale. `\\xCtEth` + group (Hisp, Asian, "
          "Black, White) + sample (All, ThreeSix) + Exam / Grant / Cites / LLM / PGrant / PLLM / Out (outcome test, "
          "group − White) / OutLevel / N; `\\xCtEthShare…`, `\\xCtEthMatched…` = shares (%). `\\xCtWhy` + "
          "center + Cites (log citations of granted, ×100), PriorArt (prior-art citations, net of grant), Pred "
          "(predicted log citations), PredSD (in SD of the score), PredNames, Resid (citations at the same "
          "prediction), Dec / DecClass / DecAu (decision gap; class cells; art-unit × year rate), Female (%). "
          "Gaps in percentage points; outcome tests in citations.", ""]
    L += [f"- `\\{k}` = {v}" for k, v in sorted(N.items())]
    (OUT / "c3x.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:80]))


if __name__ == "__main__":
    main()
