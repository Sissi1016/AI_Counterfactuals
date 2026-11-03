"""Where strict and lenient examiners' applications come from, and which industries are stricter.

Examiners' own locations are not recorded (PatEx names the examiner and the art unit, not the duty station), so the
maps place applications by the first inventor's state. The main sample (38's all_sample: applications filed 2010–2014 in the eight
technology centers (02, 04); within each art unit × filing year × USPC class cell (the assignment cell of 12) with at
least five examiners, examiners are ordered by leave-one-out leniency and split into fifths of equal caseload (as in
35). For each state: the share of its applications decided by the strictest fifth (red map) and by the most lenient
fifth (green map). Under random assignment within cells both shares equal the cells' shares in every state (about
20%); the state's deviation from that expectation is tested with standard errors clustered by examiner (all of an
examiner's applications in a cell share its fifth), and states with |z| > 3 are outlined.
Map (c) uses all five fifths at once: for each state, the mean over its applications of the examiner's leave-one-out
leniency minus the mean leniency of the cell's applications (percentage points; zero in every state under random
assignment within cells), i.e. the coefficients of a regression of the within-cell demeaned leniency on state
indicators, with standard errors clustered by examiner. States with |z| < 1.96 are drawn faded.

Industry: (a) the distribution of examiners' grant rates within each technology center (examiners with at least 50
applications); (b) grant rates of the 30 largest CPC classes (main CPC label of the application, HUPD).

Map geometry: U.S. states, Albers projection with Alaska and Hawaii inset, from the us-atlas 3.0.1 package
(states-albers-10m.json; ISC license), saved as data/processed/us_states_albers_10m.json (docs/data.md).
Runs on CPUs (DuckDB, CFE_THREADS threads, default 4).
Input: data/processed/{applications_leniency,patex_application_data,patex_all_inventors,all_sample}.parquet
Output: output/geography_industry.md,
        output/tables/{geography_states,geo_state_leniency,industry_tc,industry_cpc}.csv,
        output/figures/journal/{fig_map_strict_lenient,fig_industry}.{pdf,png}
"""
import importlib.util
import json
import os
import pathlib

import duckdb
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
spec = importlib.util.spec_from_file_location("jf20", ROOT / "code/20_journal_figures.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
TC_NAMES = {"1600": "Biotechnology, organic chemistry", "1700": "Chemical and materials engineering",
            "2100": "Computer architecture, software", "2400": "Computer networks, video, security",
            "2600": "Communications", "2800": "Semiconductors, electrical, optical",
            "3600": "Transportation, construction, business methods", "3700": "Mechanical engineering, medical devices"}
CPC_NAMES = {
    "A01": "Agriculture", "A23": "Foods", "A43": "Footwear", "A45": "Hand and travelling articles",
    "A47": "Furniture, domestic articles", "A61": "Medical and veterinary science", "A62": "Life-saving, fire-fighting",
    "A63": "Sports, games", "B01": "Physical and chemical processes", "B05": "Spraying, coating",
    "B08": "Cleaning", "B21": "Metal working", "B22": "Casting, powder metallurgy", "B23": "Machine tools",
    "B25": "Hand tools", "B26": "Cutting", "B29": "Working of plastics", "B32": "Layered products",
    "B41": "Printing", "B60": "Vehicles in general", "B62": "Land vehicles", "B63": "Ships",
    "B64": "Aircraft, space", "B65": "Conveying, packing", "B81": "Microstructures", "B82": "Nanotechnology",
    "C01": "Inorganic chemistry", "C02": "Water treatment", "C03": "Glass", "C04": "Cements, ceramics",
    "C07": "Organic chemistry", "C08": "Polymers", "C09": "Dyes, paints, adhesives", "C10": "Petroleum, fuels",
    "C11": "Oils, detergents", "C12": "Biochemistry, genetic engineering", "C22": "Metallurgy",
    "C23": "Coating metals", "C25": "Electrolysis", "C30": "Crystal growth", "E04": "Building",
    "E05": "Locks, hinges", "E21": "Drilling, mining", "F01": "Engines in general", "F02": "Combustion engines",
    "F04": "Pumps", "F16": "Engineering elements", "F21": "Lighting", "F24": "Heating, ventilation",
    "F25": "Refrigeration", "F28": "Heat exchange", "G01": "Measuring, testing", "G02": "Optics",
    "G03": "Photography, lithography", "G05": "Control systems", "G06": "Computing", "G07": "Checking devices",
    "G08": "Signalling", "G09": "Displays, education, advertising", "G10": "Acoustics, speech",
    "G11": "Information storage", "G16": "Health and other informatics", "H01": "Electric elements",
    "H02": "Electric power", "H03": "Electronic circuits", "H04": "Communications", "H05": "Electric techniques",
    "H10": "Semiconductor devices", "Y02": "Climate change mitigation", "Y10": "Former USPC cross-references"}
FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
        "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
MIN_EXAMINERS, Z_OUTLINE, MIN_APPS, Z_SIG = 5, 3.0, 50, 1.96
FADE = 0.45                # map (c): share of white mixed into states with |z| < Z_SIG
THREADS = int(os.environ.get("CFE_THREADS", "4"))
STRICT = mcolors.LinearSegmentedColormap.from_list("strict", ["#fbe9e7", "#e57368", "#b3261e", "#6e0d0a"])
LENIENT = mcolors.LinearSegmentedColormap.from_list("lenient", ["#e8f5e9", "#6fbf73", "#2e7d32", "#0f4614"])
RDGN = mcolors.LinearSegmentedColormap.from_list("rdgn", ["#b3261e", "#e8a29b", "#d9d9d4", "#8cc591", "#2e7d32"])


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    tcs = ", ".join(f"'{t}'" for t in TCS)
    return con.execute(f"""
        SELECT a.application_number, a.examiner_id, a.au_year || '_' || coalesce(u.uspc_class, 'none') AS cell,
               a.tech_center, a.lenient_grant, a.granted, upper(substr(a.main_cpc_label, 1, 3)) AS cpc, i.state
        FROM '{P}/applications_leniency.parquet' a
        LEFT JOIN (SELECT application_number, uspc_class FROM '{P}/patex_application_data.parquet') u
            USING (application_number)
        LEFT JOIN (SELECT application_number, upper(trim(inventor_region_code)) AS state
                   FROM '{P}/patex_all_inventors.parquet'
                   WHERE inventor_rank = '1' AND inventor_country_code = 'US') i USING (application_number)
        WHERE a.tech_center IN ({tcs}) AND a.lenient_grant IS NOT NULL
          AND a.filing_year BETWEEN 2010 AND 2014
          AND a.application_number IN (SELECT application_number FROM '{P}/all_sample.parquet')
        ORDER BY a.application_number""").df()   # fixed row order: sums identical across runs and threads


def fifths(d: pd.DataFrame) -> pd.DataFrame:
    """Within-cell fifth of each application's examiner (1 = strictest, 5 = most lenient), equal caseloads, with the
    one-pass hole repair of 35; and each cell's share of cases in the strictest and most lenient fifths."""
    ex = d.groupby(["cell", "examiner_id"]).agg(z=("lenient_grant", "mean"), cases=("granted", "size")).reset_index()
    ex["n_ex"] = ex.groupby("cell").z.transform("size")
    ex = ex[ex.n_ex >= MIN_EXAMINERS].sort_values(["cell", "z", "examiner_id"], kind="mergesort")
    q_all = np.empty(len(ex), int)
    starts = np.flatnonzero(np.r_[True, ex.cell.to_numpy()[1:] != ex.cell.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(ex)]
    cases = ex.cases.to_numpy(float)
    for a, b in zip(starts, ends):
        cum = np.cumsum(cases[a:b]) / cases[a:b].sum()
        q = np.clip(np.ceil(cum * 5 - 1e-9).astype(int), 1, 5)
        n = len(q)
        q[0] = 1
        for i in range(1, n):
            q[i] = max(min(max(q[i], q[i - 1]), q[i - 1] + 1), 5 - (n - 1 - i))
        q_all[a:b] = q
    ex["q"] = q_all
    tot = ex.groupby("cell").cases.transform("sum")
    for k, col in [(1, "p_strict"), (5, "p_lenient")]:
        ex[col] = ex.cases.where(ex.q == k, 0).groupby(ex.cell).transform("sum") / tot
    return d.merge(ex[["cell", "examiner_id", "q", "p_strict", "p_lenient"]], on=["cell", "examiner_id"])


def states(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d.state.isin(FIPS.values())].copy()
    x["strict"], x["lenient"] = (x.q == 1).astype(float), (x.q == 5).astype(float)
    s = x.groupby("state").agg(applications=("q", "size"), share_strict=("strict", "mean"),
                               expected_strict=("p_strict", "mean"), share_lenient=("lenient", "mean"),
                               expected_lenient=("p_lenient", "mean"), grant_rate=("granted", "mean")).reset_index()
    for k in ["strict", "lenient"]:
        # deviation from the cells' shares, with variance from examiner clusters within the state
        x["r"] = x[k] - x[f"p_{k}"]
        cl = x.groupby(["state", "examiner_id"]).r.sum().pow(2).groupby(level=0).sum()
        g = x.groupby(["state", "examiner_id"]).size().groupby(level=0).size()
        var = cl * g / (g - 1).clip(lower=1)
        dev = x.groupby("state").r.sum()
        s[f"z_{k}"] = s.state.map(dev / np.sqrt(var))
    return s


def state_leniency(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Map (c): the state means of the examiner's leniency minus the mean leniency of the cell's applications (the
    coefficients of the within-cell demeaned leniency on state indicators), with standard errors clustered by
    examiner (G/(G-1) correction, G = the state's examiners, as in states()). Also returns the application-level SD of
    the demeaned leniency, for scale."""
    dev = d.lenient_grant - d.groupby("cell").lenient_grant.transform("mean")
    x = d.assign(dev=dev)[d.state.isin(FIPS.values())]
    m = x.groupby("state").dev.agg(["size", "mean"])
    e = x.dev - x.state.map(m["mean"])
    cl = e.groupby([x.state, x.examiner_id]).sum()
    g = cl.groupby(level=0).size()
    se = np.sqrt(cl.pow(2).groupby(level=0).sum() * g / (g - 1).clip(lower=1)) / m["size"]
    out = pd.DataFrame({"state": m.index, "n": m["size"].to_numpy(), "mean_dev_pp": 100 * m["mean"].to_numpy(),
                        "se": 100 * se.reindex(m.index).to_numpy()})
    out["z"] = out.mean_dev_pp / out.se
    return out, 100 * dev.std()


# ---------------------------------------------------------------- map geometry (TopoJSON, pre-projected)

def state_shapes() -> dict:
    topo = json.loads((P / "us_states_albers_10m.json").read_text())
    (sx, sy), (tx, ty) = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        xy = np.cumsum(np.array(arc, float), axis=0)
        arcs.append(np.column_stack([xy[:, 0] * sx + tx, xy[:, 1] * sy + ty]))

    def ring(idx):
        pts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in idx]
        return np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
    out = {}
    for g in topo["objects"]["states"]["geometries"]:
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        out[FIPS[g["id"]]] = [ring(p[0]) for p in polys]   # outer rings
    return out


def faded(rgba, amount: float = FADE) -> tuple:
    """The color mixed with white (states whose deviation is not significant)."""
    r = np.asarray(mcolors.to_rgba(rgba))
    return tuple(np.r_[r[:3] + (1 - r[:3]) * amount, r[3]])


def draw_map(ax, shapes: dict, values: dict, cmap, norm, outline: set, fade: set = frozenset()) -> None:
    for st, rings in shapes.items():
        v = values.get(st, np.nan)
        face = "#eeeeea" if np.isnan(v) else cmap(norm(v))
        if st in fade:
            face = faded(face)
        for r in rings:
            ax.add_patch(Polygon(r, closed=True, facecolor=face, edgecolor="white", lw=0.35, zorder=2))
            if st in outline:
                ax.add_patch(Polygon(r, closed=True, facecolor="none", edgecolor=J.INK, lw=0.9, zorder=3))
    ax.set_xlim(-10, 975)
    ax.set_ylim(620, -5)
    ax.set_aspect("equal")
    ax.axis("off")


def strip(fig, x: float, y: float, w: float, cmap, norm, fade: bool):
    """One row of map (c)'s key: the diverging scale, at full strength or faded."""
    ax = J.axes(fig, x, y, w, 2.4)
    v = np.linspace(norm.vmin, norm.vmax, 256)
    rgba = np.array([faded(cmap(norm(t))) if fade else cmap(norm(t)) for t in v])
    ax.imshow(rgba[None], aspect="auto", extent=(norm.vmin, norm.vmax, 0, 1), interpolation="bilinear")
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
    return ax


def fig_maps(s: pd.DataFrame, sl: pd.DataFrame) -> None:
    shapes = state_shapes()
    H, top = 152, 76                       # figure height; maps (a)(b) sit above y = top (mm)
    fig = J.figure(180, H)
    # each map's scale is centred on its expected share (the cells' share, the same in every state under random
    # assignment) and spans the same width, so equal shading means an equal deviation on both maps
    half = max((s[f"share_{c}"] - s[f"expected_{c}"].mean()).abs().max() for c in ["strict", "lenient"])
    for k, (col, cmap, x0, label, title) in enumerate([
            ("strict", STRICT, 2, "a", "Share decided by the strictest fifth of examiners"),
            ("lenient", LENIENT, 92, "b", "Share decided by the most lenient fifth of examiners")]):
        mid = s[f"expected_{col}"].mean()
        norm = mcolors.Normalize(mid - half, mid + half)
        ax = J.axes(fig, x0, top + 10, 86, 56)
        vals = dict(zip(s.state, s[f"share_{col}"]))
        out = set(s.state[s[f"z_{col}"].abs() > Z_OUTLINE])
        draw_map(ax, shapes, vals, cmap, norm, out)
        J.letter(fig, x0, H - 2, label, title + ", by first inventor's state (%)")
        cax = J.axes(fig, x0 + 18, top + 5, 50, 2.2)
        cb = fig.colorbar(__import__("matplotlib.cm").cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal")
        ticks = np.linspace(mid - half, mid + half, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([J.pp(t) for t in ticks])
        cb.outline.set_linewidth(0.4)
        cax.tick_params(length=1.5, labelsize=5.5)

    # (c) all five fifths at once: the average leniency of the examiners drawn, relative to the cell mean; the scale
    # is symmetric around zero (the value random assignment implies in every state), faded where |z| < 1.96
    lim = sl.mean_dev_pp.abs().max()
    norm = mcolors.Normalize(-lim, lim)
    sig = set(sl.state[sl.z.abs() >= Z_SIG])
    ax = J.axes(fig, 2, 2, 104, 66)
    draw_map(ax, shapes, dict(zip(sl.state, sl.mean_dev_pp)), RDGN, norm, sig, set(sl.state) - sig)
    J.letter(fig, 2, top - 2, "c", "Average leniency of the examiners drawn, relative to the mean of their cells, "
                                   "by first inventor's state (percentage points)")
    x, w = 124, 50
    W, Hf = fig.get_size_inches() / J.MM
    fig.text(x / W, 52 / Hf, "Examiner leniency minus cell mean (pp)", fontsize=6, ha="left", va="bottom",
             color=J.INK)
    for y, fade, lab in [(46, False, f"|z| ≥ {Z_SIG:.2f}"), (42.6, True, f"|z| < {Z_SIG:.2f}")]:
        cax = strip(fig, x, y, w, RDGN, norm, fade)
        fig.text((x - 1.5) / W, (y + 1.2) / Hf, lab, fontsize=5.5, ha="right", va="center", color=J.INK)
        if fade:
            ticks = np.arange(-np.floor(lim / 0.5), np.floor(lim / 0.5) + 1) * 0.5
            cax.set_xticks(ticks)
            cax.set_xticklabels([f"{t:+.1f}".replace("-", "−").replace("+0.0", "0") for t in ticks])
            cax.tick_params(axis="x", length=1.5, labelsize=5.5)
        else:
            cax.set_xticks([])
    fig.text(x / W, 34.5 / Hf, "← stricter examiners", fontsize=5.5, ha="left", va="top", color=J.INK2)
    fig.text((x + w) / W, 34.5 / Hf, "more lenient examiners →", fontsize=5.5, ha="right", va="top", color=J.INK2)
    n_sig = len(sig)
    fig.text(x / W, 27 / Hf, f"Full color, outlined: |z| ≥ {Z_SIG:.2f} ({n_sig} of {len(sl)} states)\n"
                              f"Faded: |z| < {Z_SIG:.2f}; z from SEs clustered by examiner",
             fontsize=5.5, ha="left", va="top", color=J.INK2, linespacing=1.5)
    J.save(fig, "fig_map_strict_lenient")


# ---------------------------------------------------------------- industry

def industry(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ex = d.groupby("examiner_id").agg(n_e=("granted", "size"), grant_rate=("granted", "mean"),
                                      main_tc=("tech_center", lambda v: v.value_counts().index[0])).reset_index()
    ex = ex[(ex.n_e >= MIN_APPS) & ex.main_tc.isin(TCS)]
    tc = ex.groupby("main_tc").grant_rate.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).reset_index()
    tc = tc.rename(columns={"main_tc": "tech_center", "count": "examiners", "10%": "p10", "25%": "p25",
                            "50%": "median", "75%": "p75", "90%": "p90"})
    apps = d.groupby("tech_center").agg(applications=("granted", "size"), app_grant_rate=("granted", "mean"))
    tc = tc.merge(apps.reset_index(), on="tech_center")
    tc["name"] = tc.tech_center.map(TC_NAMES)
    c = d[d.cpc.str.match(r"^[A-HY]\d\d$", na=False)].groupby("cpc").agg(
        applications=("granted", "size"), grant_rate=("granted", "mean"),
        examiner_leniency=("lenient_grant", "mean")).reset_index()
    c = c.sort_values("applications", ascending=False).head(30)
    c["name"] = c.cpc.map(CPC_NAMES).fillna("")
    return tc.sort_values("median"), c.sort_values("grant_rate")


def fig_industry(tc: pd.DataFrame, c: pd.DataFrame, overall: float) -> None:
    fig = J.figure(180, 92)
    norm = mcolors.Normalize(overall - 0.15, overall + 0.15)
    ax = J.axes(fig, 52, 10, 36, 74)
    y = np.arange(len(tc))
    for i, r in enumerate(tc.itertuples()):
        col = RDGN(norm(r.median))
        ax.plot([r.p10, r.p90], [i, i], color=col, lw=0.8, solid_capstyle="round")
        ax.plot([r.p25, r.p75], [i, i], color=col, lw=3.2, solid_capstyle="butt", alpha=0.85)
        ax.plot([r.median], [i], marker="|", ms=6, mew=1.2, color=J.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.tech_center} {r.name}" for r in tc.itertuples()], fontsize=5.6)
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["20", "40", "60", "80", "100"])
    ax.set_xlabel("Examiner grant rate (%): 10th–90th, 25th–75th percentile, median")
    J.hgrid(ax)
    J.letter(fig, 2, 90, "a", "Technology centers: examiners' grant rates")
    ax = J.axes(fig, 132, 10, 44, 74)
    y = np.arange(len(c))
    ax.barh(y, c.grant_rate, color=[RDGN(norm(v)) for v in c.grant_rate], height=0.72)
    ax.axvline(overall, color=J.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.cpc} {r.name}" for r in c.itertuples()], fontsize=5.2)
    ax.set_xlim(0.3, 0.95)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(["30", "50", "70", "90"])
    ax.set_xlabel("Grant rate of applications (%)")
    ax.tick_params(axis="y", length=0)
    J.hgrid(ax)
    J.letter(fig, 92, 90, "b", "The 30 largest CPC classes (red: stricter)")
    J.save(fig, "fig_industry")


def main() -> None:
    d = fifths(load())
    s = states(d)
    s.to_csv(OUT / "tables/geography_states.csv", index=False)
    tc, c = industry(d)
    tc.to_csv(OUT / "tables/industry_tc.csv", index=False)
    c.to_csv(OUT / "tables/industry_cpc.csv", index=False)
    sl, sd_app = state_leniency(d)
    sl.to_csv(OUT / "tables/geo_state_leniency.csv", index=False)
    fig_maps(s, sl)
    overall = d.granted.mean()
    fig_industry(tc, c, overall)
    us = d[d.state.isin(FIPS.values())]
    big = sl.loc[sl.mean_dev_pp.abs().idxmax()]
    L = ["# Where strict and lenient examiners' applications come from; which industries are stricter", "",
         f"Applications 2010–2014 in the eight technology centers with an examiner in a cell of at least five: "
         f"{len(d):,}; with a U.S. first inventor in a state or DC: {len(us):,}.", "",
         f"- Strictest fifth: state shares {100 * s.share_strict.min():.1f}–{100 * s.share_strict.max():.1f}% "
         f"(expected {100 * s.expected_strict.min():.1f}–{100 * s.expected_strict.max():.1f}%); |z| > 1.96 in "
         f"{int((s.z_strict.abs() > 1.96).sum())} of {len(s)} states, |z| > {Z_OUTLINE:g} in "
         f"{int((s.z_strict.abs() > Z_OUTLINE).sum())}.",
         f"- Most lenient fifth: state shares {100 * s.share_lenient.min():.1f}–{100 * s.share_lenient.max():.1f}% "
         f"(expected {100 * s.expected_lenient.min():.1f}–{100 * s.expected_lenient.max():.1f}%); |z| > 1.96 in "
         f"{int((s.z_lenient.abs() > 1.96).sum())}, |z| > {Z_OUTLINE:g} in "
         f"{int((s.z_lenient.abs() > Z_OUTLINE).sum())}.",
         f"- All fifths at once (map c), mean leniency of the examiners drawn minus the cell mean: |z| >= {Z_SIG} in "
         f"{int((sl.z.abs() >= Z_SIG).sum())} of {len(sl)} states; largest |deviation| {big.state} "
         f"{big.mean_dev_pp:+.2f} pp (z = {big.z:.2f}, {big.n:,} applications); SD across states "
         f"{sl.mean_dev_pp.std():.2f} pp (application-level SD of the demeaned leniency: {sd_app:.1f} pp); "
         f"output/tables/geo_state_leniency.csv.", "",
         "| State | Applications | Strictest fifth (%) | Expected | z | Most lenient fifth (%) | Expected | z |",
         "|---|---|---|---|---|---|---|---|"]
    for r in s.sort_values("applications", ascending=False).itertuples():
        L.append(f"| {r.state} | {r.applications:,} | {100 * r.share_strict:.1f} | {100 * r.expected_strict:.1f} | "
                 f"{r.z_strict:.1f} | {100 * r.share_lenient:.1f} | {100 * r.expected_lenient:.1f} | "
                 f"{r.z_lenient:.1f} |")
    L += ["", "## Technology centers (examiners with at least 50 applications)", "",
          "| TC | Field | Examiners | Examiner grant rate: p10 | median | p90 | Application grant rate |",
          "|---|---|---|---|---|---|---|"]
    for r in tc.itertuples():
        L.append(f"| {r.tech_center} | {r.name} | {int(r.examiners):,} | {100 * r.p10:.1f} | {100 * r.median:.1f} | "
                 f"{100 * r.p90:.1f} | {100 * r.app_grant_rate:.1f} |")
    L += ["", "## The 30 largest CPC classes", "", "| CPC | Field | Applications | Grant rate (%) |",
          "|---|---|---|---|"]
    for r in c.itertuples():
        L.append(f"| {r.cpc} | {r.name} | {r.applications:,} | {100 * r.grant_rate:.1f} |")
    (OUT / "geography_industry.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
