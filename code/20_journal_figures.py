"""Journal-style figures (Nature / Science / PNAS conventions) built from the result tables of 12–19.

Conventions: 180 mm double-column width; Helvetica-type sans (Nimbus Sans) at 5–7 pt; bold lowercase panel letters;
hairline axes; vector PDF with embedded fonts plus a 600 dpi PNG. One colour code throughout: orange = an evaluator
that learns the organization's decisions, blue = one that learns outcomes, black = the human record (truth),
grey = no model or random. Figure legends are written to output/figures/journal/legends.md.

Figure 1: the check (strict vs lenient examiners), the evaluators' signed errors on the marginal applications, and
          what the grant-trained evaluator assumes about the examiners' rejections.
Figure 2: what the evaluator learns vs whom it learns from; the decision–outcome frontier.
Figure 3: AI examiners copy their examiner's bar, not a taste.
Figure 4: identification checks (permutation balance test; the error under alternative designs).
Figure 5: contraction of the most lenient examiners' grants by each evaluator (no instrument needed).

Runs on CPUs in seconds. Input: output/tables/*.csv. Output: output/figures/journal/fig{1..5}.{pdf,png}, legends.md
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "output/tables"
OUT = ROOT / "output/figures/journal"
MM = 1 / 25.4
TC = "3600"

# palette (validated: blue/orange categorical pair; blue ordinal ramp) and neutral inks
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#52514e", "#ebeae6"
DEC, OUTC = "#eb6834", "#2a78d6"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE_D, BLUE_L, REJ = "#1c5cab", "#86b6ef", "#e6e5e0"
HUMAN = "Human record (truth)"
NO_MODEL = "No model: mean outcome of granted applications"

FONT_DIR = ROOT / ".cache/fonts"
FONT = "Nimbus Sans TT"


def truetype_fonts() -> None:
    """Journals ask for TrueType (Type 42) text in PDFs; Nimbus Sans ships as OpenType/CFF. Convert it once (glyph
    outlines to quadratic curves, family renamed) into a user cache and register it with matplotlib."""
    from matplotlib import font_manager
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for style in ["Regular", "Bold"]:
        out = FONT_DIR / f"NimbusSansTT-{style}.ttf"
        if not out.exists():
            src = next((f.fname for f in font_manager.fontManager.ttflist
                        if f.fname.endswith(f"NimbusSans-{style}.otf")), None)
            if src is None:  # the URW base-35 fonts are missing: fall back to matplotlib's own sans
                global FONT
                FONT = "DejaVu Sans"
                print("Nimbus Sans not found (install the URW base-35 fonts); using DejaVu Sans")
                return
            f = TTFont(src)
            order = f.getGlyphOrder()
            gs = f.getGlyphSet()
            glyphs = {}
            for name in gs.keys():
                pen = TTGlyphPen(gs)
                gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
                glyphs[name] = pen.glyph()
            f["loca"], f["glyf"] = newTable("loca"), newTable("glyf")
            f["glyf"].glyphOrder, f["glyf"].glyphs = order, glyphs
            del f["CFF "]
            f["glyf"].compile(f)
            for name, g in glyphs.items():
                if hasattr(g, "xMin"):
                    f["hmtx"][name] = (f["hmtx"][name][0], g.xMin)
            maxp = f["maxp"] = newTable("maxp")
            maxp.tableVersion = 0x00010000
            for k in ["maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                      "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"]:
                setattr(maxp, k, 1 if k == "maxZones" else 0)
            f["post"].formatType, f["post"].extraNames, f["post"].mapping = 2.0, [], {}
            f["post"].glyphOrder = order
            for rec in f["name"].names:
                if rec.nameID in (1, 4, 16):
                    rec.string = str(rec).replace("Nimbus Sans", FONT)
                elif rec.nameID == 6:
                    rec.string = str(rec).replace("NimbusSans", "NimbusSansTT")
            f.sfntVersion = "\000\001\000\000"
            f.save(out)
        font_manager.fontManager.addfont(str(out))


truetype_fonts()
plt.rcParams.update({
    "font.family": FONT, "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.spines.top": False,
    "axes.spines.right": False, "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.unicode_minus": True, "legend.frameon": False, "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2, "axes.titlesize": 6.5, "axes.titlepad": 3,
})


# ---------------------------------------------------------------- helpers

def figure(w_mm: float, h_mm: float):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")
    return fig


def axes(fig, left: float, bottom: float, width: float, height: float):
    """Axes placed in millimetres from the figure's lower-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def letter(fig, x_mm: float, y_mm: float, s: str, title: str | None = None) -> None:
    W, H = fig.get_size_inches() / MM
    fig.text(x_mm / W, y_mm / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
    if title:
        fig.text((x_mm + 3.2) / W, y_mm / H - 0.2 / H, title, fontsize=6.5, va="top", ha="left", color=INK)


def hgrid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(OUT / f"{name}.png", dpi=600, metadata={"Software": None})
    plt.close(fig)


def dot(ax, x, y, color, lo=None, hi=None, filled=True, marker="o", size=3.6, z=4, lw=0.9):
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=z - 1)
    ax.plot([x], [y], marker=marker, ms=size, color=color if filled else "white", mec=color if not filled else "white",
            mew=0.9 if not filled else 0.6, zorder=z, ls="none")


def bracket(ax, x: float, y0: float, y1: float, label: str) -> None:
    """A bracket at data x joining rows y0 and y1, ticks pointing left toward the rows, label to its right."""
    ax.plot([x - 0.35, x, x, x - 0.35], [y0, y0, y1, y1], color=INK2, lw=0.5, zorder=2)
    ax.text(x + 0.35, (y0 + y1) / 2, label, ha="left", va="center", fontsize=5.5, color=INK2)


def pp(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}".replace("-", "−")


# ---------------------------------------------------------------- data

def load() -> dict:
    rd = lambda n: pd.read_csv(T / f"{n}.csv", dtype={"sample": str, "tech_center": str})  # noqa: E731
    return {n: rd(n) for n in ["marginal_split", "marginal_estimates", "marginal_contrasts", "marginal_curve",
                               "llm_marginal_estimates", "probes_bootstrap", "contraction_evaluators",
                               "ai_examiners", "ai_examiners_agreement", "ai_examiners_count_distribution",
                               "ai_examiners_lottery", "ai_examiners_summary", "assignment_permutation_pvalues",
                               "assignment_resid_estimates", "assignment_permutation_summary",
                               "contraction_curves", "contraction_quintiles"]}


def est(d: pd.DataFrame, series: str, kind: str, group: str = "All", outcome: str = "renew_4y",
        sample: str = TC) -> pd.Series:
    t = d[(d["sample"] == sample) & (d.outcome == outcome) & (d.series == series) & (d.kind == kind)
          & (d.group == group)]
    return t.iloc[0]


# ---------------------------------------------------------------- figure 1

def fig1(D: dict) -> None:
    fig = figure(180, 95)
    sp = D["marginal_split"]
    h = sp[(sp["sample"] == TC) & (sp.outcome == "renew_4y") & (sp.series == HUMAN)].iloc[0]

    # a: strict vs lenient examiners, per 100 applications
    ax = axes(fig, 27, 61, 76, 21)
    rows = [("Strict examiners", h.strict_granted, h.strict_granted_y, 1.0),
            ("Lenient examiners", h.lenient_granted, h.lenient_granted_y, 0.0)]
    bh = 0.46
    for lab, g, r, y in rows:
        for x0, w, c in [(0, r, BLUE_D), (r, g - r, BLUE_L), (g, 100 - g, REJ)]:
            ax.add_patch(Rectangle((x0 + 0.25, y - bh / 2), max(w - 0.5, 0), bh, color=c, lw=0, zorder=2))
        ax.text(r / 2, y, f"{r:.1f}", color="white", ha="center", va="center", fontsize=6, zorder=3)
        ax.text(-2, y, lab, ha="right", va="center", fontsize=6.5, color=INK)
    for g, y in [(h.lenient_granted, 0), (h.strict_granted, 1)]:
        ax.text((g + 100) / 2, y, f"rejected {100 - g:.1f}", ha="center", va="center", fontsize=6, color=INK2,
                zorder=3)
    # the marginal applications: granted by lenient, rejected by strict examiners
    g0, g1 = h.strict_granted, h.lenient_granted
    for x in [g0]:
        ax.plot([x, x], [1 - bh / 2, -bh / 2 - 0.02], color=INK, lw=0.5, ls=(0, (1.5, 1.2)), zorder=4)
    yb = -0.33
    ax.plot([g0, g0, g1, g1], [yb + 0.07, yb, yb, yb + 0.07], color=INK, lw=0.6, zorder=4, clip_on=False)
    dg, dr = g1 - g0, h.lenient_granted_y - h.strict_granted_y
    ax.text((g0 + g1) / 2, yb - 0.1, f"marginal applications: {dg:.1f} per 100, of which {dr:.1f} renewed "
            f"→ {100 * dr / dg:.0f}%", ha="center", va="top", fontsize=6, color=INK, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.45)
    ax.axis("off")
    # legend above
    lx = 0
    for c, lab in [(BLUE_D, "granted, renewed at year 4"), (BLUE_L, "granted, lapsed"),
                   (REJ, "rejected: outcome never observed")]:
        ax.add_patch(Rectangle((lx, 1.33), 2.2, 0.16, color=c, lw=0, clip_on=False))
        ax.text(lx + 3.2, 1.41, lab, va="center", ha="left", fontsize=6, color=INK, clip_on=False)
        lx += 3.2 + 1.25 * len(lab) + 4
    letter(fig, 2, 92, "a", "The check: per 100 applications in technology center 3600")

    # c: what the grant-trained evaluator assumes about the examiners' rejections
    ce = D["contraction_evaluators"]
    ce = ce[(ce["sample"] == TC) & (ce.cells == "Art unit × year × USPC class") & (ce.quintile == "2–5 (mean)")
            & (ce.measure == "renew_rate")].set_index("evaluator")
    me = D["marginal_estimates"]
    truth = est(me, HUMAN, "marginal")
    ax = axes(fig, 143, 58, 33, 25)
    items = [("Examiners' marginal cases", truth.estimate, truth.estimate - 1.96 * truth.se,
              truth.estimate + 1.96 * truth.se, INK, True)]
    for name, col, lab in [("Train on grant", DEC, "Trained on grant"),
                           ("Train on renewal (granted only)", OUTC, "Trained on renewal")]:
        imp = est(me, name, "implied")
        c = ce.loc[name]
        items += [(f"{lab}: implied", imp.estimate, imp.estimate - 1.96 * imp.se, imp.estimate + 1.96 * imp.se,
                   col, False),
                  (f"{lab}: own rejections", c.estimate, c.lo, c.hi, col, True)]
    ys = [5.2, 3.9, 3.0, 1.7, 0.8]
    ax.axvspan(100 * (truth.estimate - 1.96 * truth.se), 100 * (truth.estimate + 1.96 * truth.se), color=BAND, lw=0,
               zorder=0)
    for (lab, x, lo, hi, col, filled), y in zip(items, ys):
        dot(ax, 100 * x, y, col, 100 * lo, 100 * hi, filled=filled)
        ax.text(79.3, y, lab, ha="right", va="center", fontsize=6, color=INK, clip_on=False)
    ax.set_xlim(80, 93)
    ax.set_ylim(0.2, 5.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Renewal rate (%)", labelpad=1.5)
    hgrid(ax)
    ax.plot([], [], marker="o", ms=3.6, color=MUTED, mec="white", mew=0.6, ls="none", label="observed")
    ax.plot([], [], marker="o", ms=3.6, color="white", mec=MUTED, mew=0.9, ls="none", label="implied by the evaluator")
    ax.legend(loc="lower left", bbox_to_anchor=(-1.12, 1.0), ncol=2, columnspacing=1.0)
    letter(fig, 110, 92, "c", "Whose rejections does the evaluator see?")

    # b: signed errors on the marginal applications, all and female first inventors
    lm = D["llm_marginal_estimates"]
    rows = [("No model (mean of granted)", me, NO_MODEL, MUTED, "o"),
            ("Trained on renewal", me, "Train on renewal (granted only)", OUTC, "o"),
            ("Trained on citations", me, "Train on citations (granted only)", OUTC, "o"),
            ("Trained on first action", me, "Train on first action", DEC, "o"),
            ("Trained on grant", me, "Train on grant", DEC, "o"),
            ("LLM, zero-shot", lm, "Zero-shot", DEC, "D"),
            ("LLM, lenient examiners' examples", lm, "Few-shot, lenient examiners' decisions", DEC, "D"),
            ("LLM, strict examiners' examples", lm, "Few-shot, strict examiners' decisions", DEC, "D")]
    ypos = [8.4, 7.2, 6.2, 5.0, 4.0, 2.6, 1.6, 0.6]
    for k, (group, title) in enumerate([("All", "All applications"), ("Female", "Female first inventor")]):
        ax = axes(fig, 45 + k * 68, 10, 60, 37)
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for (lab, src, s, col, mk), y in zip(rows, ypos):
            e = est(src, s, "error", group)
            dot(ax, 100 * e.estimate, y, col, 100 * (e.estimate - 1.96 * e.se), 100 * (e.estimate + 1.96 * e.se),
                marker=mk, size=3.6 if mk == "o" else 3.2)
            if k == 0:
                ax.text(-0.04, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6,
                        color=INK)
        ax.set_xlim(-17, 9) if group == "Female" else ax.set_xlim(-8, 4)
        ax.set_ylim(0, 9.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        hgrid(ax)
        ax.set_title(title, loc="left", fontsize=6.5, color=INK)
        ax.set_xlabel("Signed error: implied − actual renewal (pp)", labelpad=1.5)
        ax.text(0.01, 0.995, "underrates", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=MUTED)
        lo, hi = ax.get_xlim()
        ax.text(-lo / (hi - lo) + 0.02, 0.995, "overrates", transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, color=MUTED)
    letter(fig, 2, 55, "b", "What each evaluator implies for the marginal applications (renewal at year 4)")
    save(fig, "fig1")


# ---------------------------------------------------------------- figure 2

def fig2(D: dict) -> None:
    fig = figure(180, 78)
    b = D["probes_bootstrap"]
    b = b[b.tech_center == TC].set_index("spec")
    hum = b.loc["Human examiners"]
    blocks = [
        ("a", "What the evaluator learns", 44, 27, [
            ("Human examiners: final grant", "Human examiners", INK, True),
            ("Human examiners: first action", "Human examiners (first action)", INK, False),
            ("Trained on first action", "Train on first action", DEC, True),
            ("Trained on final grant", "Train on grant", DEC, True),
            ("Trained on grant + prosecution", "Train on grant + prosecution history", DEC, True),
            ("Trained on renewal", "Train on renewal (granted only)", OUTC, True),
            ("Trained on citations", "Train on citations (granted only)", OUTC, True)]),
        ("b", "Whom it learns from (all trained on final grant)", 7, 22, [
            ("All examiners", "Train on grant", DEC, True),
            ("Lenient examiners", "Train on grant, lenient examiners", DEC, True),
            ("Strict examiners", "Train on grant, strict examiners", DEC, True),
            ("Examiners with a large own gap", "Train on grant, high-gap examiners", DEC, True),
            ("Examiners with a small own gap", "Train on grant, low-gap examiners", DEC, True)])]
    for lt, title, bottom, height, rows in blocks:
        ax = axes(fig, 43, bottom, 58, height)
        ax.axvspan(100 * hum.gap_lo, 100 * hum.gap_hi, color=BAND, lw=0, zorder=0)
        ax.axvline(100 * hum.gap, color=INK2, lw=0.6, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.4, zorder=1)
        n = len(rows)
        for i, (lab, spec, col, filled) in enumerate(rows):
            r = b.loc[spec]
            y = n - i
            dot(ax, 100 * r.gap, y, col, 100 * r.gap_lo, 100 * r.gap_hi, filled=filled)
            ax.text(-0.03, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6,
                    color=INK)
            if not spec.startswith("Human"):
                ax.text(1.1, y, f"{r.ratio:.2f}×".replace("-", "−"), transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=5.5, color=INK2, clip_on=False)
        ax.set_xlim(-19, 4)
        ax.set_ylim(0.4, n + 0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        hgrid(ax)
        if lt == "a":
            ax.set_xticklabels([])
            ax.text(1.1, n + 0.75, "× human", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=5.5, color=INK2, clip_on=False)
            c = b.loc["Contrast: Train on grant − Train on citations (granted only)"]
            ax.text(-18.5, 1.05, f"grant vs citations: {pp(c.gap)} pp", fontsize=5.5, color=INK2, va="center")
        else:
            ax.set_xlabel("Gender gap in grant decisions, female − male first inventor (pp)", labelpad=1.5)
            c = b.loc["Contrast: Train on grant, high-gap examiners − Train on grant, low-gap examiners"]
            c2 = b.loc["Contrast: Train on grant, lenient examiners − Train on grant, strict examiners"]
            bracket(ax, -3.9, 4, 3, f"{pp(c2.gap)} pp")
            bracket(ax, -3.9, 2, 1, f"{pp(c.gap)} pp")
        letter(fig, 2, bottom + height + 5.5, lt, title)
    fig.text(43.5 / 180, 75.0 / 78, "grey band: human examiners' gap, 95% CI", fontsize=5.5, color=INK2, va="top")

    # c: the decision–outcome frontier
    cv = D["marginal_curve"]
    cv = cv[cv["sample"] == TC].sort_values("weight_on_grant")
    ax = axes(fig, 127, 11, 49, 57)
    x = 100 * cv.renew_4y_error.to_numpy()
    y = cv.ratio.to_numpy()
    hg = abs(cv.human_gap.iloc[0])
    ax.axhline(1, color=INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.plot(x, y, color=MUTED, lw=0.9, zorder=2)
    for i in range(0, len(x) - 1, 2):
        ax.annotate("", xy=((x[i] + x[i + 1]) / 2 + 0.001, (y[i] + y[i + 1]) / 2 + 0.001), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=6), zorder=2)
    ax.scatter(x[1:-1], y[1:-1], s=6, color=MUTED, zorder=3, lw=0)
    for i, col, lab, dx, dy, ha in [(0, OUTC, "trained on renewal", -0.3, -0.07, "right"),
                                    (len(x) - 1, DEC, "trained on grant", 0.3, -0.04, "left")]:
        r = cv.iloc[i]
        ax.plot([x[i] - 196 * r.renew_4y_error_se, x[i] + 196 * r.renew_4y_error_se], [y[i], y[i]], color=col,
                lw=0.5, alpha=0.45, zorder=4)
        ax.plot([x[i], x[i]], [y[i] - 1.96 * r.gap_se / hg, y[i] + 1.96 * r.gap_se / hg], color=col, lw=0.5,
                alpha=0.45, zorder=4)
        dot(ax, x[i], y[i], col, size=4.6, z=6)
        ax.text(x[i] + dx, y[i] + dy, lab, ha=ha, va="top", fontsize=6, color=INK)
    j = int(np.argmax(y))
    ax.text(x[j] - 0.3, y[j] + 0.07, f"weight on grant {cv.weight_on_grant.iloc[j]:.1f}", ha="right",
            va="bottom", fontsize=5.5, color=INK2)
    ax.text(-7.0, 1.02, "human gap", fontsize=5.5, color=INK2, va="bottom", ha="left")
    ax.text(-0.15, 2.36, "no error", fontsize=5.5, color=INK2, va="top", ha="right")
    ax.set_xlim(-7.2, 0.9)
    ax.set_ylim(0.55, 2.4)
    ax.set_xlabel("Error on the marginal applications' renewal (pp)", labelpad=1.5)
    ax.set_ylabel("Gender gap in decisions (× human)", labelpad=2)
    hgrid(ax, "both")
    letter(fig, 112, 75, "c", "The decision–outcome frontier")
    save(fig, "fig2")


# ---------------------------------------------------------------- figure 3

def fig3(D: dict) -> None:
    fig = figure(180, 62)
    ex = D["ai_examiners"].sort_values("ai_examiner")
    sm = D["ai_examiners_summary"].iloc[0]

    # a: AI vs human leniency
    ax = axes(fig, 13, 11, 40, 42)
    xh, ya = 100 * ex.human_leniency_loo.to_numpy(), 100 * ex.ai_grant_rate_leniency.to_numpy()
    k, c0 = np.polyfit(xh, ya, 1)
    xs = np.array([xh.min() - 2, xh.max() + 2])
    ax.plot(xs, c0 + k * xs, color=MUTED, lw=0.7, zorder=2)
    ax.scatter(xh, ya, s=16, color=DEC, edgecolor="white", lw=0.6, zorder=3)
    ax.text(0.04, 0.96, f"r = {sm.corr_leniency:.2f}\n{int(sm.n_examiners)} AI examiners", transform=ax.transAxes,
            va="top", ha="left", fontsize=6, color=INK)
    ax.set_xlabel("Human examiner's grant rate (%)", labelpad=1.5)
    ax.set_ylabel("AI examiner's grant rate (%)", labelpad=2)
    hgrid(ax, "both")
    letter(fig, 1, 60, "a", "AI examiners copy the bar")

    # b: pairwise agreement, AI examiners ordered by leniency
    ag = D["ai_examiners_agreement"].set_index("ai_examiner").to_numpy(float)
    np.fill_diagonal(ag, np.nan)
    ax = axes(fig, 72, 11, 42, 42)
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    cmap.set_bad("white")
    im = ax.imshow(100 * ag, cmap=cmap, vmin=50, vmax=100, origin="upper", interpolation="nearest")
    n = ag.shape[0]
    ax.set_xticks([0, 4, 9, 14, n - 1], [1, 5, 10, 15, n])
    ax.set_yticks([0, 4, 9, 14, n - 1], [1, 5, 10, 15, n])
    for s in ["left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("AI examiner (strict → lenient)", labelpad=1.5)
    ax.set_ylabel("AI examiner (strict → lenient)", labelpad=2)
    cax = axes(fig, 116, 11, 2, 42)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.5)
    cb.set_label("Same decision (%)", fontsize=6, labelpad=2)
    letter(fig, 60, 60, "b", "Agreement falls with distance in leniency")

    # c: how many of the 20 AI examiners would grant each application
    dist = D["ai_examiners_count_distribution"]
    lot = D["ai_examiners_lottery"].set_index("group")
    ax = axes(fig, 137, 11, 40, 42)
    kx = dist.ai_examiners_granting.to_numpy()
    for col, lab, c, fill in [("male", "male first inventor", MUTED, True), ("female", "female first inventor", INK,
                                                                           False)]:
        share = 100 * dist[col].to_numpy() / dist[col].sum()
        if fill:
            ax.bar(kx, share, width=0.82, color="#c9c8c1", lw=0, zorder=2, label=lab)
        else:
            ax.step(np.r_[kx - 0.5, kx[-1] + 0.5], np.r_[share, share[-1]], where="post", color=c, lw=0.9, zorder=3,
                    label=lab)
    ax.axvspan(0.5, n - 0.5, color=BAND, alpha=0.6, lw=0, zorder=0)
    ax.text(n / 2, ax.get_ylim()[1] * 0.97, f"depends on the draw:\n{pp(lot.loc['Female', 'depends_on_examiner'])}% "
            f"female, {pp(lot.loc['Male', 'depends_on_examiner'])}% male", ha="center", va="top", fontsize=5.5,
            color=INK2)
    ax.set_xlim(-0.7, n + 0.7)
    ax.set_xticks([0, 5, 10, 15, n])
    ax.set_xlabel(f"AI examiners (of {n}) that would grant", labelpad=1.5)
    ax.set_ylabel("Applications (%)", labelpad=2)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), handlelength=1.0)
    hgrid(ax, "y")
    letter(fig, 125, 60, "c", "Same application, different AI examiner")
    save(fig, "fig3")


# ---------------------------------------------------------------- figure 4

def fig4(D: dict) -> None:
    fig = figure(180, 58)
    pv = D["assignment_permutation_pvalues"]
    pv = pv[(pv["index"] == "index_grant") & (pv.leniency == "lenient_grant")]
    names = [("au_year", "Art unit × year"), ("au_class", "× USPC class (main)"), ("au_subclass", "× USPC subclass")]
    for i, (cell, lab) in enumerate(names):
        ax = axes(fig, 12 + i * 33, 11, 28, 37)
        p = pv[pv.cells == cell].p.to_numpy()
        ax.hist(p, bins=np.linspace(0, 1, 21), weights=np.full(len(p), 20 / len(p)), color=OUTC, lw=0,
                rwidth=0.86, zorder=2)
        ax.axhline(1, color=INK2, lw=0.6, zorder=3)
        ax.set_ylim(0, 7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Density (uniform = 1)", labelpad=2)
        ax.set_title(lab, loc="left", fontsize=6, color=INK)
        ax.text(0.97, 0.9, f"{len(p):,} cells\n{100 * (p < 0.05).mean():.0f}% below 0.05", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, color=INK)
        hgrid(ax, "y")
    fig.text((12 + 49) / 180, 2.5 / 58, "Permutation p-value of within-cell balance across leniency quintiles",
             ha="center", fontsize=6.5, color=INK)
    letter(fig, 1, 56, "a", "Is assignment as good as random within cells?")

    # b: the grant-trained evaluator's error under alternative designs
    ae = D["assignment_resid_estimates"]
    ae = ae[(ae["sample"] == TC) & (ae.outcome == "renew_4y") & (ae.kind == "error")]
    designs = [("Main design (class cells)", "Leave-one-out grant rate (04)"),
               ("Residualized leniency", "Residualized leave-one-out (cell × entity size)"),
               ("Subclass cells", "Leave-one-out, art unit × year × USPC subclass cells")]
    ax = axes(fig, 141, 11, 35, 32)
    ax.axvline(0, color=INK2, lw=0.6)
    for i, (lab, inst) in enumerate(designs):
        y = len(designs) - i
        for s, col, dy in [("Train on grant", DEC, 0.14), ("Train on renewal (granted only)", OUTC, -0.14)]:
            t = ae[(ae.instrument == inst) & (ae.series == s) & (ae.group == "All")].iloc[0]
            dot(ax, 100 * t.estimate, y + dy, col, 100 * (t.estimate - 1.96 * t.se), 100 * (t.estimate + 1.96 * t.se))
        ax.text(-0.04, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color=INK)
    ax.set_ylim(0.4, len(designs) + 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-9.5, 3)
    ax.set_xticks([-8, -6, -4, -2, 0, 2])
    hgrid(ax)
    ax.set_xlabel("Signed error on marginal renewal (pp)", labelpad=1.5)
    ax.plot([], [], marker="o", ms=3.6, color=DEC, mec="white", mew=0.6, ls="none", label="trained on grant")
    ax.plot([], [], marker="o", ms=3.6, color=OUTC, mec="white", mew=0.6, ls="none", label="trained on renewal")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=0.8, ncol=1)
    letter(fig, 112, 56, "b", "The error under alternative designs")
    save(fig, "fig4")


# ---------------------------------------------------------------- figure 5

def fig5(D: dict) -> None:
    fig = figure(180, 62)
    cc = D["contraction_curves"]
    q = D["contraction_quintiles"]
    q = q[(q["sample"] == TC) & (q.cells == "Art unit × year × USPC class") & (q.quintile > 1)].sort_values("quintile")
    xmax = 100 * q.dG.max() * 1.08
    series = [("Random", MUTED, "random rejection"), ("Train on renewal (granted only)", OUTC, "trained on renewal"),
              ("Train on grant", DEC, "trained on grant")]
    for k, (title, lt) in enumerate([("Lapsed grants removed (per 100 applications)", "a"),
                                     ("Female share of the extra rejections (%)", "b")]):
        ax = axes(fig, 14 + k * 92, 11, 72, 42)
        for name, col, lab in series:
            c = cc[cc.evaluator == name]
            a, r, f = (100 * c[v].to_numpy() for v in ["rejected_share", "renewed_removed", "female_removed"])
            keep = (a <= xmax) & ((a >= 2) if k else (a >= 0))
            yv = (a - r) if k == 0 else 100 * f / np.maximum(a, 1e-9)
            ax.plot(a[keep], yv[keep], color=col, lw=1.1 if name != "Random" else 0.8, zorder=3, label=lab)
        ys = 100 * (q.dL if k == 0 else q.marg_female)
        ax.scatter(100 * q.dG, ys, s=16, color=INK, edgecolor="white", lw=0.6, zorder=5, label="stricter examiners")
        for xx, yy, qq in zip(100 * q.dG, ys, q.quintile):
            ax.annotate(f"Q{qq}", (xx, yy), textcoords="offset points", xytext=(3, -7 if k == 0 else 3), fontsize=5.5,
                        color=INK)
        if k:
            lo, hi = 100 * q.marg_female_lo, 100 * q.marg_female_hi
            ax.vlines(100 * q.dG, lo, hi, color=INK, lw=0.6, zorder=4)
            ax.set_ylim(0, 20)
            ax.set_yticks([0, 5, 10, 15, 20])
        else:
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper left", handlelength=1.4)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Extra rejections beyond the most lenient quintile (per 100 applications)", labelpad=1.5)
        hgrid(ax, "both")
        letter(fig, 2 + k * 92, 60, lt, title)
    save(fig, "fig5")


LEGENDS = """# Figure legends (draft)

**Figure 1 | Checking AI evaluators against quasi-random assignment of patent applications to examiners.**
**a**, Per 100 applications in technology center 3600, examiners below (strict) and above (lenient) the mean
leave-one-out leniency of their art unit × year × USPC class cell grant different numbers of applications (lenient
values: the strict mean plus the within-cell difference, with controls); outcomes are observed only for granted ones.
The marginal applications, granted by lenient but rejected by strict examiners, cannot be identified individually, but
their mean outcome can. **b**, Each evaluator's signed error on the marginal applications:
its implied renewal rate minus the actual one (two-stage least squares, leniency as instrument, art unit × year × USPC
class fixed effects, errors clustered by examiner; 95% CI). Orange, evaluators that learn decisions (including Qwen3-8B
asked whether the application will be granted); blue, evaluators that learn outcomes among granted applications;
grey, the mean outcome of granted applications. **c**, Renewal rates of the examiners' marginal applications (black),
the rate each evaluator implies for them (open), and the actual rate among the applications the evaluator itself would
reject in place of the stricter examiners (filled; contraction of the most lenient quintile's grants, 500 bootstrap
draws of art units). The grant-trained evaluator assumes that the examiners rejected what it would reject. Grey band:
95% CI of the actual marginal renewal rate.

**Figure 2 | What the evaluator learns matters more than whom it learns from.** **a**, **b**, Gender gap in grant
decisions (female minus male first inventor, art unit × year fixed effects) when each evaluator grants the examiners'
share of test applications; numbers at right give the gap relative to the human examiners'. 95% CIs from 1,000
examiner-clustered bootstrap draws. **c**, Evaluators that blend the renewal- and grant-trained scores, with the weight
on grant rising from 0 (blue) to 1 (orange) in steps of 0.1: their error on the marginal applications' renewal against
their gender gap in decisions. Lines at the end points: 95% CIs.

**Figure 3 | AI examiners reproduce their examiner's bar but not a taste.** One AI examiner per human examiner, for
the 20 examiners with the most pilot applications in technology center 3600, each trained on its examiner's own
decisions (median about 870). **a**, The AI examiner's grant rate on the same 3,972 applications against its human
examiner's leave-one-out grant rate. **b**, Share of the common applications on which two AI examiners decide alike,
examiners ordered from strict to lenient. **c**, How many of the 20 AI examiners would grant each application, by
gender of the first inventor; shaded, applications whose outcome depends on the AI examiner drawn.

**Figure 4 | Identification checks.** **a**, Within each assignment cell with at least five examiners, an index of
characteristics fixed at filing is regressed on examiner leniency quintiles, and the F statistic is compared with 1,000
within-cell permutations (after Kleinberg et al. 2018). Under as-good-as-random assignment the cell p-values are
uniform (line). Small subclass cells pile up at 1 because few distinct permutations exist. **b**, The evaluators'
signed error on the marginal applications' renewal under the main design, with leniency residualized on the cell's
grant rate, and with subclass fixed effects (95% CI).

**Figure 5 | Contraction of the most lenient examiners' grants.** Within art unit × year × USPC class cells with at
least five examiners, examiners are split into leniency quintiles and applications are weighted to a common mix of
cells. Starting from the grants of the most lenient quintile (Q1), an evaluator rejects additional applications, lowest
score first, within each cell; its extra rejections have observed outcomes, so no instrument is needed (after
Kleinberg et al. 2018). **a**, Grants that lapse by year 4 removed by the extra rejections. **b**, Female share of the
extra rejections, cumulative. Points: the stricter examiner quintiles relative to Q1 (b: 95% CIs from 500 bootstrap
draws of art units). Technology center 3600.
"""


def main() -> None:
    D = load()
    fig1(D)
    fig2(D)
    fig3(D)
    fig4(D)
    fig5(D)
    (OUT / "legends.md").write_text(LEGENDS)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
