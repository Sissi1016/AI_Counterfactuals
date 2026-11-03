"""Redraw all nine final appendix figures from the supplied aggregate plot data.

No raw patent data, embeddings, fitted models, or GPU are needed. The exact
manuscript graphics remain in results/appendix/figures; redraws go to
results/rebuilt/figures and may differ in fonts or spacing across systems.
"""
from __future__ import annotations

import importlib.util
import os
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ["CFE_LOSS"] = "poisson"
os.environ["CFE_SAMPLE"] = "all"

from appendix_support import ROOT

sys.path.insert(0, str(ROOT / "code"))
OUT = ROOT / "results/rebuilt/figures"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(figure, number):
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        metadata = {"Software": None} if suffix == "png" else {"CreationDate": None, "Creator": None}
        figure.savefig(OUT / f"Figure_A{number}.{suffix}", dpi=300, metadata=metadata)
    plt.close(figure)
    print(f"Rebuilt Figure A{number}")


def map_figure():
    import pandas as pd
    module = load("review_map", "36_geography_industry.py")
    module.P = ROOT / "resources"  # only state_shapes() reads P in this plotting path

    def select(fig, name):
        old_height = fig.get_size_inches()[1]
        for ax in list(fig.axes[:4]):
            ax.remove()
        fig.set_size_inches(fig.get_size_inches()[0], old_height / 2)
        for ax in fig.axes:
            x, y, w, h = ax.get_position().bounds
            ax.set_position([x, 2 * y, w, 2 * h])
        for text in list(fig.texts):
            x, y = text.get_position()
            if y > 0.5 or text.get_text() == "c":
                text.remove()
            else:
                text.set_position((x, 2 * y))
        save(fig, 1)

    module.J.save = select
    module.fig_maps(pd.read_csv(ROOT / "output/tables/geography_states.csv"),
                    pd.read_csv(ROOT / "output/tables/geo_state_leniency.csv"))


def citation_figures():
    module = load("review_zheng", "49_c1_zheng_figures.py")
    original = module.load_module
    mapping = {"fig_c1z_grant_rate": 2, "fig_c1z_reject_cites": 3, "fig_c1z_reject_grant": 4,
               "fig_c1z_calibration": 5}

    def loader(name, filename):
        result = original(name, filename)
        if filename == "20_journal_figures.py":
            result.save = lambda fig, stem: save(fig, mapping[stem])
        return result

    module.load_module = loader
    module.figures()


def contraction_figure():
    module = load("review_c1", "35_c1_kllm.py")

    def select(fig, path):
        for ax in list(fig.axes[:2]):
            ax.remove()
        fig.set_size_inches(6.6, 3.0)
        for ax, position, title in zip(fig.axes, [(0.10, 0.23, 0.34, 0.65), (0.63, 0.23, 0.34, 0.65)],
                                       ["Unobservables", "Contraction"], strict=True):
            ax.set_position(position)
            ax.set_title(title)
        save(fig, 6)

    module.save = select
    module.figure_ext = lambda *args: None
    module.figures_from_tables()


def score_figure():
    import numpy as np
    import pandas as pd
    module = load("review_score", "47_c1_margin_validation.py")
    original = module.load_module

    def loader(name, filename):
        result = original(name, filename)
        if filename == "20_journal_figures.py":
            result.save = lambda fig, stem: save(fig, 7)
        return result

    module.load_module = loader
    levels = pd.read_csv(ROOT / "output/tables/c1mv_poisson_levels.csv", dtype={"quintile": str})
    metrics = pd.read_csv(ROOT / "output/tables/c1mv_poisson_metrics.csv")
    distribution = pd.read_csv(ROOT / "output/tables/c1mv_poisson_ratio_distribution.csv")
    keys = list(levels.quantity.unique()) + [f"cdf_{who}_{point}" for who in ("s", "c") for point in module.GRID]
    K = {key: i for i, key in enumerate(keys)}
    data = {"keys": K}
    for q in ("2", "3", "4", "5", "pooled"):
        data[q] = np.zeros((1, len(keys)))
    for who, label in (("s", "score"), ("c", "guess")):
        rows = distribution[distribution.predictor == label].sort_values("left")
        assert len(rows) == len(module.GRID) - 1
        cumulative = np.r_[0, np.cumsum(rows.mass.to_numpy())]
        for point, value in zip(module.GRID, cumulative, strict=True):
            data["pooled"][0, K[f"cdf_{who}_{point}"]] = value
    indexed = metrics.set_index("quantity")
    for share in (50, 80):
        data[f"F_s_{share}"] = [indexed.loc[f"factor holding {share}% of marginal applications, score", "value"]]
    module.score_check_figure(data, metrics, summaries=levels)


def twin_figures():
    import pandas as pd
    module = load("review_twins", "45_twins_c4_extra.py")

    def select(fig, stem):
        number = {"fig_c2_twins_ab": 8, "fig_c2_twins_cd": 9}[stem]
        fig.axes[0].remove()
        for text in list(fig.texts):
            text.remove()
        if number == 8:
            fig.set_size_inches(4.6, 4.0)
            fig.axes[0].set_position([0.17, 0.15, 0.77, 0.64])
            fig.text(0.17, 0.97, "Decisions that differ, by gap in grant rates", va="top", fontsize=8)
        else:
            fig.set_size_inches(4.3, 3.7)
            fig.axes[0].set_position([0.20, 0.17, 0.74, 0.58])
        save(fig, number)

    module.J.save = select
    folder = ROOT / "output/tables"
    gt = pd.read_csv(folder / "c2x_groups.csv")
    ch = pd.read_csv(folder / "c2x_decision_change.csv")
    summary = pd.read_csv(folder / "c2x_summary.csv").iloc[0].to_dict()
    robust = pd.read_csv(folder / "c2x_decision_change_robust.csv")
    extra = {"binned": pd.read_csv(folder / "c2x_pairs_binned.csv")}
    module.figure(gt, ch, extra, summary, robust)


def proposal_figures():
    """Reuse the main-text plot functions, writing redraws separately from the references."""
    import matplotlib.pyplot as plt
    folder = ROOT / "results/rebuilt/proposal"
    folder.mkdir(parents=True, exist_ok=True)
    for number, filename in [(1, "43_c1_extra.py"), (2, "44_c3_extra.py")]:
        module = load(f"review_proposal_{number}", filename)
        original = module.load_module

        def save_proposal(figure, stem, number=number):
            for suffix in ("png", "pdf"):
                metadata = {"Software": None} if suffix == "png" else {"CreationDate": None, "Creator": None}
                figure.savefig(folder / f"Figure_{number}.{suffix}", dpi=300, metadata=metadata)
            plt.close(figure)
            print(f"Rebuilt proposal Figure {number}")

        def loader(name, source_file, original=original, callback=save_proposal):
            result = original(name, source_file)
            if source_file == "20_journal_figures.py":
                result.save = callback
            return result

        module.load_module = loader
        module.figure()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    map_figure()
    citation_figures()
    contraction_figure()
    score_figure()
    twin_figures()
    assert all((OUT / f"Figure_A{n}.png").exists() and (OUT / f"Figure_A{n}.pdf").exists() for n in range(1, 10))
    print("Rebuilt all 9 figures from aggregate results.")
    proposal_figures()
    folder = ROOT / "results/rebuilt/proposal"
    assert all((folder / f"Figure_{n}.{ext}").exists() for n in (1, 2) for ext in ("png", "pdf"))


if __name__ == "__main__":
    main()
