"""Evaluators trained on different outcomes from an LLM's own representation (Qwen3-8B).

Each pilot application's text (title, abstract, first claim, as in the prompts of 10 and 14; no names) is passed
through Qwen3-8B once; the mean of the last hidden layer over its tokens is a 4,096-dimensional representation.
Evaluators are trained on it exactly as in 12 (cross-fitted with folds grouped by examiner, penalty chosen by inner
cross-validation, recalibrated on granted applications) for four labels: first action, grant, renewal, and citations.
The marginal check (cfe.check) and the gender gap in decisions at the human grant rate then compare them with the
text-embedding evaluators of 12 (Qwen3-Embedding-0.6B).

Embedding runs on GPUs in the LLM environment (one process per GPU, see CFE_GPUS; jobs are shuffled across GPUs):
    .venv-llm/bin/python code/15_llm_features.py
Training and the check run in the main environment:
    uv run python code/15_llm_features.py --analyze

Input: data/processed/{pilot_inputs,pilot_text}.parquet; for --analyze also the inputs of cfe.check.load_record and
       output/tables/marginal_{estimates,fit}.csv (12) for the comparison
Output: data/processed/emb/q8.npy (row-aligned with pilot_inputs), data/processed/crossfit_q8.parquet,
        output/llm_features.md, output/tables/llm_features_*.csv
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
OUT = ROOT / "output"
TEXTS = P / "q8_texts.parquet"
SEED = 20261028
# GPU device indices; two GPUs (0 and 1) by default, override with CFE_GPUS.
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
SPECS = [  # names match 12, so the comparison and the decision-outcome blend line up
    ("Train on first action", "q8", "fa_allow"),
    ("Train on grant", "q8", "granted"),
    ("Train on renewal (granted only)", "q8", "renew_4y"),
    ("Train on citations (granted only)", "q8", "cites_hi"),
]
TAG = "Qwen3-8B"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- embedding (LLM environment)

def build_texts() -> pd.DataFrame:
    L10 = load_module("llm10", "10_llm_prompting.py")
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    d = inp.merge(pd.read_parquet(P / "pilot_text.parquet"), on="application_number", how="left")
    texts = [L10.app_block(r) for r in d.itertuples()]
    t = pd.DataFrame({"row": np.arange(len(d)), "application_number": d.application_number, "text": texts})
    t["job"] = np.random.default_rng(SEED).permutation(len(t))  # spread work evenly across GPUs
    t.to_parquet(TEXTS, index=False)
    return t


def worker(k: int) -> None:
    from vllm import LLM
    from vllm.config import PoolerConfig
    t = pd.read_parquet(TEXTS)
    t = t[t.job % len(GPUS) == k]
    llm = LLM(model=str(ROOT / "models/Qwen3-8B"), runner="pooling", convert="embed", max_model_len=4096,
              gpu_memory_utilization=0.85, seed=SEED,
              pooler_config=PoolerConfig(seq_pooling_type="MEAN", use_activation=False))
    outs = llm.embed(t.text.tolist(), use_tqdm=(k == 0))
    emb = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs]).astype(np.float16)
    np.save(E / f"q8_shard_{k}.npy", emb)
    np.save(E / f"q8_rows_{k}.npy", t.row.to_numpy())


def embed() -> None:
    t = pd.read_parquet(TEXTS) if TEXTS.exists() else build_texts()
    print(f"texts: {len(t):,d}", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    emb = None
    for k in range(len(GPUS)):
        part, rows = np.load(E / f"q8_shard_{k}.npy"), np.load(E / f"q8_rows_{k}.npy")
        if emb is None:
            emb = np.zeros((len(t), part.shape[1]), dtype=np.float16)
        emb[rows] = part
        (E / f"q8_shard_{k}.npy").unlink()
        (E / f"q8_rows_{k}.npy").unlink()
    np.save(E / "q8.npy", emb)
    print(f"saved {E / 'q8.npy'} {emb.shape}; run: uv run python code/15_llm_features.py --analyze")


# ---------------------------------------------------------------- training and the check (main environment)

def analyze() -> None:
    import warnings

    from sklearn.metrics import roc_auc_score

    from cfe.check import check, decision_gaps, load_record, report

    warnings.filterwarnings("ignore", message=".*singleton.*")
    m12 = load_module("m12", "12_marginal_outcomes.py")
    m12.SPECS = SPECS  # reuse 12's cross-fitting, recalibration, and blends with the LLM representation
    # stay within 16 cores (run under taskset -c 0-15), and try a short penalty grid around 12's choices
    m12.N_JOBS, m12.THREADS = 8, 2
    m12.C_GRID = [0.005, 0.0005, 0.00005, 0.000005]  # 12's choices plus one step beyond the smallest
    rec = load_record(ROOT)
    X = {"q8": np.load(E / "q8.npy").astype(np.float32)}
    cf, chosen = m12.crossfit(rec, X)
    cf.to_parquet(P / "crossfit_q8.parquet", index=False)
    d = pd.concat([rec, cf.drop(columns="application_number")], axis=1)

    names = [n for n, _, _ in SPECS]
    cal = pd.DataFrame({"application_number": d.application_number})
    for o in ["renew_4y", "cites_hi"]:
        for n in names:
            cal[f"{o}::{TAG}: {n}"] = d[f"{o}::{n}"]
    res = check(rec, cal)
    raw = d[[f"score::{n}" for n in names]].copy()
    raw.columns = [f"{TAG}: {n}" for n in names]
    gaps = decision_gaps(pd.concat([rec, raw], axis=1), list(raw.columns))
    res["gaps"] = gaps
    rows = []
    for tc in ["3600", "1600"]:
        x = d[d.tech_center == tc]
        for n, _, lab in SPECS:
            ok = x[lab].notna()
            rows.append(dict(tech_center=tc, series=n, label=lab,
                             auc_q8=roc_auc_score(x.loc[ok, lab].astype(int), x.loc[ok, f"score::{n}"]),
                             C=", ".join(f"{c:g}" for c in sorted(chosen[(chosen.tech_center == tc)
                                                                         & (chosen.series == n)].C.unique()))))
    auc = pd.DataFrame(rows)
    for k in ["estimates", "contrasts", "gaps", "balance"]:
        res[k].to_csv(OUT / "tables" / f"llm_features_{k}.csv", index=False)
    auc.to_csv(OUT / "tables" / "llm_features_auc.csv", index=False)

    # side by side with the text-embedding evaluators of 12
    e12 = pd.read_csv(OUT / "tables/marginal_estimates.csv", dtype={"sample": str})
    f12 = pd.read_csv(OUT / "tables/marginal_fit.csv", dtype={"sample": str})
    e8 = res["estimates"]
    comp = []
    for tc in ["3600", "1600"]:
        for n, _, lab in SPECS:
            def err(e, series, group, o="renew_4y"):
                r = e[(e["sample"] == tc) & (e.outcome == o) & (e.series == series) & (e.kind == "error")
                      & (e.group == group)].iloc[0]
                return r.estimate, r.se
            row = dict(tech_center=tc, series=n,
                       auc_06b=f12[(f12["sample"] == tc) & (f12.series == n)].auc_own.iloc[0],
                       auc_8b=auc[(auc.tech_center == tc) & (auc.series == n)].auc_q8.iloc[0])
            for key, src, s in [("06b", e12, n), ("8b", e8, f"{TAG}: {n}")]:
                for g in ["All", "Female"]:
                    b, se = err(src, s, g)
                    row[f"renew_err_{g.lower()}_{key}"], row[f"renew_err_{g.lower()}_{key}_se"] = b, se
            comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "tables" / "llm_features_comparison.csv", index=False)

    def ci(b, se):
        return f"{100 * b:.1f} [{100 * (b - 1.96 * se):.1f}, {100 * (b + 1.96 * se):.1f}]"
    md = ["# Evaluators trained on different outcomes from Qwen3-8B's representation", "",
          "Same training and check as 12, with the mean last-layer state of Qwen3-8B (4,096 dimensions) in place of "
          "Qwen3-Embedding-0.6B. AUC: cross-fitted, on the evaluator's own label. Errors: signed error on the "
          "marginal applications' renewal (pp).", "",
          "| TC | Trained on | AUC 0.6B | AUC 8B | Error 0.6B | Error 8B | Error, female, 0.6B | Error, female, 8B |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        md.append(f"| {r.tech_center} | {r.series} | {r.auc_06b:.3f} | {r.auc_8b:.3f} | "
                  f"{ci(r.renew_err_all_06b, r.renew_err_all_06b_se)} | {ci(r.renew_err_all_8b, r.renew_err_all_8b_se)} | "
                  f"{ci(r.renew_err_female_06b, r.renew_err_female_06b_se)} | "
                  f"{ci(r.renew_err_female_8b, r.renew_err_female_8b_se)} |")
    md += ["", report(res, "Full check, Qwen3-8B evaluators").split("\n", 1)[1]]
    (OUT / "llm_features.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:14]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.worker is not None:
        worker(a.worker)
    elif a.analyze:
        analyze()
    else:
        embed()


if __name__ == "__main__":
    main()
