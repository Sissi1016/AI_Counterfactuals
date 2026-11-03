"""Has the LLM seen these applications? A membership test (Min-K% Prob) and whether it changes the results.

Qwen3-8B was trained on web text that includes published patent applications and granted patents. If it remembers
an application, or its granted version, it could "predict" a grant or the citations from memory rather than from
the text. Following Shi et al. (2024, Detecting Pretraining Data from Large Language Models), each application's
title and abstract is scored by the mean log-probability of its 20% least likely tokens under Qwen3-8B (Min-20%
Prob): text seen in training has fewer very unlikely tokens and scores higher.

Tests (main environment, --analyze):
1. Does the score differ between granted and rejected applications in the same art unit and year? Memorized
   granted patents (whose text is also published at grant) would score higher.
2. Within each technology center, split applications at the median score. Compare the LLM's ranking quality for
   the grant decision (AUC) in the two halves, and repeat the marginal check with citations (21) and the gender
   gap in decisions in each half. If memory drove the LLM's answers, the more familiar half would look different.

Runs on GPUs in the LLM environment (one process per GPU, CFE_GPUS, default 0,1) and then on CPUs.
Run: .venv-llm/bin/python code/23_memorization.py; uv run python code/23_memorization.py --analyze
Input: data/processed/{pilot_inputs,pilot_text,citation_scores,cites_noself}.parquet, the record of cfe.check
Output: data/processed/memorization.parquet, output/memorization.md, output/tables/memorization_*.csv
"""
import argparse
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
SEED, K = 20261028, 0.2
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
# These GPUs draw ~575 W each at full load; small batches keep one card well under the server's power budget.
MAX_SEQS = int(os.environ.get("CFE_MAX_SEQS", "8"))
N_PER_TC = int(os.environ.get("CFE_MEM_N", "0"))  # applications scored per technology center (0 = all)
MIN_CHARS = 100


def texts() -> pd.DataFrame:
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    d = inp.merge(pd.read_parquet(P / "pilot_text.parquet", columns=["application_number", "title", "abstract"]),
                  on="application_number", how="left")
    t = (d.title.fillna("").str.strip() + "\n" + d.abstract.fillna("").str.strip()).str.strip()
    out = pd.DataFrame({"row": np.arange(len(d)), "application_number": d.application_number, "text": t})
    out["job"] = np.random.default_rng(SEED).permutation(len(out))
    if N_PER_TC:
        tc = pd.read_parquet(P / "pilot_sample.parquet", columns=["application_number", "tech_center"])
        out = out.merge(tc, on="application_number", how="left")
        keep = out.groupby("tech_center").job.rank(method="first") <= N_PER_TC
        out = out[keep].drop(columns="tech_center")
    return out


def worker(k: int) -> None:
    from vllm import LLM, SamplingParams
    t = texts()
    t = t[(t.job % len(GPUS) == k) & (t.text.str.len() >= MIN_CHARS)]
    llm = LLM(model=str(ROOT / "models/Qwen3-8B"), max_model_len=1024, gpu_memory_utilization=0.85, seed=SEED,
              enable_prefix_caching=False, max_num_seqs=MAX_SEQS, max_num_batched_tokens=1024)
    tok = llm.get_tokenizer()
    prompts = [tok.decode(tok.encode(x)[:600]) for x in t.text]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0),
                        use_tqdm=(k == 0))
    rows = []
    for r, o in zip(t.row, outs):
        lp = np.array([d[i].logprob for d, i in zip(o.prompt_logprobs[1:], o.prompt_token_ids[1:])], dtype=float)
        lp_sorted = np.sort(lp)
        kk = max(1, int(np.ceil(K * len(lp))))
        rows.append(dict(row=r, n_tokens=len(lp), mean_logprob=lp.mean(), min_k=lp_sorted[:kk].mean()))
    pd.DataFrame(rows).to_parquet(P / f"memorization_shard_{k}.parquet", index=False)


def score() -> None:
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    t = texts()[["row", "application_number"]]  # the scored subsample (CFE_MEM_N)
    parts = [pd.read_parquet(P / f"memorization_shard_{k}.parquet") for k in range(len(GPUS))]
    m = t.merge(pd.concat(parts), on="row", how="left").drop(columns="row")
    m.to_parquet(P / "memorization.parquet", index=False)
    for k in range(len(GPUS)):
        (P / f"memorization_shard_{k}.parquet").unlink()
    print(f"scored {m.min_k.notna().sum():,d} of {len(m):,d}; run: uv run python code/23_memorization.py --analyze")


# ---------------------------------------------------------------- analysis (main environment)

def analyze() -> None:
    import dataclasses
    import warnings

    from sklearn.metrics import roc_auc_score

    from cfe.check import HUMAN, NO_MODEL, PATENTS, assign_folds, decision_gaps, load_record, marginal, ols_coef, \
        recalibrate
    warnings.filterwarnings("ignore")
    outcome = "cites_ns"
    cit = dataclasses.replace(PATENTS, outcomes=((outcome, "Citations within five years of grant, excluding "
                                                           "self-citations"),))
    d = load_record(ROOT)
    c = pd.read_parquet(P / "cites_noself.parquet", columns=["application_number", "cites_5y_noself"])
    d = d.merge(c, on="application_number", how="left")
    d[outcome] = np.where(d.granted == 1, d.cites_5y_noself.fillna(0), np.nan)
    d = d.merge(pd.read_parquet(P / "memorization.parquet"), on="application_number", how="left")
    sc = pd.read_parquet(P / "citation_scores.parquet")
    names = ["LLM, zero-shot", "Train on grant", "Train on citations (granted only)"]
    dd = recalibrate(d.merge(sc[["application_number"] + names], on="application_number", how="left"), names,
                     assign_folds(d, cit), cit)
    dd["familiar"] = np.nan
    for tc in cit.units:
        m = (dd.tech_center == tc) & dd.min_k.notna()
        dd.loc[m, "familiar"] = (dd.loc[m, "min_k"] > dd.loc[m, "min_k"].median()).astype(float)

    # 1. granted vs rejected, within art unit × year
    rows = []
    for tc in cit.units:
        s = dd[(dd.tech_center == tc) & dd.min_k.notna()].copy()
        s["z"] = (s.min_k - s.min_k.mean()) / s.min_k.std()
        b, se = ols_coef(s, "z", "granted", fe="au_year", design=cit)
        rows.append(dict(tech_center=tc, scored=len(s), mean_min_k=s.min_k.mean(), sd_min_k=s.min_k.std(),
                         granted_minus_rejected_sd=b, se=se, auc_min_k_for_grant=roc_auc_score(s.granted, s.min_k)))
    gr = pd.DataFrame(rows)

    # 2. the LLM in the more and the less familiar half
    rows, est = [], []
    for half, lab in [(1.0, "More familiar half"), (0.0, "Less familiar half")]:
        s = dd[dd.familiar == half]
        for tc in cit.units:
            x = s[s.tech_center == tc]
            rows.append(dict(half=lab, tech_center=tc, n=len(x),
                             auc_llm_grant=roc_auc_score(x.granted, x["LLM, zero-shot"]),
                             auc_grant_model=roc_auc_score(x.granted, x["Train on grant"])))
        e, cc = marginal(s, [NO_MODEL] + names, design=cit)
        e["half"], cc["half"] = lab, lab
        g = decision_gaps(s, names, cit)
        g["half"] = lab
        est.append((e, cc, g))
    auc = pd.DataFrame(rows)
    e = pd.concat([x[0] for x in est])
    cc = pd.concat([x[1] for x in est])
    g = pd.concat([x[2] for x in est])
    for name, t in [("granted_vs_rejected", gr), ("auc_by_half", auc), ("estimates_by_half", e),
                    ("contrasts_by_half", cc), ("gaps_by_half", g)]:
        t.to_csv(OUT / f"tables/memorization_{name}.csv", index=False)

    def ci(b, se, nd=2, scale=1.0):
        return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"
    L = ["# Has the LLM seen these applications? (Min-20% Prob, Shi et al. 2024)", "",
         "Score: mean log-probability of the 20% least likely tokens of the title and abstract under Qwen3-8B; "
         "higher = more familiar to the model.", "",
         "| TC | Scored | Mean score | Granted − rejected (SD, art unit × year) | AUC of the score for grant |",
         "|---|---|---|---|---|"]
    for r in gr.itertuples():
        L.append(f"| {r.tech_center} | {r.scored:,} | {r.mean_min_k:.2f} | "
                 f"{ci(r.granted_minus_rejected_sd, r.se, 3)} | {r.auc_min_k_for_grant:.3f} |")
    L += ["", "## The LLM in the more and the less familiar half (split at the median within each TC)", "",
          "| Half | TC | Applications | AUC: LLM zero-shot | AUC: grant-trained evaluator |", "|---|---|---|---|---|"]
    for r in auc.itertuples():
        L.append(f"| {r.half} | {r.tech_center} | {r.n:,} | {r.auc_llm_grant:.3f} | {r.auc_grant_model:.3f} |")
    L += ["", "Marginal check with citations, TC 3600 (citations per patent):", "",
          "| Half | Truth | LLM error, all | LLM error, female | Gender gap in LLM decisions (× human) |",
          "|---|---|---|---|---|"]
    for half in e.half.unique():
        x = e[(e.half == half) & (e["sample"] == "3600")]
        t = x[(x.series == HUMAN) & (x.kind == "marginal") & (x.group == "All")].iloc[0]
        a = x[(x.series == "LLM, zero-shot") & (x.kind == "error") & (x.group == "All")].iloc[0]
        f = x[(x.series == "LLM, zero-shot") & (x.kind == "error") & (x.group == "Female")].iloc[0]
        gg = g[(g.half == half) & (g["sample"] == "3600") & (g.series == "LLM, zero-shot")].iloc[0]
        L.append(f"| {half} | {ci(t.estimate, t.se)} | {ci(a.estimate, a.se)} | {ci(f.estimate, f.se)} | "
                 f"{gg.ratio:.2f} |")
    (OUT / "memorization.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


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
        score()


if __name__ == "__main__":
    main()
