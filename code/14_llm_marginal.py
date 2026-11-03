"""LLM evaluators on the marginal applications: Qwen3-8B scores every pilot application, and the check from 12 tells
whether its implied outcome for the applications that strict and lenient examiners decide differently is right.

Conditions (prompt format as in 10: title, abstract, first claim; no names)
- zero: zero-shot.
- few_lenient / few_strict: K past applications from the same technology center with their outcomes, drawn only from
  lenient or only from strict examiners. Examples come from the other cross-fitting folds of 12 (folds are grouped by
  examiner) and leniency is measured on those folds only, so no prompt shows a decision of the applicant's own
  examiner.

Scoring runs on GPUs in the LLM environment and reuses the worker of 10 (one process per GPU, see CFE_GPUS):
    .venv-llm/bin/python code/14_llm_marginal.py
The analysis runs in the main environment:
    uv run python code/14_llm_marginal.py --analyze
It treats each condition's log-odds of "Yes" as an evaluator score and reports the marginal check (cfe.check: implied
renewal and high-citation rates of the marginal applications, signed errors by applicant group, balance), and the
gender gap in decisions made at the human grant rate.

Input: data/processed/{pilot_sample,pilot_text,crossfit_preds}.parquet
Output: data/processed/llm_prompts_allpilot.parquet, data/processed/llm_scores_allpilot.parquet,
        output/llm_marginal.md, output/tables/llm_marginal_*.csv
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

TAG = "_allpilot"
os.environ["CFE_LLM_TAG"] = TAG  # 10's worker reads llm_prompts{TAG} and writes llm_shard{TAG}_k
ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
_spec = importlib.util.spec_from_file_location("llm10", ROOT / "code/10_llm_prompting.py")
L10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L10)
TCS = ["3600", "1600"]
CONDS = {"zero": "Zero-shot", "few_lenient": "Few-shot, lenient examiners' decisions",
         "few_strict": "Few-shot, strict examiners' decisions"}


def build_prompts() -> pd.DataFrame:
    s = pd.read_parquet(P / "pilot_sample.parquet")
    t = pd.read_parquet(P / "pilot_text.parquet")
    f = pd.read_parquet(P / "crossfit_preds.parquet", columns=["application_number", "fold"])
    d = s.merge(t, on="application_number").merge(f, on="application_number")
    rng = np.random.default_rng(L10.SEED)
    pools = {}
    for tc in TCS:
        for k in sorted(d.fold.unique()):
            other = d[(d.tech_center == tc) & (d.fold != k)]
            other = other.assign(lenient=other.examiner_id.map(L10.examiner_lenient(other)))
            for flag in (True, False):
                pools[(tc, k, flag)] = other[other.lenient == flag]
    rows = []
    for i, r in enumerate(d.itertuples()):
        base = L10.app_block(r)
        conds = {"zero": base}
        for flag, cname in [(True, "few_lenient"), (False, "few_strict")]:
            ex = pools[(r.tech_center, r.fold, flag)].sample(L10.K, random_state=int(rng.integers(1 << 31)))
            shots = "\n\n".join(
                f"Example {j + 1}:\n{L10.app_block(e)}\nOutcome: {'Granted' if e.granted else 'Not granted'}"
                for j, e in enumerate(ex.itertuples()))
            conds[cname] = (f"Here are past applications from the same technology area and their outcomes.\n\n"
                            f"{shots}\n\nNow the new application:\n{base}")
        for c, body in conds.items():
            rows.append({"application_number": r.application_number, "cond": c,
                         "prompt": f"{body}\n\n{L10.QUESTION}"})
        if i % 10000 == 0:
            print(f"prompts for {i:,d} / {len(d):,d} applications", flush=True)
    pr = pd.DataFrame(rows)
    # shuffle job ids: 10's worker k takes job % N_GPU == k, and in row order the long few-shot prompts would all
    # land on the same GPUs
    pr["job"] = np.random.default_rng(L10.SEED).permutation(len(pr))
    pr.to_parquet(P / f"llm_prompts{TAG}.parquet", index=False)
    return pr


def score() -> None:
    path = P / f"llm_prompts{TAG}.parquet"
    pr = pd.read_parquet(path, columns=["job"]) if path.exists() else build_prompts()
    print(f"prompts: {len(pr):,d}", flush=True)
    procs = [subprocess.Popen([sys.executable, str(ROOT / "code/10_llm_prompting.py"), "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(L10.GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    sc = pd.concat([pd.read_parquet(P / f"llm_shard{TAG}_{k}.parquet") for k in range(L10.N_GPU)])
    sc.to_parquet(P / f"llm_scores{TAG}.parquet", index=False)
    for k in range(L10.N_GPU):
        (P / f"llm_shard{TAG}_{k}.parquet").unlink()
    print("scores saved; run the analysis with the main env: uv run python code/14_llm_marginal.py --analyze")


def analyze() -> None:
    import warnings

    from sklearn.metrics import roc_auc_score

    from cfe.check import check, load_record, report

    warnings.filterwarnings("ignore", message=".*singleton.*")
    sc = pd.read_parquet(P / f"llm_scores{TAG}.parquet")
    mass = sc.p_yes + sc.p_no
    print("Yes/No probability mass:", mass.describe()[["mean", "min", "50%"]].round(4).to_dict())
    missing = (sc.p_yes == 0).groupby(sc.cond).mean().round(3).to_dict()
    print("Share with Yes outside the top 20, by condition:", missing)
    floor = 1e-12  # as in 10: log-odds, with a floor when Yes or No is missing from the top 20
    sc["score"] = np.log(sc.p_yes.clip(lower=floor)) - np.log(sc.p_no.clip(lower=floor))
    wide = sc.pivot(index="application_number", columns="cond", values="score")[list(CONDS)]
    wide = wide.rename(columns=CONDS).reset_index()
    rec = load_record(ROOT)
    res = check(rec, wide)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for k in ["estimates", "contrasts", "gaps", "balance"]:
        res[k].to_csv(OUT / "tables" / f"llm_marginal_{k}.csv", index=False)
    d = rec.merge(wide, on="application_number")
    rows = []
    for tc in TCS:
        x = d[d.tech_center == tc]
        g = x[x.granted == 1]
        for c in CONDS.values():
            rows.append(dict(tech_center=tc, evaluator=c, auc_grant=roc_auc_score(x.granted, x[c]),
                             auc_renewal=roc_auc_score(g.renew_4y.astype(int), g[c]),
                             auc_cites=roc_auc_score(g.cites_hi.astype(int), g[c])))
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / "tables" / "llm_marginal_auc.csv", index=False)
    md = report(res, "LLM evaluators (Qwen3-8B) on the marginal applications")
    md += ("\n## Ranking quality\n\nAUC of each condition's log-odds of \"Yes\": for the grant decision (all "
           "applications) and for renewal and high citations (granted applications).\n\n"
           + auc.round(3).to_markdown(index=False) + "\n")
    (OUT / "llm_marginal.md").write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else score()


if __name__ == "__main__":
    main()
