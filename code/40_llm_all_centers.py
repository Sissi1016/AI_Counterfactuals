"""Qwen3-8B, zero-shot, on the main sample: the eight technology centers, applications filed 2010–2014.

The prompt is 10's and 14's zero-shot prompt (title, abstract, first claim; no names; "Will this patent application
ultimately be granted as a U.S. patent?"). Pilot applications (TC 3600 and 1600) reuse 14's zero-shot scores; the
others are scored here with 10's worker (one process per GPU, CFE_GPUS):
    .venv-llm/bin/python code/40_llm_all_centers.py
The analysis runs in the main environment:
    uv run python code/40_llm_all_centers.py --analyze
Score: log P(Yes) − log P(No). Decisions: within each center, the LLM grants the share of applications the examiners
grant. Reported by center and pooled: AUC of the examiners' grants, and the gender gap in decisions (female − male
first inventor, art unit × year fixed effects, errors clustered by examiner) against the examiners' gap.

Input: data/processed/{all_sample,all_text,llm_scores_allpilot}.parquet
Output: data/processed/llm_scores_allcenters.parquet, output/llm_all_centers.md, output/tables/llm_all_centers.csv
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

TAG = "_allcenters"
os.environ["CFE_LLM_TAG"] = TAG  # 10's worker reads llm_prompts{TAG} and writes llm_shard{TAG}_k
ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
_spec = importlib.util.spec_from_file_location("llm10", ROOT / "code/10_llm_prompting.py")
L10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L10)


def build_prompts() -> pd.DataFrame:
    s = pd.read_parquet(P / "all_sample.parquet", columns=["application_number"])
    done = pd.read_parquet(P / "llm_scores_allpilot.parquet", columns=["application_number", "cond"])
    done = set(done[done.cond == "zero"].application_number)
    t = pd.read_parquet(P / "all_text.parquet", columns=["application_number", "title", "abstract", "claims"])
    d = s.merge(t, on="application_number")
    d = d[~d.application_number.isin(done)].reset_index(drop=True)
    rows = [{"application_number": r.application_number, "cond": "zero",
             "prompt": f"{L10.app_block(r)}\n\n{L10.QUESTION}"} for r in d.itertuples()]
    pr = pd.DataFrame(rows)
    pr["job"] = np.random.default_rng(L10.SEED).permutation(len(pr))
    pr.to_parquet(P / f"llm_prompts{TAG}.parquet", index=False)
    print(f"prompts: {len(pr):,d} (pilot applications reuse 14's scores: {len(done):,d})", flush=True)
    return pr


def score() -> None:
    path = P / f"llm_prompts{TAG}.parquet"
    if not path.exists():
        build_prompts()
    procs = [subprocess.Popen([sys.executable, str(ROOT / "code/10_llm_prompting.py"), "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(L10.GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    sc = pd.concat([pd.read_parquet(P / f"llm_shard{TAG}_{k}.parquet") for k in range(L10.N_GPU)])
    sc.to_parquet(P / f"llm_scores{TAG}.parquet", index=False)
    for k in range(L10.N_GPU):
        (P / f"llm_shard{TAG}_{k}.parquet").unlink()
    print("scores saved; analysis: uv run python code/40_llm_all_centers.py --analyze")


def analyze() -> None:
    import pyfixest as pf
    from sklearn.metrics import roc_auc_score
    old = pd.read_parquet(P / "llm_scores_allpilot.parquet")
    sc = pd.concat([old[old.cond == "zero"], pd.read_parquet(P / f"llm_scores{TAG}.parquet")])
    sc["score"] = np.log(sc.p_yes.clip(lower=1e-12)) - np.log(sc.p_no.clip(lower=1e-12))
    d = pd.read_parquet(P / "all_sample.parquet").merge(sc[["application_number", "score"]], on="application_number")
    tb = np.random.default_rng(L10.SEED).random(len(d))
    d["dec"] = 0
    for tc in TCS:
        m = (d.tech_center == tc).to_numpy()
        order = np.lexsort((tb[m], -d.score.to_numpy()[m]))
        k = int(round(d.granted[m].mean() * m.sum()))
        dec = np.zeros(m.sum(), int)
        dec[order[:k]] = 1
        d.loc[m, "dec"] = dec
    rows = []
    for smp, units in [(tc, [tc]) for tc in TCS] + [("All centers", TCS)]:
        s = d[d.tech_center.isin(units)]
        h = pf.feols("granted ~ female | au_year", data=s, vcov={"CRV1": "examiner_id"})
        g = pf.feols("dec ~ female | au_year", data=s, vcov={"CRV1": "examiner_id"})
        rows.append(dict(sample=smp, applications=len(s), auc=roc_auc_score(s.granted, s.score),
                         human_gap=h.coef()["female"], human_gap_se=h.se()["female"],
                         llm_gap=g.coef()["female"], llm_gap_se=g.se()["female"],
                         ratio=g.coef()["female"] / h.coef()["female"]))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "tables/llm_all_centers.csv", index=False)

    def ci(b, se):
        return f"{100 * b:.1f} [{100 * (b - 1.96 * se):.1f}, {100 * (b + 1.96 * se):.1f}]"
    L = ["# Qwen3-8B, zero-shot, in all eight technology centers (applications filed 2010–2014)", "",
         "Decisions at each center's grant rate; gaps: female − male first inventor, art unit × year fixed effects, "
         "errors clustered by examiner (pp).", "",
         "| Sample | Applications | AUC (grant) | Examiners' gap | LLM's gap | × examiners' |", "|---|---|---|---|---|---|"]
    for x in r.itertuples():
        L.append(f"| {x.sample} | {x.applications:,} | {x.auc:.3f} | {ci(x.human_gap, x.human_gap_se)} | "
                 f"{ci(x.llm_gap, x.llm_gap_se)} | {x.ratio:.2f} |")
    (OUT / "llm_all_centers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else score()


if __name__ == "__main__":
    main()
