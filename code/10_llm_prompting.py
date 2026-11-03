"""LLM prompting experiments (Qwen3-8B, vLLM, sharded across GPUs; see GPUS): zero-shot, name-swap audit, and
few-shot prompts with different "teachers".

Question: will this application ultimately be granted? We read the probabilities that the model's first output token
is "Yes" or "No": prob = P(Yes) / (P(Yes) + P(No)); rankings use the log-odds log P(Yes) − log P(No).
Conditions
- zero_noname: zero-shot, given only the title, abstract, and first claim.
- zero_f / zero_m: zero-shot plus the inventor list, with the first inventor's first name replaced by a female or a
  male name (the paired name lists from 08).
- few_lenient / few_strict: the prompt includes K past training-set examples (same technology center, drawn at
  random), taken only from lenient or only from strict examiners, each with its actual outcome (Granted / Not
  granted). No names.
Examiner leniency is computed on the training set only (as in 09).
Sample: a random draw from the test set, N_3600 applications in TC 3600 and N_1600 in TC 1600 (environment variables
CFE_LLM_N3600 / CFE_LLM_N1600, where "all" means the full test set; CFE_LLM_TAG adds a suffix to this run's files).
The analysis step adds 95% intervals for key contrasts from an examiner-clustered bootstrap.

Run: .venv-llm/bin/python code/10_llm_prompting.py
     uv run python code/10_llm_prompting.py --analyze   (analysis step, main environment)
Output: data/processed/llm_scores.parquet, output/llm_prompting.md, output/tables/llm_prompting.csv
"""
import argparse
import math
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
MODEL = str(ROOT / "models/Qwen3-8B")
SEED, K = 20261028, 6
N_3600 = os.environ.get("CFE_LLM_N3600", "2000")  # "all" means the full test set
N_1600 = os.environ.get("CFE_LLM_N1600", "1000")
TAG = os.environ.get("CFE_LLM_TAG", "")  # runs of different sizes keep separate files, e.g. TAG="_full"
# GPU device indices; two GPUs (0 and 1) by default, override with CFE_GPUS.
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
N_GPU = len(GPUS)
TOK_TITLE, TOK_ABSTRACT, TOK_CLAIM = 40, 300, 500  # ~850 tokens per application; 6 examples + the new one ~6,000
SYSTEM = "You are an experienced patent examiner at the United States Patent and Trademark Office."
QUESTION = ("Will this patent application ultimately be granted as a U.S. patent? "
            "Answer with a single word: Yes or No.")


def first_claim(claims: str) -> str:
    m = re.split(r"\s2\s?\.\s", claims, maxsplit=1)
    return m[0]


_TOK = None
_CACHE: dict[str, str] = {}


def cut(text: str, n: int) -> str:
    """Truncate by tokens: sequences and chemical formulas run ~1 token per character, so character cuts are unreliable."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    ids = _TOK(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else _TOK.decode(ids[:n]) + " ..."


def app_block(r, names: str | None = None) -> str:
    key = r.application_number
    if key not in _CACHE:
        _CACHE[key] = (f"Title: {cut(r.title, TOK_TITLE)}\nAbstract: {cut(r.abstract, TOK_ABSTRACT)}\n"
                       f"First claim: {cut(first_claim(r.claims), TOK_CLAIM)}")
    head = f"{names}\n" if names else ""
    return head + _CACHE[key]


def examiner_lenient(tr: pd.DataFrame) -> pd.Series:
    tr = tr.copy()
    tr["r"] = tr["granted"] - tr.groupby("au_year")["granted"].transform("mean")
    ex = tr.groupby("examiner_id").agg(n=("r", "size"), len_=("r", "mean"))
    w = np.repeat(ex["len_"].to_numpy(), ex["n"].to_numpy())
    return ex["len_"] > np.median(w)


def take(x: pd.DataFrame, n: str) -> pd.DataFrame:
    return x if n == "all" else x.sample(int(n), random_state=SEED)


def build_prompts() -> pd.DataFrame:
    s = pd.read_parquet(P / "pilot_sample.parquet")
    t = pd.read_parquet(P / "pilot_text.parquet")
    inp = pd.read_parquet(P / "pilot_inputs.parquet",
                          columns=["application_number", "test", "s1_f", "s1_m"])
    d = s.merge(t, on="application_number").merge(inp, on="application_number")
    rng = np.random.default_rng(SEED)
    tr = d[~d.test].copy()
    tr["lenient"] = tr["examiner_id"].map(examiner_lenient(tr))
    te = pd.concat([
        take(d[d.test & (d.tech_center == "3600")], N_3600),
        take(d[d.test & (d.tech_center == "1600")], N_1600),
    ])
    pools = {(tc, flag): tr[(tr.tech_center == tc) & (tr.lenient == flag)] for tc in ["3600", "1600"]
             for flag in [True, False]}
    rows = []
    for r in te.itertuples():
        base = app_block(r)
        names_f = r.s1_f.split("\n\n", 1)[0]
        names_m = r.s1_m.split("\n\n", 1)[0]
        conds = {"zero_noname": base, "zero_f": app_block(r, names_f), "zero_m": app_block(r, names_m)}
        for flag, cname in [(True, "few_lenient"), (False, "few_strict")]:
            ex = pools[(r.tech_center, flag)].sample(K, random_state=int(rng.integers(1 << 31)))
            shots = "\n\n".join(
                f"Example {i + 1}:\n{app_block(e)}\nOutcome: {'Granted' if e.granted else 'Not granted'}"
                for i, e in enumerate(ex.itertuples()))
            conds[cname] = (f"Here are past applications from the same technology area and their outcomes.\n\n"
                            f"{shots}\n\nNow the new application:\n{base}")
        for c, body in conds.items():
            rows.append({"application_number": r.application_number, "cond": c,
                         "prompt": f"{body}\n\n{QUESTION}"})
    pr = pd.DataFrame(rows)
    pr["job"] = np.arange(len(pr))
    pr.to_parquet(P / f"llm_prompts{TAG}.parquet", index=False)
    te[["application_number", "tech_center", "au_year", "examiner_id", "female", "granted", "fa_allow"]] \
        .to_parquet(P / f"llm_sample{TAG}.parquet", index=False)
    return pr


def worker(k: int) -> None:
    from vllm import LLM, SamplingParams
    pr = pd.read_parquet(P / f"llm_prompts{TAG}.parquet")
    pr = pr[pr.job % N_GPU == k]
    llm = LLM(model=MODEL, max_model_len=12288, gpu_memory_utilization=0.85, seed=SEED)
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in pr.prompt]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    outs = llm.chat(msgs, sp, use_tqdm=(k == 0), chat_template_kwargs={"enable_thinking": False})
    py, pn, n_top = [], [], []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        # "Yes", " yes", "YES", etc. are separate tokens and must be summed. Do not de-duplicate by lowercase in a dict:
        # a low-probability variant would overwrite the high-probability one.
        s = {"yes": 0.0, "no": 0.0}
        for v in lp.values():
            key = v.decoded_token.strip().lower()
            if key in s:
                s[key] += math.exp(v.logprob)
        py.append(s["yes"])
        pn.append(s["no"])
        n_top.append(o.outputs[0].text.strip())
    pr = pr.assign(p_yes=py, p_no=pn, top_token=n_top).drop(columns="prompt")
    pr.to_parquet(P / f"llm_shard{TAG}_{k}.parquet", index=False)


def analyze() -> None:
    import pyfixest as pf  # only installed in the main environment; run the analysis step there
    from sklearn.metrics import roc_auc_score
    sc = pd.read_parquet(P / f"llm_scores{TAG}.parquet")
    sm = pd.read_parquet(P / f"llm_sample{TAG}.parquet")
    mass = sc.p_yes + sc.p_no
    print("Yes/No probability mass (should be close to 1):", mass.describe()[["mean", "min", "50%"]].round(4).to_dict())
    print("Share with Yes outside the top 20 candidates, by condition:",
          (sc.p_yes == 0).groupby(sc.cond).mean().round(3).to_dict())
    sc["prob"] = sc.p_yes / mass.replace(0, np.nan)
    # The model is often extremely confident, so probabilities pile up near 0 or 1. Rankings and the name-swap audit
    # use log-odds; when Yes is missing from the top 20, a floor value stands in for it.
    floor = 1e-12
    sc["score"] = np.log(sc.p_yes.clip(lower=floor)) - np.log(sc.p_no.clip(lower=floor))
    rows, ci_rows = [], []
    conds = ["zero_noname", "zero_f", "zero_m", "few_lenient", "few_strict"]
    for tc in ["3600", "1600"]:
        base = sm[sm.tech_center == tc]
        rate = base.granted.mean()
        h = pf.feols("granted ~ female | au_year", data=base, vcov={"CRV1": "examiner_id"})
        hg = float(h.coef()["female"])
        rows.append(dict(tech_center=tc, cond="Human examiners", mean_score=rate, auc_granted=np.nan,
                         gap_decision=hg, se=float(h.se()["female"]), ratio=1.0))
        decs = base[["application_number", "au_year", "examiner_id", "female", "granted"]].copy()
        for c in conds:
            x = base.merge(sc[sc.cond == c][["application_number", "score", "prob"]], on="application_number")
            x = x.dropna(subset=["score"])
            # break ties with tiny random noise so that a whole tied group is not pushed in or out at the cutoff
            r = x.score + np.random.default_rng(SEED).normal(0, 1e-9, len(x))
            x["dec"] = (r >= np.quantile(r, 1 - rate)).astype(int)
            decs = decs.merge(x[["application_number", "dec"]].rename(columns={"dec": c}), on="application_number")
            f = pf.feols("dec ~ female | au_year", data=x, vcov={"CRV1": "examiner_id"})
            rows.append(dict(tech_center=tc, cond=c, mean_score=x.prob.mean(),
                             auc_granted=roc_auc_score(x.granted, x.score),
                             gap_decision=float(f.coef()["female"]), se=float(f.se()["female"]),
                             ratio=float(f.coef()["female"]) / hg))
        ci_rows += bootstrap_contrasts(decs, conds, tc)
        for col, label in [("prob", "Name swap: female − male (probability)"),
                           ("score", "Name swap: female − male (log-odds)")]:
            w = sc[sc.application_number.isin(base.application_number)].pivot(
                index="application_number", columns="cond", values=col)
            diff = (w["zero_f"] - w["zero_m"]).dropna()
            rows.append(dict(tech_center=tc, cond=label, mean_score=diff.mean(), auc_granted=np.nan,
                             gap_decision=np.nan, se=diff.std(ddof=1) / np.sqrt(len(diff)), ratio=np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tables" / f"llm_prompting{TAG}.csv", index=False)
    with open(OUT / f"llm_prompting{TAG}.md", "w") as f:
        f.write("# LLM prompting experiments (Qwen3-8B)\n\n"
                "mean_score: mean predicted grant probability; gap_decision: female gap when decisions are made at the "
                "human grant rate; "
                "ratio: model gap / human examiner gap.\nIn the name-swap rows, mean_score is the mean female-name minus "
                "male-name difference (in probability or log-odds), and se is its standard error.\n\n")
        f.write(res.round(4).to_markdown(index=False) + "\n")
        ci = pd.DataFrame(ci_rows)
        ci.to_csv(OUT / "tables" / f"llm_prompting_ci{TAG}.csv", index=False)
        f.write("\n## Key contrasts (examiner-clustered bootstrap, 1,000 draws; percentage points)\n\n"
                + ci.round(1).to_markdown(index=False) + "\n")
    print((OUT / f"llm_prompting{TAG}.md").read_text())


def bootstrap_contrasts(decs: pd.DataFrame, conds: list[str], tc: str, B: int = 1000) -> list[dict]:
    """Each condition's decision gap minus the human gap, plus differences between conditions.

    95% intervals from an examiner-clustered bootstrap; reuses the weighted within-group regression in 11.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("b11", ROOT / "code/11_bootstrap.py")
    b11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b11)
    cols = ["granted"] + conds
    Y = decs[cols].to_numpy(float)
    f = decs["female"].to_numpy(float)
    grp = pd.factorize(decs["au_year"])[0]
    ex = pd.factorize(decs["examiner_id"])[0]
    n_ex = ex.max() + 1
    point = b11.fe_gaps(Y, f, grp, np.ones(len(decs)))
    rng = np.random.default_rng(SEED)
    draws = np.array([b11.fe_gaps(Y, f, grp, np.bincount(rng.integers(0, n_ex, n_ex), minlength=n_ex)[ex]
                                  .astype(float)) for _ in range(B)])
    k = {c: i for i, c in enumerate(cols)}
    pairs = [(c, "granted", f"{c} − human examiners") for c in conds] + [
        ("few_strict", "few_lenient", "Strict-examiner examples − lenient-examiner examples"),
        ("zero_f", "zero_m", "Female name − male name (decision gap)"),
        ("zero_noname", "zero_f", "No names − female name")]
    out = []
    for a, b, label in pairs:
        dd = (draws[:, k[a]] - draws[:, k[b]]) * 100
        out.append(dict(tech_center=tc, contrast=label, diff=(point[k[a]] - point[k[b]]) * 100,
                        lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), n=len(decs)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-sample and rebuild the prompts (default: reuse the existing llm_prompts.parquet)")
    a = ap.parse_args()
    if a.analyze:
        analyze()
        return
    if a.worker is not None:
        worker(a.worker)
        return
    have = (P / f"llm_prompts{TAG}.parquet").exists() and (P / f"llm_sample{TAG}.parquet").exists()
    pr = pd.read_parquet(P / f"llm_prompts{TAG}.parquet") if have and not a.rebuild else build_prompts()
    print(f"prompts: {len(pr):,d}", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                   "CUDA_VISIBLE_DEVICES": str(g)}) for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    sc = pd.concat([pd.read_parquet(P / f"llm_shard{TAG}_{k}.parquet") for k in range(N_GPU)])
    sc.to_parquet(P / f"llm_scores{TAG}.parquet", index=False)
    for k in range(N_GPU):
        (P / f"llm_shard{TAG}_{k}.parquet").unlink()
    print("scores saved; run the analysis with the main env: uv run python code/10_llm_prompting.py --analyze")


if __name__ == "__main__":
    main()
