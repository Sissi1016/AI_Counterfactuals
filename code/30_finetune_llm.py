"""Fine-tuned LLM evaluators: Qwen3-8B with LoRA, trained on the organization's record and checked like the others.

Targets (a scalar head on the last token; LoRA rank 16 on the attention and MLP projections):
- grant: the final grant decision, all applications (binary cross-entropy);
- cites: citations within five years excluding self-citations, granted applications (mean squared error on the
  standardized log(1 + citations));
Cross-fitting: two halves of examiners within each technology center (the examiner-grouped folds of 12, even vs odd),
a model per target and half, trained for one epoch on a random subsample of at most N_TRAIN applications of its half
and predicting every application of the other half, so no application is scored by a model that saw its examiner.
Input: title, abstract, and first claim (the text of 10 and 15), at most MAX_LEN tokens.

Run: uv run python code/30_finetune_llm.py --prepare        (main environment: the training table)
     .venv-llm/bin/python code/30_finetune_llm.py            (one worker per GPU, CFE_GPUS, default 0,1)
     uv run python code/30_finetune_llm.py --analyze        (the citation check of 21 for the fine-tuned scores)
Output: data/processed/ft_data.parquet, data/processed/ft_preds.parquet, output/finetune.md,
        output/tables/finetune_*.csv
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
MODEL = ROOT / "models/Qwen3-8B"
SEED = 20261028
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
N_TRAIN = int(os.environ.get("CFE_FT_N", "12000"))
N_PRED = int(os.environ.get("CFE_FT_PRED", "0"))  # 0 = all of the other half (a small number for a test run)
MAX_LEN, BS, ACC, LR, WARMUP = 320, 8, 2, 1e-4, 50
TARGETS = ["grant", "cites"]
JOBS = [(t, h) for t in TARGETS for h in (0, 1)]
if os.environ.get("CFE_FT_ONLY"):  # e.g. "grant:0" for a test run
    JOBS = [(x.split(":")[0], int(x.split(":")[1])) for x in os.environ["CFE_FT_ONLY"].split(",")]


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the training table (main environment)

def prepare() -> None:
    m25 = load_module("m25", "25_draft_tests.py")
    d = m25.load()
    d["tiebreak"] = np.random.default_rng(SEED).random(len(d))
    cf = np.zeros(len(d), bool)
    for tc in m25.CIT.units:  # the draft's construction, as in 25
        s = d[(d.tech_center == tc) & d.lenient_grant.notna()]
        gs = s.loc[s.lenient_half == 0, "granted"].mean()
        gl = s.loc[s.lenient_half == 1, "granted"].mean()
        f = (gl - gs) / gl
        for _, x in s[(s.lenient_half == 1) & (s.granted == 1)].groupby("cell_id"):
            x = x.sort_values([m25.OUTCOME, "tiebreak"])
            cf[d.index.get_indexer(x.index[:int(round(f * len(x)))])] = True
    draft = ((d.lenient_half == 0) & (d.granted == 1)).to_numpy() | cf
    t = pd.read_parquet(P / "q8_texts.parquet", columns=["application_number", "text"])
    out = pd.DataFrame({"application_number": d.application_number, "tech_center": d.tech_center,
                        "half": (d.fold.to_numpy() % 2).astype(int), "granted": d.granted.astype(float),
                        "log_cites": np.log1p(d[m25.OUTCOME].to_numpy(float)), "draft": draft})
    out = out.merge(t, on="application_number", how="left")
    assert out.text.notna().all()
    out.to_parquet(P / "ft_data.parquet", index=False)
    print(out.groupby(["tech_center", "half"]).agg(n=("granted", "size"), granted=("granted", "sum"),
                                                    draft=("draft", "sum")))


# ---------------------------------------------------------------- training (LLM environment, one process per GPU)

def worker(k: int) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    d = pd.read_parquet(P / "ft_data.parquet")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    for target, half in JOBS[k::len(GPUS)]:
        t0 = time.time()
        rng = np.random.default_rng(SEED + 10 * TARGETS.index(target) + half)
        if target == "grant":
            tr = d[d.half == half]
        elif target == "cites":
            tr = d[(d.half == half) & (d.granted == 1)]
        else:
            tr = d[(d.half == half) & d.draft]
        tr = tr.iloc[rng.permutation(len(tr))[:N_TRAIN]]
        if target == "grant":
            y = tr.granted.to_numpy(np.float32)
        else:
            mu, sd = tr.log_cites.mean(), tr.log_cites.std()
            y = ((tr.log_cites - mu) / sd).to_numpy(np.float32)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.bfloat16)
        model.config.pad_token_id = tok.pad_token_id
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        steps = int(np.ceil(len(tr) / (BS * ACC)))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / max(steps, 1)))
        lossf = torch.nn.BCEWithLogitsLoss() if target == "grant" else torch.nn.MSELoss()
        model.train()
        texts = tr.text.tolist()
        run = 0.0
        for b in range(0, len(tr), BS):
            enc = tok(texts[b:b + BS], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc).logits.squeeze(-1).float()
            loss = lossf(out, torch.tensor(y[b:b + BS], device="cuda")) / ACC
            loss.backward()
            run = 0.98 * run + 0.02 * loss.item() * ACC
            if (b // BS + 1) % ACC == 0 or b + BS >= len(tr):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if k == 0 and (b // BS) % 100 == 0:
                print(f"[{target} h{half}] batch {b // BS}/{len(tr) // BS} loss {run:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        # predict every application of the other half
        model.eval()
        te = d[d.half != half]
        if N_PRED:
            te = te.iloc[:N_PRED]
        order = np.argsort(te.text.str.len().to_numpy())
        te = te.iloc[order]
        preds = []
        with torch.no_grad():
            for b in range(0, len(te), 64):
                enc = tok(te.text.iloc[b:b + 64].tolist(), truncation=True, max_length=MAX_LEN, padding=True,
                          return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        pd.DataFrame({"application_number": te.application_number.to_numpy(), "target": target, "half_model": half,
                      "pred": np.concatenate(preds)}).to_parquet(P / f"ft_shard_{target}_{half}.parquet", index=False)
        print(f"[{target} h{half}] trained on {len(tr):,}, predicted {len(te):,} in {time.time() - t0:.0f}s",
              flush=True)
        del model, opt
        torch.cuda.empty_cache()


def train() -> None:
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    parts = [pd.read_parquet(P / f"ft_shard_{t}_{h}.parquet") for t, h in JOBS]
    w = pd.concat(parts).pivot(index="application_number", columns="target", values="pred").reset_index()
    w.to_parquet(P / "ft_preds.parquet", index=False)
    for t, h in JOBS:
        (P / f"ft_shard_{t}_{h}.parquet").unlink()
    print(f"saved {P / 'ft_preds.parquet'} {w.shape}; run: uv run python code/30_finetune_llm.py --analyze")


# ---------------------------------------------------------------- the check (main environment)




def analyze() -> None:
    import warnings

    from sklearn.metrics import roc_auc_score

    from cfe.check import HUMAN, check
    warnings.filterwarnings("ignore")
    m21 = load_module("m21", "21_citations.py")
    d = m21.load()
    ft = pd.read_parquet(P / "ft_preds.parquet").rename(columns={
        "grant": "LLM fine-tuned on grant", "cites": "LLM fine-tuned on citations",
        "draft": "LLM fine-tuned on the draft's C1 sample"})
    ft = ft.drop(columns=["LLM fine-tuned on the draft's C1 sample"], errors="ignore")
    small = pd.read_parquet(P / "citation_scores.parquet",
                            columns=["application_number", "Train on grant", "Train on citations (granted only)",
                                     "LLM, zero-shot"])
    sc = ft.merge(small, on="application_number", how="left")
    rec = d.drop(columns=list(m21.CF.values()))
    res = check(rec, sc, m21.CIT)
    for k in ["estimates", "contrasts", "gaps"]:
        res[k].to_csv(OUT / f"tables/finetune_{k}.csv", index=False)
    x = d[["application_number", "tech_center", "granted", m21.OUTCOME]].merge(sc, on="application_number")
    series = [c for c in sc.columns if c != "application_number" and "::" not in c]
    rows = []
    for tc in m21.CIT.units:
        m = x[x.tech_center == tc]
        g = m[m.granted == 1]
        for s in series:
            rows.append(dict(tech_center=tc, evaluator=s, auc_grant=roc_auc_score(m.granted, m[s]),
                             corr_log_cites=np.corrcoef(g[s], np.log1p(g[m21.OUTCOME]))[0, 1]))
    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / "tables/finetune_fit.csv", index=False)

    def ci(b, se, nd=2, scale=1.0):
        return f"{scale * b:.{nd}f} [{scale * (b - 1.96 * se):.{nd}f}, {scale * (b + 1.96 * se):.{nd}f}]"
    est, gaps = res["estimates"], res["gaps"]
    L = ["# Fine-tuned LLM evaluators (Qwen3-8B, LoRA), citations as the outcome", "",
         f"Two examiner-grouped halves per technology center; each model trained on at most {N_TRAIN:,} applications "
         "of one half and scoring the other. Scores mapped to citations by the Poisson recalibration of cfe.check.",
         "", "| TC | Evaluator | AUC for grant | Corr. with log citations (granted) |", "|---|---|---|---|"]
    for r in fit.itertuples():
        L.append(f"| {r.tech_center} | {r.evaluator} | {r.auc_grant:.3f} | {r.corr_log_cites:.3f} |")
    L += ["", "| TC | Row | Implied | Error, all | Error, female | Gender gap in decisions (pp) | × human |",
          "|---|---|---|---|---|---|---|"]
    for tc in m21.CIT.units:
        e = est[est["sample"] == tc]
        t = e[(e.series == HUMAN) & (e.kind == "marginal") & (e.group == "All")].iloc[0]
        L.append(f"| {tc} | Truth | — | {ci(t.estimate, t.se)} | | | |")
        for s in series:
            a = e[(e.series == s) & (e.kind == "error") & (e.group == "All")].iloc[0]
            f = e[(e.series == s) & (e.kind == "error") & (e.group == "Female")].iloc[0]
            im = e[(e.series == s) & (e.kind == "implied") & (e.group == "All")].iloc[0].estimate
            g = gaps[(gaps["sample"] == tc) & (gaps.series == s)]
            gtxt = f"{ci(g.iloc[0].gap, g.iloc[0].se, 1, 100)} | {g.iloc[0].ratio:.2f}" if len(g) else "— | —"
            L.append(f"| {tc} | {s} | {im:.2f} | {ci(a.estimate, a.se)} | {ci(f.estimate, f.se)} | {gtxt} |")
    (OUT / "finetune.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.analyze:
        analyze()
    elif a.worker is not None:
        worker(a.worker)
    else:
        train()


if __name__ == "__main__":
    main()
