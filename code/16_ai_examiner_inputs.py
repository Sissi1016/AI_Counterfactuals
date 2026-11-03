"""Inputs for the AI examiners (17): the decision records of the 20 examiners with the most pilot applications in
TC 3600, beyond the pilot sample, embedded the same way as the pilot.

Applications: all their non-continuation applications in HUPD filed 2004–2018 that are not in the pilot (learning an
examiner's decisions does not need the pilot's other restrictions, such as a U.S. first inventor of known gender).
Text comes from the HUPD yearly archives (07's scan); inputs are built as in 08 (title, abstract, claims) and 15
(title, abstract, first claim), and embedded with Qwen3-Embedding-0.6B and Qwen3-8B, one process per GPU.

Run: .venv-llm/bin/python code/16_ai_examiner_inputs.py [--model s0|q8]   (default: both; the reported twins use s0)
Output: data/processed/ai_examiner_apps.parquet, data/processed/emb/ai_ex_{s0,q8}.npy (row-aligned)
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
APPS = P / "ai_examiner_apps.parquet"
SEED = 20261028
TC, N_EXAMINERS, MAXCH = "3600", 20, 6000
YEARS = range(2004, 2019)
# GPU device indices; two GPUs (0 and 1) by default, override with CFE_GPUS.
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # worker processes look functions up by module name
    spec.loader.exec_module(mod)
    return mod


def top_examiners() -> list:
    """The examiners with the most pilot applications in TC (ties broken by order of appearance, as in 17)."""
    s = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"]).merge(
        pd.read_parquet(P / "pilot_sample.parquet", columns=["application_number", "examiner_id", "tech_center"]),
        on="application_number", how="left")
    s = s[s.tech_center == TC]
    return s.groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort").head(N_EXAMINERS).index.tolist()


def build_apps() -> pd.DataFrame:
    top = top_examiners()
    con = duckdb.connect()
    con.register("top", pd.DataFrame({"examiner_id": top}))
    a = con.execute(f"""
        SELECT l.application_number, l.examiner_id, l.art_unit, l.au_year, l.filing_year, l.granted, l.fa_allow
        FROM '{P}/applications_leniency.parquet' l JOIN top USING (examiner_id)
        WHERE l.in_hupd AND NOT l.continuation
          AND l.application_number NOT IN (SELECT application_number FROM '{P}/pilot_sample.parquet')""").df()
    L07 = load_module("text07", "07_extract_text.py")
    wanted = frozenset(a.application_number)
    with ProcessPoolExecutor(max_workers=len(YEARS)) as ex:
        parts = list(ex.map(L07.scan, YEARS, [wanted] * len(YEARS)))
    t = pd.DataFrame([r for part in parts for r in part]).drop_duplicates("application_number")
    d = a.merge(t, on="application_number", how="inner").sort_values("application_number").reset_index(drop=True)
    print(f"applications: {len(a):,d} in PatEx, {len(d):,d} with text", flush=True)
    L10 = load_module("llm10", "10_llm_prompting.py")
    d["s0"] = ("Title: " + d["title"] + "\n\nAbstract: " + d["abstract"] + "\n\nClaims: " + d["claims"]).str[:MAXCH]
    d["q8"] = [L10.app_block(r) for r in d.itertuples()]
    d["job"] = np.random.default_rng(SEED).permutation(len(d))
    keep = ["application_number", "examiner_id", "art_unit", "au_year", "filing_year", "granted", "fa_allow", "s0", "q8",
            "job"]
    d[keep].to_parquet(APPS, index=False)
    return d[keep]


def worker(k: int, model: str) -> None:
    from vllm import LLM
    from vllm.config import PoolerConfig
    d = pd.read_parquet(APPS)
    d = d[d.job % len(GPUS) == k]
    if model == "s0":
        llm = LLM(model=str(ROOT / "models/Qwen3-Embedding-0.6B"), runner="pooling", max_model_len=4096,
                  gpu_memory_utilization=0.6, seed=SEED)
        out = llm.embed(d.s0.tolist(), use_tqdm=(k == 0), tokenization_kwargs={"truncation": True, "max_length": 4000})
    else:
        llm = LLM(model=str(ROOT / "models/Qwen3-8B"), runner="pooling", convert="embed", max_model_len=4096,
                  gpu_memory_utilization=0.85, seed=SEED,
                  pooler_config=PoolerConfig(seq_pooling_type="MEAN", use_activation=False))
        out = llm.embed(d.q8.tolist(), use_tqdm=(k == 0))
    np.save(E / f"ai_ex_{model}_shard_{k}.npy", np.asarray([o.outputs.embedding for o in out], dtype=np.float16))
    np.save(E / f"ai_ex_{model}_rows_{k}.npy", d.index.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--model", choices=["s0", "q8"])
    a = ap.parse_args()
    if a.worker is not None:
        worker(a.worker, a.model)
        return
    d = pd.read_parquet(APPS) if APPS.exists() else build_apps()
    for model in [a.model] if a.model else ["s0", "q8"]:
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k), "--model", model],
                                  env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
                 for k, g in enumerate(GPUS)]
        if any(p.wait() for p in procs):
            sys.exit(f"a {model} worker failed")
        emb = None
        for k in range(len(GPUS)):
            part, rows = np.load(E / f"ai_ex_{model}_shard_{k}.npy"), np.load(E / f"ai_ex_{model}_rows_{k}.npy")
            if emb is None:
                emb = np.zeros((len(d), part.shape[1]), dtype=np.float16)
            emb[rows] = part
            (E / f"ai_ex_{model}_shard_{k}.npy").unlink()
            (E / f"ai_ex_{model}_rows_{k}.npy").unlink()
        np.save(E / f"ai_ex_{model}.npy", emb)
        print(f"saved ai_ex_{model}.npy {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
