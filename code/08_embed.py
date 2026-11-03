"""Build model inputs and embed them with Qwen3-Embedding-0.6B (vLLM, sharded across GPUs; see GPUS).

Input variants
- s0: title + abstract + claims (first MAXCH characters), with no identity information.
- s1: s0 prefixed with "Inventors: First Last; ...".
- s1_f / s1_m (test set only): the first inventor's first name is replaced with a female or a male name and
  everything else is left unchanged; used for the name-swap audit. The female and male names are the 100 most common
  unambiguous names among U.S. inventors (WGND share >= 0.99), paired by rank.

Train/test split: a random 20% of applications form the test set (fixed seed).

Run: .venv-llm/bin/python code/08_embed.py
Output: data/processed/pilot_inputs.parquet, data/processed/emb/{s0,s1,s1_f,s1_m}.npy (float16, row-aligned with
        the inputs)
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
E = P / "emb"
MODEL = str(ROOT / "models/Qwen3-Embedding-0.6B")
SEED, TEST_SHARE, MAXCH, N_NAMES = 20261028, 0.2, 6000, 100
MAX_TOKENS = 4000  # biotech sequences and chemical formulas run ~1 token per character, so also truncate by tokens
# GPU device indices; two GPUs (0 and 1) by default, override with CFE_GPUS.
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
N_GPU = len(GPUS)


def name_lists() -> tuple[list[str], list[str]]:
    con = duckdb.connect()
    q = """
      SELECT name_full, count(*) n FROM '{p}/inventor_gender.parquet'
      WHERE code = 'US' AND gender = '{g}' AND {col} >= 0.99 AND regexp_matches(name_full, '^[a-z]{{3,}}$')
      GROUP BY 1 ORDER BY 2 DESC LIMIT {k}"""
    f = con.execute(q.format(p=P, g="F", col="p_f", k=N_NAMES)).df()["name_full"].str.title().tolist()
    m = con.execute(q.format(p=P, g="M", col="p_m", k=N_NAMES)).df()["name_full"].str.title().tolist()
    return f, m


def inventors(js: str) -> list[tuple[str, str]]:
    out = []
    for x in json.loads(js):
        first = (x.get("inventor_name_first") or "").strip().title()
        last = (x.get("inventor_name_last") or "").strip().title()
        if first or last:
            out.append((first, last))
    return out


def fmt(invs: list[tuple[str, str]]) -> str:
    return "Inventors: " + "; ".join(f"{a} {b}".strip() for a, b in invs)


def build_inputs() -> pd.DataFrame:
    s = pd.read_parquet(P / "pilot_sample.parquet")
    t = pd.read_parquet(P / "pilot_text.parquet")
    d = s.merge(t, on="application_number", how="inner").sort_values("application_number").reset_index(drop=True)
    d["test"] = np.random.default_rng(SEED).random(len(d)) < TEST_SHARE
    d["s0"] = ("Title: " + d["title"] + "\n\nAbstract: " + d["abstract"] + "\n\nClaims: " + d["claims"]).str[:MAXCH]
    invs = d["inventor_list"].map(inventors)
    d["s1"] = invs.map(fmt) + "\n\n" + d["s0"]
    fem, mal = name_lists()
    k = np.random.default_rng(SEED + 1).integers(0, N_NAMES, len(d))
    d["audit_name_f"] = [fem[i] for i in k]
    d["audit_name_m"] = [mal[i] for i in k]
    swap = lambda iv, nm: fmt([(nm, iv[0][1])] + iv[1:]) if iv else fmt([(nm, "")])
    d["s1_f"] = [swap(iv, nm) + "\n\n" + s0 for iv, nm, s0 in zip(invs, d["audit_name_f"], d["s0"])]
    d["s1_m"] = [swap(iv, nm) + "\n\n" + s0 for iv, nm, s0 in zip(invs, d["audit_name_m"], d["s0"])]
    d.loc[~d["test"], ["s1_f", "s1_m"]] = None
    keep = ["application_number", "test", "s0", "s1", "s1_f", "s1_m", "audit_name_f", "audit_name_m"]
    d[keep].to_parquet(P / "pilot_inputs.parquet", index=False)
    return d[keep]


def jobs(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for v in ["s0", "s1", "s1_f", "s1_m"]:
        idx = d.index[d[v].notna()]
        parts.append(pd.DataFrame({"variant": v, "row": idx}))
    j = pd.concat(parts, ignore_index=True)
    j["job"] = np.arange(len(j))
    return j


def worker(k: int) -> None:
    from vllm import LLM
    d = pd.read_parquet(P / "pilot_inputs.parquet")
    j = jobs(d)
    j = j[j["job"] % N_GPU == k]
    texts = [d.at[r, v] for v, r in zip(j["variant"], j["row"])]
    llm = LLM(model=MODEL, runner="pooling", max_model_len=4096, gpu_memory_utilization=0.6, seed=SEED)
    out = llm.embed(texts, use_tqdm=(k == 0),
                    tokenization_kwargs={"truncation": True, "max_length": MAX_TOKENS})
    np.save(E / f"shard_{k}.npy", np.asarray([o.outputs.embedding for o in out], dtype=np.float16))
    j[["job"]].to_parquet(E / f"shard_{k}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    a = ap.parse_args()
    if a.worker is not None:
        worker(a.worker)
        return
    E.mkdir(parents=True, exist_ok=True)
    d = build_inputs()
    print(f"inputs: {len(d):,d} rows, test {d['test'].sum():,d}", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                   "CUDA_VISIBLE_DEVICES": str(g)}) for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    j = jobs(d)
    emb = np.zeros((len(j), 0), dtype=np.float16)
    for k in range(N_GPU):
        x = np.load(E / f"shard_{k}.npy")
        ids = pd.read_parquet(E / f"shard_{k}.parquet")["job"].to_numpy()
        if emb.shape[1] == 0:
            emb = np.zeros((len(j), x.shape[1]), dtype=np.float16)
        emb[ids] = x
    for v in ["s0", "s1", "s1_f", "s1_m"]:
        m = (j["variant"] == v).to_numpy()
        np.save(E / f"{v}.npy", emb[m])
        print(v, emb[m].shape)
    for k in range(N_GPU):
        (E / f"shard_{k}.npy").unlink()
        (E / f"shard_{k}.parquet").unlink()


if __name__ == "__main__":
    main()
