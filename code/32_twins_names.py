"""C3 at the level of the same examiner: AI twins that see the inventors' names, and what changes when the first
inventor's name is changed.

The proposal's C3 asks whether an application would have been granted had gender been hidden from the same examiner.
17 builds AI twins of the 20 TC 3600 examiners with the most pilot applications from text without names. Here the
twins are rebuilt from text with the inventor list ("Inventors: First Last; ..." + text, as 08's s1), and each twin
decides the same 3,972 applications three times: with the actual names, with a female, and with a male first name
for the first inventor (everything else unchanged; the names of 08's audit, paired by rank).

Reported per twin and pooled: the change in grant probability (female − male name), the share of applications whose
decision changes when only the name changes, and, against the text-only twins of 17, the share of decisions that change
when names are shown, by the first inventor's gender.

GPU part (LLM environment, one GPU): embeds the twins' training applications with names and the common applications
with swapped names using Qwen3-Embedding-0.6B. Then --analyze (main environment).
Run: uv run python code/32_twins_names.py --prepare; CFE_GPUS=0 .venv-llm/bin/python code/32_twins_names.py;
     uv run python code/32_twins_names.py --analyze
Input: data/processed/{ai_examiner_apps,pilot_inputs,pilot_text,patex_all_inventors}.parquet, emb/{s0,s1,ai_ex_s0}.npy
Output: data/processed/emb/{ai_ex_s1,common_s1_f,common_s1_m}.npy, data/processed/twins_names_inputs.parquet,
        output/twins_names.md, output/tables/twins_names_*.csv
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
SEED = 20261028
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0").split(",")]
INPUTS = P / "twins_names_inputs.parquet"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_inputs() -> pd.DataFrame:
    import duckdb
    m17 = load_module("m17", "17_ai_examiners.py")
    from cfe.check import load_record
    d = load_record(ROOT)
    tc_rows = d[d.tech_center == m17.TC]
    top = tc_rows.groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort").head(
        m17.N_EXAMINERS).index
    common = d[d.examiner_id.isin(top)][["application_number"]]
    inp = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number", "s0", "audit_name_f",
                                                              "audit_name_m"])
    txt = pd.read_parquet(P / "pilot_text.parquet", columns=["application_number", "inventor_list"])
    c = common.merge(inp, on="application_number").merge(txt, on="application_number")

    m08 = load_module("m08", "08_embed.py")
    iv = c.inventor_list.map(m08.inventors)
    fmt = m08.fmt
    swap = lambda v, nm: fmt([(nm, v[0][1])] + v[1:]) if v else fmt([(nm, "")])  # noqa: E731
    c["s1_f"] = [swap(v, nm) + "\n\n" + s0 for v, nm, s0 in zip(iv, c.audit_name_f, c.s0)]
    c["s1_m"] = [swap(v, nm) + "\n\n" + s0 for v, nm, s0 in zip(iv, c.audit_name_m, c.s0)]
    # the twins' other decisions: names from PatEx in rank order, first and middle names as in HUPD's lists
    xa = pd.read_parquet(P / "ai_examiner_apps.parquet", columns=["application_number", "s0"])
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.register("xa", xa[["application_number"]])
    names = con.execute(f"""
        SELECT application_number, concat_ws(' ', nullif(trim(inventor_name_first), ''),
                                             nullif(trim(inventor_name_middle), '')) AS a,
               inventor_name_last AS b, TRY_CAST(inventor_rank AS INT) r
        FROM '{P}/patex_all_inventors.parquet' JOIN xa USING (application_number) ORDER BY 1, 4""").df()
    names["a"], names["b"] = names.a.fillna("").str.strip().str.title(), names.b.fillna("").str.strip().str.title()
    lists = names.groupby("application_number").apply(lambda x: fmt(list(zip(x.a, x.b))), include_groups=False)
    lists = lists.rename("names")
    xa = xa.merge(lists.reset_index(), on="application_number", how="left")
    xa["s1"] = xa.names.fillna("Inventors:") + "\n\n" + xa.s0
    rows = [pd.DataFrame({"set": "ai_ex_s1", "row": np.arange(len(xa)), "application_number": xa.application_number,
                          "text": xa.s1}),
            pd.DataFrame({"set": "common_s1_f", "row": np.arange(len(c)), "application_number": c.application_number,
                          "text": c.s1_f}),
            pd.DataFrame({"set": "common_s1_m", "row": np.arange(len(c)), "application_number": c.application_number,
                          "text": c.s1_m})]
    t = pd.concat(rows, ignore_index=True)
    t["job"] = np.random.default_rng(SEED).permutation(len(t))
    t.to_parquet(INPUTS, index=False)
    print(t.groupby("set").size())
    return t


def worker(k: int) -> None:
    from vllm import LLM
    m08 = load_module("m08", "08_embed.py")
    t = pd.read_parquet(INPUTS)
    t = t[t.job % len(GPUS) == k]
    llm = LLM(model=str(ROOT / "models/Qwen3-Embedding-0.6B"), runner="pooling", max_model_len=4096,
              gpu_memory_utilization=0.6, seed=SEED)
    out = llm.embed(t.text.tolist(), use_tqdm=(k == 0),
                    tokenization_kwargs={"truncation": True, "max_length": m08.MAX_TOKENS})
    np.save(E / f"tn_shard_{k}.npy", np.asarray([o.outputs.embedding for o in out], dtype=np.float16))
    t[["job"]].to_parquet(E / f"tn_shard_{k}.parquet", index=False)


def embed() -> None:
    t = pd.read_parquet(INPUTS)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    emb = None  # indexed by job (a permutation of the rows)
    for k in range(len(GPUS)):
        x = np.load(E / f"tn_shard_{k}.npy")
        ids = pd.read_parquet(E / f"tn_shard_{k}.parquet").job.to_numpy()
        if emb is None:
            emb = np.zeros((len(t), x.shape[1]), dtype=np.float16)
        emb[ids] = x
        (E / f"tn_shard_{k}.npy").unlink()
        (E / f"tn_shard_{k}.parquet").unlink()
    for s in ["ai_ex_s1", "common_s1_f", "common_s1_m"]:
        x = t[t.set == s].sort_values("row")
        np.save(E / f"{s}.npy", emb[x.job.to_numpy()])
        print(s, len(x))


# ---------------------------------------------------------------- analysis (main environment)

def mean_se(x: np.ndarray, cl) -> float:
    """Standard error of a mean, clustered by examiner."""
    from cfe.estimation import cluster_se
    return float(cluster_se(((x - x.mean()) / len(x))[:, None], cl)[0])


def analyze() -> None:
    import warnings

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from cfe.check import load_record
    from cfe.estimation import cluster_se, tsls
    warnings.filterwarnings("ignore")
    m17 = load_module("m17", "17_ai_examiners.py")
    d = load_record(ROOT)
    t = pd.read_parquet(INPUTS)
    xa = pd.read_parquet(P / "ai_examiner_apps.parquet", columns=["application_number", "examiner_id", "granted"])
    xs1 = t[t.set == "ai_ex_s1"].sort_values("row").application_number.to_numpy()
    assert (xs1 == xa.application_number.to_numpy()).all()
    common_ids = t[t.set == "common_s1_f"].sort_values("row").application_number.to_numpy()
    tc_rows = np.where(d.tech_center == m17.TC)[0]
    counts = d.iloc[tc_rows].groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort")
    top = counts.head(m17.N_EXAMINERS).index.tolist()
    others = tc_rows[~d.examiner_id.iloc[tc_rows].isin(top).to_numpy()]
    pool = d[d.examiner_id.isin(top)].copy()
    pool_rows = pool.index.to_numpy()
    swap_rows = pd.Series(np.arange(len(common_ids)), index=common_ids).loc[pool.application_number].to_numpy()
    # names hidden: 17's inputs (title, abstract, claims); names shown: the inventor list in front, as 08's s1
    versions = {"hidden": ("s0", "ai_ex_s0", {}),
                "shown": ("s1", "ai_ex_s1", {"female": "common_s1_f", "male": "common_s1_m"})}
    kinds = {"leniency": [0], "text": slice(None)}  # 17: leniency only (main) and with text dimensions
    prob, auc, shared_auc = {}, {}, {}
    for ver, (pe, xe, sw) in versions.items():
        emb = np.load(E / f"{pe}.npy", mmap_mode="r")
        shared, _, shared_auc[ver] = m17.fit_shared(np.asarray(emb[others], dtype=np.float32),
                                                    d.granted.to_numpy()[others].astype(int),
                                                    pd.factorize(d.examiner_id.iloc[others])[0])
        E_tc = np.asarray(emb[tc_rows], dtype=np.float32)
        sc_all = StandardScaler().fit(E_tc)
        Z = PCA(m17.N_PC, random_state=SEED).fit(sc_all.transform(E_tc))

        def feats(E_):
            return np.column_stack([shared(E_), Z.transform(sc_all.transform(E_))])
        X = {"actual": feats(np.asarray(emb[pool_rows], dtype=np.float32))}
        for lab, s in sw.items():
            X[lab] = feats(np.load(E / f"{s}.npy").astype(np.float32)[swap_rows])
        Xx = feats(np.load(E / f"{xe}.npy").astype(np.float32))
        for i, e in enumerate(top):
            own, ext = (pool.examiner_id == e).to_numpy(), (xa.examiner_id == e).to_numpy()
            y, yx = pool.granted.to_numpy()[own].astype(int), xa.granted.to_numpy()[ext].astype(int)
            for kind, cols in kinds.items():
                oof, m, sc, auc[ver, kind, e] = m17.fit_cv(Xx[ext][:, cols], yx, X["actual"][own][:, cols], y,
                                                           SEED + i)
                for v, Xv in X.items():
                    p = m.predict_proba(sc.transform(Xv[:, cols]))[:, 1]
                    if v == "actual":
                        p[own] = oof
                    prob[ver, kind, v, e] = p
    fem = pool.female.to_numpy() == 1
    f = fem.astype(float)
    lev = pool.groupby("examiner_id").lenient_grant.first()
    order = sorted(top, key=lambda e: (np.isnan(lev[e]), lev[e]))  # AI twin 1–20 as in 17

    def gaps(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, psi = tsls(D.astype(float), f, f, pool.au_year)
        return b, cluster_se(psi, pool.examiner_id)
    rows, summ = [], []
    for kind in kinds:
        D, thr = {}, {}
        for ver in versions:
            P_ = np.column_stack([prob[ver, kind, "actual", e] for e in order])
            thr[ver] = np.array([np.quantile(P_[:, j], 1 - P_[:, j].mean()) for j in range(len(order))])
            D[ver] = P_ >= thr[ver]
        # the bar stays where the actual names put it; only the first inventor's first name changes
        PF = np.column_stack([prob["shown", kind, "female", e] for e in order])
        PM = np.column_stack([prob["shown", kind, "male", e] for e in order])
        DF, DM = PF >= thr["shown"], PM >= thr["shown"]
        g_hid, se_hid = gaps(D["hidden"])
        g_sh, se_sh = gaps(D["shown"])
        for j, e in enumerate(order):
            dd = DF[:, j].astype(float) - DM[:, j]
            rows.append(dict(kind=kind, ai_examiner=j + 1, auc_hidden=auc["hidden", kind, e],
                             auc_shown=auc["shown", kind, e], gap_hidden=g_hid[j], gap_hidden_se=se_hid[j],
                             gap_shown=g_sh[j], gap_shown_se=se_sh[j],
                             dprob_female_minus_male=np.mean(PF[:, j] - PM[:, j]),
                             dgrant_female_minus_male=dd.mean(), dgrant_se=mean_se(dd, pool.examiner_id),
                             decisions_changed_by_name=np.mean(DF[:, j] != DM[:, j]),
                             changed_when_shown_female=np.mean(D["shown"][fem, j] != D["hidden"][fem, j]),
                             changed_when_shown_male=np.mean(D["shown"][~fem, j] != D["hidden"][~fem, j])))
        dd = (DF.astype(float) - DM).mean(1)  # per application, averaged over the 20 twins
        pf_pm = (PF - PM).mean(1)
        summ.append(dict(kind=kind, shared_auc_hidden=shared_auc["hidden"], shared_auc_shown=shared_auc["shown"],
                         gap_hidden_median=np.median(g_hid), gap_shown_median=np.median(g_sh),
                         dprob_mean=pf_pm.mean(), dprob_se=mean_se(pf_pm, pool.examiner_id),
                         dgrant_mean=dd.mean(), dgrant_se=mean_se(dd, pool.examiner_id),
                         changed_by_name_mean=np.mean(DF != DM),
                         changed_when_shown_female=np.mean(D["shown"][fem] != D["hidden"][fem]),
                         changed_when_shown_male=np.mean(D["shown"][~fem] != D["hidden"][~fem]),
                         n_common=len(pool), n_female=int(fem.sum())))
    tw, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    tw.to_csv(OUT / "tables/twins_names_by_twin.csv", index=False)
    summ.to_csv(OUT / "tables/twins_names_summary.csv", index=False)
    hb, hpsi = tsls(pool[["granted"]].to_numpy(float), f, f, pool.au_year)
    hse = cluster_se(hpsi, pool.examiner_id)[0]

    def pp(x, nd=1):
        return f"{100 * x:.{nd}f}"

    def ci(b, se, nd=1):
        return f"{pp(b, nd)} [{pp(b - 1.96 * se, nd)}, {pp(b + 1.96 * se, nd)}]"
    L = ["# AI twins that see the inventors' names (C3 for the same examiner)", "",
         f"The 20 AI twins of 17 (TC {m17.TC}), rebuilt from text with the inventor list in front (as 08's s1), each "
         f"decide the same {len(pool):,} applications ({int(fem.sum())} with a female first inventor) three times: "
         "with the actual names, and with a female and with a male first name for the first inventor (everything "
         "else unchanged; the names of 08's audit). The bar is set by the decisions with the actual names. "
         "\"Leniency\" twins (main in 17) use the common grant score with the examiner's own intercept and slope; "
         "\"text\" twins add 32 text dimensions.", "",
         f"Common grant score, examiner-grouped AUC: names hidden {shared_auc['hidden']:.3f}, shown "
         f"{shared_auc['shown']:.3f}. Human examiners on the same applications: gender gap {ci(hb[0], hse)} pp.", "",
         "| Twins | Gender gap, names hidden (median, pp) | Names shown | Δ grant probability, female − male name "
         "(pp) | Δ grant decisions, female − male name (pp) | Decisions changed by the name | Decisions changed by "
         "showing names: female, male |", "|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.kind} | {pp(r.gap_hidden_median)} | {pp(r.gap_shown_median)} | "
                 f"{ci(r.dprob_mean, r.dprob_se, 2)} | {ci(r.dgrant_mean, r.dgrant_se, 2)} | "
                 f"{pp(r.changed_by_name_mean, 2)}% | {pp(r.changed_when_shown_female)}%, "
                 f"{pp(r.changed_when_shown_male)}% |")
    L += ["", "## By twin (leniency twins)", "",
          "| AI twin | AUC, hidden | AUC, shown | Gap hidden (pp) | Gap shown (pp) | Δ prob F − M (pp) | "
          "Δ decisions F − M (pp) | Changed by the name |", "|---|---|---|---|---|---|---|---|"]
    for r in tw[tw.kind == "leniency"].itertuples():
        L.append(f"| {r.ai_examiner} | {r.auc_hidden:.3f} | {r.auc_shown:.3f} | {pp(r.gap_hidden)} | "
                 f"{pp(r.gap_shown)} | {pp(r.dprob_female_minus_male, 2)} | "
                 f"{ci(r.dgrant_female_minus_male, r.dgrant_se, 2)} | {pp(r.decisions_changed_by_name, 2)}% |")
    (OUT / "twins_names.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        build_inputs()
    elif a.worker is not None:
        worker(a.worker)
    elif a.analyze:
        analyze()
    else:
        embed()


if __name__ == "__main__":
    main()
