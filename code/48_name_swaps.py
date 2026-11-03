"""Name swaps (C3): does an AI evaluator's decision change when only the first inventor's name changes?

C3 asks whether an application would have been decided differently had the evaluator not known the first inventor's
gender or ethnicity. For an AI evaluator the question has an experimental answer: show it the same application
twice, identical except for the first inventor's name, and compare the decisions. Two evaluators:

1. The eight-center text evaluator "Train on grant + names" of 38: an L2-penalized logistic regression on the
   Qwen3-Embedding-0.6B vector of "Inventors: First Last; ..." + title + abstract + claims, cross-fitted in five
   folds grouped by examiner within each center. 38 saves the out-of-fold scores, not the fitted models. Each fold's
   model is linear in the vector (standardization and logistic index), so it is recovered from that fold's held-out
   applications by least squares of the saved score on the vector and a constant: the fit is exact to rounding error
   and predicts held-out halves of the fold equally well (output/tables/nsw_fold_recovery.csv). Every application is
   scored by its own out-of-fold model. Sample: 20,000 applications of 38's main sample (eight centers, filed
   2010–2014) with at least one inventor listed, drawn at random within center in proportion to the centers' sizes.
   Decisions: as in 38, each center grants the share of applications its examiners grant, with the threshold fixed
   where the original scores of all of the center's applications put it, so that only the name moves.
2. The 20 AI twins of TC 3600 examiners (17), rebuilt from text with the inventor list in 32, on the 3,972
   applications of those examiners: 32's fits, 32's bars (set by the decisions with the actual names), and 32's
   female/male first-name vectors; the surname swaps are new.

Variants (only the first inventor's name differs; every other byte of the input is the same):
- original: the actual names, re-embedded in the same run as the swaps (the stored vectors of 38 and 08 give a
  placebo: the identical text embedded twice);
- surname -> Hispanic, Asian, or White (the control): the N_SURNAMES most frequent surnames in the 2010 Census
  surname table that are borne at least 90% by Hispanic, Asian/Pacific Islander, or non-Hispanic White Americans
  (letters only; surnames the table writes without their apostrophe, such as OBRIEN, are skipped); one rank drawn at
  random per application (fixed seed) and the same rank used for the three groups;
- first name -> female or male (evaluator 1; evaluator 2 has them from 32): 08's audit names (the 100 most common
  unambiguous female and male first names of U.S. inventors, paired by rank), one rank drawn per application;
- neutral (evaluator 1): the male first name and the White surname of the two draws above.

Reported: the share of decisions that change, the change in the grant rate (pp) and in the predicted grant
probability (pp), for each swap against the original and for Hispanic − White, Asian − White, and female − male;
95% CIs clustered by examiner; pooled and in TC 3600 (evaluator 1); pooled over the twins and the spread across
twins, with CIs as in 32 (evaluator 2; no twin-level rows). For evaluator 1 also the female − male, Hispanic − White,
and Asian − White gaps in decisions (art unit × year cells, as 38) with the actual names and with the first
inventor's name neutralized: what remains runs through the text and the co-inventors' names, not the first
inventor's name.

Run: uv run python code/48_name_swaps.py --prepare           (main environment, four threads, about two minutes)
     CFE_GPUS=0 .venv-llm/bin/python code/48_name_swaps.py   (embedding, one GPU, about 25 minutes)
     uv run python code/48_name_swaps.py --analyze           (main environment, about ten minutes)
Input: data/processed/{all_sample,all_text,all_scores,census_surnames_2010,inventor_gender,pilot_inputs,pilot_text,
       pilot_sample,twins_names_inputs,ai_examiner_apps,patex_all_inventors}.parquet,
       data/processed/emb/{all_s1,s1,ai_ex_s1,common_s1_f,common_s1_m}.npy
Output: data/processed/nsw_{variants,texts}.parquet, data/processed/nsw_emb.npy, output/name_swaps.md,
        output/tables/nsw_*.csv, output/latex/nsw_tab_swaps.tex, output/latex/nsw_numbers.tex (\\xNs...)
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
P = ROOT / "data/processed"
E = P / "emb"
OUT = ROOT / "output"
T = OUT / "tables"
TEX = OUT / "latex"
SEED = 20261101  # this script's draws (sample, names)
SEED32 = 20261028  # 32's (and 17's) seed: the twins are refitted exactly as there
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0").split(",")]
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
GN = "Train on grant + names"
N_SAMPLE, N_SURNAMES, SHARE = 20_000, 20, 90.0
SURNAME_GROUPS = {"hispanic": "pcthispanic", "asian": "pctapi", "white": "pctwhite"}
NO_APOSTROPHE = {"OBRIEN", "OCONNOR", "ONEIL", "ONEILL", "ODONNELL", "OCONNELL", "OHARA", "OBRIAN"}
VARIANTS = P / "nsw_variants.parquet"
TEXTS = P / "nsw_texts.parquet"
EMB = P / "nsw_emb.npy"
EV_VARIANTS = ["orig", "hispanic", "asian", "white", "female", "male", "neutral"]
TW_VARIANTS = ["orig", "hispanic", "asian", "white"]
CHUNK = 10_000  # texts per saved chunk of the embedding


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. inputs (main environment)

def surname_lists() -> dict:
    cs = pd.read_parquet(P / "census_surnames_2010.parquet")
    cs = cs[cs.name.str.fullmatch("[A-Z]+") & ~cs.name.isin(NO_APOSTROPHE)]
    out = {}
    for key, col in SURNAME_GROUPS.items():
        x = cs[cs[col] >= SHARE].sort_values(["count", "name"], ascending=[False, True]).head(N_SURNAMES)
        out[key] = x.name.str.title().tolist()
    return out


def largest_remainder(n: pd.Series, total: int) -> pd.Series:
    q = total * n / n.sum()
    base = np.floor(q).astype(int)
    extra = (q - base).sort_values(ascending=False, kind="mergesort").index[: total - base.sum()]
    base[extra] += 1
    return base


def prepare() -> None:
    import duckdb
    m08 = load_module("m08", "08_embed.py")
    m17 = load_module("m17", "17_ai_examiners.py")
    from cfe.check import load_record
    con = duckdb.connect()
    con.execute("SET threads = 4")

    # evaluator 1: a stratified random sample of 38's main sample
    s = pd.read_parquet(P / "all_sample.parquet", columns=["application_number", "tech_center"])
    iv = pd.read_parquet(P / "all_text.parquet", columns=["application_number", "inventor_list"])
    s = s.merge(iv, on="application_number", how="inner", validate="1:1")
    s = s[s.inventor_list.map(lambda js: len(m08.inventors(js)) > 0)].reset_index(drop=True)
    quota = largest_remainder(s.tech_center.value_counts().sort_index(), N_SAMPLE)
    rng = np.random.default_rng(SEED)
    pick = np.concatenate([rng.choice(np.where(s.tech_center == tc)[0], quota[tc], replace=False) for tc in TCS])
    smp = s.iloc[np.sort(pick)][["application_number"]]
    con.register("smp", smp)
    t = con.execute(f"""SELECT application_number, title, abstract, claims, inventor_list
                        FROM '{P}/all_text.parquet' JOIN smp USING (application_number)
                        ORDER BY application_number""").df()
    t["s0"] = ("Title: " + t["title"] + "\n\nAbstract: " + t["abstract"] + "\n\nClaims: "
               + t["claims"]).str[:m08.MAXCH]  # 38's inputs()
    ev = t[["application_number", "s0", "inventor_list"]].assign(set="evaluator")

    # evaluator 2: the twins' common applications, as in 32
    d = load_record(ROOT)
    tc_rows = d[d.tech_center == m17.TC]
    top = tc_rows.groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort").head(
        m17.N_EXAMINERS).index
    common = d[d.examiner_id.isin(top)][["application_number"]]
    pin = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number", "s0", "s1"])
    ptx = pd.read_parquet(P / "pilot_text.parquet", columns=["application_number", "inventor_list"])
    tw = common.merge(pin[["application_number", "s0"]], on="application_number").merge(ptx, on="application_number")
    tw = tw.assign(set="twins")

    # one surname rank and one first-name rank per application, the same draw in both sets
    sur = surname_lists()
    fem, mal = m08.name_lists()
    apps = np.sort(pd.unique(pd.concat([ev.application_number, tw.application_number])))
    r2 = np.random.default_rng(SEED + 1)
    k_sur = pd.Series(r2.integers(0, N_SURNAMES, len(apps)), index=apps)
    k_fn = pd.Series(r2.integers(0, len(fem), len(apps)), index=apps)

    fmt = m08.fmt
    rows = []
    for x, variants in [(ev, EV_VARIANTS), (tw, TW_VARIANTS)]:
        for a, s0, js, st in zip(x.application_number, x.s0, x.inventor_list, x.set):
            invs = m08.inventors(js)
            first, last = invs[0] if invs else ("", "")
            rest = invs[1:]
            ks, kf = k_sur[a], k_fn[a]
            names = {"orig": (first, last), "hispanic": (first, sur["hispanic"][ks]),
                     "asian": (first, sur["asian"][ks]), "white": (first, sur["white"][ks]),
                     "female": (fem[kf], last), "male": (mal[kf], last), "neutral": (mal[kf], sur["white"][ks])}
            orig = fmt(invs) + "\n\n" + s0
            tail = orig[len("Inventors: ") + len(f"{first} {last}".strip()):] if invs else orig[len("Inventors: "):]
            for v in variants:
                txt = fmt([names[v]] + rest) + "\n\n" + s0
                if v == "orig" and invs:
                    assert txt == orig
                assert txt.endswith(tail), (a, v)  # only the first inventor's name differs
                rows.append((st, a, v, names[v][0], names[v][1], txt))
    V = pd.DataFrame(rows, columns=["set", "application_number", "variant", "first_name", "last_name", "text"])
    # the originals are the inputs that 38 (evaluator 1) and 08 (evaluator 2) embedded
    o = V[(V.set == "twins") & (V.variant == "orig")].merge(pin[["application_number", "s1"]], on="application_number")
    assert (o.text == o.s1).all(), "twins: original text differs from 08's s1"
    o = V[(V.set == "evaluator") & (V.variant == "orig")].merge(pin[["application_number", "s1"]],
                                                                 on="application_number")
    assert (o.text == o.s1).all(), "evaluator: original text differs from 08's s1 for pilot applications"
    V["text_id"], uniq = pd.factorize(V.text)
    texts = pd.DataFrame({"text_id": np.arange(len(uniq)), "text": uniq})
    texts["job"] = np.random.default_rng(SEED + 2).permutation(len(texts))  # spread long texts over the GPUs
    texts.to_parquet(TEXTS, index=False)
    V.drop(columns="text").to_parquet(VARIANTS, index=False)
    print("sample by center:", quota.to_dict())
    print(V.groupby(["set", "variant"]).size().to_string())
    print(f"unique texts to embed: {len(texts):,}")
    for k, v in sur.items():
        print(k, v)


# ---------------------------------------------------------------- 2. vectors (LLM environment)

def worker(k: int) -> None:
    """Embeds this GPU's texts in chunks; finished chunks are kept, so a stopped run resumes where it stopped."""
    from vllm import LLM
    m08 = load_module("m08", "08_embed.py")
    t = pd.read_parquet(TEXTS)
    t = t[t.job % len(GPUS) == k].sort_values("job")
    chunks = [t.iloc[i:i + CHUNK] for i in range(0, len(t), CHUNK)]
    todo = [c for c in range(len(chunks)) if not (P / f"nsw_shard_{k}_{c}.parquet").exists()]
    print(f"worker {k}: {len(todo)} of {len(chunks)} chunks to embed", flush=True)
    if not todo:
        return
    llm = LLM(model=str(ROOT / "models/Qwen3-Embedding-0.6B"), runner="pooling", max_model_len=4096,
              gpu_memory_utilization=0.6, seed=SEED32)
    for c in todo:
        x = chunks[c]
        out = llm.embed(x.text.tolist(), use_tqdm=(k == 0),
                        tokenization_kwargs={"truncation": True, "max_length": m08.MAX_TOKENS})
        np.save(P / f"nsw_shard_{k}_{c}.npy", np.asarray([o.outputs.embedding for o in out], dtype=np.float16))
        x[["text_id"]].to_parquet(P / f"nsw_shard_{k}_{c}.parquet", index=False)  # written last: marks it done
        print(f"worker {k}: chunk {c + 1} of {len(chunks)} done", flush=True)


def embed() -> None:
    t = pd.read_parquet(TEXTS, columns=["text_id"])
    print(f"embedding {len(t):,} texts on GPUs {GPUS}", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    emb = None
    parts = sorted(P.glob("nsw_shard_*_*.parquet"))
    for f in parts:
        x = np.load(f.with_suffix(".npy"))
        ids = pd.read_parquet(f).text_id.to_numpy()
        if emb is None:
            emb = np.zeros((len(t), x.shape[1]), dtype=np.float16)
        emb[ids] = x
    assert np.abs(emb).sum(1).min() > 0, "a text was not embedded"
    np.save(EMB, emb)
    for f in parts:
        f.with_suffix(".npy").unlink()
        f.unlink()
    print("saved", EMB, emb.shape)


# ---------------------------------------------------------------- 3. analysis (main environment)

def mean_se(x: np.ndarray, cl) -> float:
    """Standard error of a mean, clustered (32's mean_se)."""
    from cfe.estimation import cluster_se
    return float(cluster_se(((x - x.mean()) / len(x))[:, None], cl)[0])


def compare(Da, Db, Pa, Pb, cl) -> dict:
    """a against b: share of decisions that differ, change in the grant rate and in the grant probability.
    Matrices (applications × twins) are averaged over twins per application first (as in 32)."""
    ch = (Da != Db).astype(float)
    dd = Da.astype(float) - Db
    dp = Pa - Pb
    if dd.ndim == 2:
        ch, dd, dp = ch.mean(1), dd.mean(1), dp.mean(1)
    return dict(n=len(dd), changed=ch.mean(), changed_se=mean_se(ch, cl), dgrant=dd.mean(), dgrant_se=mean_se(dd, cl),
                dprob=dp.mean(), dprob_se=mean_se(dp, cl), grant_a=np.mean(Da), grant_b=np.mean(Db))


EV_CONTRASTS = [("Placebo: original re-embedded vs stored vector", "orig", "stored"),
                ("Surname Hispanic vs original", "hispanic", "orig"), ("Surname Asian vs original", "asian", "orig"),
                ("Surname White (control) vs original", "white", "orig"),
                ("First name female vs original", "female", "orig"), ("First name male vs original", "male", "orig"),
                ("Neutral name (male first name, White surname) vs original", "neutral", "orig"),
                ("Female − male first name", "female", "male"), ("Hispanic − White surname", "hispanic", "white"),
                ("Asian − White surname", "asian", "white")]
TW_CONTRASTS = [("Placebo: original re-embedded vs stored vector", "orig", "actual_final"),
                ("Surname Hispanic vs original", "hispanic", "orig"), ("Surname Asian vs original", "asian", "orig"),
                ("Surname White (control) vs original", "white", "orig"),
                ("Female − male first name", "female", "male"), ("Hispanic − White surname", "hispanic", "white"),
                ("Asian − White surname", "asian", "white")]


def evaluator(V: pd.DataFrame, Xn: np.ndarray) -> tuple:
    import dataclasses

    from threadpoolctl import threadpool_limits

    from cfe.check import PATENTS, assign_folds
    d = pd.read_parquet(P / "all_sample.parquet")
    sc = pd.read_parquet(P / "all_scores.parquet", columns=["application_number", GN])
    assert d.application_number.is_monotonic_increasing and (sc.application_number == d.application_number).all()
    s_all = sc[GN].to_numpy(float)
    X = np.load(E / "all_s1.npy", mmap_mode="r")
    assert X.shape[0] == len(d)
    fold = assign_folds(d, dataclasses.replace(PATENTS, units=tuple(TCS)))  # 38's folds
    rng = np.random.default_rng(SEED + 3)
    beta, diag = {}, []
    with threadpool_limits(4):
        for tc in TCS:
            rows = np.where(d.tech_center == tc)[0]
            for k in np.unique(fold[rows]):
                te = rows[fold[rows] == k]
                A = np.column_stack([np.asarray(X[te], dtype=np.float64), np.ones(len(te))])
                y = s_all[te]
                b = np.linalg.lstsq(A, y, rcond=None)[0]
                h = rng.random(len(te)) < 0.5
                bh = np.linalg.lstsq(A[h], y[h], rcond=None)[0]
                diag.append(dict(tech_center=tc, fold=int(k), n=len(te), score_sd=y.std(),
                                 max_abs_residual=np.abs(A @ b - y).max(),
                                 max_abs_error_other_half=np.abs(A[~h] @ bh - y[~h]).max()))
                beta[tc, k] = b
    diag = pd.DataFrame(diag)
    print(diag.to_string(index=False))
    assert diag.max_abs_error_other_half.max() < 1e-3 * diag.score_sd.min(), "fold models not recovered"

    thr, rate = {}, {}
    for tc in TCS:
        m = (d.tech_center == tc).to_numpy()
        rate[tc] = d.loc[m, "granted"].mean()
        thr[tc] = pd.Series(s_all[m]).quantile(1 - rate[tc])  # cfe.check.decision_gaps, as in 38
    ev = V[V.set == "evaluator"].pivot(index="application_number", columns="variant", values="text_id").astype(int)
    pos = np.searchsorted(d.application_number.to_numpy(), ev.index.to_numpy())
    assert (d.application_number.to_numpy()[pos] == ev.index.to_numpy()).all()
    s = d.iloc[pos].reset_index(drop=True)
    tcs, fk = s.tech_center.to_numpy(), fold[pos]
    S = {"stored": s_all[pos]}
    for v in EV_VARIANTS + ["stored_recovered"]:
        Xv = np.asarray(X[pos], dtype=np.float64) if v == "stored_recovered" else Xn[ev[v].to_numpy()].astype(
            np.float64)
        out = np.empty(len(s))
        for (tc, k), b in beta.items():
            m = (tcs == tc) & (fk == k)
            out[m] = Xv[m] @ b[:-1] + b[-1]
        S[v] = out
    print("recovered vs saved scores of the sample, max abs difference:",
          f"{np.abs(S['stored_recovered'] - S['stored']).max():.2e}")
    th = np.array([thr[t] for t in tcs])
    D = {v: (x >= th) for v, x in S.items()}
    Pr = {v: 1 / (1 + np.exp(-x)) for v, x in S.items()}
    rows = []
    for smp, mask in [("All centers", np.ones(len(s), bool)), ("3600", tcs == "3600")]:
        cl = s.examiner_id.to_numpy()[mask]
        for lab, a, b in EV_CONTRASTS:
            rows.append(dict(sample=smp, contrast=lab, **compare(D[a][mask], D[b][mask], Pr[a][mask], Pr[b][mask],
                                                                 cl)))
    res = pd.DataFrame(rows)
    return res, diag, s, D, Pr, rate


def neutralization(s: pd.DataFrame, D: dict, Pr: dict) -> pd.DataFrame:
    """Gaps in the evaluator's decisions (and probabilities) with the actual and with neutralized names."""
    from cfe.estimation import cluster_se, tsls
    m44 = load_module("m44", "44_c3_extra.py")
    s = s.copy()
    s["group"] = m44.ethnicity_groups(s).to_numpy()  # 33's rule: surname, 2010 Census, share >= 60%
    series = ["orig", "stored", "white", "male", "female", "neutral"]
    neutral = {"Female − male": ["male", "female", "neutral"], "Hispanic − White": ["white", "neutral"],
               "Asian − White": ["white", "neutral"]}
    rows = []
    for smp, units in [("All centers", TCS), ("3600", ["3600"])]:
        in_s = s.tech_center.isin(units).to_numpy()
        for comp, keep in [("Female − male", in_s),
                           ("Hispanic − White", in_s & s.group.isin(["hispanic", "white"]).to_numpy()),
                           ("Asian − White", in_s & s.group.isin(["asian", "white"]).to_numpy())]:
            x = s[keep]
            g = (x.female == 1 if comp == "Female − male" else x.group != "white").to_numpy(float)
            for scale, src in [("decisions", D), ("probability", Pr)]:
                cols = ["examiners"] + series
                Y = np.column_stack([x.granted.to_numpy(float)] + [src[v][keep].astype(float) for v in series])
                b, psi = tsls(Y, g, g, x.au_year)
                se = cluster_se(psi, x.examiner_id)
                j = {c: i for i, c in enumerate(cols)}
                for c in cols:
                    r = dict(sample=smp, comparison=comp, scale=scale, series=c, n=len(x), n_group=int(g.sum()),
                             gap=b[j[c]], se=se[j[c]])
                    if c in neutral[comp]:
                        o, n_ = j["orig"], j[c]
                        ratio = b[n_] / b[o]
                        r["remaining"] = ratio
                        r["remaining_se"] = cluster_se(((psi[:, n_] - ratio * psi[:, o]) / b[o])[:, None],
                                                       x.examiner_id)[0]
                        r["through_name"] = b[o] - b[n_]
                        r["through_name_se"] = cluster_se((psi[:, o] - psi[:, n_])[:, None], x.examiner_id)[0]
                    rows.append(r)
    return pd.DataFrame(rows)


def twins(V: pd.DataFrame, Xn: np.ndarray) -> tuple:
    """32's analysis with the names shown, plus the surname variants."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from cfe.check import load_record
    m17 = load_module("m17", "17_ai_examiners.py")
    d = load_record(ROOT)
    t32 = pd.read_parquet(P / "twins_names_inputs.parquet", columns=["set", "row", "application_number"])
    xa = pd.read_parquet(P / "ai_examiner_apps.parquet", columns=["application_number", "examiner_id", "granted"])
    xs1 = t32[t32.set == "ai_ex_s1"].sort_values("row").application_number.to_numpy()
    assert (xs1 == xa.application_number.to_numpy()).all()
    common_ids = t32[t32.set == "common_s1_f"].sort_values("row").application_number.to_numpy()
    tc_rows = np.where(d.tech_center == m17.TC)[0]
    counts = d.iloc[tc_rows].groupby("examiner_id").size().sort_values(ascending=False, kind="mergesort")
    top = counts.head(m17.N_EXAMINERS).index.tolist()
    others = tc_rows[~d.examiner_id.iloc[tc_rows].isin(top).to_numpy()]
    pool = d[d.examiner_id.isin(top)].copy()
    pool_rows = pool.index.to_numpy()
    swap_rows = pd.Series(np.arange(len(common_ids)), index=common_ids).loc[pool.application_number].to_numpy()
    tw = V[V.set == "twins"].pivot(index="application_number", columns="variant", values="text_id").astype(int).loc[
        pool.application_number]
    emb = np.load(E / "s1.npy", mmap_mode="r")
    shared, _, shared_auc = m17.fit_shared(np.asarray(emb[others], dtype=np.float32),
                                           d.granted.to_numpy()[others].astype(int),
                                           pd.factorize(d.examiner_id.iloc[others])[0])
    E_tc = np.asarray(emb[tc_rows], dtype=np.float32)
    sc_all = StandardScaler().fit(E_tc)
    Z = PCA(m17.N_PC, random_state=SEED32).fit(sc_all.transform(E_tc))

    def feats(E_):
        return np.column_stack([shared(E_), Z.transform(sc_all.transform(E_))])
    X = {"actual": feats(np.asarray(emb[pool_rows], dtype=np.float32)),
         "female": feats(np.load(E / "common_s1_f.npy").astype(np.float32)[swap_rows]),
         "male": feats(np.load(E / "common_s1_m.npy").astype(np.float32)[swap_rows])}
    for v in TW_VARIANTS:
        X[v] = feats(Xn[tw[v].to_numpy()].astype(np.float32))
    Xx = feats(np.load(E / "ai_ex_s1.npy").astype(np.float32))
    kinds = {"leniency": [0], "text": slice(None)}
    prob = {}
    for i, e in enumerate(top):
        own, ext = (pool.examiner_id == e).to_numpy(), (xa.examiner_id == e).to_numpy()
        y, yx = pool.granted.to_numpy()[own].astype(int), xa.granted.to_numpy()[ext].astype(int)
        for kind, cols in kinds.items():
            oof, m, sc, _ = m17.fit_cv(Xx[ext][:, cols], yx, X["actual"][own][:, cols], y, SEED32 + i)
            for v, Xv in X.items():
                p = m.predict_proba(sc.transform(Xv[:, cols]))[:, 1]
                if v == "actual":
                    prob[kind, "actual_final", e] = p.copy()
                    p[own] = oof
                prob[kind, v, e] = p
    lev = pool.groupby("examiner_id").lenient_grant.first()
    order = sorted(top, key=lambda e: (np.isnan(lev[e]), lev[e]))  # AI twin 1–20 as in 17 and 32
    cl = pool.examiner_id.to_numpy()
    pooled, spread = [], []
    for kind in kinds:
        Pm = {v: np.column_stack([prob[kind, v, e] for e in order]) for v in ["actual_final", *X]}
        thr = np.array([np.quantile(Pm["actual"][:, j], 1 - Pm["actual"][:, j].mean()) for j in range(len(order))])
        Dm = {v: x >= thr for v, x in Pm.items()}  # the bar stays where the actual names put it (32)
        for lab, a, b in TW_CONTRASTS:
            pooled.append(dict(kind=kind, contrast=lab, **compare(Dm[a], Dm[b], Pm[a], Pm[b], cl)))
            per = pd.DataFrame([compare(Dm[a][:, j], Dm[b][:, j], Pm[a][:, j], Pm[b][:, j], cl)
                                for j in range(len(order))])
            lo, hi = per.dgrant - 1.96 * per.dgrant_se, per.dgrant + 1.96 * per.dgrant_se
            spread.append(dict(kind=kind, contrast=lab, twins=len(per), dgrant_min=per.dgrant.min(),
                               dgrant_median=per.dgrant.median(), dgrant_max=per.dgrant.max(),
                               twins_ci_below_zero=int((hi < 0).sum()), twins_ci_above_zero=int((lo > 0).sum()),
                               dprob_min=per.dprob.min(), dprob_median=per.dprob.median(), dprob_max=per.dprob.max(),
                               changed_min=per.changed.min(), changed_median=per.changed.median(),
                               changed_max=per.changed.max()))
    pooled, spread = pd.DataFrame(pooled), pd.DataFrame(spread)
    # check against 32: the female − male first-name swap reproduces its pooled numbers
    s32 = pd.read_csv(T / "twins_names_summary.csv").set_index("kind")
    for kind in kinds:
        r = pooled[(pooled.kind == kind) & (pooled.contrast == "Female − male first name")].iloc[0]
        print(f"twins ({kind}), female − male: Δ grants {r.dgrant:.5f} (32: {s32.loc[kind, 'dgrant_mean']:.5f}), "
              f"changed {r.changed:.5f} (32: {s32.loc[kind, 'changed_by_name_mean']:.5f})")
        assert abs(r.dgrant - s32.loc[kind, "dgrant_mean"]) < 2e-3, "32's twins not reproduced"
    info = dict(n_common=len(pool), n_twins=len(order), shared_auc=shared_auc,
                s32={k: (s32.loc[k, "dgrant_mean"], s32.loc[k, "changed_by_name_mean"]) for k in kinds})
    return pooled, spread, info


# ---------------------------------------------------------------- 4. report

def num(x: float, nd: int = 1, scale: float = 100) -> str:
    """A number for LaTeX: scaled, rounded (Python's round on a float, as in txt), with a typographic minus."""
    v = round(scale * float(x), nd)
    if v == 0:
        v = 0.0
    return f"{v:.{nd}f}".replace("-", "$-$")


def txt(x: float, nd: int = 1, scale: float = 100) -> str:
    v = round(scale * float(x), nd)
    return f"{0.0 if v == 0 else v:.{nd}f}".replace("-", "−")


def ci_txt(b, se, nd=1, scale=100) -> str:
    return f"{txt(b, nd, scale)} [{txt(b - 1.96 * se, nd, scale)}, {txt(b + 1.96 * se, nd, scale)}]"


def ci_tex(b, se, nd=1, scale=100) -> str:
    return f"[{num(b - 1.96 * se, nd, scale)}, {num(b + 1.96 * se, nd, scale)}]"


def report(ev, diag, s, rate, gaps, tw, spread, info, sur, fem, mal) -> None:
    def row(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        assert m.sum() == 1, kw
        return df[m].iloc[0]
    n_all, n36 = int((s.tech_center.notna()).sum()), int((s.tech_center == "3600").sum())
    nex = s.examiner_id.nunique()

    # ---- macros
    M = {"N": f"{n_all:,}", "NThreeSix": f"{n36:,}", "NExaminers": f"{nex:,}", "NSurnames": str(N_SURNAMES),
         "NFirstNames": str(len(fem)), "SurnameShare": f"{SHARE:.0f}", "TwN": f"{info['n_common']:,}",
         "TwNTwins": str(info["n_twins"])}
    key = {"Female − male first name": "Fm", "Hispanic − White surname": "Hw", "Asian − White surname": "Aw",
           "Surname Hispanic vs original": "Hisp", "Surname Asian vs original": "Asian",
           "Surname White (control) vs original": "White", "First name female vs original": "Female",
           "First name male vs original": "Male",
           "Neutral name (male first name, White surname) vs original": "Neutral",
           "Placebo: original re-embedded vs stored vector": "Placebo"}
    for smp, pre in [("All centers", "Ev"), ("3600", "EvThreeSix")]:
        for lab, k in key.items():
            r = row(ev, sample=smp, contrast=lab)
            M[f"{pre}{k}Grant"], M[f"{pre}{k}GrantCI"] = num(r.dgrant, 2), ci_tex(r.dgrant, r.dgrant_se, 2)
            M[f"{pre}{k}Prob"], M[f"{pre}{k}ProbCI"] = num(r.dprob, 2), ci_tex(r.dprob, r.dprob_se, 2)
            M[f"{pre}{k}Changed"] = num(r.changed, 2)
    for kind, pre in [("leniency", "Tw"), ("text", "TwText")]:
        for lab in TW_CONTRASTS:
            k = key[lab[0]]
            r = row(tw, kind=kind, contrast=lab[0])
            M[f"{pre}{k}Grant"], M[f"{pre}{k}GrantCI"] = num(r.dgrant, 2), ci_tex(r.dgrant, r.dgrant_se, 2)
            M[f"{pre}{k}Prob"], M[f"{pre}{k}ProbCI"] = num(r.dprob, 2), ci_tex(r.dprob, r.dprob_se, 2)
            M[f"{pre}{k}Changed"] = num(r.changed, 2)
            sp = row(spread, kind=kind, contrast=lab[0])
            M[f"{pre}{k}GrantMin"], M[f"{pre}{k}GrantMax"] = num(sp.dgrant_min, 2), num(sp.dgrant_max, 2)
            M[f"{pre}{k}TwinsBelow"] = str(sp.twins_ci_below_zero)
            M[f"{pre}{k}TwinsAbove"] = str(sp.twins_ci_above_zero)
    gk = {"Female − male": "Fem", "Hispanic − White": "Hisp", "Asian − White": "Asian"}
    sk = {"examiners": "Exam", "orig": "Orig", "white": "Surname", "male": "MaleName", "female": "FemaleName",
          "neutral": "Neutral"}
    for smp, pre in [("All centers", "Gap"), ("3600", "GapThreeSix")]:
        for comp, g in gk.items():
            for ser, k in sk.items():
                if comp == "Female − male" and ser == "white" or comp != "Female − male" and ser in ["male", "female"]:
                    continue
                r = row(gaps, sample=smp, comparison=comp, scale="decisions", series=ser)
                M[f"{pre}{g}{k}"], M[f"{pre}{g}{k}CI"] = num(r.gap), ci_tex(r.gap, r.se)
                if ser not in ["examiners", "orig"]:
                    M[f"{pre}{g}{k}Remaining"] = num(r.remaining, 0)
                    M[f"{pre}{g}{k}RemainingCI"] = ci_tex(r.remaining, r.remaining_se, 0)
                    M[f"{pre}{g}{k}ThroughName"] = num(r.through_name)
                    M[f"{pre}{g}{k}ThroughNameCI"] = ci_tex(r.through_name, r.through_name_se)
            M[f"{pre}{g}NGroup"] = f"{row(gaps, sample=smp, comparison=comp, scale='decisions', series='orig').n_group:,}"
    for k in M:
        assert k.isalpha(), k
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "nsw_numbers.tex").write_text("% Generated by code/48_name_swaps.py; do not edit by hand.\n" + "".join(
        f"\\newcommand{{\\xNs{k}}}{{{v}}}\n" for k, v in sorted(M.items())))

    # ---- table
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Eight-center evaluator (trained on grant + names)} & "
         r"\multicolumn{2}{c}{TC 3600 AI twins (pooled over 20)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"First inventor's name & $\Delta$ grants (pp) & Decisions changed (\%) & $\Delta$ grants (pp) & "
         r"Decisions changed (\%) \\", r"\midrule"]
    for lab, nm in [("Female − male first name", "Female vs.\\ male first name"),
                    ("Hispanic − White surname", "Hispanic vs.\\ White surname"),
                    ("Asian − White surname", "Asian vs.\\ White surname")]:
        a, t = row(ev, sample="All centers", contrast=lab), row(tw, kind="leniency", contrast=lab)
        L.append(f"{nm} & {num(a.dgrant, 2)} {{\\footnotesize {ci_tex(a.dgrant, a.dgrant_se, 2)}}} & "
                 f"{num(a.changed, 2)} & {num(t.dgrant, 2)} {{\\footnotesize {ci_tex(t.dgrant, t.dgrant_se, 2)}}} & "
                 f"{num(t.changed, 2)} \\\\")
    plc = "Placebo: original re-embedded vs stored vector"
    pa, pt = row(ev, sample="All centers", contrast=plc), row(tw, kind="leniency", contrast=plc)
    L += [r"\addlinespace", f"Placebo: same text embedded twice & {num(pa.dgrant, 2)} & {num(pa.changed, 2)} & "
                            f"{num(pt.dgrant, 2)} & {num(pt.changed, 2)} \\\\",
          r"\midrule", f"Applications & \\multicolumn{{2}}{{c}}{{{n_all:,} (eight centers)}} & "
                       f"\\multicolumn{{2}}{{c}}{{{info['n_common']:,} $\\times$ {info['n_twins']} twins}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    (TEX / "nsw_tab_swaps.tex").write_text("\n".join(L) + "\n")

    # ---- report
    def evline(r):
        return (f"| {r.contrast} | {ci_txt(r.dgrant, r.dgrant_se, 2)} | {ci_txt(r.dprob, r.dprob_se, 2)} | "
                f"{ci_txt(r.changed, r.changed_se, 2)} | {txt(r.grant_a)}% vs {txt(r.grant_b)}% |")
    R = ["# Name swaps: the AI evaluator's direct response to the first inventor's name (C3)", "",
         "Same application, everything identical except the first inventor's name (code/48_name_swaps.py). "
         "Δ grants: change in the share granted (pp); Δ probability: change in the predicted grant probability "
         "(pp); decisions changed: share of applications whose decision differs (%). 95% CIs clustered by examiner.",
         "", "## Names", "",
         f"Surnames: the {N_SURNAMES} most frequent in the 2010 Census surname table borne at least {SHARE:.0f}% by "
         "each group (letters only; one rank drawn at random per application with a fixed seed, the same rank for "
         "the three groups).", ""]
    for k, lab in [("hispanic", "Hispanic"), ("asian", "Asian/Pacific Islander"), ("white", "White (control)")]:
        R.append(f"- {lab}: {', '.join(sur[k])}")
    R += ["", f"First names: 08's audit names ({len(fem)} female and {len(mal)} male, the most common unambiguous "
              "first names of U.S. inventors, paired by rank; one rank drawn per application). The neutral name "
              "combines the drawn male first name with the drawn White surname.", "",
          "## 1. Eight-center evaluator (38's \"Train on grant + names\")", "",
          f"A random {n_all:,} applications of 38's main sample (filed 2010–2014, eight centers, at least one "
          f"inventor listed), drawn within center in proportion to its size ({n36:,} in TC 3600; {nex:,} "
          "examiners). Each application is scored by its own out-of-fold model of 38 (five folds grouped by "
          "examiner within center). 38 saved the out-of-fold scores, not the models; as each fold's model is "
          "linear in the embedding, it is recovered from the fold's held-out applications by least squares (largest "
          f"absolute residual {diag.max_abs_residual.max():.1e}, largest error on a held-out half of the fold "
          f"{diag.max_abs_error_other_half.max():.1e}, against a score SD of at least {diag.score_sd.min():.2f}; "
          "tables/nsw_fold_recovery.csv). Each center grants the share its examiners grant ("
          + ", ".join(f"{tc} {100 * rate[tc]:.1f}%" for tc in TCS) + "), with the threshold fixed where the "
          "original scores of all of its applications put it; only the name moves. Swaps are compared with the "
          "original re-embedded in the same run; the placebo compares that with 38's stored vector of the same "
          "text (numerical noise of the embedding).", ""]
    for smp in ["All centers", "3600"]:
        R += [f"### {'Pooled over the eight centers' if smp == 'All centers' else 'TC 3600'}", "",
              "| Swap | Δ grants (pp) | Δ probability (pp) | Decisions changed (%) | Grant rate, a vs b |",
              "|---|---|---|---|---|"]
        R += [evline(r) for r in ev[ev["sample"] == smp].itertuples()]
        R.append("")
    R += ["### How much of the evaluator's gap runs through the first inventor's name", "",
          "Gap in the evaluator's decisions (pp; group − reference, art unit × year cells, clustered by examiner) "
          "with the actual names and with the first inventor's name neutralized for everyone; \"remaining\" = "
          "neutralized gap ÷ actual gap. Ethnicity of the first inventor from the surname by 33's rule (2010 Census, "
          "share ≥ 60%); gender as in 03. The evaluator's own decisions with the actual names are those from the "
          "re-embedded original.", "",
          "| Sample | Comparison | Group n | Examiners | Evaluator, actual names | Neutralized | Gap | Remaining (%) | "
          "Through the name (pp) |", "|---|---|---|---|---|---|---|---|---|"]
    lab_n = {"white": "surname → White", "male": "first name → male", "female": "first name → female",
             "neutral": "full name → male first, White surname"}
    for smp in ["All centers", "3600"]:
        for comp in gk:
            g = gaps[(gaps["sample"] == smp) & (gaps.comparison == comp) & (gaps.scale == "decisions")].set_index(
                "series")
            for ser in [c for c in ["white", "male", "female", "neutral"] if pd.notna(g.loc[c].get("remaining"))]:
                r = g.loc[ser]
                R.append(f"| {smp} | {comp} | {int(r.n_group):,} | {ci_txt(g.loc['examiners'].gap, g.loc['examiners'].se)}"
                         f" | {ci_txt(g.loc['orig'].gap, g.loc['orig'].se)} | {lab_n[ser]} | {ci_txt(r.gap, r.se)} | "
                         f"{ci_txt(r.remaining, r.remaining_se, 0)} | {ci_txt(r.through_name, r.through_name_se)} |")
    R += ["", "The same on the probability scale (pp) is in tables/nsw_gaps.csv (scale = probability).", "",
          "## 2. AI twins of 20 TC 3600 examiners (17, with names as in 32)", "",
          f"Each twin decides the same {info['n_common']:,} applications with each variant; bars as in 32 (set by "
          "the decisions with the actual names, own applications out of fold); swaps compared with the same "
          "twin's decision on the re-embedded original. Pooled: per application averaged over the twins; CIs "
          "clustered by the application's examiner, as in 32. Leniency twins (main in 17): the common grant score "
          f"(names shown, examiner-grouped AUC {info['shared_auc']:.3f}) with the examiner's own intercept and "
          "slope; text twins add 32 text dimensions. The female − male rows reproduce 32 up to numerical noise in "
          "the refit (Δ grants, changed: leniency twins " + ", ".join(f"{txt(v, 2)}" for v in info["s32"]["leniency"])
          + "; text twins " + ", ".join(f"{txt(v, 2)}" for v in info["s32"]["text"]) + " in 32).", ""]
    for kind in ["leniency", "text"]:
        R += [f"### {kind.capitalize()} twins", "",
              "| Swap | Δ grants (pp) | Δ probability (pp) | Decisions changed (%) | Δ grants across twins: min, "
              "median, max (pp) | Twins with CI below / above 0 |", "|---|---|---|---|---|---|"]
        for r in tw[tw.kind == kind].itertuples():
            sp = row(spread, kind=kind, contrast=r.contrast)
            R.append(f"| {r.contrast} | {ci_txt(r.dgrant, r.dgrant_se, 2)} | {ci_txt(r.dprob, r.dprob_se, 2)} | "
                     f"{ci_txt(r.changed, r.changed_se, 2)} | {txt(sp.dgrant_min, 2)}, {txt(sp.dgrant_median, 2)}, "
                     f"{txt(sp.dgrant_max, 2)} | {sp.twins_ci_below_zero} / {sp.twins_ci_above_zero} |")
        R.append("")
    R += ["## Caveats", "",
          "- The swap measures the model's direct response to the first inventor's name with the text held fixed. "
          "Gender and ethnicity also reach the evaluator through the text (field, wording, co-inventors): 38's text "
          "alone predicts the first inventor's gender with an AUC of 0.75 in TC 3600. A small direct effect of the "
          "name does not mean that the evaluator's decisions are free of a gap; the neutralization rows give the "
          "gap that remains when no first inventor's name carries the signal.",
          "- Only the first inventor's name changes; co-inventors' names, which also signal gender and ethnicity, "
          "are kept, so the neutralized gaps still include their contribution.",
          "- Surnames stand for groups through the most common names borne mostly by one group; names that are less "
          "typical, or first names that signal ethnicity, may move the evaluators differently. Hispanic and Asian "
          "surnames are compared with common White surnames rather than with the original name, so that the "
          "contrast is not confounded with the rarity of the original name.",
          "- The evaluators are linear in the embedding of the full input (evaluator 1) or of a common score and "
          "32 text dimensions (evaluator 2); a model that reads the names differently (an LLM prompted with them) "
          "may respond more strongly. The placebo row bounds the numerical noise of re-embedding identical text.",
          "- The 20,000-application sample gives the change in decisions precisely, the gaps with the actual names "
          "less so (compare the full-sample gaps of 38 and 44); the gap through the name is estimated from paired "
          "decisions on the same applications and is precise.",
          "- Gaps are disparities in decisions, not evidence about any examiner's intent; twin results are pooled "
          "over the 20 twins (no twin-level rows).", "",
          "Files: tables/nsw_evaluator.csv, nsw_gaps.csv, nsw_twins.csv, nsw_twins_spread.csv, nsw_fold_recovery.csv, "
          "nsw_names.csv; latex/nsw_tab_swaps.tex, nsw_numbers.tex (\\xNs...)."]
    (OUT / "name_swaps.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


def analyze() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    m08 = load_module("m08", "08_embed.py")
    V = pd.read_parquet(VARIANTS)
    Xn = np.load(EMB)
    assert len(Xn) == V.text_id.max() + 1
    ev, diag, s, D, Pr, rate = evaluator(V, Xn)
    gaps = neutralization(s, D, Pr)
    tw, spread, info = twins(V, Xn)
    sur = surname_lists()
    fem, mal = m08.name_lists()
    names = pd.concat([pd.DataFrame({"list": f"surname_{k}", "rank": np.arange(1, len(v) + 1), "name": v})
                       for k, v in sur.items()] +
                      [pd.DataFrame({"list": f"first_{k}", "rank": np.arange(1, len(v) + 1), "name": v})
                       for k, v in [("female", fem), ("male", mal)]])
    names.to_csv(T / "nsw_names.csv", index=False)
    ev.to_csv(T / "nsw_evaluator.csv", index=False)
    gaps.to_csv(T / "nsw_gaps.csv", index=False)
    tw.to_csv(T / "nsw_twins.csv", index=False)
    spread.to_csv(T / "nsw_twins_spread.csv", index=False)
    diag.to_csv(T / "nsw_fold_recovery.csv", index=False)
    report(ev, diag, s, rate, gaps, tw, spread, info, sur, fem, mal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.worker is not None:
        worker(a.worker)
    elif a.analyze:
        analyze()
    else:
        embed()


if __name__ == "__main__":
    main()
