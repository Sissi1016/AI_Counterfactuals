"""The check in all eight technology centers, with the embedding-based evaluators.

The pilot (06–29) covers TC 3600 and 1600; the human side of the check already covers all eight centers (24). This
script runs the evaluator side everywhere: applications filed 2010–2014 in the eight centers with the pilot's
restrictions (06: in HUPD, not a continuation, first-named inventor in the U.S. with known gender), citations within
five years of grant excluding self-citations (24), texts from the HUPD tarballs (07), Qwen3-Embedding-0.6B vectors of
title, abstract, and claims, without and with the inventor list (08; pilot applications reuse their vectors), and
cross-fitted evaluators in five folds grouped by examiner within each center (12, 21):
- Train on grant / grant + names / first action: L2-penalized logistic regressions, penalty by inner cross-validation;
- Train on citations / citations + names: ridge regressions of log(1 + citations) on granted applications.
The check of cfe.check then gives, by center and pooled: the marginal applications' citations (2SLS), each evaluator's
implied value and signed error, by gender, and the gender gap in decisions at the human grant rate. The LLM
evaluators (zero-shot, fine-tuned) stay in the pilot.

Run: uv run python code/38_all_centers.py --prepare
     uv run python -I code/38_all_centers.py --extract      (reads the HUPD archives; -I as in 07)
     CFE_GPUS=0,1 .venv-llm/bin/python code/38_all_centers.py --embed
     uv run python code/38_all_centers.py --analyze
Output: data/processed/{all_sample,all_text}.parquet, data/processed/emb/{all_s0,all_s1}.npy,
        output/all_centers.md, output/tables/all_centers_*.csv, output/figures/journal/fig_all_centers.{pdf,png}
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
TCS = ["1600", "1700", "2100", "2400", "2600", "2800", "3600", "3700"]
YEARS = (2010, 2014)
GPUS = [int(x) for x in os.environ.get("CFE_GPUS", "0,1").split(",")]
SAMPLE, TEXT, JOBS = P / "all_sample.parquet", P / "all_text.parquet", P / "all_embed_jobs.parquet"
LOGIT = [("Train on grant", "s0", "granted"), ("Train on grant + names", "s1", "granted"),
         ("Train on first action", "s0", "fa_allow")]
RIDGE = [("Train on citations (granted only)", "s0"), ("Train on citations + names", "s1")]
K_INNER, THREADS = 4, 4
C_GRID = [0.5, 0.05, 0.005, 0.0005, 0.00005]  # as in 12
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]  # as in 21


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # worker processes look functions up by module name
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. sample (main environment)

def prepare() -> None:
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads = 16")
    tcs = ", ".join(f"'{t}'" for t in TCS)
    con.execute(f"""
    COPY (
      SELECT a.application_number, a.examiner_id, a.art_unit, a.au_year, a.tech_center, a.filing_year,
             a.fa_allow, a.granted, a.lenient_grant, a.lenient_fa,
             (a.small_entity_indicator IN ('SMALL', 'MICRO'))::INT AS small_entity,
             g.n_inventors, (g.first_inventor_gender = 'F')::INT AS female,
             a.au_year || '_' || coalesce(u.uspc_class, 'none') AS au_class,
             CASE WHEN a.granted = 1 THEN coalesce(c.cites_5y_noself, 0) END AS cites_ns
      FROM '{P}/applications_leniency.parquet' a
      JOIN '{P}/app_gender.parquet' g USING (application_number)
      LEFT JOIN (SELECT application_number, uspc_class FROM '{P}/patex_application_data.parquet') u
          USING (application_number)
      LEFT JOIN '{P}/cites_noself_all.parquet' c USING (application_number)
      WHERE a.tech_center IN ({tcs}) AND a.filing_year BETWEEN {YEARS[0]} AND {YEARS[1]}
        AND a.in_hupd AND NOT a.continuation
        AND g.first_inventor_country = 'US' AND g.first_inventor_gender IN ('F', 'M')
      ORDER BY a.application_number
    ) TO '{SAMPLE}' (FORMAT parquet, COMPRESSION zstd)""")
    print(con.execute(f"""SELECT tech_center, count(*) n, count(DISTINCT examiner_id) examiners,
        round(avg(female), 3) female, round(avg(granted), 3) granted, round(avg(cites_ns), 2) cites
        FROM '{SAMPLE}' GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


# ---------------------------------------------------------------- 2. texts (main environment, isolated)

def extract() -> None:
    from concurrent.futures import ProcessPoolExecutor
    m07 = load_module("m07", "07_extract_text.py")
    s = pd.read_parquet(SAMPLE, columns=["application_number"])
    pilot = pd.read_parquet(P / "pilot_text.parquet")
    have = pilot[pilot.application_number.isin(s.application_number)]
    wanted = frozenset(set(s.application_number) - set(have.application_number))
    years = list(m07.YEARS)
    with ProcessPoolExecutor(max_workers=len(years)) as ex:
        parts = list(ex.map(m07.scan, years, [wanted] * len(years)))
    new = pd.DataFrame([r for part in parts for r in part]).drop_duplicates("application_number")
    out = pd.concat([have, new], ignore_index=True).sort_values("application_number")
    out.to_parquet(TEXT, index=False)
    print(f"texts: {len(out):,} of {len(s):,} ({len(out) / len(s):.1%}); new {len(new):,}, "
          f"from the pilot {len(have):,}")


# ---------------------------------------------------------------- 3. vectors (LLM environment)

def inputs() -> pd.DataFrame:
    """The applications with text, in application_number order (the row order of the vectors)."""
    m08 = load_module("m08", "08_embed.py")
    s = pd.read_parquet(SAMPLE, columns=["application_number"])
    t = pd.read_parquet(TEXT)
    d = s.merge(t, on="application_number").sort_values("application_number").reset_index(drop=True)
    d["s0"] = ("Title: " + d["title"] + "\n\nAbstract: " + d["abstract"] + "\n\nClaims: "
               + d["claims"]).str[:m08.MAXCH]
    d["s1"] = d["inventor_list"].map(m08.inventors).map(m08.fmt) + "\n\n" + d["s0"]
    return d[["application_number", "s0", "s1"]]


def worker(k: int) -> None:
    from vllm import LLM
    m08 = load_module("m08", "08_embed.py")
    d = inputs()
    j = pd.read_parquet(JOBS)
    j = j[j.job % len(GPUS) == k]
    texts = [d.at[r, v] for v, r in zip(j.variant, j.row)]
    llm = LLM(model=str(ROOT / "models/Qwen3-Embedding-0.6B"), runner="pooling", max_model_len=4096,
              gpu_memory_utilization=0.6, seed=SEED)
    out = llm.embed(texts, use_tqdm=(k == 0), tokenization_kwargs={"truncation": True, "max_length": m08.MAX_TOKENS})
    np.save(E / f"all_shard_{k}.npy", np.asarray([o.outputs.embedding for o in out], dtype=np.float16))
    j[["job"]].to_parquet(E / f"all_shard_{k}.parquet", index=False)


def embed() -> None:
    d = inputs()
    pilot = pd.read_parquet(P / "pilot_inputs.parquet", columns=["application_number"])
    pos = pd.Series(np.arange(len(pilot)), index=pilot.application_number)
    in_pilot = d.application_number.isin(pos.index).to_numpy()
    rows = np.where(~in_pilot)[0]
    j = pd.DataFrame({"variant": np.repeat(["s0", "s1"], len(rows)), "row": np.tile(rows, 2)})
    j["job"] = np.random.default_rng(SEED).permutation(len(j))  # shuffled so long texts spread over the GPUs
    j.to_parquet(JOBS, index=False)
    print(f"to embed: {len(j):,} texts ({len(rows):,} applications × 2); reused from the pilot: {in_pilot.sum():,}",
          flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)],
                              env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g)})
             for k, g in enumerate(GPUS)]
    if any(p.wait() for p in procs):
        sys.exit("a worker failed")
    for v in ["s0", "s1"]:
        old = np.load(E / f"{v}.npy", mmap_mode="r")
        out = np.zeros((len(d), old.shape[1]), dtype=np.float16)
        out[in_pilot] = old[pos[d.application_number[in_pilot]].to_numpy()]
        for k in range(len(GPUS)):
            x = np.load(E / f"all_shard_{k}.npy")
            ids = pd.read_parquet(E / f"all_shard_{k}.parquet").job.to_numpy()
            jj = j.set_index("job").loc[ids]
            m = (jj.variant == v).to_numpy()
            out[jj.row.to_numpy()[m]] = x[m]
        assert np.abs(out).sum(1).min() > 0, "a row was not embedded"
        np.save(E / f"all_{v}.npy", out)
    for k in range(len(GPUS)):
        (E / f"all_shard_{k}.npy").unlink()
        (E / f"all_shard_{k}.parquet").unlink()
    print("saved", E / "all_s0.npy", E / "all_s1.npy")


# ---------------------------------------------------------------- 4. the check (main environment)

def logit_task(X, y, ex, tr, te, seed):
    """12's spec_task: L2 logistic regression, penalty by inner cross-validated AUC (folds grouped by examiner)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with threadpool_limits(THREADS):
        oof = {c: np.full(len(tr), np.nan) for c in C_GRID}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            fa = tr[ia][~np.isnan(y[tr[ia]])]
            sc = StandardScaler().fit(X[fa])
            Xa, Xb = sc.transform(X[fa]), sc.transform(X[tr[ib]])
            for c in C_GRID:
                oof[c][ib] = LogisticRegression(C=c, max_iter=3000).fit(Xa, y[fa].astype(int)).decision_function(Xb)
        ok = ~np.isnan(y[tr])
        auc = {c: roc_auc_score(y[tr][ok].astype(int), oof[c][ok]) for c in C_GRID}
        best = max(auc, key=auc.get)
        fit = tr[ok]
        sc = StandardScaler().fit(X[fit])
        m = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X[fit]), y[fit].astype(int))
        return m.decision_function(sc.transform(X[te]))


def ridge_task(X, y, ex, tr, te, seed):
    """21's ridge_task: ridge regression of log(1 + citations) on granted applications, penalty by inner CV."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with threadpool_limits(THREADS):
        mse = {a: 0.0 for a in ALPHAS}
        for ia, ib in GroupKFold(K_INNER, shuffle=True, random_state=seed).split(tr, groups=ex[tr]):
            sc = StandardScaler().fit(X[tr[ia]])
            Xa, Xb = sc.transform(X[tr[ia]]), sc.transform(X[tr[ib]])
            for a in ALPHAS:
                mse[a] += ((Ridge(alpha=a).fit(Xa, y[tr[ia]]).predict(Xb) - y[tr[ib]]) ** 2).sum()
        best = min(mse, key=mse.get)
        sc = StandardScaler().fit(X[tr])
        return Ridge(alpha=best).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))

def analyze() -> None:
    import dataclasses
    import warnings

    from joblib import Parallel, delayed

    from cfe.check import HUMAN, NO_MODEL, assign_folds, check
    warnings.filterwarnings("ignore")
    m21 = load_module("m21", "21_citations.py")
    keys = inputs()[["application_number"]]
    d = keys.merge(pd.read_parquet(SAMPLE), on="application_number", how="left")
    X = {"s0": np.load(E / "all_s0.npy").astype(np.float32), "s1": np.load(E / "all_s1.npy").astype(np.float32)}
    design = dataclasses.replace(m21.CIT, units=tuple(TCS),
                                 samples=tuple((t, (t,)) for t in TCS) + (("All centers", tuple(TCS)),))
    fold = assign_folds(d, design)
    ex = pd.factorize(d.examiner_id)[0]
    lab = {c: d[c].to_numpy(float) for c in ["granted", "fa_allow"]}
    ylog = np.log1p(d.cites_ns.to_numpy(float))
    g = d.granted.to_numpy() == 1
    folds = []
    for tc in TCS:
        rows = np.where(d.tech_center == tc)[0]
        for k in np.unique(fold[rows]):
            folds.append((rows[fold[rows] != k], rows[fold[rows] == k]))
    jobs = [("logit", i, name, x, y) for i in range(len(folds)) for name, x, y in LOGIT] + \
           [("ridge", i, name, x, None) for i in range(len(folds)) for name, x in RIDGE]

    calls = []
    for kind, i, name, x, y in jobs:
        tr, te = folds[i]
        if kind == "logit":
            calls.append(delayed(logit_task)(X[x], lab[y], ex, tr, te, SEED + i))
        else:
            calls.append(delayed(ridge_task)(X[x], ylog, ex, tr[g[tr]], te, SEED + i))
    out = Parallel(n_jobs=4, verbose=5, max_nbytes="1M")(calls)
    sc = pd.DataFrame({"application_number": d.application_number})
    for (kind, i, name, x, y), s in zip(jobs, out):
        if name not in sc:
            sc[name] = np.nan
        sc.loc[folds[i][1], name] = s
    sc.to_parquet(P / "all_scores.parquet", index=False)
    res = check(d, sc, design)
    T = OUT / "tables"
    for k in ["estimates", "contrasts", "gaps"]:
        res[k].to_csv(T / f"all_centers_{k}.csv", index=False)
    report(d, res, [n for n, _, _ in LOGIT] + [n for n, _ in RIDGE], HUMAN, NO_MODEL)


def report(d: pd.DataFrame, res: dict, names: list, HUMAN: str, NO_MODEL: str) -> None:
    est, gaps = res["estimates"], res["gaps"]
    o = "cites_ns"

    def e(s, series, kind, group="All"):
        x = est[(est["sample"] == s) & (est.series == series) & (est.kind == kind) & (est.group == group)
                & (est.outcome == o)]
        return x.iloc[0] if len(x) else None

    def ci(b, se, nd=2, sc=1.0):
        return f"{sc * b:.{nd}f} [{sc * (b - 1.96 * se):.{nd}f}, {sc * (b + 1.96 * se):.{nd}f}]"
    rows = []
    for s in [*TCS, "All centers"]:
        x = d if s == "All centers" else d[d.tech_center == s]
        t = e(s, HUMAN, "marginal")
        hg = gaps[(gaps["sample"] == s) & (gaps.series == "Human decision makers")]
        r = dict(sample=s, applications=len(x), examiners=x.examiner_id.nunique(), female=x.female.mean(),
                 grant_rate=x.granted.mean(), mean_cites=x.cites_ns.mean(), marginal=t.estimate, marginal_se=t.se,
                 human_gap=hg.gap.iloc[0] if len(hg) else np.nan, human_gap_se=hg.se.iloc[0] if len(hg) else np.nan)
        for n in names:
            a, f = e(s, n, "error"), e(s, n, "error", "Female")
            gg = gaps[(gaps["sample"] == s) & (gaps.series == n)]
            r.update({f"err::{n}": a.estimate, f"err_se::{n}": a.se, f"errF::{n}": f.estimate, f"errF_se::{n}": f.se,
                      f"gap::{n}": gg.gap.iloc[0] if len(gg) else np.nan,
                      f"gap_se::{n}": gg.se.iloc[0] if len(gg) else np.nan,
                      f"ratio::{n}": gg.ratio.iloc[0] if len(gg) else np.nan})
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "tables/all_centers_summary.csv", index=False)
    L = ["# The check in all eight technology centers (embedding-based evaluators)", "",
         f"Applications filed {YEARS[0]}–{YEARS[1]} with the pilot's restrictions and text in HUPD: {len(d):,}. "
         "Outcome: citations within five years of grant, excluding self-citations. Gaps: female − male, decisions at "
         "the human grant rate, art unit × year cells; × human = the evaluator's gap over the examiners'.", "",
         "| Center | Applications | Female | Granted | Marginal citations (truth) | Examiners' gender gap (pp) | "
         "Error: grant-trained | Error: citation-trained | Gap × human: grant | citations | grant + names | "
         "citations + names |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append(f"| {r['sample']} | {int(r.applications):,} | {100 * r.female:.1f}% | {100 * r.grant_rate:.1f}% | "
                 f"{ci(r.marginal, r.marginal_se)} | {ci(r.human_gap, r.human_gap_se, 1, 100)} | "
                 f"{ci(r['err::Train on grant'], r['err_se::Train on grant'])} | "
                 f"{ci(r['err::Train on citations (granted only)'], r['err_se::Train on citations (granted only)'])} | "
                 f"{r['ratio::Train on grant']:.2f} | {r['ratio::Train on citations (granted only)']:.2f} | "
                 f"{r['ratio::Train on grant + names']:.2f} | {r['ratio::Train on citations + names']:.2f} |")
    L += ["", "Signed error for female first inventors' marginal applications:", "",
          "| Center | Grant-trained | Citation-trained | First action |", "|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append(f"| {r['sample']} | {ci(r['errF::Train on grant'], r['errF_se::Train on grant'])} | "
                 f"{ci(r['errF::Train on citations (granted only)'], r['errF_se::Train on citations (granted only)'])}"
                 f" | {ci(r['errF::Train on first action'], r['errF_se::Train on first action'])} |")
    (OUT / "all_centers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(S)


def figure(S: pd.DataFrame) -> None:
    J = load_module("jf20", "20_journal_figures.py")
    rows = [*TCS, "All centers"]
    S = S.set_index("sample").loc[rows]
    fig = J.figure(120, 66)
    y = np.arange(len(rows))[::-1].astype(float)
    ax = J.axes(fig, 26, 11, 90, 52)
    for col, key, dy, lab in [(J.INK, None, 0.22, "Examiners"), (J.DEC, "Train on grant", 0.0, "Trained on grants"),
                              (J.OUTC, "Train on citations (granted only)", -0.22, "Trained on citations")]:
        for yy, (_, r) in zip(y, S.iterrows()):
            b, se = (r.human_gap, r.human_gap_se) if key is None else (r[f"gap::{key}"], r[f"gap_se::{key}"])
            J.dot(ax, 100 * b, yy + dy, col, 100 * (b - 1.96 * se), 100 * (b + 1.96 * se), size=2.8)
        ax.plot([], [], marker="o", ls="none", color=col, ms=3, label=lab)
    ax.axvline(0, color=J.INK2, lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"TC {r}" if r != "All centers" else r for r in rows])
    ax.set_xlabel("Female − male, decisions at the examiners' grant rate (pp)")
    ax.legend(loc="upper left", fontsize=5.5)
    J.hgrid(ax)
    J.save(fig, "fig_all_centers")


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ["prepare", "extract", "embed", "analyze", "figure"]:
        ap.add_argument(f"--{a}", action="store_true")
    ap.add_argument("--worker", type=int)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.extract:
        extract()
    if a.worker is not None:
        worker(a.worker)
    elif a.embed:
        embed()
    if a.analyze:
        analyze()
    if a.figure:  # redraw from the saved summary
        figure(pd.read_csv(OUT / "tables/all_centers_summary.csv", dtype={"sample": str}))


if __name__ == "__main__":
    main()
