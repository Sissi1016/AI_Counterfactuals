"""C1 as in Kleinberg, Lakkaraju, Leskovec, Ludwig, and Mullainathan (2018, QJE), with patents; extensions; and a
counterfactual score for every rejected application.

Follows the paper and the authors' replication code (QJE Dataverse, doi:10.7910/DVN/VWDGHT: model_fitting.R,
contraction_plot.R, imputation_plot.R). Mapping: defendant → application, release → grant, detain → reject,
judge → examiner. Their outcome (failure to appear) is observed only for the released; ours only for granted
applications. Two outcomes:
- cites: citations within five years of grant excluding self-citations (the proposal's C1 outcome). It is a good
  outcome, so a rule rejects the applications with the lowest predicted citations first;
- uncited: a granted patent that no one else cites within five years, the literal analogue of their binary outcome
  (a rule rejects the highest predicted risk first).

The paper, step by step:
1. Partition: a lock box (filed in the last six months; all applications of examiners drawn at random from the 25
   with the largest caseloads until 10%; a random 10%; paper Figure I), then 20% test and the rest split evenly into
   training and imputation sets, stratified on the outcome (model_fitting.R).
2. Prediction: gradient-boosted trees on the granted applications of the training set; five-fold cross-validation
   over trees of 2, 3, or 4 splits, learning rates 0.025, 0.05, 0.1, and 100–2,500 trees, at least 20 observations per
   leaf; selected by accuracy (binary) or RMSE (counts), caret's defaults; refitted on the whole training set. The
   imputation model is fitted the same way on the imputation set. Inputs: what the examiner sees (64 principal
   components of the Qwen3-Embedding-0.6B embedding of title, abstract, and claims, fitted on the training set; entity
   size; number of inventors); not gender.
3. Figure II (test set): percentile bins of the prediction; the examiners' grant rate; observed against predicted
   outcomes of granted applications.
4. Table II (test set): boosting against a linear model with the same inputs (logit / OLS).
5. Figure III: permutation balance test within cells (index: the prediction).
6. Figure IV: a model of the most lenient quintile's grants applied to the grants of stricter quintiles.
7. Figure V, Table III (contraction_plot.R), test and imputation sets together, and the lock box: cells with at least
   five examiners; within each cell, examiners ordered by leniency and split into quintiles of equal caseload; for each
   stricter quintile, the change in the rejection rate and the relative change in the outcome per application against
   the most lenient quintile, averaged over cells weighted by cell size. A rule rejects, within each cell, additional
   grants of the most lenient quintile in order of its prediction. Random order as the benchmark.
8. Table IV (model_fitting.R, imputation_plot.R; bounds from the paper): test set, all applications reranked by the
   prediction; outcomes of applications the examiners rejected are the imputation model's predictions, and, beyond
   the most lenient quintile's grant rate within bins of the imputed value, a worse value set by α.
9. Table V (omitted payoffs): the citation algorithm's contraction judged on other outcomes (renewal at four years,
   top-decile citations, prior-art citations).
10. Tables VI and VII (disparate impact; their race, our inventor gender): the female share of the extra rejections
    under contraction, and the gender gap in grants under reranking, for the algorithm and for the algorithm constrained
    to the examiners' female share or to equal rates for women and men.
11. Figure VI (misranking): where in the distribution of predicted citations each stricter quintile's extra rejections
    fall, against where the algorithm's would.
12. Table VIII: the predicted examiner (a model of the grant decision) by contraction.
Rules compared with the examiners: the algorithm; the predicted examiner; Qwen3-8B fine-tuned on citations (30; cross-
fitted in examiner halves, so no application is scored by a model that saw its examiner); random.

Extensions (not in the paper):
A. Prior-art citations (06c): later applications cite an application's publication whether or not it was granted, so
   this outcome is observed for rejected applications too. It (i) tests the imputation of Table IV directly: reranking
   gains on prior-art citations computed as the paper does (rejected applications imputed from granted ones) against
   the gains computed from the observed values; (ii) measures α, the ratio of granted to rejected applications'
   outcome at the same prediction; (iii) trains an algorithm on all applications, without selective labels.
B. Least-selected labels: an algorithm trained only on the grants of the most lenient quintile.
C. Levels at the margin: the extra rejections of a stricter quintile are granted by the most lenient one, so Table
   III's quantities give their mean citations (citations lost per application over extra rejections), and the same
   quantities for the counterfactual scores of rejected applications give the scores' mean for the same applications.
   The same comparison for prior-art citations, which the extra rejections have both as granted (most lenient quintile)
   and as rejected (stricter quintile), measures the grant's own effect on prior-art citations.
D. Counterfactual scores: every rejected application gets an out-of-sample prediction of the citations it would have
   received if granted (the imputation model, or for imputation-set applications the training model), its percentile
   among granted applications' predictions, a version that also uses the application's prior-art citations, and
   Kleinberg et al.'s α-bounds. Written to output/tables/c1_counterfactual_scores.csv.

Differences from the paper, and why: (a) examiners are ordered by leave-one-out leniency (04) rather than by their
grant rate on the evaluated cases: NYC judges hear about 70 cases per cell, an examiner a handful, so ranking on the
same cases would sort on the decisions themselves; (b) cells are art unit × year × USPC class (the assignment cell of
12 and 19), theirs county × arraignment part × weekday × year; (c) both the examiner points and the rule's curve are
weighted by cell size (in contraction_plot.R the curve's weight, cell_sum$cell.cases, is not a column of cell_sum, so
the curve is an unweighted mean); (d) relative changes are ratios of cell-weighted means rather than cell-weighted
means of per-cell ratios, and a rule rejects fractional numbers of cases (linear interpolation) rather than
round(pp × n_q5 / 100): a quintile holds a handful of applications in a cell (about 180 defendants in theirs), so
per-cell ratios divide by outcome means near zero and rounding moves the rejection rate in steps of 10–20 points;
(e) scikit-learn's histogram gradient boosting instead of R's gbm (no 50% subsampling of rows per tree).
Intervals (not in the paper): bootstrap of art units, CFE_B draws (default 500).

Samples (CFE_SAMPLE): "all" (the main sample), the eight technology centers pooled (38's all_sample, 443,742
applications filed 2010–2014; one partition and one model, with the center and the filing year as inputs; outputs
with the suffix _all); "pilot", TC 3600 and TC 1600 separately, which adds the fine-tuned LLM as a rule (its scores
exist for the pilot only). Both also run contraction on the cells that pass the permutation test (Figure III).

Variant: CFE_LOSS=poisson fits the count models (citations, prior-art citations) with a Poisson loss (log link, so
predictions are positive), selected by the mean Poisson deviance over the same grid; outputs carry the suffix _poisson.

Modes without refitting: --figure redraws the figures from output/tables; --robust redoes the robustness contractions
and --post extension D (counterfactual scores and their validation) from the cached predictions (c1_preds*.parquet);
--linear refits only Table II's linear benchmark (with CFE_LOSS=poisson a Poisson GLM, as the boosted models' loss);
--fig4 adds Figure IV's residuals relative to the most lenient quintile (refitting only that quintile's model);
--figdata redoes Figure VI (rules within cells) and the top-coded levels at the margin from the cached predictions;
--recompute redoes the contraction (expected random rule), the robustness contractions and Figure II/IV's bins.

Runs on CPUs only (CFE_THREADS processes, default 4). Input: the record of 21,
data/processed/{applications_2004_2018,prior_art_cites,ft_preds,patex_application_data}.parquet, emb/s0.npy
Output: output/c1_kllm.md, output/tables/c1_kllm_*.csv, output/tables/c1_counterfactual_scores.csv,
        output/figures/journal/fig_c1_kllm{,_ext}.{pdf,png}
"""
import importlib.util
import os
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor  # noqa: E402
from sklearn.metrics import log_loss, mean_poisson_deviance, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
SAMPLE = os.environ.get("CFE_SAMPLE", "all")
# CFE_LOSS=poisson: the count models (citations, prior-art citations) with a Poisson loss and the mean Poisson deviance
# as the cross-validation criterion, so predictions are positive; all outputs then carry the suffix _poisson
LOSS = os.environ.get("CFE_LOSS", "poisson")
SFX = ("_all" if SAMPLE == "all" else "") + ("_poisson" if LOSS == "poisson" else "")
OUT = ROOT / "output"
SEED = 20261028
THREADS = int(os.environ.get("CFE_THREADS", "4"))
B = int(os.environ.get("CFE_B", "500"))
M = int(os.environ.get("CFE_M", "1000"))
Q, MIN_EXAMINERS, CELL, N_PC = 5, 5, "au_class", 64
LOCK_START, N_BUSY, LOCK_SHARE = "2014-07-01", 25, 0.10
DEPTHS, RATES = (2, 3, 4), (0.025, 0.05, 0.1)
TREES = np.arange(100, int(os.environ.get("CFE_MAX_TREES", "2500")) + 1, 100)
PP = np.round(np.arange(0, 40.01, 0.5), 2)   # extra rejections, % of the lenient quintile's caseload
ALPHAS = [1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
GLM_ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0)   # ridge penalties, Poisson GLM benchmark (CFE_LOSS=poisson)
TOPS = [0.01, 0.05, 0.10, 0.25]
N_BINS = 20
OUTCOMES = {"cites": "Citations within five years (excl. self-citations)",
            "uncited": "Granted patent not cited by others within five years"}
OTHER = {"renew": "Renewed at four years", "top10": "Top-decile citations",
         "pa_granted": "Prior-art citations of the granted applications"}
ALGO = {"cites": "Algorithm (trained on citations)", "uncited": "Algorithm (trained on no citations)"}
JUDGE, LLM_FT, RANDOM = "Predicted examiner (trained on grants)", "LLM fine-tuned on citations (Qwen3-8B)", "Random"
LENIENT = "Algorithm trained on the most lenient quintile's grants"
PA_ALL = "Algorithm trained on prior-art citations (all applications)"
PA_SEL = "Algorithm trained on prior-art citations (granted only)"
MATCH, EQUAL = "Algorithm, examiners' female share", "Algorithm, equal rates for women and men"
CF, CF_PA = "Counterfactual score (text)", "Counterfactual score (text + prior-art citations)"


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load() -> pd.DataFrame:
    d = load_module("m21", "21_citations.py").load()  # row i of d is row i of emb/s0.npy
    n = len(d)
    a = pd.read_parquet(P / "applications_2004_2018.parquet", columns=["application_number", "filing_date"])
    pa = pd.read_parquet(P / "prior_art_cites.parquet")
    ft = pd.read_parquet(P / "ft_preds.parquet", columns=["application_number", "cites"]).rename(
        columns={"cites": "ft_cites"})
    for t in [a, pa, ft]:
        d = d.merge(t, on="application_number", how="left")
    assert len(d) == n
    d["filing_date"] = pd.to_datetime(d.filing_date)
    g = d.granted == 1
    d["cites"] = d.cites_ns
    d["uncited"] = np.where(g, (d.cites_ns == 0).astype(float), np.nan)
    d["renew"] = np.where(g, d.renew_4y, np.nan)
    q90 = d[g].groupby(["tech_center", "filing_year"]).cites_ns.quantile(0.9).rename("q90")
    d = d.join(q90, on=["tech_center", "filing_year"])
    d["top10"] = np.where(g, (d.cites_ns >= d.q90).astype(float), np.nan)
    d["prior_art"] = np.where(d.published.fillna(False), d.prior_art_cites, np.nan)
    d["prior_art_ex"] = np.where(d.published.fillna(False), d.prior_art_cites_examiner, np.nan)
    d["pa_granted"] = np.where(g, d.prior_art, np.nan)
    d["female"] = d.female.fillna(0).astype(float)
    return d


def load_all() -> pd.DataFrame:
    """The eight centers (38), in the row order of emb/all_s0.npy, with the columns load() adds for the pilot."""
    import duckdb
    keys = pd.read_parquet(P / "all_text.parquet", columns=["application_number"])
    d = pd.read_parquet(P / "all_sample.parquet")
    d = keys.merge(d, on="application_number").sort_values("application_number").reset_index(drop=True)
    con = duckdb.connect()
    con.execute("SET threads = 16")
    mf = sorted((ROOT / "data/maintenance_fee/txt").glob("MaintFeeEvents_2*.txt"))[-1]
    extra = con.execute(f"""
    WITH m AS (
      SELECT string_split(trim(regexp_replace(line, '\\s+', ' ', 'g')), ' ') AS f
      FROM read_csv('{mf}', delim = '\x01', header = false, columns = {{'line': 'VARCHAR'}})),
    r AS (SELECT ltrim(f[1], '0') AS pn, max((f[7] IN ('M1551', 'M2551', 'M3551', 'F170'))::INT) AS renew_4y
          FROM m GROUP BY 1)
    SELECT a.application_number, a.filing_date, coalesce(r.renew_4y, 0) AS renew_4y
    FROM '{P}/applications_2004_2018.parquet' a LEFT JOIN r ON r.pn = ltrim(a.patent_number, '0')
    WHERE a.application_number IN (SELECT application_number FROM '{P}/all_sample.parquet')""").df()
    n = len(d)
    d = d.merge(extra, on="application_number", how="left")
    d = d.merge(pd.read_parquet(P / "prior_art_cites.parquet"), on="application_number", how="left")
    d = d.merge(pd.read_parquet(P / "leniency_window.parquet", columns=["application_number", "lenient_win"]),
                on="application_number", how="left")   # 04b
    if (P / "docketed_once.parquet").exists():   # 41
        d = d.merge(pd.read_parquet(P / "docketed_once.parquet", columns=["application_number", "docketed_once"]),
                    on="application_number", how="left")
    assert len(d) == n
    d["ft_cites"] = np.nan
    d["filing_date"] = pd.to_datetime(d.filing_date)
    g = d.granted == 1
    d["renew_4y"] = np.where(g, d.renew_4y, np.nan)
    d["cites"] = d.cites_ns
    d["uncited"] = np.where(g, (d.cites_ns == 0).astype(float), np.nan)
    d["renew"] = d.renew_4y
    q90 = d[g].groupby(["tech_center", "filing_year"]).cites_ns.quantile(0.9).rename("q90")
    d = d.join(q90, on=["tech_center", "filing_year"])
    d["top10"] = np.where(g, (d.cites_ns >= d.q90).astype(float), np.nan)
    d["prior_art"] = np.where(d.published.fillna(False), d.prior_art_cites, np.nan)
    d["prior_art_ex"] = np.where(d.published.fillna(False), d.prior_art_cites_examiner, np.nan)
    d["pa_granted"] = np.where(g, d.prior_art, np.nan)
    d["female"] = d.female.fillna(0).astype(float)
    return d


# ---------------------------------------------------------------- 1. partition

def partition(x: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    lock = (x.filing_date >= LOCK_START).to_numpy().copy()
    size = x.groupby("examiner_id").size().sort_index().sort_values(ascending=False, kind="mergesort")
    busy = size.head(N_BUSY).index.to_numpy().copy()
    rng.shuffle(busy)
    by_ex = np.zeros(n, bool)
    for e in busy:
        if by_ex.sum() >= LOCK_SHARE * n:
            break
        by_ex |= (x.examiner_id == e).to_numpy()
    lock |= by_ex | (rng.random(n) < LOCK_SHARE)
    part = np.full(n, "lockbox", dtype=object)
    strat = ((x.granted == 1) & (x.cites_ns == 0)).to_numpy()   # their createDataPartition(df$fta): 0 if detained
    for s in [False, True]:
        idx = np.where(~lock & (strat == s))[0]
        idx = idx[rng.permutation(len(idx))]
        n_test = int(np.ceil(0.2 * len(idx)))
        rest = idx[n_test:]
        part[idx[:n_test]] = "test"
        part[rest[: int(np.ceil(0.5 * len(rest)))]] = "train"
        part[rest[int(np.ceil(0.5 * len(rest))):]] = "impute"
    return part


# ---------------------------------------------------------------- 2. prediction functions

def _path(X, y, a, b, kind, depth, rate, seed):
    """Validation score at each number of trees for one fold and one (depth, rate)."""
    Est = HistGradientBoostingRegressor if kind == "gaussian" else HistGradientBoostingClassifier
    loss = dict(loss=LOSS) if kind == "gaussian" else {}
    m = Est(max_leaf_nodes=depth + 1, learning_rate=rate, min_samples_leaf=20, early_stopping=False,
            warm_start=True, random_state=seed, **loss)
    out = []
    with threadpool_limits(1):
        for n in TREES:
            m.set_params(max_iter=int(n))
            m.fit(X[a], y[a])
            if kind == "gaussian":
                if LOSS == "poisson":
                    out.append((-mean_poisson_deviance(y[b], np.maximum(m.predict(X[b]), 1e-12)), 0.0))
                else:
                    out.append((-np.sqrt(np.mean((m.predict(X[b]) - y[b]) ** 2)), 0.0))
            else:
                p = m.predict_proba(X[b])[:, 1]
                out.append((np.mean((p > 0.5) == y[b]), -log_loss(y[b], np.clip(p, 1e-9, 1 - 1e-9))))
    return out


def fit_gbm(X: np.ndarray, y: np.ndarray, kind: str, seed: int, prm: dict | None = None):
    """caret::train(method = 'gbm') as in model_fitting.R: 5-fold CV over the grid, then a fit on all of X."""
    cv = np.nan
    if prm is None:
        strat = y if kind == "binary" else (y > 0)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat))
        grid = [(dp, r) for dp in DEPTHS for r in RATES]
        res = Parallel(n_jobs=THREADS)(delayed(_path)(X, y, a, b, kind, dp, r, seed)
                                       for dp, r in grid for a, b in folds)
        sc = np.array(res).reshape(len(grid), len(folds), len(TREES), 2).mean(1)   # grid × trees × (main, tie)
        best = max(((i, j) for i in range(len(grid)) for j in range(len(TREES))), key=lambda t: tuple(sc[t]))
        prm = dict(depth=grid[best[0]][0], rate=grid[best[0]][1], trees=int(TREES[best[1]]))
        cv = float(sc[best][0])
    Est = HistGradientBoostingRegressor if kind == "gaussian" else HistGradientBoostingClassifier
    loss = dict(loss=LOSS) if kind == "gaussian" else {}
    with threadpool_limits(THREADS):
        m = Est(max_leaf_nodes=prm["depth"] + 1, learning_rate=prm["rate"], max_iter=prm["trees"],
                min_samples_leaf=20, early_stopping=False, random_state=seed, **loss).fit(X, y)
    if kind == "gaussian":
        return (lambda Z: m.predict(Z)), prm, cv
    return (lambda Z: m.predict_proba(Z)[:, 1]), prm, cv


def fit_linear(X: np.ndarray, y: np.ndarray, kind: str):
    """Table II's linear benchmark with the boosted model's objective: OLS, or with CFE_LOSS=poisson a Poisson GLM (log
    link) whose ridge penalty is chosen by 5-fold cross-validation on the mean Poisson deviance; logit for 'uncited'."""
    sc = StandardScaler().fit(X)
    with threadpool_limits(THREADS):
        if kind == "gaussian" and LOSS == "poisson":
            Z = sc.transform(X)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, y > 0))
            dev = [np.mean([mean_poisson_deviance(y[b], np.maximum(PoissonRegressor(alpha=a_, max_iter=1000).fit(
                Z[a], y[a]).predict(Z[b]), 1e-12)) for a, b in folds]) for a_ in GLM_ALPHAS]
            alpha = GLM_ALPHAS[int(np.argmin(dev))]
            m = PoissonRegressor(alpha=alpha, max_iter=1000).fit(Z, y)
            f = lambda W: m.predict(sc.transform(W))  # noqa: E731
            f.alpha, f.cv = alpha, -float(np.min(dev))
            return f
        if kind == "gaussian":
            m = LinearRegression().fit(sc.transform(X), y)
            return lambda Z: m.predict(sc.transform(Z))
        m = LogisticRegression(C=1e6, max_iter=5000).fit(sc.transform(X), y)
        return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


# ---------------------------------------------------------------- 7. contraction (contraction_plot.R)

def quintiles(e: pd.DataFrame, min_ex: int = MIN_EXAMINERS) -> pd.Series:
    """Quintile of each case's examiner within its cell (5 = most lenient), equal caseload shares, holes filled as
    in contraction_plot.R; cells with at least min_ex examiners among the evaluated cases."""
    e = e[e.lenient_grant.notna()]
    ex = e.groupby([CELL, "examiner_id"]).agg(z=("lenient_grant", "first"), cases=("granted", "size")).reset_index()
    ex["n_ex"] = ex.groupby(CELL).z.transform("size")
    ex = ex[ex.n_ex >= min_ex].sort_values([CELL, "z", "examiner_id"], kind="mergesort")
    parts = []
    for _, t in ex.groupby(CELL, sort=False):
        cum = t.cases.cumsum().to_numpy() / t.cases.sum()
        q = np.clip(np.ceil(cum * 5 - 1e-9).astype(int), 1, 5)   # cut(cum, seq(0, 1, .2), include.lowest = T)
        # contraction_plot.R fills an empty quintile m with the first examiner of quintile m + 1 (an empty 5 with the
        # last examiner). In small cells that can empty another quintile, so the same rule is applied in one pass:
        # quintiles rise by at most one from examiner to examiner and reach 5 at the last (no-op without holes)
        n = len(q)
        q[0] = 1
        for i in range(1, n):
            q[i] = max(min(max(q[i], q[i - 1]), q[i - 1] + 1), 5 - (n - 1 - i))
        parts.append(t.assign(q=q))
    ex = pd.concat(parts)
    return e[[CELL, "examiner_id"]].merge(ex[[CELL, "examiner_id", "q"]], how="left").set_index(e.index).q


def boot_mult(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(B + 1) × n multipliers: row 0 the estimate, rows 1.. bootstrap draws of the groups (art units)."""
    u, inv = np.unique(groups, return_inverse=True)
    mult = np.ones((B + 1, len(groups)))
    for b in range(1, B + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
    return mult


def pc(v, lo):
    return float(np.nanpercentile(v, 2.5 if lo else 97.5))


def contraction(e: pd.DataFrame, outcome: str, bad: dict, rng: np.random.Generator, gender: bool = False) -> tuple:
    """e: evaluated cases with quintile q. bad: {rule: score, rejected first = largest}; the first rule is the
    algorithm that the gender-constrained rules (MATCH, EQUAL) reorder. Per-cell quantities as in contraction_plot.R,
    then cell-size-weighted means for the estimate and for bootstrap draws of art units. Also the female share of each
    rule's extra rejections (Table VI). The random rule is its expected value (rejecting k of a cell's most lenient
    quintile's grants at random removes k times their mean outcome), so its interval reflects only the resampling of
    art units; rnd (one draw) only breaks ties among equal scores of the other rules."""
    cells = list(e.groupby(CELL, sort=True))
    n_cell = np.array([len(c) for _, c in cells], float)
    au = np.array([c.art_unit.iloc[0] for _, c in cells])
    algo = list(bad)[0]
    rules = list(bad) + [RANDOM] + ([MATCH, EQUAL] if gender else [])
    nc = len(cells)
    dd, do, dfm = np.full((nc, 4), np.nan), np.full((nc, 4), np.nan), np.full((nc, 4), np.nan)   # quintiles 4..1
    base = np.zeros(nc)                                                                       # quintile 5 per case
    red = {r: np.full((nc, len(PP)), np.nan) for r in rules}   # removed outcome per q5 case
    redf = {r: np.full((nc, len(PP)), np.nan) for r in rules}  # removed female grants per q5 case
    rnd = rng.random(len(e))
    pos = pd.Series(np.arange(len(e)), index=e.index)
    for i, (_, c) in enumerate(cells):
        y = np.nan_to_num(c[outcome].to_numpy(float))
        g = c.granted.to_numpy(float)
        f = c.female.to_numpy(float)
        q = c.q.to_numpy(int)
        rel = np.array([g[q == k].mean() for k in range(1, 6)])
        out = np.array([y[q == k].mean() for k in range(1, 6)])
        fo = np.array([(f * g)[q == k].mean() for k in range(1, 6)])
        dd[i] = (1 - rel[[3, 2, 1, 0]]) - (1 - rel[4])
        do[i] = out[[3, 2, 1, 0]] - out[4]
        dfm[i] = fo[4] - fo[[3, 2, 1, 0]]
        base[i] = out[4]
    target = float((n_cell @ dfm).sum() / (n_cell @ dd).sum())   # examiners' female share of extra rejections
    for i, (_, c) in enumerate(cells):
        y = np.nan_to_num(c[outcome].to_numpy(float))
        g = c.granted.to_numpy(float)
        f = c.female.to_numpy(float)
        q = c.q.to_numpy(int)
        five = q == 5
        k = five.sum() / 100 * PP                       # cases to reject, fractional
        gi = np.where(five & (g == 1))[0]
        p = pos[c.index].to_numpy()
        for r in rules:
            if r in (MATCH, EQUAL):
                s = bad[algo][p]
                lists = []
                for sel in [f[gi] == 1, f[gi] != 1]:
                    ii = gi[sel]
                    ii = ii[np.lexsort((rnd[p][ii], -s[ii]))]
                    lists.append((len(ii), np.concatenate([[0], np.cumsum(y[ii])])))
                share = target if r == MATCH else lists[0][0] / max(len(gi), 1)
                kf, km = k * share, k * (1 - share)
                ok = (kf <= lists[0][0] + 1e-9) & (km <= lists[1][0] + 1e-9)
                v = (np.interp(kf, np.arange(lists[0][0] + 1), lists[0][1]) +
                     np.interp(km, np.arange(lists[1][0] + 1), lists[1][1]))
                red[r][i] = np.where(ok, v, np.nan) / five.sum()
                redf[r][i] = np.where(ok, kf, np.nan) / five.sum()
                continue
            ok = k <= len(gi) + 1e-9
            if r == RANDOM:   # its expected value: each rejection removes the mean outcome of the quintile's grants
                red[r][i] = np.where(ok, k * (y[gi].mean() if len(gi) else 0.0), np.nan) / five.sum()
                redf[r][i] = np.where(ok, k * (f[gi].mean() if len(gi) else 0.0), np.nan) / five.sum()
                continue
            ii = gi[np.lexsort((rnd[p][gi], -bad[r][p][gi]))]
            ar = np.arange(len(ii) + 1)
            red[r][i] = np.where(ok, np.interp(k, ar, np.concatenate([[0], np.cumsum(y[ii])])), np.nan) / five.sum()
            redf[r][i] = np.where(ok, np.interp(k, ar, np.concatenate([[0], np.cumsum(f[ii])])), np.nan) / five.sum()
    mult = np.ones((B + 1, nc))
    u, inv = np.unique(au, return_inverse=True)
    for b in range(1, B + 1):
        mult[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]

    def wmean(v, w):   # weighted mean over cells, ignoring missing (na.rm = T)
        ok = ~np.isnan(v)
        return (np.where(ok, v, 0) * w).sum(0) / np.maximum((ok * w).sum(0), 1e-12)
    W = mult * n_cell
    X = np.array([wmean(dd, w[:, None]) for w in W])          # draws × 4: extra rejection rate
    Y = np.array([wmean(do, w[:, None]) / (w @ base) * w.sum() for w in W])   # draws × 4: relative change
    F = np.array([(w @ dfm) / (w @ dd) for w in W])           # draws × 4: female share of the extra rejections
    # the rule's relative change: removed outcome over quintile 5's outcome, both over the cells where the rejections
    # are possible (a cell runs out of grants to reject); its female share: removed female grants over removed grants
    curves = {r: np.array([-(np.nan_to_num(red[r]) * w[:, None]).sum(0) /
                           np.maximum(((~np.isnan(red[r])) * (w * base)[:, None]).sum(0), 1e-12) for w in W])
              for r in rules}
    fcurves = {r: np.array([(np.nan_to_num(redf[r]) * w[:, None]).sum(0) /
                            np.maximum(((~np.isnan(redf[r])) * w[:, None]).sum(0) * PP / 100, 1e-12) for w in W])
               for r in rules}
    qrows = [dict(outcome=outcome, quintile=j + 2, extra_rejections=X[0, j], change=Y[0, j],
                  extra_rejections_lo=pc(X[1:, j], 1), extra_rejections_hi=pc(X[1:, j], 0),
                  change_lo=pc(Y[1:, j], 1), change_hi=pc(Y[1:, j], 0),
                  female_share=F[0, j], female_share_lo=pc(F[1:, j], 1), female_share_hi=pc(F[1:, j], 0))
             for j in range(4)]
    qrows.append(dict(outcome=outcome, quintile="2–5 (mean)", extra_rejections=X[0].mean(), change=Y[0].mean(),
                      extra_rejections_lo=pc(X[1:].mean(1), 1), extra_rejections_hi=pc(X[1:].mean(1), 0),
                      change_lo=pc(Y[1:].mean(1), 1), change_hi=pc(Y[1:].mean(1), 0), female_share=F[0].mean(),
                      female_share_lo=pc(F[1:].mean(1), 1), female_share_hi=pc(F[1:].mean(1), 0)))
    erows = []
    for r in rules:
        at, need, fat = (np.full((B + 1, 4), np.nan) for _ in range(3))
        for b in range(B + 1):
            cv, fc = curves[r][b], fcurves[r][b]
            ok = ~np.isnan(cv)
            xs, ys = PP[ok] / 100, cv[ok]
            for j in range(4):
                if X[b, j] <= xs[-1]:
                    at[b, j] = np.interp(X[b, j], xs, ys)
                    fat[b, j] = np.interp(X[b, j], PP[1:] / 100, fc[1:])
                hit = np.where(ys <= Y[b, j])[0]   # first point where the rule's change reaches the examiners'
                if len(hit) and hit[0] > 0:
                    i1 = hit[0]
                    need[b, j] = xs[i1 - 1] + (xs[i1] - xs[i1 - 1]) * (ys[i1 - 1] - Y[b, j]) / (ys[i1 - 1] - ys[i1])
                elif len(hit):
                    need[b, j] = 0.0
        for j in list(range(4)) + ["mean"]:
            sl = (slice(None), slice(None)) if j == "mean" else (slice(None), j)

            def m_(a):
                v = a[sl]
                return v.mean(1) if j == "mean" else v
            a0, d0 = m_(at), m_(at) - m_(Y)
            erows.append(dict(outcome=outcome, rule=r, quintile="2–5 (mean)" if j == "mean" else j + 2,
                              change_at_examiner_rejections=a0[0], lo=pc(a0[1:], 1), hi=pc(a0[1:], 0),
                              diff=d0[0], diff_lo=pc(d0[1:], 1), diff_hi=pc(d0[1:], 0),
                              rejections_for_examiner_change=m_(need)[0], need_ratio=(m_(need) / m_(X))[0],
                              female_share=m_(fat)[0], female_share_lo=pc(m_(fat)[1:], 1),
                              female_share_hi=pc(m_(fat)[1:], 0)))
    return qrows, erows, dict(curves={r: curves[r][0] for r in rules}, X=X[0], Y=Y[0])


# ---------------------------------------------------------------- 8. reranking (model_fitting.R; bounds: paper IV.C)

def reranking(t: pd.DataFrame, outcome: str, keep: dict, imp: np.ndarray, q5: np.ndarray, rng: np.random.Generator,
              actual: np.ndarray | None = None) -> list:
    """t: test set; keep: {rule: score, granted first = largest}; imp: imputation model; q5: most lenient quintile.
    actual: the outcome of every application where it is observed for the rejected too (prior-art citations); adds
    the row 'observed', which uses it in place of the imputation."""
    g = t.granted.to_numpy(float)
    y = np.nan_to_num(t[outcome].to_numpy(float))
    bins = pd.qcut(pd.Series(imp).rank(method="first"), N_BINS, labels=False).to_numpy()
    r_all = pd.Series(g).groupby(bins).mean()
    r_len = pd.Series(g[q5]).groupby(bins[q5]).mean().reindex(r_all.index).fillna(r_all)
    pi = np.clip((r_len - r_all) / (1 - r_all).clip(lower=1e-9), 0, 1).to_numpy()[bins]
    mult = boot_mult(t.art_unit.to_numpy(), rng)
    tb = rng.random(len(t))
    names = ["outcome_at_examiner_rate", "outcome_change", "rejection_rate_at_examiner_outcome", "rejection_change",
             "examiner_outcome", "examiner_rejection_rate"]
    variants = []
    for alpha in ALPHAS:
        if outcome != "uncited":
            worse = np.zeros_like(imp) if np.isinf(alpha) else imp / alpha
        else:
            worse = np.ones_like(imp) if np.isinf(alpha) else np.minimum(1, alpha * imp)
        variants.append((alpha, np.where(g == 1, y, pi * imp + (1 - pi) * worse)))
    if actual is not None:
        variants.append(("observed", np.nan_to_num(actual)))
    rows = []
    for alpha, yy in variants:
        for name, s in keep.items():
            order = np.lexsort((tb, -s))
            res = []
            for m_ in mult:
                N = m_.sum()
                Gr, Yj = (m_ * g).sum() / N, (m_ * g * yy).sum() / N
                cw = np.cumsum(m_[order]) / N
                cy = np.cumsum(m_[order] * yy[order]) / N
                Ya = cy[min(np.searchsorted(cw, Gr), len(cy) - 1)]
                ga = cw[min(np.searchsorted(cy, Yj), len(cw) - 1)]   # min(which(cum >= judge total))
                res.append((Ya, (Ya - Yj) / Yj, 1 - ga, ((1 - ga) - (1 - Gr)) / (1 - Gr), Yj, 1 - Gr))
            res = np.array(res)
            r = dict(outcome=outcome, rule=name, alpha=alpha)
            for j, k in enumerate(names):
                r[k] = res[0, j]
                r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(res[1:, j], 2.5), np.nanpercentile(res[1:, j], 97.5)
            rows.append(r)
    return rows


def rerank_gender(t: pd.DataFrame, score: np.ndarray, imp: np.ndarray, rng: np.random.Generator) -> list:
    """Table VII: all test applications reranked at the examiners' grant rate (α = 1: rejected applications get the
    imputation model's prediction). Rules: the algorithm; the algorithm with the examiners' female share of grants;
    the algorithm with the same grant rate for women and men."""
    g = t.granted.to_numpy(float)
    y = np.where(g == 1, np.nan_to_num(t.cites.to_numpy(float)), imp)
    f = t.female.to_numpy(float) == 1
    mult = boot_mult(t.art_unit.to_numpy(), rng)
    tb = rng.random(len(t))
    order_all = np.lexsort((tb, -score))

    def top(m_, idx, n_grant):
        """Indicator of the n_grant (weighted) highest-scored cases among idx."""
        o = order_all[np.isin(order_all, idx)]
        a = np.zeros(len(g))
        a[o[np.cumsum(m_[o]) <= n_grant + 1e-9]] = 1
        return a
    fi, mi = np.where(f)[0], np.where(~f)[0]
    rows = {k: [] for k in ["Examiners", ALGO["cites"], MATCH, EQUAL]}
    for m_ in mult:
        N, Nf, Nm = m_.sum(), m_[fi].sum(), m_[mi].sum()
        G = (m_ * g).sum()
        sF = (m_ * g * f).sum() / G
        Yj = (m_ * g * y).sum() / N
        sel = {"Examiners": g, ALGO["cites"]: top(m_, np.arange(len(g)), G),
               MATCH: top(m_, fi, G * sF) + top(m_, mi, G * (1 - sF)),
               EQUAL: top(m_, fi, G / N * Nf) + top(m_, mi, G / N * Nm)}
        for k, a in sel.items():
            Ya = (m_ * a * y).sum() / N
            gap = (m_ * a * f).sum() / Nf - (m_ * a * ~f).sum() / Nm
            fr = (m_ * (1 - a) * f).sum() / (m_ * (1 - a)).sum()
            rows[k].append(((Ya - Yj) / Yj, gap, fr, (m_ * a).sum() / N))
    out = []
    for k, v in rows.items():
        v = np.array(v)
        r = dict(rule=k)
        for j, nm in enumerate(["outcome_change", "gap", "female_share_rejected", "grant_rate"]):
            r[nm], r[f"{nm}_lo"], r[f"{nm}_hi"] = v[0, j], pc(v[1:, j], 1), pc(v[1:, j], 0)
        out.append(r)
    return out


# ---------------------------------------------------------------- extensions

def margin_levels(e: pd.DataFrame, scores: dict, rng: np.random.Generator) -> list:
    """Extension C. Per cell and stricter quintile k, against the most lenient (5): dg = g5 − gk; the extra rejections'
    citations ΔY = Y5 − Yk (citations per application, rejected = 0); for a counterfactual score s of rejected
    applications, Δs = mean(s·(1 − g))_k − mean(s·(1 − g))_5; likewise for prior-art citations, observed for the
    rejected. Ratios of cell-weighted sums give the means over the extra rejections."""
    cells = list(e.groupby(CELL, sort=True))
    n_cell = np.array([len(c) for _, c in cells], float)
    names = ["observed citations"] + list(scores) + ["observed prior-art citations",
                                                      "observed prior-art citations, as granted"]
    D = np.zeros((len(cells), 4))
    V = {k: np.zeros((len(cells), 4)) for k in names}
    pos = pd.Series(np.arange(len(e)), index=e.index)
    for i, (_, c) in enumerate(cells):
        g = c.granted.to_numpy(float)
        q = c.q.to_numpy(int)
        y = np.nan_to_num(c.cites.to_numpy(float))
        pa = np.nan_to_num(c.prior_art.to_numpy(float))
        p = pos[c.index].to_numpy()
        a = q == 5
        for j, k in enumerate([4, 3, 2, 1]):
            b = q == k
            D[i, j] = g[a].mean() - g[b].mean()
            V[names[0]][i, j] = (y * g)[a].mean() - (y * g)[b].mean()
            for s, v in scores.items():
                v = v[p]
                V[s][i, j] = (v * (1 - g))[b].mean() - (v * (1 - g))[a].mean()
            V[names[-2]][i, j] = (pa * (1 - g))[b].mean() - (pa * (1 - g))[a].mean()   # as rejected (quintile k)
            V[names[-1]][i, j] = (pa * g)[a].mean() - (pa * g)[b].mean()               # as granted (quintile 5)
    mult = boot_mult(np.array([c.art_unit.iloc[0] for _, c in cells]), rng)
    W = mult * n_cell
    rows = []
    for k in names:
        est = (W @ V[k]) / (W @ D)                          # draws × 4
        pool = (W @ V[k].sum(1)) / (W @ D.sum(1))           # draws
        for j in range(4):
            rows.append(dict(series=k, quintile=j + 2, value=est[0, j], lo=pc(est[1:, j], 1), hi=pc(est[1:, j], 0)))
        rows.append(dict(series=k, quintile="2–5 (pooled)", value=pool[0], lo=pc(pool[1:], 1), hi=pc(pool[1:], 0)))
        if k in scores:
            dif = (W @ (V[k] - V[names[0]]).sum(1)) / (W @ D.sum(1))
            rows.append(dict(series=f"{k} − observed citations", quintile="2–5 (pooled)", value=dif[0],
                             lo=pc(dif[1:], 1), hi=pc(dif[1:], 0)))
    # the grant's own effect on prior-art citations: the same applications granted (quintile 5) and rejected (k)
    dif = (W @ (V[names[-1]] - V[names[-2]]).sum(1)) / (W @ D.sum(1))
    rat = (W @ V[names[-1]].sum(1)) / (W @ V[names[-2]].sum(1))
    rows.append(dict(series="grant effect on prior-art citations", quintile="2–5 (pooled)", value=dif[0],
                     lo=pc(dif[1:], 1), hi=pc(dif[1:], 0)))
    rows.append(dict(series="grant effect on prior-art citations (ratio)", quintile="2–5 (pooled)", value=rat[0],
                     lo=pc(rat[1:], 1), hi=pc(rat[1:], 0)))
    return rows


def matched_rule(e: pd.DataFrame, score: np.ndarray, k: int, tb: np.ndarray) -> np.ndarray:
    """A rule working within each cell on the most lenient quintile's caseload: k = n_5 × (rate of quintile k's
    rejections − quintile 5's) cases, so that in every cell it makes exactly quintile k's extra rejections; the most
    lenient quintile's grants with the lowest score (k > 0) or, where quintile k rejects less, its rejections with the
    highest score granted (k < 0); fractional cases as in contraction() (the last one counts in part). Returns, per
    case of quintile 5, the share of it the rule rejects beyond that quintile's decisions (negative: grants)."""
    q, g = e.q.to_numpy(int), e.granted.to_numpy(float)
    f = np.zeros(len(e))
    for idx in e.groupby(CELL, sort=True).indices.values():
        five, kk = idx[q[idx] == 5], idx[q[idx] == k]
        n_ = (g[five].mean() - g[kk].mean()) * len(five)
        if n_ > 0:
            cand = five[g[five] == 1]
            cand = cand[np.lexsort((tb[cand], score[cand]))]       # lowest score first
        elif n_ < 0:
            cand = five[g[five] == 0]
            cand = cand[np.lexsort((tb[cand], -score[cand]))]      # highest score first
        else:
            cand = five[:0]
        f[cand] = np.sign(n_) * np.clip(abs(n_) - np.arange(len(cand)), 0, 1)
    return f


def misranking(e: pd.DataFrame, score: np.ndarray, judge: np.ndarray | None = None) -> pd.DataFrame:
    """Figure VI: share of each stricter quintile's extra rejections from each bin of predicted citations, against
    where a rule's would fall: the algorithm (lowest predicted citations first) and the predicted examiner (lowest
    predicted grant probability first), each working within every cell to match that quintile's extra rejections
    there (matched_rule), on the same bins of predicted citations. Cases are weighted n_c / n_cq so that every quintile
    has the same mix of cells. algorithm_pooled: the earlier version, which let the algorithm take the
    lowest-predicted applications bin by bin across the pooled sample."""
    b = pd.qcut(pd.Series(score).rank(method="first"), N_BINS, labels=False).to_numpy()
    n_c = e.groupby(CELL)[CELL].transform("size").to_numpy(float)
    n_cq = e.groupby([CELL, "q"])[CELL].transform("size").to_numpy(float)
    w = n_c / n_cq
    t = pd.DataFrame(dict(q=e.q.to_numpy(int), bin=b, w=w, g=e.granted.to_numpy(float)))
    t["wg"] = t.w * t.g
    s = t.groupby(["q", "bin"])[["w", "wg"]].sum()
    tot = t.groupby("q").w.sum()
    grants = (s.wg / tot.reindex(s.index.get_level_values(0)).to_numpy()).unstack(fill_value=0)   # q × bin
    share = (s.w / tot.reindex(s.index.get_level_values(0)).to_numpy()).unstack(fill_value=0)
    tb = np.random.default_rng(SEED + 52).random(len(e))   # ties among equal scores (own stream)
    five = t.q.to_numpy() == 5
    rows = []
    for k in [4, 3, 2, 1]:
        ex = (grants.loc[5] - grants.loc[k]).to_numpy()
        total = ex.sum()
        avail = grants.loc[5].to_numpy()
        alg = np.zeros(N_BINS)
        left = total
        for i in range(N_BINS):
            alg[i] = min(avail[i], max(left, 0))
            left -= alg[i]
        rules = {"algorithm": score} | ({} if judge is None else {"judge": judge})
        mass = {}
        for r, sc in rules.items():
            f = matched_rule(e, np.asarray(sc, float), k, tb)
            m = pd.Series((w * f)[five]).groupby(b[five]).sum().reindex(range(N_BINS), fill_value=0).to_numpy()
            mass[r] = m / m.sum()
        for i in range(N_BINS):
            rows.append(dict(quintile=6 - k, bin=i + 1, examiners=ex[i] / total, algorithm=mass["algorithm"][i],
                             **({"judge": mass["judge"][i]} if "judge" in mass else {}),
                             algorithm_pooled=alg[i] / total,
                             grant_rate_lenient=grants.loc[5].iloc[i] / share.loc[5].iloc[i],
                             grant_rate_quintile=grants.loc[k].iloc[i] / share.loc[k].iloc[i]))
    return pd.DataFrame(rows)


def top_codes(x: pd.DataFrame) -> pd.Series:
    """The 99th percentile of granted applications' citations within center × filing year (as 43's top-coding)."""
    g = x.granted == 1
    q99 = x[g].groupby(["tech_center", "filing_year"]).cites.quantile(0.99).rename("q99")
    return x[["tech_center", "filing_year"]].join(q99, on=["tech_center", "filing_year"]).q99


def levels_topcoded(ev: pd.DataFrame, caps: pd.Series, rng: np.random.Generator) -> list:
    """Extension C with citations top-coded at the 99th percentile of granted applications in the same center ×
    filing year, and the counterfactual score capped at the same values (like with like): per-quintile means over the
    extra rejections are ratios of cell differences and, with raw citations, can have intervals below zero."""
    z = ev.assign(cites=np.minimum(ev.cites, caps.loc[ev.index]))
    cf = np.minimum(z.cf.to_numpy(), caps.loc[ev.index].to_numpy())
    rows = margin_levels(z, {CF: cf}, rng)
    g = (ev.granted == 1).to_numpy()
    for lab, v in [("mean citations of granted applications, top-coded", z.cites), ("mean citations of granted "
                   "applications, raw", ev.cites)]:
        rows.append(dict(series=lab, quintile="all", value=float(v[g].mean()), lo=np.nan, hi=np.nan))
    return rows


def alpha_table(t: pd.DataFrame, pred: np.ndarray, label: str) -> list:
    """Extension A(ii): within filing year × decile of a prediction, prior-art citations of granted over rejected
    applications, averaged with the rejected applications' weights (α of Table IV for this outcome)."""
    z = pd.DataFrame(dict(y=np.nan_to_num(t.prior_art.to_numpy(float)), g=t.granted.to_numpy(int),
                          yr=t.filing_year.to_numpy(), p=pred))
    z["dec"] = z.groupby("yr").p.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
    m = z.groupby(["yr", "dec", "g"]).y.agg(["mean", "size"]).unstack("g")
    m = m.dropna()
    num = (m[("size", 0)] * m[("mean", 1)]).sum()
    den = (m[("size", 0)] * m[("mean", 0)]).sum()
    rows = [dict(prediction=label, decile="all", granted=num / m[("size", 0)].sum(), rejected=den / m[("size", 0)].sum(),
                 alpha=num / den)]
    by = z.groupby(["dec", "g"]).y.mean().unstack("g")
    for dcl, r in by.iterrows():
        rows.append(dict(prediction=label, decile=int(dcl) + 1, granted=r[1], rejected=r[0], alpha=r[1] / r[0]))
    return rows


def percentile_among_granted(x: pd.DataFrame, col: str) -> np.ndarray:
    """Percentile of each application's x[col] among the values of the granted applications of the same technology
    center and filing year (share of those granted at or below it, in %)."""
    cell = (x.tech_center.astype(str) + "_" + x.filing_year.astype(str)).to_numpy()
    v, g = x[col].to_numpy(), (x.granted == 1).to_numpy()
    pct = np.full(len(x), np.nan)
    for c in np.unique(cell):
        s = cell == c
        ref = np.sort(v[s & g])
        pct[s] = 100 * np.searchsorted(ref, v[s], side="right") / len(ref)
    return pct


def counterfactual_outputs(x: pd.DataFrame, tc: str) -> dict:
    """Extension D: the counterfactual scores of rejected applications and their validation. x: all applications of
    the sample with the partition (part) and the out-of-sample scores cf (text) and cf_pa (text + prior-art
    citations). Depends only on the predictions and the data (no random draws), so --post recomputes it from the cached
    predictions. Each score's percentile is among the granted applications' values of the same score."""
    g = (x.granted == 1).to_numpy()
    x = x.assign(cf_pct=percentile_among_granted(x, "cf"), cf_pa_pct=percentile_among_granted(x, "cf_pa"))
    gx, rx = x[g & (x.part != "lockbox")], x[~g & (x.part != "lockbox")]
    val = []
    for name, col in [(CF, "cf"), (CF_PA, "cf_pa")]:
        val.append(dict(tc=tc, score=name, corr_cites_granted=np.corrcoef(gx[col], gx.cites)[0, 1],
                        spearman_cites_granted=spearmanr(gx[col], gx.cites)[0],
                        spearman_prior_art_granted=spearmanr(gx[col], gx.prior_art.fillna(0))[0],
                        spearman_prior_art_rejected=spearmanr(rx[col], rx.prior_art.fillna(0))[0],
                        mean_granted=gx[col].mean(), mean_rejected=rx[col].mean(), mean_cites_granted=gx.cites.mean(),
                        share_rejected_above_granted_median=float(np.mean(rx[f"{col}_pct"] > 50)),
                        share_rejected_top_quarter=float(np.mean(rx[f"{col}_pct"] > 75)),
                        mean_prior_art_granted=gx.prior_art.mean(), mean_prior_art_rejected=rx.prior_art.mean()))
    out = dict(cf_valid=pd.DataFrame(val))
    # deciles of the score within center × filing year, so that every decile has the same mix of centers and years
    # (citation levels and their truncation differ between centers and years)
    early = x[(x.filing_year <= 2012) & (x.part != "lockbox")].copy()
    early["dec"] = early.groupby(["tech_center", "filing_year"]).cf.transform(
        lambda v: pd.qcut(v.rank(method="first"), 10, labels=False)) + 1
    out["cf_deciles"] = early.groupby(["dec", "granted"]).agg(cf=("cf", "mean"), prior_art=("prior_art", "mean"),
                                                             n=("cf", "size")).reset_index().assign(tc=tc)
    # Kleinberg et al.'s bound for each rejected application: the imputed value up to the most lenient quintile's grant
    # rate in its bin of the imputed value, and the value divided by α beyond it; reported as cf / α for α = 2
    rej = x[~g]
    out["scores"] = pd.DataFrame(dict(application_number=rej.application_number, tech_center=rej.tech_center,
                                      filing_year=rej.filing_year, part=rej.part, cf_cites=rej.cf,
                                      cf_cites_alpha2=rej.cf / 2, cf_percentile_among_granted=rej.cf_pct,
                                      cf_cites_with_prior_art=rej.cf_pa,
                                      cf_with_prior_art_percentile_among_granted=rej.cf_pa_pct,
                                      prior_art_cites=rej.prior_art, prior_art_cites_examiner=rej.prior_art_ex))
    # Figure A4c: densities of the score for rejected and granted applications outside the lock box, on 39 bins from 0
    # to the 99th percentile of their scores (the x-range is cut there; densities integrate to one over it)
    nl = x[x.part != "lockbox"]
    edges = np.linspace(0, float(np.nanpercentile(nl.cf, 99)), 40)
    dens = {gv: np.histogram(nl.cf[nl.granted == gv], bins=edges, density=True)[0] for gv in [0, 1]}
    out["cf_density"] = pd.DataFrame(dict(bin_lo=edges[:-1], bin_hi=edges[1:], density_rejected=dens[0],
                                          density_granted=dens[1], n_rejected=int((nl.granted == 0).sum()),
                                          n_granted=int((nl.granted == 1).sum())))
    return out


def cached(d: pd.DataFrame, tc: str) -> pd.DataFrame:
    """The sample of tc with the cached predictions of the main run (c1_preds), in the main run's row order and with
    d's index (rows of the embeddings), and the counterfactual scores built from them as in run_tc."""
    pr = pd.read_parquet(P / f"c1_preds{SFX}_{tc}.parquet")
    x = (d if tc == "all" else d[d.tech_center == tc]).drop(columns=[c for c in pr.columns if c in d.columns
                                                                    and c != "application_number"])
    x = x.reset_index().merge(pr, on="application_number", how="inner").set_index("index")
    x.index.name = None
    im = (x.part == "impute").to_numpy()
    x["cf"] = np.where(im, x.ml_cites, x.imp_cites)
    x["cf_pa"] = np.where(im, x.ml_cites_pa, x.imp_cites_pa)
    return x


# ---------------------------------------------------------------- one technology center

ROBUST = {"ten": "Cells with at least ten examiners", "fifteen": "Cells with at least fifteen examiners",
          "window": "Leniency from the same period (filing years t - 2 to t + 2)",
          "dock": "Applications docketed to an examiner only once"}
PRED_COLS = ["ml_cites", "ml_lenient", "ml_pa_all", "ml_judge", "ft_cites"]


def robust(x: pd.DataFrame, tc: str) -> tuple:
    """Table III on the test and imputation sets with fewer single-examiner quintiles (cells with at least ten or
    fifteen examiners) and with leniency from the same period (04b). Own random stream, so the main results do not
    depend on it."""
    rng = np.random.default_rng(SEED + 31)
    ev = x[x.part.isin(["test", "impute"])]
    res_q, res_e, cells = [], [], []
    for key, lab in ROBUST.items():
        e = ev.copy()
        if key == "window":
            if "lenient_win" not in e or e.lenient_win.isna().all():
                continue
            e["lenient_grant"] = e.lenient_win
            e["q"] = quintiles(e)
        elif key == "dock":   # 41
            if "docketed_once" not in e:
                continue
            e = e[e.docketed_once.fillna(False).astype(bool)].copy()
            e["q"] = quintiles(e)
        else:
            e["q"] = quintiles(e, 10 if key == "ten" else 15)
        e = e[e.q.notna()]
        bad = {ALGO["cites"]: -e.ml_cites.to_numpy(), LENIENT: -e.ml_lenient.to_numpy(),
               PA_ALL: -e.ml_pa_all.to_numpy(), JUDGE: -e.ml_judge.to_numpy()}
        if e.ft_cites.notna().all():
            bad[LLM_FT] = -e.ft_cites.to_numpy()
        a, b_, _ = contraction(e, "cites", bad, rng)
        res_q += [dict(r, tc=tc, sample=lab) for r in a]
        res_e += [dict(r, tc=tc, sample=lab) for r in b_]
        ne = e.groupby([CELL, "q"]).examiner_id.nunique()
        cells.append(dict(tc=tc, sample=lab, n=len(e), cells=e[CELL].nunique(),
                          examiners_per_cell_quintile=ne.median(), single_examiner_quintiles=float((ne == 1).mean())))
    return pd.DataFrame(res_q), pd.DataFrame(res_e), pd.DataFrame(cells)


def design_matrix(x: pd.DataFrame, emb: np.ndarray, tc: str, tr: np.ndarray) -> np.ndarray:
    """What the examiner sees: principal components of the text embedding (fitted on the training set), entity size,
    number of inventors; with all centers pooled, the center and the filing year. x.index: rows of emb."""
    E = np.asarray(emb[x.index.to_numpy()], dtype=np.float32)
    pca = PCA(N_PC, random_state=SEED).fit(E[tr])
    X = np.hstack([pca.transform(E), x[["small_entity", "n_inventors"]].to_numpy(np.float32)])
    if tc == "all":   # the examiner knows the field and the year
        X = np.hstack([X, pd.get_dummies(x.tech_center).to_numpy(np.float32),
                       x[["filing_year"]].to_numpy(np.float32)])
    return X


def fig4_relative(tst: pd.DataFrame, r: np.ndarray, rng: np.random.Generator) -> dict:
    """Figure IV: each stricter quintile's mean residual (observed − the most lenient quintile's model) minus the most
    lenient quintile's own (a level error common to all quintiles cancels), with the same bootstrap draws of art units
    for both. {quintile label 2..5: dict(resid_rel, resid_rel_lo, resid_rel_hi)}."""
    u, inv = np.unique(tst.art_unit.to_numpy(), return_inverse=True)
    q = tst.q.to_numpy()
    est = np.zeros((B + 1, 5))
    for b in range(B + 1):
        m_ = np.ones(len(r)) if b == 0 else np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        for k in range(1, 6):
            s = q == k
            est[b, k - 1] = (m_ * s * r).sum() / max((m_ * s).sum(), 1)
    out = {}
    for k in range(1, 5):   # q = k is reported as quintile 6 − k (1 = most lenient)
        d = est[:, k - 1] - est[:, 4]
        out[6 - k] = dict(resid_rel=d[0], resid_rel_lo=pc(d[1:], 1), resid_rel_hi=pc(d[1:], 0))
    return out


def fig4_post() -> None:
    """python 35_c1_kllm.py --fig4: add Figure IV's residuals relative to the most lenient quintile (fig4_relative) to
    the existing c1_kllm*_fig4.csv from the cached predictions, refitting only the lenient quintile's model as in run_tc
    (CFE_THREADS=16 and 16 BLAS threads reproduce the main run's); the existing columns are checked and kept."""
    d = load_all() if SAMPLE == "all" else load()
    emb = np.load(P / ("emb/all_s0.npy" if SAMPLE == "all" else "emb/s0.npy"), mmap_mode="r")
    tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
    T = OUT / "tables"
    old = pd.read_csv(T / f"c1_kllm{SFX}_fig4.csv", dtype={"tc": str})
    mo = pd.read_csv(T / f"c1_kllm{SFX}_models.csv", dtype={"tc": str})
    new = []
    for tc in tcs:
        x = cached(d, tc)
        tr = (x.part == "train").to_numpy()
        X = design_matrix(x, emb, tc, tr)
        ev = x[x.part.isin(["test", "impute"])].copy()
        ev["q"] = quintiles(ev)
        ev = ev[ev.q.notna()]
        rng_rel = np.random.default_rng(SEED + 50)
        for o in OUTCOMES:
            kind = "gaussian" if o == "cites" else "binary"
            pm = mo[(mo.tc == tc) & (mo.model == f"training: {o}")].iloc[0]
            prm = dict(depth=int(pm.depth), rate=float(pm.rate), trees=int(pm.trees))
            trn = ev[(ev.part == "impute") & (ev.granted == 1) & (ev.q == 5)]
            f_len, _, _ = fit_gbm(X[x.index.get_indexer(trn.index)], np.nan_to_num(trn[o].to_numpy(float)), kind,
                                  SEED + 5, prm)
            tst = ev[(ev.part == "test") & (ev.granted == 1)].copy()
            tst["len_pred"] = f_len(X[x.index.get_indexer(tst.index)])
            r = (tst[o] - tst.len_pred).to_numpy()
            rel = fig4_relative(tst, r, rng_rel)
            z = old[(old.tc == tc) & (old.outcome == o)].copy()
            for k in range(5, 0, -1):
                s = (tst.q == k).to_numpy()
                row = z[z.quintile == 6 - k].iloc[0]
                assert np.isclose(row.resid, r[s].mean(), rtol=0, atol=1e-9) and np.isclose(
                    row.predicted, tst.len_pred[s].mean(), rtol=0, atol=1e-9), (tc, o, k, row.resid, r[s].mean())
            for c_ in ["resid_rel", "resid_rel_lo", "resid_rel_hi"]:
                z[c_] = z.quintile.map(lambda j, c_=c_: rel[j][c_] if j in rel else np.nan)
            new.append(z)
    pd.concat(new).to_csv(T / f"c1_kllm{SFX}_fig4.csv", index=False)
    print(pd.concat(new).round(3).to_string())


def fit_table2(te: pd.DataFrame, tc: str) -> tuple:
    """Figure II's bins, the fit statistics, and Table II (boosting against the linear benchmark) on the test set."""
    te = te.copy()
    tg = te[te.granted == 1]
    # Figure II (imputation_plot.R): percentile bins over all test cases; observed among the granted
    te["bin"] = pd.qcut(te.ml_cites.rank(method="first"), 100, labels=False)
    fig2 = te.groupby("bin").agg(predicted=("ml_cites", "mean"), grant_rate=("granted", "mean")).join(
        te[te.granted == 1].groupby("bin").agg(observed=("cites", "mean"), predicted_granted=("ml_cites", "mean")))
    low1 = te.ml_cites <= te.ml_cites.quantile(0.01)
    fit = dict(tc=tc, corr_cites=np.corrcoef(tg.ml_cites, tg.cites)[0, 1],
               corr_cites_linear=np.corrcoef(tg.lin_cites, tg.cites)[0, 1],
               corr_cites_llm=spearmanr(tg.ft_cites, tg.cites)[0],
               spearman_cites=spearmanr(tg.ml_cites, tg.cites)[0],
               auc_uncited=roc_auc_score(tg.uncited, tg.ml_uncited),
               auc_uncited_linear=roc_auc_score(tg.uncited, tg.lin_uncited),
               auc_grant_by_predicted_cites=roc_auc_score(te.granted, te.ml_cites),
               auc_grant_predicted_examiner=roc_auc_score(te.granted, te.ml_judge),
               lowest1_grant_rate=te.granted[low1].mean(),
               lowest1_cites=te.cites[low1 & (te.granted == 1)].mean(),
               lowest1_uncited=te.uncited[low1 & (te.granted == 1)].mean(),
               mean_cites=tg.cites.mean(), uncited_share=tg.uncited.mean(), grant_rate=te.granted.mean())
    # Table II
    t2 = []
    gg = te.granted == 1
    for o in OUTCOMES:
        sgn = -1 if o == "cites" else 1
        a, b = sgn * te[f"ml_{o}"], sgn * te[f"lin_{o}"]
        for top in TOPS:
            fa, fb = a >= a.quantile(1 - top), b >= b.quantile(1 - top)
            t2.append(dict(tc=tc, outcome=o, top=top, n_flagged=int(fa.sum()), overlap=(fa & fb).sum() / fa.sum(),
                           both=te.loc[fa & fb & gg, o].mean(), ml_only=te.loc[fa & ~fb & gg, o].mean(),
                           linear_only=te.loc[fb & ~fa & gg, o].mean(), all_ml=te.loc[fa & gg, o].mean(),
                           all_linear=te.loc[fb & gg, o].mean()))
    # the same for the highest predicted citations (the proposal's 'highest-citation percentile')
    for top in TOPS:
        a, b = te.ml_cites, te.lin_cites
        fa, fb = a >= a.quantile(1 - top), b >= b.quantile(1 - top)
        t2.append(dict(tc=tc, outcome="cites (highest)", top=top, n_flagged=int(fa.sum()),
                       overlap=(fa & fb).sum() / fa.sum(), both=te.loc[fa & fb & gg, "cites"].mean(),
                       ml_only=te.loc[fa & ~fb & gg, "cites"].mean(), linear_only=te.loc[fb & ~fa & gg, "cites"].mean(),
                       all_ml=te.loc[fa & gg, "cites"].mean(), all_linear=te.loc[fb & gg, "cites"].mean()))
    return fit, fig2.reset_index(), pd.DataFrame(t2)


def contraction_block(x: pd.DataFrame, tc: str, rng: np.random.Generator, f_stat, llm: bool) -> dict:
    """Contraction (contraction_plot.R) on the test and imputation sets together and on the lock box, for citations,
    'uncited' and Table V's other outcomes; Figure III's permutation test in the contraction cells (index: predicted
    citations; own stream); contraction on the cells that pass it. Draws from rng in run_tc's order."""
    res_q, res_e, curves, out = [], [], {}, {}
    for part, mask in [("test + imputation", x.part.isin(["test", "impute"])), ("lock box", x.part == "lockbox")]:
        e = x[mask].copy()
        e["q"] = quintiles(e)
        e = e[e.q.notna()]
        out[f"n_{part}"] = (len(e), e[CELL].nunique())
        if part == "test + imputation":
            ev = e
        for o in list(OUTCOMES) + (list(OTHER) if part == "test + imputation" else []):
            if o == "uncited":
                bad = {ALGO[o]: e.ml_uncited.to_numpy(), JUDGE: -e.ml_judge.to_numpy()}
            elif o == "cites":
                bad = {ALGO[o]: -e.ml_cites.to_numpy(), LENIENT: -e.ml_lenient.to_numpy(),
                       PA_ALL: -e.ml_pa_all.to_numpy(), LLM_FT: -e.ft_cites.to_numpy(), JUDGE: -e.ml_judge.to_numpy()}
                if not llm:
                    del bad[LLM_FT]
            else:   # Table V: the citation algorithm judged on other outcomes
                bad = {ALGO["cites"]: -e.ml_cites.to_numpy(), JUDGE: -e.ml_judge.to_numpy()}
            a, b_, c = contraction(e, o, bad, rng, gender=(o == "cites"))
            res_q += [dict(r, tc=tc, sample=part) for r in a]
            res_e += [dict(r, tc=tc, sample=part) for r in b_]
            curves[(part, o)] = c
    # Figure III: permutation balance test in the contraction cells (index: predicted citations)
    prng = np.random.default_rng(SEED + 7)
    pv = []
    for _, c in ev.groupby(CELL, sort=True):
        gq = c.q.to_numpy(int) - 1
        yy = c.ml_cites.to_numpy(float)
        f0 = f_stat(yy, gq, Q)[0]
        fp = f_stat(yy, gq[np.argsort(prng.random((M, len(gq))), axis=1)], Q)
        pv.append((1 + (fp >= f0 - 1e-12).sum()) / (1 + M))
    out["perm_p"] = np.array(pv)
    # robustness: contraction on the cells that pass the permutation test (p >= 0.05)
    ok_cells = np.array([c for c, pv_ in zip(sorted(ev[CELL].unique()), pv) if pv_ >= 0.05])
    e = ev[ev[CELL].isin(ok_cells)].copy()
    out["n_balanced cells"] = (len(e), e[CELL].nunique())
    bad = {ALGO["cites"]: -e.ml_cites.to_numpy(), LENIENT: -e.ml_lenient.to_numpy(), PA_ALL: -e.ml_pa_all.to_numpy(),
           JUDGE: -e.ml_judge.to_numpy()}
    if llm:
        bad[LLM_FT] = -e.ft_cites.to_numpy()
    a, b_, c = contraction(e, "cites", bad, rng)
    res_q += [dict(r, tc=tc, sample="balanced cells") for r in a]
    res_e += [dict(r, tc=tc, sample="balanced cells") for r in b_]
    return dict(out, res_q=res_q, res_e=res_e, curves=curves, ev=ev)


def run_tc(d: pd.DataFrame, emb: np.ndarray, tc: str, f_stat) -> dict:
    rng = np.random.default_rng(SEED + (9999 if tc == "all" else int(tc)))
    x = (d if tc == "all" else d[d.tech_center == tc]).copy()
    x["part"] = partition(x, rng)
    tr = (x.part == "train").to_numpy()
    im = (x.part == "impute").to_numpy()
    g = (x.granted == 1).to_numpy()
    X = design_matrix(x, emb, tc, tr)
    llm = x.ft_cites.notna().all()
    pa = np.nan_to_num(x.prior_art.to_numpy(float))
    Xpa = np.hstack([X, np.log1p(np.c_[pa, np.nan_to_num(x.prior_art_ex.to_numpy(float))]).astype(np.float32)])
    out = dict(tc=tc, parts=x.part.value_counts().to_dict(), n=len(x))
    pr, models = {}, []

    def fit(name, Z, mask, y, kind, seed, prm=None, both=True):
        """Training-set model ml_<name>; with both, the imputation-set model imp_<name> too."""
        f, p_, cv = fit_gbm(Z[mask & tr], y[mask & tr], kind, seed, prm)
        pr[f"ml_{name}"] = f(Z)
        models.append(dict(tc=tc, model=f"training: {name}", n=int((mask & tr).sum()), cv=cv, **p_))
        if both:
            fi, pi_, cvi = fit_gbm(Z[mask & im], y[mask & im], kind, seed + 1, prm)
            pr[f"imp_{name}"] = fi(Z)
            models.append(dict(tc=tc, model=f"imputation: {name}", n=int((mask & im).sum()), cv=cvi, **pi_))
        print(tc, name, p_, flush=True)
        return p_
    for o in OUTCOMES:
        kind = "gaussian" if o == "cites" else "binary"
        y = np.nan_to_num(x[o].to_numpy(float))
        out[f"prm_{o}"] = fit(o, X, g, y, kind, SEED)
        fl = fit_linear(X[tr & g], y[tr & g], kind)
        pr[f"lin_{o}"] = fl(X)
        if hasattr(fl, "alpha"):
            models.append(dict(tc=tc, model=f"linear (Poisson GLM): {o}", n=int((tr & g).sum()), cv=fl.cv,
                               depth=np.nan, rate=fl.alpha, trees=np.nan))
    fit("judge", X, np.ones(len(x), bool), g.astype(int), "binary", SEED + 3, both=False)
    fit("cites_pa", Xpa, g, np.nan_to_num(x.cites.to_numpy(float)), "gaussian", SEED + 10)   # extension D
    fit("pa_all", X, np.ones(len(x), bool), pa, "gaussian", SEED + 12, both=False)           # extension A(iii)
    fit("pa_sel", X, g, pa, "gaussian", SEED + 14)                                          # extension A(i)
    # extension B: the most lenient quintile's grants in the training set
    xt = x[tr].copy()
    xt["q"] = quintiles(xt)
    q5t = x.index.isin(xt.index[xt.q == 5])
    fit("lenient", X, g & q5t, np.nan_to_num(x.cites.to_numpy(float)), "gaussian", SEED + 16, out["prm_cites"],
        both=False)
    for k, v in pr.items():
        x[k] = v
    # counterfactual scores: out of sample for every application (imputation-set rows: the training model)
    x["cf"] = np.where(im, x.ml_cites, x.imp_cites)
    x["cf_pa"] = np.where(im, x.ml_cites_pa, x.imp_cites_pa)
    te = x[x.part == "test"].copy()
    out["fit"], out["fig2"], out["table2"] = fit_table2(te, tc)

    # contraction (Tables III, V, VI, VIII), Figure III's permutation test, balanced cells
    blk = contraction_block(x, tc, rng, f_stat, llm)
    res_q, res_e, curves, ev = blk["res_q"], blk["res_e"], blk["curves"], blk["ev"]
    out.update({k: blk[k] for k in ["n_test + imputation", "n_lock box", "n_balanced cells", "perm_p"]})
    out["quintiles"], out["contraction"], out["curves"] = pd.DataFrame(res_q), pd.DataFrame(res_e), curves

    # Figure IV: model of the most lenient quintile's grants (imputation half), applied to each quintile (test half)
    f4, f4b = [], []
    rng_rel = np.random.default_rng(SEED + 50)   # own stream: the main one is unchanged
    for o in OUTCOMES:
        kind = "gaussian" if o == "cites" else "binary"
        trn = ev[(ev.part == "impute") & (ev.granted == 1) & (ev.q == 5)]
        f_len, _, _ = fit_gbm(X[x.index.get_indexer(trn.index)], np.nan_to_num(trn[o].to_numpy(float)), kind,
                              SEED + 5, out[f"prm_{o}"])
        tst = ev[(ev.part == "test") & (ev.granted == 1)].copy()
        tst["len_pred"] = f_len(X[x.index.get_indexer(tst.index)])
        u, inv = np.unique(tst.art_unit.to_numpy(), return_inverse=True)
        r = (tst[o] - tst.len_pred).to_numpy()
        rel = fig4_relative(tst, r, rng_rel)
        for k in range(5, 0, -1):
            s = (tst.q == k).to_numpy()
            draws = []
            for _ in range(B):
                m_ = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
                draws.append((m_ * s * r).sum() / max((m_ * s).sum(), 1))
            f4.append(dict(tc=tc, outcome=o, quintile=6 - k, n=int(s.sum()), n_train=len(trn),
                           observed=tst.loc[s, o].mean(), predicted=tst.len_pred[s].mean(), resid=r[s].mean(),
                           resid_lo=np.percentile(draws, 2.5), resid_hi=np.percentile(draws, 97.5),
                           **rel.get(6 - k, dict(resid_rel=np.nan, resid_rel_lo=np.nan, resid_rel_hi=np.nan))))
        tst["decile"] = pd.qcut(tst.len_pred.rank(method="first"), 10, labels=False)
        b_ = tst.groupby(["q", "decile"]).agg(predicted=("len_pred", "mean"), observed=(o, "mean"),
                                              n=(o, "size")).reset_index()
        b_.insert(0, "quintile", 6 - b_.pop("q").astype(int))   # 1 = most lenient, as fig4
        f4b.append(b_.assign(outcome=o, tc=tc))
    out["fig4"], out["fig4_bins"] = pd.DataFrame(f4), pd.concat(f4b)

    # Table IV, and extension A(i): the same reranking for prior-art citations, imputed as in the paper and observed
    q5 = te.index.isin(ev.index[ev.q == 5])
    t4 = []
    for o in OUTCOMES:
        sgn = 1 if o == "cites" else -1   # granted first = largest
        keep = {ALGO[o]: sgn * te[f"ml_{o}"].to_numpy(), JUDGE: te.ml_judge.to_numpy()}
        if o == "cites":
            keep.update({LENIENT: te.ml_lenient.to_numpy(), PA_ALL: te.ml_pa_all.to_numpy()})
            if llm:
                keep[LLM_FT] = te.ft_cites.to_numpy()
        t4 += [dict(r, tc=tc) for r in reranking(te, o, keep, te[f"imp_{o}"].to_numpy(), q5, rng)]
    keep = {ALGO["cites"]: te.ml_cites.to_numpy(), PA_SEL: te.ml_pa_sel.to_numpy(), PA_ALL: te.ml_pa_all.to_numpy()}
    t4 += [dict(r, tc=tc) for r in reranking(te.assign(pa_obs=te.pa_granted), "pa_obs", keep, te.imp_pa_sel.to_numpy(),
                                             q5, rng, actual=te.prior_art.to_numpy())]
    out["table4"] = pd.DataFrame(t4)

    # Table VII (gender, reranking) and Figure VI (misranking)
    out["table7"] = pd.DataFrame([dict(r, tc=tc) for r in rerank_gender(te, te.ml_cites.to_numpy(),
                                                                        te.imp_cites.to_numpy(), rng)])
    out["fig6"] = misranking(ev, ev.ml_cites.to_numpy(), ev.ml_judge.to_numpy()).assign(tc=tc)

    # extension C: levels at the margin
    out["levels"] = pd.DataFrame([dict(r, tc=tc) for r in margin_levels(
        ev, {CF: ev.cf.to_numpy(), CF_PA: ev.cf_pa.to_numpy()}, rng)])
    out["levels_topcoded"] = pd.DataFrame([dict(r, tc=tc) for r in levels_topcoded(
        ev, top_codes(x), np.random.default_rng(SEED + 53))])

    # extension A(ii): α measured on prior-art citations (test set)
    out["alpha"] = pd.DataFrame([dict(r, tc=tc) for r in alpha_table(te, te.imp_pa_sel.to_numpy(), "prior-art model")
                                 + alpha_table(te, te.cf.to_numpy(), "counterfactual citation score")])

    # extension D: the counterfactual scores and their validation (no random draws; --post redoes it from the cache)
    out.update(counterfactual_outputs(x, tc))
    out["models"] = models
    keep_cols = ["application_number", "part"] + [c for c in x.columns if c.startswith(("ml_", "imp_", "lin_"))]
    x[keep_cols].to_parquet(P / f"c1_preds{SFX}_{tc}.parquet")   # for --robust
    out["robust_q"], out["robust"], out["robust_cells"] = robust(x, tc)
    return out


# ---------------------------------------------------------------- figures

STY = {"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.titleweight": "bold",
       "axes.titlelocation": "left", "axes.linewidth": 0.6, "pdf.fonttype": 42}
COL = {ALGO["cites"]: ("#2a78d6", "-", "Algorithm"), LENIENT: ("#0e7490", "--", "Lenient-trained"),
       PA_ALL: ("#15803d", "-.", "Prior-art-trained"), LLM_FT: ("#7c3aed", "-", "LLM, fine-tuned"),
       JUDGE: ("#eb6834", "-", "Predicted examiner"), RANDOM: ("#999999", "--", "Random")}


def save(fig, path):
    for a in fig.axes:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    for ext, meta in [("pdf", {"CreationDate": None}), ("png", {"Software": None})]:
        fig.savefig(path.with_suffix(f".{ext}"), metadata=meta, dpi=300)
    plt.close(fig)


def figure(r: dict, path: pathlib.Path) -> None:
    plt.rcParams.update(STY)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw={"wspace": 0.6})
    f2 = r["fig2"]
    ax[0].scatter(f2.predicted, 100 * f2.grant_rate, s=4, color="#2a78d6")
    ax[0].set_xlabel("Predicted citations")
    ax[0].set_ylabel("Examiners' grant rate (%)")
    ax[0].set_title("a  Grants by prediction")
    xb = f2.predicted_granted if "predicted_granted" in f2 else f2.predicted   # x and y over the same (granted)
    ax[1].scatter(xb, f2.observed, s=4, color="#2a78d6")
    lim = [0, float(np.nanmax([xb.max(), f2.observed.max()]))]
    diag = {1: ax[1].plot(lim, lim, color="#999", lw=0.6, ls="--")[0]}
    ax[1].set_xlabel("Predicted citations")
    ax[1].set_ylabel("Observed citations (granted)")
    ax[1].set_title("b  Calibration")
    b4 = r["fig4_bins"]
    b4 = b4[b4.outcome == "cites"]
    qcol = b4.quintile if "quintile" in b4 else 6 - b4.q   # 1 = most lenient
    for j, col in zip(range(1, 6), ["#0b0b0b", "#2a78d6", "#5aa0e6", "#eb6834", "#c2410c"]):
        s = b4[qcol == j]
        ax[2].plot(s.predicted, s.observed, marker="o", ms=2, lw=0.8, color=col,
                   label="Q1 (most lenient)" if j == 1 else f"Q{j}")
    diag[2] = ax[2].plot(lim, lim, color="#999", lw=0.6, ls="--")[0]
    ax[2].set_xlabel("Predicted (lenient quintile's model)")
    ax[2].set_ylabel("Observed citations")
    ax[2].set_title("c  Unobservables")
    ax[2].legend(frameon=False, fontsize=5.5, ncol=2)
    # d: citations lost (% of the most lenient quintile's citations per application), positive, as Table III
    c = r["curves"][("test + imputation", "cites")]
    xmax = 100 * max(c["X"]) * 1.3
    top_ = 0.0
    for name, (col, ls, lab) in COL.items():
        if name in c["curves"]:
            ax[3].plot(PP, -100 * c["curves"][name], color=col, ls=ls, lw=1, label=lab)
            top_ = max(top_, float(np.nanmax(-100 * c["curves"][name][PP <= xmax])))
    qd = r.get("quintiles")
    if qd is not None:   # the examiners' points with their 95% intervals (bootstrap of art units)
        qd = qd[(qd["sample"] == "test + imputation") & (qd.outcome == "cites") &
                (qd.quintile.astype(str) != "2–5 (mean)")]
        qd = qd.assign(k=qd.quintile.astype(int)).sort_values("k")
        ax[3].errorbar(100 * qd.extra_rejections, -100 * qd.change,
                       yerr=[100 * (qd.change_hi - qd.change), 100 * (qd.change - qd.change_lo)], fmt="o",
                       color="#0b0b0b", ms=3, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
        low = qd.loc[qd.change_hi.idxmax()]   # labelled directly, beside the lowest interval's lower end
        ax[3].text(100 * low.extra_rejections + 1.2, -100 * low.change_hi, "Stricter\nquintiles\n(95% CI)",
                   fontsize=4.6, va="bottom", ha="left", color="#0b0b0b", linespacing=1.1)
        top_ = max(top_, float(-100 * qd.change_lo.min()))
        bot_ = float(-100 * qd.change_hi.max())
    else:
        ax[3].scatter(100 * c["X"], -100 * c["Y"], color="#0b0b0b", s=10, zorder=3, label="Stricter quintiles")
        bot_ = min(0.0, float(-100 * max(c["Y"])))
    ax[3].set_xlim(0, xmax)
    ax[3].set_ylim(min(bot_, 0) - 0.05 * (top_ - min(bot_, 0)), top_ * 1.55)   # the legend above the data
    ax[3].set_xlabel("Extra rejections (pp)")
    ax[3].set_ylabel("Citations lost (%)")
    ax[3].set_title("d  Contraction")
    ax[3].legend(frameon=False, fontsize=4.6, loc="upper left", handlelength=1.6, labelspacing=0.2)
    if f2.predicted.max() > 5 * f2.predicted.median():   # a heavy right tail (pooled centers)
        pos = (f2.predicted > 0).all() and (b4.predicted > 0).all() and (f2.observed > 0).all()
        if pos:   # all predictions positive (Poisson): log axes with readable ticks
            from matplotlib.ticker import FixedLocator, NullLocator
            tk = [0.5, 1, 2, 5, 10, 20, 50]
            for a_, both in [(ax[0], False), (ax[1], True), (ax[2], True)]:
                axes_ = [a_.xaxis, a_.yaxis] if both else [a_.xaxis]
                for ax_ in axes_:
                    (a_.set_xscale if ax_ is a_.xaxis else a_.set_yscale)("log")
                    ax_.set_major_locator(FixedLocator(tk))
                    ax_.set_minor_locator(NullLocator())
                    ax_.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            lo_ = 0.8 * float(min(f2.predicted.min(), xb.min(), f2.observed.min(), b4.predicted.min(),
                                  b4.observed.min()))
            hi_ = 1.25 * float(max(f2.predicted.max(), xb.max(), f2.observed.max(), b4.predicted.max(),
                                   b4.observed.max()))
            ax[0].set_xlim(lo_, hi_)
            for a_ in ax[1:3]:
                a_.set_xlim(lo_, hi_)
                a_.set_ylim(lo_, hi_)
            for k_ in [1, 2]:   # the 45-degree line on log axes
                diag[k_].set_data([lo_, hi_], [lo_, hi_])
            ax[2].legend(frameon=False, fontsize=5, ncol=1, loc="upper left", labelspacing=0.2, handlelength=1.2)
        else:   # some predictions are negative (squared error): log-like axes
            for a_ in ax[:3]:
                a_.set_xscale("symlog", linthresh=1)
            for a_ in ax[1:3]:
                a_.set_yscale("symlog", linthresh=1)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.22)
    save(fig, path)


def figure_ext(r: dict, path: pathlib.Path) -> None:
    plt.rcParams.update(STY)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw={"wspace": 0.6})
    f6 = r["fig6"]
    s = f6[f6.quintile == 5]
    # a: the strictest quintile's extra rejections and the rules' (working within cells, Table III's rule)
    ser = [("examiners", "#0b0b0b", "Strictest quintile"), ("algorithm", "#2a78d6", "Algorithm")]
    if "judge" in s and s.judge.notna().all():
        ser.append(("judge", "#eb6834", "Predicted examiner"))
    wd = 0.84 / len(ser)
    for j_, (col_, colr, lab) in enumerate(ser):
        ax[0].bar(s.bin + (j_ - (len(ser) - 1) / 2) * wd, 100 * s[col_], width=wd, color=colr, label=lab)
    ax[0].set_xlabel("Predicted citations, bin (1 = lowest)")
    ax[0].set_ylabel("Share of extra rejections (%)")
    ax[0].set_title("a  Who is rejected")
    ax[0].set_ylim(0, 100 * float(s[[c_ for c_, _, _ in ser]].max().max()) * 1.45)
    ax[0].legend(frameon=False, fontsize=5, loc="upper right", handlelength=1.0, ncol=1)
    # b: levels at the margin, per quintile with citations and the score top-coded at the same caps (99th percentile
    # of granted applications, center × year), and the pooled quintiles 2–5 with raw values
    lv = r["levels"]
    pooled = lv[lv.quintile.astype(str) == "2–5 (pooled)"].set_index("series")
    lt = r.get("levels_topcoded")
    tq = None
    if lt is not None and len(lt):
        tq = lt[(lt.quintile.astype(str) != "2–5 (pooled)") & lt.series.isin(["observed citations", CF])]
        if not (tq.lo > 0).all():   # a mean count cannot be negative: show the pooled estimates only
            tq = None
    labs = {"observed citations": "Observed", CF: "Score (text)"}
    for j_, (name, col, mk) in enumerate([("observed citations", "#0b0b0b", "o"), (CF, "#2a78d6", "s")]):
        if tq is not None:
            t = tq[tq.series == name]
            xs = t.quintile.astype(int).to_numpy() + (j_ - 0.5) * 0.18
            ax[1].errorbar(xs, t.value, yerr=[(t.value - t.lo).clip(lower=0), (t.hi - t.value).clip(lower=0)],
                           fmt=mk, ms=3, lw=0.8, color=col, label=labs[name] + ", top-coded")
        z = pooled.loc[name]
        ax[1].errorbar([6.4 + (j_ - 0.5) * 0.18], [z.value], yerr=[[max(z.value - z.lo, 0)], [max(z.hi - z.value, 0)]],
                       fmt=mk, ms=3, lw=0.8, color=col, mfc="white", label=labs[name] + ", raw (2–5 pooled)")
    cv_ = r.get("cf_valid")
    if tq is not None and lt is not None and (lt.series == "mean citations of granted applications, top-coded").any():
        m_ = float(lt[lt.series == "mean citations of granted applications, top-coded"].value.iloc[0])
        ax[1].plot([1.6, 5.4], [m_, m_], color="#999999", lw=0.7, ls="--", zorder=1,
                   label=f"Mean of granted patents, top-coded ({m_:.2f})")
    raw_ = None if lt is None else lt[lt.series == "mean citations of granted applications, raw"]
    if raw_ is not None and len(raw_):   # the same population as the top-coded line: the contraction sample
        cv_ = pd.DataFrame(dict(mean_cites_granted=[float(raw_.value.iloc[0])]))
    if cv_ is not None and len(cv_):   # the constant benchmark for the raw values: mean citations of granted patents
        m_ = float(cv_.mean_cites_granted.iloc[0])
        ax[1].plot([6.0, 6.8], [m_, m_], color="#999999", lw=0.7, ls=":", zorder=1,
                   label=f"Mean of granted patents, raw ({m_:.2f})")
    ax[1].axvline(5.75, color="#cccccc", lw=0.5)
    ax[1].set_xticks([2, 3, 4, 5, 6.4])
    ax[1].set_xticklabels(["2", "3", "4", "5", "2–5"])
    ax[1].set_xlim(1.5, 6.9)
    ax[1].set_xlabel("Stricter quintile")
    ax[1].set_ylabel("Citations per extra rejection")
    ax[1].set_title("b  Levels at the margin")
    lo_, hi_ = ax[1].get_ylim()
    ax[1].set_ylim(0, hi_ + 0.9 * hi_)   # room for the legend above the intervals
    ax[1].legend(frameon=False, fontsize=4.2, loc="upper left", labelspacing=0.15, handlelength=1.4)
    # c: densities of the score outside the lock box, x cut at the 99th percentile (counterfactual_outputs)
    cd = r["cf_density"]
    if "bin_lo" in cd:
        edges = np.r_[cd.bin_lo.to_numpy(), cd.bin_hi.to_numpy()[-1]]
        ax[2].stairs(cd.density_rejected.to_numpy(), edges, color="#c2410c", lw=1, label="Rejected: score")
        ax[2].stairs(cd.density_granted.to_numpy(), edges, color="#2a78d6", lw=1, label="Granted: score")
    else:   # an older sample of applications
        cd = cd[cd.part != "lockbox"]
        bins = np.linspace(0, float(np.nanpercentile(cd.cf, 99)), 40)
        for gv, col, lab in [(0, "#c2410c", "Rejected: score"), (1, "#2a78d6", "Granted: score")]:
            ax[2].hist(cd.cf[cd.granted == gv], bins=bins, density=True, histtype="step", color=col, lw=1, label=lab)
    ax[2].set_xlabel("Counterfactual citations")
    ax[2].set_ylabel("Density")
    ax[2].set_title("c  Scores")
    ax[2].legend(frameon=False, fontsize=5.5)
    dc = r["cf_deciles"]
    for gv, col, lab in [(0, "#c2410c", "Rejected"), (1, "#2a78d6", "Granted")]:
        t = dc[dc.granted == gv]
        ax[3].plot(t.dec, t.prior_art, marker="o", ms=2.5, lw=0.9, color=col, label=lab)
    ax[3].set_xlabel("Score decile\n(within center × year)")
    ax[3].set_ylabel("Prior-art citations")
    ax[3].set_title("d  Prior-art check")
    ax[3].legend(frameon=False, fontsize=5.5)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.25)
    save(fig, path)


# ---------------------------------------------------------------- report

def ci(v, lo, hi, s=1.0, nd=2) -> str:
    return f"{s * v:.{nd}f} [{s * lo:.{nd}f}, {s * hi:.{nd}f}]"


def ratio(v) -> str:
    return "beyond 40 pp" if np.isnan(v) else f"{v:.2f}"


def fig6_lines(f6: pd.DataFrame) -> list:
    """The report's Figure VI table (also rewritten in place by --figdata)."""
    jd = "judge" in f6
    L = ["**Figure VI** (share of the extra rejections from the lowest quarter of predicted citations, bins 1–5 of "
         "20; the rules work within each cell to match the quintile's extra rejections there):", "",
         "| Quintile | Examiners (%) | Algorithm (%) |" + (" Predicted examiner (%) |" if jd else ""),
         "|---|---|---|" + ("---|" if jd else "")]
    for k, t in f6.groupby("quintile"):
        lo = t.bin <= 5
        L.append(f"| {k} | {100 * t.examiners[lo].sum():.1f} | {100 * t.algorithm[lo].sum():.1f} |"
                 + (f" {100 * t.judge[lo].sum():.1f} |" if jd else ""))
    return L


def fit_lines(f: dict, t2: pd.DataFrame) -> list:
    """The report's Figure II paragraph and Table II (also rewritten in place by --linear)."""
    lin = "Poisson GLM" if LOSS == "poisson" else "linear"
    L = ["**Figure II.** "
         f"Correlation of predicted and observed citations among granted test applications: boosting "
         f"{f['corr_cites']:.3f} (Spearman {f['spearman_cites']:.3f}), {lin} {f['corr_cites_linear']:.3f}, "
         f"fine-tuned LLM (Spearman) {f['corr_cites_llm']:.3f}; AUC for 'uncited': {f['auc_uncited']:.3f} "
         f"(logit {f['auc_uncited_linear']:.3f}). The lowest-predicted 1% are granted at "
         f"{100 * f['lowest1_grant_rate']:.1f}% (all: {100 * f['grant_rate']:.1f}%) and, when granted, receive "
         f"{f['lowest1_cites']:.2f} citations ({100 * f['lowest1_uncited']:.0f}% uncited; all granted: "
         f"{f['mean_cites']:.2f}, {100 * f['uncited_share']:.0f}%). AUC of the examiners' grants: on predicted "
         f"citations {f['auc_grant_by_predicted_cites']:.3f}, on the predicted examiner "
         f"{f['auc_grant_predicted_examiner']:.3f}.", "",
         f"**Table II** (boosting against the {'Poisson GLM' if LOSS == 'poisson' else 'linear model (OLS)'} for "
         "citations, the logit for 'uncited'; flagged: lowest predicted citations / highest predicted risk of no "
         "citations / highest predicted citations; mean outcome of the granted among them).", "",
         "| Outcome | Flagged | ML/linear overlap | Both | ML only | Linear only | All ML | All linear |",
         "|---|---|---|---|---|---|---|---|"]
    for x in t2.itertuples():
        L.append(f"| {x.outcome} | {100 * x.top:.0f}% | {100 * x.overlap:.1f}% | {x.both:.3f} | {x.ml_only:.3f} | "
                 f"{x.linear_only:.3f} | {x.all_ml:.3f} | {x.all_linear:.3f} |")
    return L


def contraction_lines(qq: pd.DataFrame, cc: pd.DataFrame, part: str, o: str) -> list:
    """The report's table for one sample and outcome of the contraction (also rewritten in place by --recompute)."""
    tab = "Table V" if o in OTHER else "Tables III, VI, VIII" if o == "cites" else "Table III"
    L = [f"**{tab} ({part}, {o}).** Relative to the most lenient quintile. Change: outcome per "
         "application, relative. A rule's change is at the examiners' extra rejections; "
         "'rejections for the examiners' change' is where the rule's outcome falls as much as the "
         "examiners' (× the examiners' extra rejections). Female: share of the extra rejections "
         "with a female first inventor. Random: the expected value of random rejection.", "",
         "| Quintile | Examiners: extra rejections (pp) | Examiners: change (%) | Examiners: female (%) | "
         "Rule | Rule: change (%) | Rule − examiners (pp) | Rejections for the examiners' change (×) | "
         "Rule: female (%) |", "|---|---|---|---|---|---|---|---|---|"]
    for x in qq.itertuples():
        first = True
        for y in cc[cc.quintile.astype(str) == str(x.quintile)].itertuples():
            head = "| | | | "
            if first:
                head = (f"| {x.quintile} | "
                        f"{ci(x.extra_rejections, x.extra_rejections_lo, x.extra_rejections_hi, 100, 1)}"
                        f" | {ci(x.change, x.change_lo, x.change_hi, 100, 1)} | "
                        f"{ci(x.female_share, x.female_share_lo, x.female_share_hi, 100, 1)} | ")
            first = False
            L.append(f"{head}{y.rule} | {ci(y.change_at_examiner_rejections, y.lo, y.hi, 100, 1)} | "
                     f"{ci(y.diff, y.diff_lo, y.diff_hi, 100, 1)} | {ratio(y.need_ratio)} | "
                     f"{ci(y.female_share, y.female_share_lo, y.female_share_hi, 100, 1)} |")
    return L


CF_HEAD = "**Extension D: counterfactual scores**"


def cf_valid_lines(v: pd.DataFrame) -> list:
    """The report's table of extension D (also rewritten in place by --post)."""
    L = [f"{CF_HEAD} (out of sample; lock box excluded; each score's percentile among the granted applications' values "
         "of the same score, same center and filing year).", "",
         "| Score | Corr. with citations (granted) | Spearman: citations (granted) | Spearman: prior art (granted) | "
         "Spearman: prior art (rejected) | Mean: granted | Mean: rejected | Rejected above granted median (%) | "
         "Rejected in granted top quarter (%) |", "|---|---|---|---|---|---|---|---|---|"]
    for x in v.itertuples():
        L.append(f"| {x.score} | {x.corr_cites_granted:.3f} | {x.spearman_cites_granted:.3f} | "
                 f"{x.spearman_prior_art_granted:.3f} | {x.spearman_prior_art_rejected:.3f} | {x.mean_granted:.2f} | "
                 f"{x.mean_rejected:.2f} | {100 * x.share_rejected_above_granted_median:.1f} | "
                 f"{100 * x.share_rejected_top_quarter:.1f} |")
    return L


def report(res: dict, titles: pd.DataFrame) -> None:
    L = ["# C1 as in Kleinberg et al. (2018, QJE), extensions, and counterfactual scores", "",
         "Paper and the authors' code (QJE Dataverse doi:10.7910/DVN/VWDGHT). Defendant → application, release → "
         "grant, detain → reject. Main outcome: citations within five years excluding self-citations (a rule rejects "
         "the lowest predicted first); literal analogue of their binary outcome: a granted patent no one else cites "
         "within five years ('uncited'). Intervals (not in the paper): bootstrap of art units.", ""]
    for tc, r in res.items():
        f = r["fit"]
        n_ti, n_lb = r["n_test + imputation"], r["n_lock box"]
        nb = r["n_balanced cells"]
        L += ["## All eight technology centers" if tc == "all" else f"## TC {tc}", "",
              f"Cells passing the permutation test (p ≥ 0.05): {nb[1]} cells, {nb[0]:,} applications.", "",
              f"Partition: {r['n']:,} applications; " + ", ".join(f"{k} {v:,}" for k, v in sorted(r["parts"].items()))
              + f". Contraction: test + imputation {n_ti[0]:,} applications in {n_ti[1]} cells; lock box "
              f"{n_lb[0]:,} in {n_lb[1]} cells.", "",
              "Models (5-fold CV; depth = splits per tree; CV score: accuracy, or for counts "
              + ("−mean Poisson deviance (Poisson loss)" if LOSS == "poisson" else "−RMSE") + "):", "",
              "| Model | Training cases | Depth | Rate | Trees | CV score |", "|---|---|---|---|---|---|"]
        for m in r["models"]:
            L.append(f"| {m['model']} | {m['n']:,} | {m['depth']} | {m['rate']} | {m['trees']} | {m['cv']:.4f} |")
        L += [""] + fit_lines(f, r["table2"])
        p = r["perm_p"]
        L += ["", f"**Figure III.** Permutation test in {len(p)} cells: {100 * np.mean(p < 0.05):.1f}% of p-values "
                  f"below 0.05, {100 * np.mean(p < 0.10):.1f}% below 0.10 (5% and 10% expected).", "",
              "**Figure IV** (model of the most lenient quintile's grants, imputation half; applied to each quintile's "
              "grants, test half; 1 = most lenient):", "",
              "| Outcome | Quintile | Grants | Observed | Predicted | Observed − predicted |",
              "|---|---|---|---|---|---|"]
        for x in r["fig4"].itertuples():
            L.append(f"| {x.outcome} | {x.quintile} | {x.n:,} | {x.observed:.3f} | {x.predicted:.3f} | "
                     f"{ci(x.resid, x.resid_lo, x.resid_hi, nd=3)} |")
        for part in ["test + imputation", "lock box", "balanced cells"]:
            for o in (list(OUTCOMES) + list(OTHER) if part == "test + imputation" else
                      list(OUTCOMES) if part == "lock box" else ["cites"]):
                qq = r["quintiles"].query("sample == @part and outcome == @o")
                cc = r["contraction"].query("sample == @part and outcome == @o")
                L += [""] + contraction_lines(qq, cc, part, o)
        L += ["", "**Table IV** (test set; all applications reranked; outcomes of rejected applications imputed, "
                  "bounded by α; for pa_obs, prior-art citations, the row 'observed' uses the rejected applications' "
                  "observed values instead).", "",
              "| Outcome | Rule | α | Outcome per application at the examiners' grant rate | Change (%) | "
              "Rejection rate at the examiners' outcome | Change (%) |", "|---|---|---|---|---|---|---|"]
        for x in r["table4"].itertuples():
            a = x.alpha if isinstance(x.alpha, str) else "∞" if np.isinf(x.alpha) else f"{x.alpha:g}"
            ch = ci(x.outcome_change, x.outcome_change_lo, x.outcome_change_hi, 100, 1)
            rc = ci(x.rejection_change, x.rejection_change_lo, x.rejection_change_hi, 100, 1)
            L.append(f"| {x.outcome} | {x.rule} | {a} | {x.outcome_at_examiner_rate:.3f} (examiners "
                     f"{x.examiner_outcome:.3f}) | {ch} | {100 * x.rejection_rate_at_examiner_outcome:.1f}% (examiners "
                     f"{100 * x.examiner_rejection_rate:.1f}%) | {rc} |")
        L += ["", "**Table VII** (test set, reranked at the examiners' grant rate, α = 1).", "",
              "| Rule | Citations per application, change (%) | Female − other grant rate (pp) | "
              "Female share of rejections (%) |", "|---|---|---|---|"]
        for x in r["table7"].itertuples():
            L.append(f"| {x.rule} | {ci(x.outcome_change, x.outcome_change_lo, x.outcome_change_hi, 100, 1)} | "
                     f"{ci(x.gap, x.gap_lo, x.gap_hi, 100, 1)} | "
                     f"{ci(x.female_share_rejected, x.female_share_rejected_lo, x.female_share_rejected_hi, 100, 1)} |")
        f6 = r["fig6"]
        L += [""] + fig6_lines(f6)
        L += ["", "**Extension C: levels at the margin** (mean over the extra rejections of each stricter quintile).", "",
              "| Series | Quintile | Value |", "|---|---|---|"]
        for x in r["levels"].itertuples():
            L.append(f"| {x.series} | {x.quintile} | {ci(x.value, x.lo, x.hi)} |")
        L += ["", "**Extension A(ii): α on prior-art citations** (granted over rejected, same prediction; within filing "
                  "year × decile).", "", "| Prediction | Decile | Granted | Rejected | α |", "|---|---|---|---|---|"]
        for x in r["alpha"].itertuples():
            L.append(f"| {x.prediction} | {x.decile} | {x.granted:.2f} | {x.rejected:.2f} | {x.alpha:.2f} |")
        L += [""] + cf_valid_lines(r["cf_valid"])
        sc = r["scores"].merge(titles, on="application_number", how="left")
        sc = sc[sc.part != "lockbox"].sort_values("cf_cites", ascending=False)
        L += ["", "Rejected applications with the highest counterfactual scores (test and imputation sets):", "",
              "| Application | Filed | Title | Score | Percentile | Prior-art citations |", "|---|---|---|---|---|---|"]
        for x in sc[sc.part.isin(["test", "impute"])].head(10).itertuples():
            L.append(f"| {x.application_number} | {x.filing_year} | {str(x.invention_title).title()[:80]} | "
                     f"{x.cf_cites:.1f} | {x.cf_percentile_among_granted:.0f} | {x.prior_art_cites:.0f} |")
        L.append("")
    (OUT / f"c1_kllm{SFX}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def figures_from_tables() -> None:
    """Redraw the two figures from output/tables (python 35_c1_kllm.py --figure)."""
    T = OUT / "tables"
    tc = "all" if SAMPLE == "all" else "3600"

    def rd(k):
        x = pd.read_csv(T / f"c1_kllm{SFX}_{k}.csv", dtype={"tc": str})
        return x[x.tc.astype(str) == tc]
    q = rd("quintiles")
    q = q[(q["sample"] == "test + imputation") & (q.outcome == "cites") & (q.quintile.astype(str) != "2–5 (mean)")]
    cv = rd("curves")
    cv = cv[(cv["sample"] == "test + imputation") & (cv.outcome == "cites")]
    curves = {("test + imputation", "cites"): dict(
        curves={r: g.sort_values("pp").change.to_numpy() for r, g in cv.groupby("rule")},
        X=q.extra_rejections.to_numpy(), Y=q.change.to_numpy())}
    r = dict(fig2=rd("fig2"), fig4_bins=rd("fig4_bins"), curves=curves, fig6=rd("fig6"), levels=rd("levels"),
             cf_density=rd("cf_density"), cf_deciles=rd("cf_deciles"), quintiles=rd("quintiles"),
             cf_valid=rd("cf_valid"))
    if (T / f"c1_kllm{SFX}_levels_topcoded.csv").exists():
        r["levels_topcoded"] = rd("levels_topcoded")
    figure(r, OUT / f"figures/journal/fig_c1_kllm{SFX}")
    figure_ext(r, OUT / f"figures/journal/fig_c1_kllm{SFX}_ext")


def write_extension_d(res: dict) -> pd.DataFrame:
    """Writes extension D's tables (cf_valid, cf_deciles, cf_density, the scores, the highest-scored rejected
    applications); returns the titles of the scored applications."""
    T = OUT / "tables"
    for k in ["cf_valid", "cf_deciles", "cf_density"]:
        pd.concat([res[tc][k].assign(tc=tc) for tc in res]).to_csv(T / f"c1_kllm{SFX}_{k}.csv", index=False)
    scores = pd.concat([res[tc]["scores"] for tc in res])
    scores.round(4).to_csv(T / f"c1_counterfactual_scores{SFX}.csv", index=False)
    titles = pd.read_parquet(P / "patex_application_data.parquet", columns=["application_number", "invention_title"])
    titles = titles[titles.application_number.isin(scores.application_number)]
    top = []
    for tc in res:
        s = res[tc]["scores"]
        s = s[s.part.isin(["test", "impute"])].sort_values("cf_cites", ascending=False).head(20)
        top.append(s.merge(titles, on="application_number", how="left"))
    pd.concat(top).to_csv(T / f"c1_kllm{SFX}_top_rejected.csv", index=False)
    return titles


def post() -> None:
    """python 35_c1_kllm.py --post: extension D (counterfactual scores, their validation, deciles, density, the
    highest-scored rejected applications) from the cached predictions of the main run, without refitting; also its
    table in output/c1_kllm*.md. Other outputs are left as they are (extension C, levels, uses random draws of the main
    run's stream, and its inputs, the scores themselves, do not change)."""
    d = load_all() if SAMPLE == "all" else load()
    tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
    res = {tc: counterfactual_outputs(cached(d, tc), tc) for tc in tcs}
    write_extension_d(res)
    md = OUT / f"c1_kllm{SFX}.md"
    if md.exists():   # replace each center's extension D table, in the order of tcs
        L, new, i = md.read_text().split("\n"), [], 0
        while L:
            line = L.pop(0)
            if line.startswith(CF_HEAD) and i < len(tcs):
                while L and not L[0].startswith("Rejected applications with the highest"):
                    L.pop(0)
                new += cf_valid_lines(res[tcs[i]]["cf_valid"]) + [""]
                i += 1
            else:
                new.append(line)
        md.write_text("\n".join(new))
    for tc in tcs:
        print(res[tc]["cf_valid"].T.to_string())


def relinear() -> None:
    """python 35_c1_kllm.py --linear: refit only Table II's linear benchmark (with CFE_LOSS=poisson the Poisson GLM)
    on the cached predictions' design (principal components refitted as in run_tc: CFE_THREADS=16 and 16 BLAS
    threads reproduce the main run's), and rewrite Table II, the fit statistics, the cache's lin_cites column, the
    models table's linear row and the report's Figure II paragraph and Table II. The boosted models are not refitted."""
    d = load_all() if SAMPLE == "all" else load()
    emb = np.load(P / ("emb/all_s0.npy" if SAMPLE == "all" else "emb/s0.npy"), mmap_mode="r")
    tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
    T = OUT / "tables"
    fits, t2s, lin_rows = {}, {}, []
    for tc in tcs:
        x = cached(d, tc)
        tr, g = (x.part == "train").to_numpy(), (x.granted == 1).to_numpy()
        X = design_matrix(x, emb, tc, tr)
        y = np.nan_to_num(x.cites.to_numpy(float))
        fl = fit_linear(X[tr & g], y[tr & g], "gaussian")
        x["lin_cites"] = fl(X)
        if hasattr(fl, "alpha"):
            lin_rows.append(dict(tc=tc, model="linear (Poisson GLM): cites", n=int((tr & g).sum()), cv=fl.cv,
                                 depth=np.nan, rate=fl.alpha, trees=np.nan))
        fits[tc], _, t2s[tc] = fit_table2(x[x.part == "test"], tc)
        pr = pd.read_parquet(P / f"c1_preds{SFX}_{tc}.parquet")
        lin = pr[["application_number"]].merge(x[["application_number", "lin_cites"]], how="left")
        pr["lin_cites"] = lin.lin_cites.to_numpy()
        pr.to_parquet(P / f"c1_preds{SFX}_{tc}.parquet")
    pd.DataFrame([fits[tc] for tc in tcs]).to_csv(T / f"c1_kllm{SFX}_fit.csv", index=False)
    pd.concat([t2s[tc] for tc in tcs]).to_csv(T / f"c1_kllm{SFX}_table2.csv", index=False)
    mo = pd.read_csv(T / f"c1_kllm{SFX}_models.csv", dtype={"tc": str})
    mo = pd.concat([mo[~mo.model.str.startswith("linear")], pd.DataFrame(lin_rows)])
    mo.to_csv(T / f"c1_kllm{SFX}_models.csv", index=False)
    md = OUT / f"c1_kllm{SFX}.md"
    if md.exists():   # each center's Figure II paragraph and Table II, in the order of tcs
        L, k = md.read_text().split("\n"), 0
        for tc in tcs:
            i = next(n for n in range(k, len(L)) if L[n].startswith("**Figure II.**"))
            j = next(n for n in range(i, len(L)) if L[n].startswith("**Figure III.**"))
            new = fit_lines(fits[tc], t2s[tc]) + [""]
            L = L[:i] + new + L[j:]
            k = i + len(new)
        md.write_text("\n".join(L))
    for tc in tcs:
        print(t2s[tc].round(3).to_string(), "\n", lin_rows)


def figdata() -> None:
    """python 35_c1_kllm.py --figdata: Figure VI (misranking, with the rules working within cells) and the top-coded
    levels at the margin from the cached predictions, without refitting; rewrites c1_kllm*_fig6.csv,
    c1_kllm*_levels_topcoded.csv and the report's Figure VI table. The examiners' shares are checked against the
    existing fig6 table."""
    d = load_all() if SAMPLE == "all" else load()
    tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
    T = OUT / "tables"
    old = pd.read_csv(T / f"c1_kllm{SFX}_fig6.csv", dtype={"tc": str})
    f6s, lts = [], []
    for tc in tcs:
        x = cached(d, tc)
        ev = x[x.part.isin(["test", "impute"])].copy()
        ev["q"] = quintiles(ev)
        ev = ev[ev.q.notna()]
        f6 = misranking(ev, ev.ml_cites.to_numpy(), ev.ml_judge.to_numpy()).assign(tc=tc)
        o = old[old.tc == tc].sort_values(["quintile", "bin"])
        assert np.allclose(o.examiners.to_numpy(), f6.sort_values(["quintile", "bin"]).examiners.to_numpy(),
                           rtol=0, atol=1e-12)
        f6s.append(f6)
        lts.append(pd.DataFrame([dict(r, tc=tc) for r in levels_topcoded(ev, top_codes(x),
                                                                        np.random.default_rng(SEED + 53))]))
    pd.concat(f6s).to_csv(T / f"c1_kllm{SFX}_fig6.csv", index=False)
    pd.concat(lts).to_csv(T / f"c1_kllm{SFX}_levels_topcoded.csv", index=False)
    md = OUT / f"c1_kllm{SFX}.md"
    if md.exists():   # each center's Figure VI table, in the order of tcs
        L, k = md.read_text().split("\n"), 0
        for f6 in f6s:
            i = next(n for n in range(k, len(L)) if L[n].startswith("**Figure VI**"))
            j = next(n for n in range(i + 2, len(L)) if L[n] == "")
            new = fig6_lines(f6)
            L = L[:i] + new + L[j:]
            k = i + len(new)
        md.write_text("\n".join(L))
    for f6 in f6s:
        print(fig6_lines(f6))
    print(pd.concat(lts).round(3).to_string())


def recompute() -> None:
    """python 35_c1_kllm.py --recompute: from the cached predictions, without refitting, what depends on the random
    rule's definition or feeds the figures: the contraction (Tables III, V, VI, VIII, their curves, the lock box, the
    permutation test and the balanced cells), replaying run_tc's random stream (the partition, then the contraction:
    the replayed partition is checked against the cached one), the robustness contractions (robust(), own stream),
    Figure II's bins and Figure IV's bins (quintile 1 = most lenient). The rows of all rules but the random one must
    not change (checked); the report's contraction tables are rewritten in place."""
    d = load_all() if SAMPLE == "all" else load()
    f_stat = load_module("m18", "18_assignment_checks.py").f_stat
    tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
    T = OUT / "tables"
    old_c = pd.read_csv(T / f"c1_kllm{SFX}_contraction.csv", dtype={"tc": str})
    res = {}
    for tc in tcs:
        rng = np.random.default_rng(SEED + (9999 if tc == "all" else int(tc)))
        x = cached(d, tc)
        assert (partition(x, rng) == x.part.to_numpy()).all(), "partition not reproduced"
        blk = contraction_block(x, tc, rng, f_stat, x.ft_cites.notna().all())
        rq, re_, rc = robust(x, tc)
        fit, fig2, t2 = fit_table2(x[x.part == "test"], tc)
        res[tc] = dict(quintiles=pd.DataFrame(blk["res_q"]), contraction=pd.DataFrame(blk["res_e"]),
                       curves=blk["curves"], fig2=fig2.assign(tc=tc), robust_q=rq, robust=re_, robust_cells=rc)
    for k in ["quintiles", "contraction", "fig2"]:
        pd.concat([res[tc][k].assign(tc=tc) for tc in tcs]).to_csv(T / f"c1_kllm{SFX}_{k}.csv", index=False)
    write_robust([res[tc]["robust_q"] for tc in tcs], [res[tc]["robust"] for tc in tcs],
                 [res[tc]["robust_cells"] for tc in tcs])
    curves = [pd.DataFrame(dict(tc=tc, sample=part, outcome=o, rule=name, pp=PP, change=v))
              for tc in tcs for (part, o), c in res[tc]["curves"].items() for name, v in c["curves"].items()]
    pd.concat(curves).to_csv(T / f"c1_kllm{SFX}_curves.csv", index=False)
    b4 = pd.read_csv(T / f"c1_kllm{SFX}_fig4_bins.csv", dtype={"tc": str})
    if "q" in b4:   # Figure IV's bins: quintile 1 = most lenient, as fig4
        b4.insert(0, "quintile", 6 - b4.pop("q").astype(int))
        b4.to_csv(T / f"c1_kllm{SFX}_fig4_bins.csv", index=False)
    # checks: every rule but the random one unchanged
    new_c = pd.read_csv(T / f"c1_kllm{SFX}_contraction.csv", dtype={"tc": str})
    key = ["tc", "sample", "outcome", "rule", "quintile"]
    m = old_c.merge(new_c, on=key, suffixes=("_old", ""))
    nr = m.rule != RANDOM
    for col in ["change_at_examiner_rejections", "lo", "hi", "diff", "diff_lo", "diff_hi", "female_share"]:
        dmax = float(np.nanmax(np.abs(m[col][nr] - m[f"{col}_old"][nr])))
        assert dmax < 1e-9, (col, dmax)
    rr = m[~nr & (m.quintile.astype(str) == "2–5 (mean)")]
    print(rr[["tc", "sample", "outcome", "change_at_examiner_rejections_old", "change_at_examiner_rejections",
              "diff_old", "diff", "diff_lo", "diff_hi"]].round(4).to_string())
    md = OUT / f"c1_kllm{SFX}.md"
    if md.exists():   # each center's contraction tables, in order
        L, k = md.read_text().split("\n"), 0
        for tc in tcs:
            r = res[tc]
            for part in ["test + imputation", "lock box", "balanced cells"]:
                for o in (list(OUTCOMES) + list(OTHER) if part == "test + imputation" else
                          list(OUTCOMES) if part == "lock box" else ["cites"]):
                    qq = r["quintiles"].query("sample == @part and outcome == @o")
                    cc = r["contraction"].query("sample == @part and outcome == @o")
                    new = contraction_lines(qq, cc, part, o)
                    i = next(n for n in range(k, len(L))
                             if L[n].startswith("**Table") and f"({part}, {o}).**" in L[n])
                    j = next(n for n in range(i + 4, len(L)) if L[n] == "")
                    L = L[:i] + new + L[j:]
                    k = i + len(new)
        md.write_text("\n".join(L))


def write_robust(q: list, e: list, c: list) -> None:
    T = OUT / "tables"
    pd.concat(q).to_csv(T / f"c1_kllm{SFX}_robust_quintiles.csv", index=False)
    pd.concat(e).to_csv(T / f"c1_kllm{SFX}_robust.csv", index=False)
    pd.concat(c).to_csv(T / f"c1_kllm{SFX}_robust_cells.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore")
    if "--figure" in os.sys.argv:
        figures_from_tables()
        return
    if "--post" in os.sys.argv:   # extension D from the cached predictions, without refitting
        post()
        return
    if "--recompute" in os.sys.argv:   # contraction with the expected random rule, figure inputs, from the cache
        recompute()
        return
    if "--figdata" in os.sys.argv:   # Figure VI within cells and top-coded levels, from the cache
        figdata()
        return
    if "--fig4" in os.sys.argv:   # Figure IV's residuals relative to the most lenient quintile
        fig4_post()
        return
    if "--linear" in os.sys.argv:   # Table II's linear benchmark refitted on the cached design
        relinear()
        return
    f_stat = load_module("m18", "18_assignment_checks.py").f_stat
    if "--robust" in os.sys.argv:   # the robustness contractions from the cached predictions, without refitting
        d = load_all() if SAMPLE == "all" else load()
        tcs = ["all"] if SAMPLE == "all" else os.environ.get("CFE_TCS", "3600,1600").split(",")
        parts = []
        for tc in tcs:
            pr = pd.read_parquet(P / f"c1_preds{SFX}_{tc}.parquet")
            x = (d if tc == "all" else d[d.tech_center == tc]).drop(columns=[c for c in pr.columns if c in d.columns
                                                                            and c != "application_number"])
            x = x.merge(pr, on="application_number", how="inner")
            parts.append(robust(x, tc))
        write_robust([p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts])
        return
    if SAMPLE == "all":
        d, emb, tcs = load_all(), np.load(P / "emb/all_s0.npy", mmap_mode="r"), ["all"]
    else:
        d, emb = load(), np.load(P / "emb/s0.npy", mmap_mode="r")
        tcs = os.environ.get("CFE_TCS", "3600,1600").split(",")
    res = {tc: run_tc(d, emb, tc, f_stat) for tc in tcs}
    T = OUT / "tables"
    pd.DataFrame([res[tc]["fit"] for tc in res]).to_csv(T / f"c1_kllm{SFX}_fit.csv", index=False)
    pd.DataFrame([m for tc in res for m in res[tc]["models"]]).to_csv(T / f"c1_kllm{SFX}_models.csv", index=False)
    for k in ["table2", "fig4", "fig4_bins", "quintiles", "contraction", "table4", "fig2", "table7", "fig6", "levels",
              "levels_topcoded", "alpha"]:
        pd.concat([res[tc][k].assign(tc=tc) for tc in res]).to_csv(T / f"c1_kllm{SFX}_{k}.csv", index=False)
    titles = write_extension_d(res)
    write_robust([res[tc]["robust_q"] for tc in res], [res[tc]["robust"] for tc in res],
                 [res[tc]["robust_cells"] for tc in res])
    pd.DataFrame([dict(tc=tc, cells=len(res[tc]["perm_p"]), share_p_below_05=np.mean(res[tc]["perm_p"] < 0.05),
                       share_p_below_10=np.mean(res[tc]["perm_p"] < 0.10)) for tc in res]).to_csv(
        T / f"c1_kllm{SFX}_permutation.csv", index=False)
    curves = []
    for tc in res:
        for (part, o), c in res[tc]["curves"].items():
            for name, v in c["curves"].items():
                curves.append(pd.DataFrame(dict(tc=tc, sample=part, outcome=o, rule=name, pp=PP, change=v)))
    pd.concat(curves).to_csv(T / f"c1_kllm{SFX}_curves.csv", index=False)
    main_tc = "all" if SAMPLE == "all" else "3600"
    if main_tc in res:
        figure(res[main_tc], OUT / f"figures/journal/fig_c1_kllm{SFX}")
        figure_ext(res[main_tc], OUT / f"figures/journal/fig_c1_kllm{SFX}_ext")
    report(res, titles)


if __name__ == "__main__":
    main()
