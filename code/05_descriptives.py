"""Descriptive statistics and identification checks.

1. Overview by technology center: applications, examiners, share of female first-named inventors, first-action
   allowance rate, grant rate, rejection rounds.
2. Gender gaps: outcomes regressed on "first-named inventor is female", with art unit × filing year fixed effects,
   clustered by examiner.
3. First stage of examiner leniency: outcomes regressed on leave-one-out leniency (same fixed effects).
4. Random-assignment check: leniency regressed on application characteristics (female, small entity, foreign
   inventor, number of inventors, continuation); coefficients should be close to 0.
5. Stability of examiner-level gender gaps: split each examiner's applications into two halves by the parity of the
   application number and compute the female − male grant-rate gap in each half. The correlation between the halves
   (Spearman-Brown corrected) is the reliability. High reliability means individual gaps are a stable trait rather
   than just noise.

Input: data/processed/applications_leniency.parquet, data/processed/app_gender.parquet
Output: output/tables/*.csv, output/descriptives.md; data/processed/examiner_gender_gap.csv (examiner level, private)
"""
import pathlib

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "output"
TC_NAMES = {
    "1600": "Biotech & organic chemistry", "1700": "Chemicals & materials", "2100": "Computer architecture & software",
    "2400": "Networks & security", "2600": "Communications", "2800": "Semiconductors & electrical",
    "3600": "Transportation, e-commerce & business methods", "3700": "Mechanical engineering & medical devices",
}
MIN_PER_GENDER = 30  # examiner-level gender gap: at least this many female and male applications in each half


def load() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET threads = 64")
    df = con.execute(f"""
      SELECT a.application_number, a.examiner_id, a.art_unit, a.tech_center, a.filing_year, a.au_year,
             a.fa_allow, a.granted, a.abandoned, a.n_rejections, a.continuation, a.in_hupd,
             a.lenient_fa, a.lenient_grant,
             (a.small_entity_indicator IN ('SMALL', 'MICRO'))::INT AS small_entity,
             g.n_inventors,
             CASE g.first_inventor_gender WHEN 'F' THEN 1 WHEN 'M' THEN 0 END AS female,
             (g.first_inventor_country <> 'US')::INT AS foreign_inv,
             g.n_female, g.n_gender_known
      FROM '{P}/applications_leniency.parquet' a
      LEFT JOIN '{P}/app_gender.parquet' g USING (application_number)
      WHERE a.tech_center IN ({",".join(repr(k) for k in TC_NAMES)})
    """).df()
    df["continuation"] = df["continuation"].astype(int)
    return df


def tc_overview(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("tech_center")
    t = pd.DataFrame({
        "Field": [TC_NAMES[k] for k in g.groups],
        "Applications": g.size(),
        "Examiners": g["examiner_id"].nunique(),
        "Share gender known": g["female"].apply(lambda s: s.notna().mean()),
        "Share female first inventor": g["female"].mean(),
        "First-action allowance rate": g["fa_allow"].mean(),
        "Grant rate": g["granted"].mean(),
        "Mean rejection rounds": g["n_rejections"].mean(),
    })
    return t.round(3)


def coef_row(fit, var: str, **extra) -> dict:
    return {**extra, "coef": fit.coef()[var], "se": fit.se()[var], "n": fit._N}


def gender_gaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.dropna(subset=["female"])
    for tc in ["ALL"] + list(TC_NAMES):
        s = d if tc == "ALL" else d[d.tech_center == tc]
        for y in ["fa_allow", "granted", "abandoned", "n_rejections"]:
            fit = pf.feols(f"{y} ~ female + small_entity + foreign_inv + n_inventors + continuation | au_year",
                           data=s, vcov={"CRV1": "examiner_id"})
            rows.append(coef_row(fit, "female", tech_center=tc, outcome=y, mean_male=s.loc[s.female == 0, y].mean()))
    return pd.DataFrame(rows)


def first_stage_and_balance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs, bal = [], []
    for y, z in [("fa_allow", "lenient_fa"), ("granted", "lenient_grant")]:
        fit = pf.feols(f"{y} ~ {z} | au_year", data=df, vcov={"CRV1": "examiner_id"})
        fs.append(coef_row(fit, z, outcome=y))
    d = df.dropna(subset=["female"])
    for x in ["female", "small_entity", "foreign_inv", "n_inventors", "continuation"]:
        fit = pf.feols(f"lenient_grant ~ {x} | au_year", data=d, vcov={"CRV1": "examiner_id"})
        bal.append(coef_row(fit, x, regressor=x, sd_leniency=d["lenient_grant"].std()))
    return pd.DataFrame(fs), pd.DataFrame(bal)


def examiner_gender_gap_reliability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=["female"]).copy()
    # demean by art unit × year first, so differences in unit composition do not leak into individual gaps
    d["granted_r"] = d["granted"] - d.groupby("au_year")["granted"].transform("mean")
    d["half"] = d["application_number"].str[-1].astype(int) % 2
    g = (d.groupby(["examiner_id", "half", "female"])["granted_r"].agg(["mean", "size"]).unstack("female"))
    g.columns = [f"{a}_{'f' if b == 1 else 'm'}" for a, b in g.columns]
    g = g[(g["size_f"] >= MIN_PER_GENDER) & (g["size_m"] >= MIN_PER_GENDER)]
    g["gap"] = g["mean_f"] - g["mean_m"]
    w = g["gap"].unstack("half").dropna()
    r = w[0].corr(w[1])
    rel = pd.DataFrame([{
        "Examiners (both halves large enough)": len(w),
        "Split-half correlation": round(r, 3),
        "Reliability (Spearman-Brown)": round(2 * r / (1 + r), 3) if r > -1 else np.nan,
        "Mean individual gap": round(pd.concat([w[0], w[1]]).mean(), 4),
        "SD of individual gap": round(((w[0] + w[1]) / 2).std(), 4),
    }])
    full = (d.groupby(["examiner_id", "female"])["granted_r"].agg(["mean", "size"]).unstack("female"))
    full.columns = [f"{a}_{'f' if b == 1 else 'm'}" for a, b in full.columns]
    full = full[(full["size_f"] >= 2 * MIN_PER_GENDER) & (full["size_m"] >= 2 * MIN_PER_GENDER)]
    full["gap"] = full["mean_f"] - full["mean_m"]
    return rel, anonymize_examiners(full.reset_index())


def anonymize_examiners(t: pd.DataFrame) -> pd.DataFrame:
    """Examiner-level estimates are noisy and must not be published under real names.

    Names are replaced by pseudonymous IDs (a fixed permutation of the sorted names, so the mapping can be rebuilt);
    the table and the name-to-ID map are stored only in data/processed/, which is not distributed.
    """
    names = pd.Series(sorted(t["examiner_id"].unique()))
    perm = np.random.default_rng(20261027).permutation(len(names))
    mapping = pd.DataFrame({"examiner_id": names, "examiner_anon": [f"E{i + 1:05d}" for i in perm]})
    mapping.to_parquet(P / "examiner_anon_map.parquet", index=False)
    return t.merge(mapping, on="examiner_id").drop(columns="examiner_id")[
        ["examiner_anon"] + [c for c in t.columns if c != "examiner_id"]]


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df = load()
    print("loaded", df.shape)
    ov = tc_overview(df)
    gaps = gender_gaps(df)
    fs, bal = first_stage_and_balance(df)
    rel, ex_gap = examiner_gender_gap_reliability(df)
    for name, t in [("tc_overview", ov), ("gender_gaps", gaps), ("first_stage", fs),
                    ("balance", bal), ("examiner_gap_reliability", rel)]:
        t.to_csv(OUT / "tables" / f"{name}.csv", index=name == "tc_overview")
    # examiner-level estimates stay with the private intermediate files: their IDs follow from the sorted names and a
    # fixed seed, so they are pseudonyms that can be undone, not anonymous identifiers
    ex_gap.to_csv(P / "examiner_gender_gap.csv", index=False)
    with open(OUT / "descriptives.md", "w") as f:
        f.write("# Descriptive statistics\n\nSample: PatEx utility applications filed 2004–2018; examiners with at least "
                "20 applications.\n\n")
        f.write("## 1. Overview by technology center\n\n" + ov.to_markdown() + "\n\n")
        f.write("## 2. Gender gaps (female first-named inventor; art unit × filing year fixed effects, clustered by examiner)\n\n"
                + gaps.round(4).to_markdown(index=False) + "\n\n")
        f.write("## 3. First stage of examiner leniency\n\n" + fs.round(4).to_markdown(index=False) + "\n\n")
        f.write("## 4. Random-assignment check (outcome: leave-one-out grant leniency)\n\n"
                + bal.round(5).to_markdown(index=False) + "\n\n")
        f.write("## 5. Reliability of examiner-level gender gaps\n\n" + rel.to_markdown(index=False) + "\n")
    print((OUT / "descriptives.md").read_text())


if __name__ == "__main__":
    main()
