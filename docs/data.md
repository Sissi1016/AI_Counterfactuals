# Data

All inputs are public. They are not redistributed here (about 100 GB in total). Place them under `data/` at the
repository root (a directory or a symlink; `data` is git-ignored) using the layout below. The scripts read the raw
files from these paths and write all intermediate files to `data/processed/`.

## Sources

| Directory | Dataset | Coverage used here | Source | Read by |
|---|---|---|---|---|
| `hupd/` | Harvard USPTO Patent Dataset (HUPD; Suzgun et al. 2022, arXiv:2207.04043), CC BY-SA 4.0 | ~4.5M utility applications filed 2004–2018 (2018 partial), with full text and final decision | <https://huggingface.co/datasets/HUPD/hupd> | 02, 07 |
| `patex/` | USPTO Patent Examination Research Dataset (PatEx), 2022 release (product ECOPAIR) | ~14M applications; examiner, art unit, and prosecution events through June 2023 | <https://www.uspto.gov/ip-policy/economic-research/research-datasets/patent-examination-research-dataset-public-pair> | 01 |
| `patentsview/` | PatentsView granted-patent disambiguated data (product PVGPATDIS) | 9.45M granted patents and 153M patent-to-patent citations, through 2025-12-31 (released 2026-04-10) | <https://data.uspto.gov/bulkdata/datasets/pvgpatdis> | 06, 12 |
| `maintenance_fee/` | USPTO Patent Maintenance Fee Events (product PTMNFEE2) | 27.5M fee events for patents granted since 1981-09-01; file of 2026-10-26 | <https://data.uspto.gov/bulkdata/datasets/ptmnfee2> | 06 |
| `wgnd/` | WIPO World Gender-Name Dictionary 2.0 (WGND 2.0) | name–gender shares by country, plus a country-free table | Harvard Dataverse, <https://doi.org/10.7910/DVN/MSEGSJ> | 03 |
| `processed/census_surnames_2010.parquet` | U.S. Census Bureau, Frequently Occurring Surnames from the 2010 Census | 162,253 surnames with at least 100 bearers; shares White, Black, Asian/Pacific Islander, Hispanic, and other | <https://www.census.gov/topics/population/genealogy/data/2010_surnames.html>; we use the copy shipped as `ethnicolr/data/census/census_2010.parquet` in the `ethnicolr` 2.0.0 wheel on PyPI (unzip the wheel; the package need not be installed) | 33 |
| `processed/us_states_albers_10m.json` | U.S. state boundaries, Albers projection with Alaska and Hawaii inset (us-atlas 3.0.1, ISC license, from the U.S. Census Bureau's cartographic boundaries) | 50 states and DC | `states-albers-10m.json` from <https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/> | 36 |
| `office_actions/` | USPTO Office Action Research Dataset (product PTOFFACT), final 2017 release | office actions mailed 2008 to June 2017; `citations.csv` (prior art cited by examiners and applicants) | <https://data.uspto.gov/bulkdata/datasets/ptoffact> | 06c |

## Expected layout

```
data/
├── hupd/
│   ├── hupd_metadata_2022-02-22.feather   # metadata for all applications, no text (~1 GB)
│   └── data/{2004..2018}.tar.gz           # one JSON per application, named by application number (62 GB)
├── patex/csv/                             # the five tables used, from PatEx_2022_ECOPAIR_csv.zip (18 GB)
│   ├── application_data.csv
│   ├── transactions.csv
│   ├── all_inventors.csv
│   ├── all_applicants.csv
│   └── event_codes.csv
├── patentsview/tsv/                       # unzipped g_patent.tsv.zip and g_us_patent_citation.tsv.zip (12 GB)
│   ├── g_patent.tsv
│   └── g_us_patent_citation.tsv
├── maintenance_fee/txt/
│   └── MaintFeeEvents_YYYYMMDD.txt        # 06 uses the latest file matching MaintFeeEvents_2*.txt
├── wgnd/
│   ├── wgnd_2_0_name-gender-code.tab
│   └── wgnd_2_0_name-gender_nocode.tab
└── processed/                             # written by the scripts (about 4 GB)
```

## Notes by dataset

**HUPD.** The metadata table supplies `decision` (`ACCEPTED`, `REJECTED`, `PENDING`; continuations carry a `CONT-`
prefix) and the main CPC label. From each application's JSON, 07 keeps the title, abstract, claims, background and
summary (first 4,000 characters each), and the inventor list. Because a tarball's year can differ from the filing
year, 07 scans the 2009–2015 tarballs for the 2010–2014 pilot. Examiners are taken from PatEx, not from HUPD.

**PatEx.** All fields are read as strings to keep leading zeros in application numbers. PatEx has no examiner ID,
so examiners are identified by `examiner_full_name`; the art unit is `examiner_art_unit`. Event codes used from
`transactions.csv`: `CTNF` (non-final rejection), `CTFR` (final rejection), `MN/=.` (notice of allowance), `CTEQ`
(Quayle action), `CTRS` (restriction requirement), `RCEX` (request for continued examination), and `AP.B` (appeal).
Field documentation: <https://www.uspto.gov/sites/default/files/documents/PatEx-2019-Technical-Doc.pdf>.

**PatentsView.** `g_us_patent_citation.tsv` has `citation_category`, which separates citations added by the
examiner (`cited by examiner`) from those submitted by the applicant (`cited by applicant`). 06 counts citations
from patents granted within five years of the cited patent's grant.

**Maintenance fees.** The file is cumulative, so only the latest one is needed. Each line is space-separated text
with Windows line endings: patent number (13 digits, zero-padded), application number, small-entity flag, filing
date, grant date, event date, event code. 06 marks a patent as renewed at year 4 if any of the event codes
`M1551`, `M2551`, `M3551`, or `F170` appears.

**WGND 2.0.** A gender is assigned when its share for the name in the inventor's country is at least 0.9; names not
listed for that country fall back to the country-free table. 03 checks the size of the country-free file
(56,962,284 bytes) and skips the fallback if the download is incomplete.

## Models

Qwen3-Embedding-0.6B (step 08) and Qwen3-8B (step 10), both Apache-2.0, available from Hugging Face
(`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-8B`) or ModelScope. Place them in `models/Qwen3-Embedding-0.6B/` and
`models/Qwen3-8B/` at the repository root (`models` is git-ignored; a symlink works), or change `MODEL` at the top of
`code/08_embed.py` and `code/10_llm_prompting.py`.

## Intermediate files (`data/processed/`)

| File | Written by | Contents |
|---|---|---|
| `patex_*.parquet` | 01 | PatEx tables as Parquet |
| `applications_2004_2018.parquet` | 02 | application-level examiner, art unit, first action, rejections, grant |
| `inventor_gender.parquet`, `app_gender.parquet` | 03 | inventor-level and application-level gender |
| `applications_leniency.parquet`, `examiners.parquet` | 04 | leave-one-out leniency; examiner summaries |
| `examiner_anon_map.parquet`, `examiner_gender_gap.csv` | 05 | examiner name-to-ID map and examiner-level gender gaps (keep private; see below) |
| `pilot_sample.parquet` | 06 | pilot sample with citation and renewal outcomes |
| `pilot_text.parquet` | 07 | pilot texts |
| `pilot_inputs.parquet`, `emb/{s0,s1,s1_f,s1_m}.npy` | 08 | model inputs, train/test split, embeddings |
| `probe_preds_{3600,1600}.parquet` | 09 | per-application probe predictions on the test set |
| `llm_prompts{TAG}.parquet`, `llm_sample{TAG}.parquet`, `llm_scores{TAG}.parquet` | 10 | prompts, sample, and Yes/No probabilities |
| `crossfit_preds.parquet` | 12 | cross-fitted model scores and predicted outcomes for every pilot application |
| `cites_noself.parquet` | 06b | five-year citations of granted pilot patents, all and without self-citations |
| `prior_art_cites.parquet` | 06c | prior-art citations of every pilot and all-center application: later applications citing its publication or patent, excluding self-citations |
| `llm_scores_allcenters.parquet` | 40 | Qwen3-8B zero-shot P(Yes), P(No) for the applications of the eight centers outside the pilot (the pilot's are in `llm_scores_allpilot.parquet`) |
| `cites_noself_all.parquet` | 24 | the same for all granted 2004–2014 applications of the eight centers |
| `citation_scores.parquet` | 21 | evaluator scores used for the citation check (incl. the citation-count models) |
| `memorization.parquet` | 23 | Min-20% Prob of title and abstract under Qwen3-8B, for a random 30,000 applications |
| `ft_data.parquet`, `ft_preds.parquet` | 30 | training table and cross-fitted scores of the fine-tuned LLM evaluators |
| `all_sample.parquet`, `all_text.parquet`, `emb/{all_s0,all_s1}.npy`, `all_scores.parquet` | 38 | sample, texts, vectors, and cross-fitted scores of all eight centers (2010–2014) |
| `twins_names_inputs.parquet`, `emb/{ai_ex_s1,common_s1_f,common_s1_m}.npy` | 32 | AI twins' inputs with inventor names, and the common applications with a female or male first name |
| `llm_prompts_allpilot.parquet`, `llm_scores_allpilot.parquet` | 14 | Qwen3-8B prompts and Yes/No probabilities for every pilot application |

## Examiner privacy

Examiner names are public in PatEx, but examiner-level estimates are noisy. The distributed results contain no examiner
identifiers. Examiner-level intermediate files, including script 05's examiner
gender gaps and the name-to-ID map, are written to `data/processed/` and are not distributed. Their IDs are
pseudonyms derived from the sorted names, not anonymous identifiers.
