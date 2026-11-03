# The Paths Less Traveled: Analysis Code and Results

This repository accompanies the proposal *The Paths Less Traveled: Using AI to
Approximate Counterfactual Outcomes and Decisions* and its technical appendix.
It contains the analysis code, aggregate results, the appendix's **33 tables and
9 figures**, and the proposal's **Table 1 and two figures**. The main sample
comprises 443,742 patent applications filed in 2010–2014 across eight utility
technology centers.

## Start with the final exhibits

Open [results/appendix/index.html](results/appendix/index.html) locally for the
final tables, captions, and embedded manuscript graphics. Individual tables are
CSV files in that directory; figures are in its `figures/` subdirectory.
The [proposal exhibit index](results/proposal/index.html) contains the final
proposal's table and graphics, with their [source map](docs/proposal_map.csv).

- [Appendix map](docs/appendix_map.csv): every final exhibit, its scripts and
  analysis sources, and the rows, columns, or panels retained.
- [Results reported in prose](docs/text_results_map.csv): sources for numerical
  claims that appear only in the text.
- [Reproduction guide](docs/reproduction.md): environments, inputs, stage order,
  and the scope of the checks performed for this package.
- [Data documentation](docs/data.md): public sources and expected input layout.
- [Code inventory](docs/code_inventory.csv): analysis scripts and imported helpers.

## Review without raw data

Python 3.12 or later is required (the recorded run used Python 3.13.9;
`.python-version` selects Python 3.13 for uv). The integrity and exhibit checks
use only the Python standard library:

```bash
python reproduce.py verify
python reproduce.py assemble
```

The verification checks every final table against the final-appendix snapshot,
972 numerical result-cell bindings, the nine manuscript image hashes, Python
syntax and numbered-script dependencies, and the distributed-file checksums.
The final proposal's 16 numerical result cells and both image hashes are checked
as well. Table assembly reads the supplied result cells; it performs no estimation.

All nine appendix figures and both proposal figures can be redrawn from the
supplied aggregate CSVs:

```bash
uv sync --frozen
uv run --frozen python reproduce.py figures
```

Redraws are written to `results/rebuilt/figures/` as PNG and PDF. They retain the
final appendix's panel selection; fonts and spacing can differ between systems.
The exact manuscript graphics remain in `results/appendix/figures/`.
Proposal redraws are written to `results/rebuilt/proposal/`.

## Rebuild from public data

Raw data and model weights are acquired separately. Follow [docs/data.md](docs/data.md)
and [docs/reproduction.md](docs/reproduction.md), then inspect the complete command
plan before running any stage:

```bash
python reproduce.py plan
uv run --frozen python reproduce.py run c1 --dry-run
uv run --frozen python reproduce.py run c1
```

`run c1` requires the prepared data, embeddings, and model scores listed in the
guide. Stages run sequentially; GPU tasks use the separate `.venv-llm` Python.
The runner explicitly sets `CFE_SAMPLE=all` and `CFE_LOSS=poisson`; a separate
stage reproduces the squared-error sensitivity reported in the text. The C1
stages (`c1` and `c1-squared`) run with `CFE_THREADS=16` and 16 BLAS/OpenMP
threads, the setting of the reported run. As appendix A8 notes, a different
thread count can change the hyperparameters selected by cross-validation and
slightly change the C1 estimates.

After regenerating the analysis outputs, `python reproduce.py refresh` parses
the updated LaTeX result cells with Pandoc and assembles the final table layout.
The original review snapshot and its checksums remain the reference for
comparison. Regenerated estimates that differ should be inspected rather than
substituted silently.

## Data availability

All inputs are public and free to download: the Harvard USPTO Patent Dataset
(HUPD), the USPTO Patent Examination Research Dataset (PatEx), PatentsView
granted-patent data, the USPTO Office Action Research Dataset, the USPTO patent
maintenance-fee events, the WIPO World Gender-Name Dictionary 2.0, and the 2010
Census surname table. They are not redistributed here because of their size
(about 100 GB) and their own license terms. [docs/data.md](docs/data.md) gives
the source, release, and expected location of each. The two open-weight models,
Qwen3-Embedding-0.6B and Qwen3-8B (Apache-2.0), are available from Hugging Face.
No proprietary or restricted data are used.

## Computational requirements

As reported in appendix A8, text embeddings take about one hour on two GPUs,
zero-shot inference with Qwen3-8B less than one hour on two GPUs, fine-tuning in
TC 3600 about two hours, and the C1 evaluation about one hour on 16 CPU cores.
Most remaining analyses run in minutes. Some data-construction scripts set
DuckDB memory limits of 120–200 GB (see the reproduction guide). The package
checks and the figure reconstruction run on an ordinary laptop.

## Application-level counterfactual scores

`output/tables/c1_counterfactual_scores_all_poisson.csv` has one row for each of
the 146,149 rejected applications in the main sample.

| Column | Contents |
| --- | --- |
| `application_number` | Public USPTO application number |
| `tech_center`, `filing_year` | Technology center and filing year |
| `part` | Sample partition: `train`, `impute`, `test`, or `lockbox` (the appendix's held-out sample) |
| `cf_cites` | Predicted five-year post-grant citations, excluding self-citations, had the application been granted; out of sample (the imputation model, or the training model for applications in the imputation set) |
| `cf_cites_alpha2` | `cf_cites` divided by 2, the bound of Kleinberg et al. (2018) with α = 2 |
| `cf_percentile_among_granted` | Percentile of `cf_cites` among the granted applications of the same technology center and filing year |
| `cf_cites_with_prior_art`, `cf_with_prior_art_percentile_among_granted` | The same for a score that also uses the application's prior-art citations |
| `prior_art_cites` | Observed prior-art citations: later applications citing this application, excluding self-citations |
| `prior_art_cites_examiner` | The part of `prior_art_cites` added by examiners (published applications only) |

The scores are model predictions, not observed outcomes. A single application's
score should be read as a range rather than a point estimate (appendix Table A13
shows how far scores fall from the outcome at the margin), and a score is not
evidence that an examiner erred in deciding that application. The file contains
no inventor names and no inferred gender, race, or ethnicity. Validation
statistics in the appendix exclude the held-out partition where stated.

## Research ethics and privacy

Examiner identities are public in PatEx, but examiner-specific estimates are
imprecise and attributable to identifiable public employees. Results are
therefore reported in aggregate. The package contains no examiner names or
identifiers, no examiner-level estimates, and no examiner-specific twin models;
the examiner name-to-ID map written by script 05 is not distributed. Inventor
gender and race or ethnicity are inferred from names, are not self-identified,
and are used only for group-level comparisons. The analysis uses public records
only.

## Repository contents

| Directory | Contents |
| --- | --- |
| `code/` | Numbered data and analysis scripts and shared statistical utilities |
| `tools/` | Review checks, final table assembly, figure reconstruction, and source-document checking |
| `output/tables/` | Underlying aggregate analysis results and the counterfactual score export |
| `output/latex/` | Generated table bodies and number macros used as numerical sources |
| `output/figures/journal/` | Analysis graphics used unchanged in the final exhibits (see below) |
| `specification/` | Reproduction stages and explicit bindings from final cells to their numerical sources |
| `reference/` | Final table and caption snapshots, figure hashes, and release provenance |
| `results/appendix/` | Final numbered exhibits and a local HTML index |
| `results/proposal/` | Final proposal Table 1, two graphics, and a local HTML index |
| `resources/` | Public U.S. state geometry used to reconstruct Figure A1 |

The multi-panel source graphics of Figures A1, A6, A8, and A9, of which the final
exhibits keep only some panels, are written by their scripts but not distributed;
`docs/appendix_map.csv` records the panels kept.

The shared `cfe` utilities and some support modules contain pilot estimators.
The final appendix's C1 and examiner outcome tests use scripts 35 and 39,
respectively. The code inventory distinguishes entry points from imported
dependencies; the numbered filenames are retained so manuscript references
remain traceable. In file names, `q8` denotes inputs built from Qwen3-8B
representations, and `lockbox` denotes the held-out sample.

Private drafts, author correspondence, raw data, model weights, local paths,
Git history, and examiner-name mappings are excluded. The proposal and appendix
Word files are supplied to the journal separately; their SHA-256 values are
recorded in `reference/final_proposal.json` and `reference/release_origin.json`.
