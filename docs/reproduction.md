# Reproducing the final appendix

The review package separates checking the submitted results from re-estimating
them. `python reproduce.py verify` uses only the Python standard library and the
distributed files. `python reproduce.py figures` uses the main statistical
environment and aggregate plot data. A complete re-estimation additionally
requires public patent data, model weights, and GPU computation.

## Environments

The main environment is specified by `pyproject.toml` and `uv.lock`. The recorded
run used Python 3.13.9, which `.python-version` selects. Install it with `uv sync --frozen`.
Python 3.12 or later is required, including for `python reproduce.py verify`. The lock file resolves
the statistical dependencies, including direct dependencies on scipy, joblib,
threadpoolctl, and fonttools. Pandoc is needed only to refresh numerical source
cells from regenerated LaTeX tables. It is not needed to inspect or verify the
submitted exhibits. Nimbus Sans from the URW base-35 fonts is optional; plotting
falls back to DejaVu Sans when the font is unavailable.

The LLM environment uses Python 3.12, vLLM 0.32.0, torch 2.13.0,
transformers 5.19.0, peft 0.21.2, and accelerate 1.15.0. Recorded package versions
are in [environment_versions.json](environment_versions.json). These versions
describe the producing environment, rather than a guarantee that every GPU,
CUDA driver, or operating system can run the same stack.

```bash
uv venv .venv-llm --python 3.12
uv pip install --python .venv-llm/bin/python -r requirements-llm.txt
```

Use `.venv-llm/bin/python` for GPU tasks and the main environment for statistical
analysis. The runner accepts `--llm-python PATH` for a different LLM environment.

## Inputs and resources

Create a `data/` directory or symlink with the layout in [data.md](data.md).
Place Qwen3-Embedding-0.6B and Qwen3-8B under `models/`. Neither is included.
The full raw inputs occupy roughly 100 GB; intermediate tables, embeddings,
and fitted models require additional space. The data guide also describes the
Census surname file, which must be supplied before the ethnicity and surname
analyses. Figure A1's small public geometry file is included with its notice.

The runner sets BLAS/OpenMP threads to one and defaults to `CFE_THREADS=4`, except in the C1 stages
(`c1` and `c1-squared`), which run with `CFE_THREADS=16` and 16 BLAS/OpenMP threads as in the reported run.
As appendix A8 notes, a different thread count can change the hyperparameters selected by cross-validation
and slightly change the C1 estimates.
Use at most 16 CPU cores for heavy jobs, for example with Linux `taskset`, and
run GPU tasks separately from heavy CPU work. Some data-construction scripts
set DuckDB memory limits of 120–200 GB and thread counts internally; inspect
these settings before running on a smaller machine. The recorded LLM run uses
GPUs `0,1`; the membership test and name-embedding tasks use one GPU. GPU layout,
library versions, and numerical settings can affect exact model replication.

## Stage order

`python reproduce.py plan` prints every command without executing it. Each
stage can be run as `uv run --frozen python reproduce.py run STAGE`; add
`--dry-run` to inspect just that stage. Stages stop on the first failed command.

| Order | Stage | Main products |
| --- | --- | --- |
| 1 | `prepare` | Application records, leniency, pilot and all-center samples, citations, texts |
| 2 | `embed` | Pilot and all-center text vectors |
| 3 | `pilot` | Pilot cross-fitted inputs, zero-shot LLM, Qwen3-8B representations, fine-tuned scores |
| 4 | `all-centers` | All-center evaluators, zero-shot LLM, surname groups |
| 5 | `c1` | Main Poisson C1, counterfactual scores, accuracy and group checks, prediction tails |
| 6 | `c1-squared` | Squared-error sensitivity retained in the final prose |
| 7 | `diagnostics` | Assignment, balance, first stages, exclusion, familiarity, prior-art diagnostics |
| 8 | `groups` | Gender and ethnicity gaps, probability comparisons, temporal and field checks |
| 9 | `twins` | C2 twins and aggregate decision-change results |
| 10 | `names` | C3 name and surname interventions |
| 11 | `tables` | Formatted numerical sources and final appendix table selections |

Then run `uv run --frozen python reproduce.py figures` to reconstruct the nine
appendix figures and both proposal figures. No GPU is involved in figure reconstruction.

The pipeline retains support calculations needed to construct inputs shared
by several analyses. Their additional diagnostic columns do not define the
final appendix's reported estimators. In particular, `code/cfe/` is a shared
dependency; C1 uses the quintile comparisons in script 35, not its legacy 2SLS
check. Script 24's citation builder is called through `tools/data_tasks.py`
without running its separate pilot estimation report.

## Final table and figure selections

`specification/appendix_layout.json` stores the final labels, row/column
selection, and explicit references for 972 numerical result cells. Bindings
identify the generated LaTeX source, source row label, column, and numerical
token. `tools/supplementary_sources.py` handles the few cells merged from
different outputs, including the prior-art and marginal-score checks,
within-field rank correlations, and names-hidden/shown comparisons. It reads
generated macros or CSVs; it does not estimate new effects.

`specification/source_cells.json` is a parsed snapshot of those sources with
their hashes. Refresh it with Pandoc after re-estimation:

```bash
python reproduce.py refresh
python tools/assemble_appendix.py --check
```

The second command compares the rebuilt tables to the author-approved reference
cell for cell and fails if estimates or table content differ. Refreshing an
analysis does not rewrite `reference/appendix_tables.json` or the release manifest.
The final table layout omits the removed tables and rows and preserves the
author's merged tables. All source values remain inspectable in `output/`.

The nine exact Word-embedded graphics are distributed as the reference images.
Reconstruction uses their final panel selections: the within-cell map for A1,
the lenient quintile's model by quintile and the contraction for A6, decision disagreement for A8, and threshold
zones for A9. Reconstructed PNG/PDF files are separate from the reference images.

## Validation of this package

The release was assembled from existing analysis results. The final 33 tables
were rebuilt from their numerical sources and compared with the final Word
appendix. All nine appendix and both proposal figure reconstructions were run from the supplied aggregate
CSVs. Exhibit correspondence, file integrity, Python syntax, numbered-script
dependencies, and command-plan generation were checked. The raw-data pipeline
and GPU training were not rerun when preparing this package.

The checks verify that this snapshot, its sources, and the final appendix agree.
They do not constitute an independent re-estimation from raw patent data.

For this release, the 33 appendix tables (1,680 cells) and proposal Table 1 (43 cells) were also compared
cell by cell with the final Word documents, all 45 table and figure captions were compared with them, and
the 11 graphics were compared byte for byte with the images embedded in them; all matched. The SHA-256
values of the two Word files are recorded in `reference/`.

The proposal exhibit sources and their appendix counterparts are listed in
`docs/proposal_map.csv`. Proposal Table 1 uses the same numerical source cells
as Appendix Table A19. F − M denotes female minus male throughout the code
and result tables. The proposal and appendix Word files are not distributed.
