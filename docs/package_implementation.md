# Package implementation

The numbered source files retain their established numbers and imported helpers
so references in the appendix can be followed directly. The public review
directory is an independent snapshot; it has no local data/model symlinks or Git
history. The entry-point and support roles are recorded in `code_inventory.csv`.

The package adds `reproduce.py` for verification, command planning, and staged
execution; source-cell specifications for the final 33 tables; and aggregate-data
reconstruction of all nine final figure selections. 

The C1 entry points in scripts 35, 43, and 47 default to Poisson loss. The runner
also sets the loss explicitly and provides a separate squared-error sensitivity
stage. Script 34's table generation omits the career-stage branch. Script 45's
entry point runs C2 only, with its single-threaded numerical fitting preserved.
Script 30 trains only the grant and citation targets used in the final appendix;
their target order and seeds are retained. Its input-construction helper remains
available as an imported dependency. Script 47 accepts stored plot summaries for
figure reconstruction; its estimation path is unchanged. The shared plotting
module caches converted fonts inside the repository's ignored cache directory.

The final table templates retain labels and layout while binding numerical cells
to generated results. Table A13 combines the score-validation and prior-art
checks; A18 includes within-field correlations; A30 retains the overall twin
summary; A31 retains three thresholded-decision rows; A32 combines name and
surname changes with the twin probability and hidden/shown comparisons. Removed
tables and panels do not appear in the final exhibit index.

The final proposal contributes a second exhibit index with Table 1 and its two
exact embedded graphics. Their source mapping is in `proposal_map.csv`; table
assembly and release verification cover both exhibit indexes.

## Changes made when preparing this release

None of the following changes any reported estimate.

- `code/05_descriptives.py` writes its examiner-level gender gaps to `data/processed/`, which is not
  distributed, instead of `output/tables/`. Their IDs are pseudonyms derived from the sorted names.
- `code/16_ai_examiner_inputs.py` accepts `--model s0|q8`. The workflow builds only the `s0` inputs of the
  reported twins and runs script 17 with `--features s0`; building both inputs made script 45 stop at its check.
- `code/45_twins_c4_extra.py` writes `c2x_groups.csv` without medians and without the gender gaps of the groups
  of five examiners, and `c2x_tab_examiners.tex` with only the pooled row used in Table A30. The distributed
  files were reduced in the same way, and the Table A30 bindings point to that row.
- Comments on hardware incidents and internal remarks were removed from scripts 08, 10, 14, 15, 16, 34, 45,
  and 46, and `code/cfe/__init__.py` no longer refers to a command-line front end that is not included.
- In `specification/workflow.json`, script 04b runs after `38 --prepare`, whose output it reads, and the C1
  stages run with `CFE_THREADS=16` and 16 BLAS/OpenMP threads, the setting of the reported run.
- Table rows follow the column grid of the Word tables: a cell spanning k columns is followed by k − 1 empty
  cells.
- The proposal captions and the SHA-256 values of the final Word files were updated, the multi-panel source
  graphics of Figures A1, A6, A8, and A9 were removed, the dependency URLs in `uv.lock` point to PyPI, and
  `.python-version` selects Python 3.13.
