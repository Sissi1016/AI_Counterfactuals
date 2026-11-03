# Result files and provenance

The numbered tables in `results/appendix/` follow the final appendix exactly.
They include the author's merged and shortened tables. Their CSV rows preserve
the manuscript's multirow headings and panel labels, so these files are intended for review and typesetting rather than as rectangular estimation datasets.
Rows follow the column grid of the Word tables: a cell that spans k columns in the manuscript is
followed by k − 1 empty cells.
`results/appendix/tables.json` preserves the same rows in structured form.

For analysis, use the machine-readable result tables in `output/tables/`.
`docs/appendix_map.csv` links each final exhibit to its scripts and generated
table sources. `docs/text_results_map.csv` identifies the results retained only
in prose, including the Qwen3-8B representation and blended-target comparisons.
Additional intermediate columns and rows support the shared analysis pipeline;
their presence does not make them results reported in the final appendix.

The main individual-score export is
`output/tables/c1_counterfactual_scores_all_poisson.csv`. It contains public
application numbers, technology center, filing year, partition, counterfactual
citation scores, sensitivity scores, percentiles among granted applications,
and observed prior-art citations. It contains predictions, not observed counterfactual outcomes. The partition labelled
`lockbox` is the appendix's held-out sample. A single score should be read as a range
rather than a point estimate (appendix Table A13), and a score is not evidence that an
examiner erred in deciding that application. The file carries no inferred gender, race,
or ethnicity. The export retains all partitions of the source run;
the validation statistics in the final appendix apply their stated sample
restrictions, including exclusion of the held-out partition where specified.

Examiner-level name mappings, individual twin tables, training text, raw data,
and model weights are not distributed. Aggregate twin summaries are sufficient
to reconstruct the submitted C2 figures and tables.

`reference/appendix_tables.json` and the nine image hashes are extracted from
the author-approved Word appendix. That document is not copied into this
repository. Its SHA-256 is recorded in
`reference/release_origin.json`. The assembled tables draw on numerical source
cells; the reference table values are used for comparison, not as a substitute
for generating those cells.
