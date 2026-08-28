# Test fixtures

## `mini_ner.conll`

A small synthetic CoNLL-2003-style BIO corpus, written so the test suite
runs from a clean clone without needing a real annotated dataset.

Columns are `token <flag> label`, space-delimited. The middle column is the
extra field the library reads to detect gold corrections: `CORRECTED`
marks a mention that a human annotator later fixed, and `-X-` (seqscore's
empty-field placeholder) is used everywhere else. Documents are separated
by `-DOCSTART-` lines, which the ingester attends to.

Shape (asserted in `test_data.py`, so edits that change these counts fail
loudly rather than quietly weakening the pipeline tests):

| | |
|---|---|
| sequences | 190 |
| documents | 6 |
| mentions | 377 |
| entity types | `LOC`, `MISC`, `ORG`, `PER` |
| `CORRECTED` mentions | 37 |

Sized so the pipeline's hardcoded HDBSCAN settings (`min_cluster_size=25`)
and the default `--cluster-neighbors` have enough points to work with,
while staying small enough to keep the suite quick.

The text is generated from sentence templates and invented entity names.
It contains no real-world annotation and is not a benchmark — it exists to
exercise code paths, not to measure quality.
