# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`nercluster` finds likely annotation errors in CoNLL-formatted NER datasets. It embeds every
entity mention with a remote OpenAI-compatible embedding endpoint (e.g. self-hosted vLLM),
applies two-stage dimension reduction (PCA/SVD/KernelPCA, then UMAP or t-SNE), clusters with
HDBSCAN, and ranks mentions by silhouette score — low-silhouette mentions are the flagged
outliers. This implements the LAW XX paper "Clustering Analysis for Error Detection in Named
Entity Recognition Datasets" (see README for the citation and the full CLI option list).

## Commands

```shell
uv sync                      # set up the venv
uv run nercluster --help     # run the CLI
uv run pytest                # run the test suite
uv run pytest tests/test_clustering.py::test_pipeline_writes_a_complete_report  # single test
uv tool install .            # install `nercluster` globally from this checkout
```

Python is `>=3.12`, tested in CI on 3.12, 3.13 and 3.14. `.python-version` pins local dev to
3.12 — the floor, so day-to-day work doesn't silently adopt newer syntax — but nothing in the
build reads it. Dependencies are managed exclusively by `uv` (`uv.lock` is committed); do not
use pip.

To reproduce a CI job locally on another interpreter:

```shell
UV_PROJECT_ENVIRONMENT=/tmp/venv314 uv run --python 3.14 --frozen pytest
```

## Testing

The suite is hermetic and needs no dataset and no network. `tests/conftest.py` stubs
`EmbeddingClient.embed` — the pipeline's only outbound call — with deterministic vectors, and
everything downstream runs for real: PCA, UMAP/t-SNE and the numba JIT beneath them, HDBSCAN,
silhouette scoring, plotting and report serialization. That is what makes it usable as a
cross-version smoke check.

`tests/data/mini_ner.conll` is a small committed synthetic corpus (377 mentions, 4 entity
types, 37 marked `CORRECTED`); see `tests/data/README.md`. Its counts are asserted in
`test_data.py`, so editing it fails loudly rather than quietly weakening the pipeline tests.

Two things to keep in mind when adding tests:

- **Assert on artifacts, not on returning.** `Cluster.__call__` logs exceptions instead of
  raising, so a run that blew up still returns normally. Tests assert the report files exist
  and parse the JSON back through `Report`; a test that only calls `.cluster()` proves nothing.
- **The stub encodes the phenomenon under test.** Mentions flagged `CORRECTED` are drawn from
  a *neighbouring* entity type's cluster, which is what an annotation error looks like in
  embedding space. That is why the outlier ranking has real signal to recover instead of the
  suite only checking that nothing crashes.

`test_against_live_embedding_endpoint` is marked `integration` and skipped unless
`NERCLUSTER_TEST_BASE_URL` points at a real OpenAI-compatible endpoint.

## Architecture

The pipeline is a single pass, orchestrated top-down:

- `nercluster.py` — Typer CLI. Every knob is a CLI option that becomes a `Clusters` field.
  `Clusters` is imported *inside* the command body, deliberately.
- `data/data.py` — `Dataset.from_path(s)` wraps `seqscore.conll` ingestion into
  `Dataset → Document → Mention` Pydantic models. `Mention` carries four indices
  (`doc_idx`, `doc_local_idx`, `seq_idx`, `mention_idx`) so a flagged outlier can be traced
  back to its position in the source file, plus `corrected`, read from the CoNLL
  `other_fields` when the first extra column is `CORRECTED` — that flag is the gold signal
  the `num_corrected` metric scores against.
- `clients/embed.py` — `EmbeddingClient` wraps the `openai` SDK against `base_url`. When
  `--prompt` is set, mentions are wrapped via `_gen_prompt` (`--with-context` additionally
  injects the full sentence under a `### Context` heading). With no prompt, the raw mention
  span is sent. All mentions go in one `embeddings.create` call, with a 20-minute timeout.
- `clustering/cluster_eval.py` — the substance. `Clusters` embeds once, then fans out over
  the cartesian product of `--pca-components × --clustering-components` (both are repeatable
  CLI options), constructing one `Cluster` per combination with a **copy** of the embeddings,
  since each run mutates them in place. `Cluster.__call__` runs `_decompose → _cluster →
  _report`.
- `clustering/models.py` — `Settings` (one run's full config, with a random 8-hex `id`) and
  `Report`/`ReportMention`/`Summary` output schemas. `Settings.path()` builds the report
  path/filename stem, encoding dataset names, model, and every hyperparameter — that naming
  scheme is how runs are distinguished under `reports/`.

Each run writes, to `reports/<datasets>/<model-family>/<model>/<stem>`: `.json` (full
`Report` dump), `.md` (human summary + sklearn classification report), `.csv` (per-mention
silhouettes and HDBSCAN assignments), `_cm.png`, and — only with `--tsne`/`--umap` — 2D/3D
projection plots, each rendered twice (gold labels vs. HDBSCAN labels).

### Two distinct uses of UMAP/t-SNE

Easy to conflate: `--clustering-implementation` (with `--clustering-components`, default 16)
is the *analytical* reduction whose output feeds HDBSCAN and the silhouette scores. The
`--tsne`/`--umap` boolean flags are only for *visualization*, re-projecting to 2D/3D purely
for the plots. Note also that clustering-stage UMAP uses `min_dist=0.00` while the plotting
projection uses `min_dist=0.25`.

### Lazy imports are load-bearing

`clustering/__init__.py` implements module-level `__getattr__` so `Clusters`/`Settings`/
`Report` resolve lazily, and `nercluster.py` imports `Clusters` inside the command function.
This keeps the scientific stack (sklearn, umap, matplotlib) out of `nercluster --help`.
Keep new heavy imports behind this boundary — do not hoist them to module scope in
`__init__.py` or `nercluster.py`.

## Gotchas

- `Cluster.__call__` wraps the whole pipeline in `try/except` and only calls `logger.error`.
  Runs therefore "succeed" silently on failure — when a run produces no reports, look for an
  ERROR line in the log rather than a traceback.
- **Only five of the ten `METRIC` values survive the pipeline**: `euclidean`, `manhattan`,
  `chebyshev`, `minkowski`, `canberra` (the set `test_every_supported_metric_runs` covers).
  The chosen metric is applied twice — as UMAP's `output_metric` and again in
  `silhouette_samples` — and each of the rest fails one of those:
  - `hellinger` — scikit-learn's `silhouette_samples` rejects it.
  - `cosine`, `correlation` — degenerate on a UMAP output space centred on the origin.
  - `braycurtis` — defined for non-negative data.
  - `haversine`/`mahalanobis` — a **missing comma** in the `METRIC` literal
    (`clustering/models.py`) fuses these into one `"haversinemahalanobis"` string, which UMAP
    rejects. Fixing the comma changes the CLI's accepted values.
- When HDBSCAN finds no clusters at all, every point is noise, and `_cm` filters those out and
  hands `ConfusionMatrixDisplay` an empty array — `ValueError: zero-size array to reduction
  operation maximum`. Swallowed by the `try/except` above, so it shows up as a missing report.
- `--whiten` is plumbed all the way to `Clusters.whiten` but never reaches the PCA
  constructors; it currently does nothing.
- HDBSCAN hyperparameters (`min_cluster_size=25`, `min_samples=10`,
  `cluster_selection_epsilon=0.1`) and t-SNE `perplexity=30` are hardcoded in
  `cluster_eval.py`, not exposed as options. `min_cluster_size=25` is why a dataset needs to
  be reasonably large before anything clusters.
- Either `--pca-components` or `--pca-cumulative-variance` must be given; the CLI raises
  otherwise.
- Report generation reads `os.getenv("SLURM_JOB_ID"/"SLURM_JOB_NAME")` — these runs are
  expected to be launched on a Slurm cluster, and the fields are simply `None` locally.

## Conventions

Fully type-annotated, Pydantic models for all structured data, `loguru` for logging (every
stage logs a `_components_ref()` prefix identifying the PCA/clustering config), and
`# pyright: ignore` for the untyped sklearn/pandas edges.
