# Clustering for NER

The `nercluster` utility runs cluster analysis on CoNLL-formatted Named Entity
Recognition datasets using dimension-reduced dense sentence embeddings. Using
silhouette scores, it detects outliers that are likely annotation errors.

This approach is detailed in our paper:

["Clustering Analysis for Error Detection in Named Entity Recognition Datasets."](https://aclanthology.org/2026.law-main.17/)

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/).
First, Ensure `uv` is installed.

### As a uv tool

The recommended way of installing `nercluster` is as a `uv tool`:

```
uv tool install git+https://github.com/bltlab/clustering-for-ner
```

This makes `nercluster` available globally on the command line in a uv-managed venv.

### Cloning

Alternatively, this GitHub repo can be cloned and manually managed.

First, clone this repo, and then `uv sync` to configure the dependencies in a venv.
The tool can then be run as `uv run nercluster` when in the project root, or simply as
`nercluster` if the venv is active.

As another option, in the project root, run `uv tool install .`
to make the `nercluster` command globally available from the cloned local repo.

## Usage

The cli command is called `nercluster`. Run it in the root directory
and uv-managed venv by using `uv run nercluster`.

If it was installed as a `uv tool`, run it directly with `nercluster`.

```shell
$ nercluster --help

Usage: nercluster [OPTIONS] DATASET_PATHS...

Arguments:
  DATASET_PATHS...  One or more paths to CoNLL-formatted BIO files. Will be
                    combined into a single dataset before clustering analysis.
                    [required]

Options:
  --model-id TEXT                 HuggingFace Model ID e.g. 'Qwen/Qwen3-32B'
                                  [required]
  --base-url TEXT                 vLLM embedding model base url e.g.
                                  'http://localhost:8000/v1'  [required]
  --metric [euclidean|manhattan|chebyshev|minkowski|canberra|braycurtis|haversine
            mahalanobis|cosine|correlation|hellinger]
                                  What metric to use with clustering
                                  algorithms  [required]
  --pca-implementation [pca|truncated|kernel]
                                  Which PCA implementation to use with
                                  dimension reduction.  [default: pca]
  --pca-components INTEGER        The number of components to reduce
                                  dimensions to.
  --pca-cumulative-variance FLOAT
                                  If dynamically identifying the number of PCA
                                  components, the cumulative variance to keep
                                  in determined dimensions.
  --clustering-implementation [tsne|umap]
                                  Which PCA implementation to use with
                                  dimension reduction.  [default: umap]
  --clustering-components INTEGER
                                  The number of components to reduce
                                  dimensions to.  [default: 16]
  --cluster-neighbors INTEGER     The number of neighbors when running the
                                  clustering algorithm for visual projections
                                  [default: 50]
  --tsne / --no-tsne              Whether to plot projections with t-SNE
                                  [default: no-tsne]
  --umap / --no-umap              Whether to plot projections with UMAP
                                  [default: no-umap]
  --top-k INTEGER                 Set top_k value  [default: 10]
  --whiten / --no-whiten          Whether to 'whiten' the values to scale
                                  components before applying PCA  [default:
                                  whiten]
  --random-state INTEGER          Random state for dimension reduction and
                                  clustering algorithms  [default: 42]
  --prompt TEXT                   The prompt to wrap the mention in for
                                  embedding, if any. Some modern embedding
                                  models, especially LLM-based ones, expect a
                                  prompt along with the text to embed.
  --with-context / --no-with-context
                                  Whether to include the context of the
                                  mention when embedding  [default: no-with-
                                  context]
  --include-labels / --no-include-labels
                                  Whether to train the UMAP dimension
                                  reduction in a supervised fashion using the
                                  labels  [default: no-include-labels]
  --api-key TEXT                  API key if required by embedding endpoint.
                                  Not required if a locally or self-hosted
                                  model.  [default: LOCAL]
  --reports-dir PATH              Path to save reports in. Defaults to
                                  'reports' and creates directory if it does
                                  not already exist.  [default: reports]
  --install-completion            Install completion for the current shell.
  --show-completion               Show completion for the current shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.```
```

Where `DATASET_PATHS` are absolute file paths CoNLL-formatted dataset files.
Can be 1 or more files. If more than one, they will be be combined into a
single dataset before clustering analysis. Useful if the clustering
should be performed on all splits of a dataset together.

## Citation

If this work is helpful for your research, please give us a cite:

```shell
@inproceedings{flynn-etal-2026-clustering,
    title = "Clustering Analysis for Error Detection in Named Entity Recognition
                Datasets",
    author = "Flynn, Matthew  and
      Obiso, Timothy  and
      Newman, Sam  and
      Lignos, Constantine",
    editor = "Liu, Yang Janet  and
      Gessler, Luke",
    booktitle = "Proceedings of the 20th Linguistic Annotation Workshop ({LAW} {XX})",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.law-main.17/",
    doi = "10.18653/v1/2026.law-main.17",
    pages = "229--240",
    ISBN = "979-8-89176-404-0",
    abstract = "This paper introduces a method for the automatic detection of
            annotation errors and corrections in named entity recognition
            datasets using a novel two-stage dimension reduction of dense
            sentence embeddings. We first find the top-\textit{n} principal
            components of an embedding and then use UMAP for second-stage,
            non-linear dimension reduction and clustering using different
            distance metrics. We analyze these clusters using silhouette scores
            to flag outlier mentions for correction. Using the corrections in
            the CoNLL{\#} dataset as a benchmark, all of the top-five outliers
            needed correction, as did 7 of the top-10. This approach also
            identified 32 of the top-50 outlier mentions that are corrections.
            This method offers a relatively low-effort way to leverage text
            embeddings and dimensionality reduction to identify likely
            annotation errors. We release related code and data at
            \url{https://github.com/bltlab/clustering-for-ner}."
}
```
