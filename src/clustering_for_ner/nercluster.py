from pathlib import Path
from typing import Annotated, Optional

import typer

from .clustering.models import CLUSTERING_IMPLEMENTATION, METRIC, PCA_IMPLEMENTATION

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def cluster(
    dataset_paths: Annotated[
        list[Path],
        typer.Argument(
            help="One or more paths to CoNLL-formatted BIO files. Will be combined into a single dataset before clustering analysis."
        ),
    ],
    model_id: Annotated[
        str, typer.Option(help="HuggingFace Model ID e.g. 'Qwen/Qwen3-32B'")
    ],
    base_url: Annotated[
        str,
        typer.Option(
            help="vLLM embedding model base url e.g. 'http://localhost:8000/v1'"
        ),
    ],
    metric: Annotated[
        METRIC, typer.Option(help="What metric to use with clustering algorithms")
    ],
    pca_implementation: Annotated[
        PCA_IMPLEMENTATION,
        typer.Option(help="Which PCA implementation to use with dimension reduction."),
    ] = "pca",
    pca_components: Annotated[
        Optional[list[int]],
        typer.Option(help="The number of components to reduce dimensions to."),
    ] = None,
    pca_cumulative_variance: Annotated[
        Optional[float],
        typer.Option(
            help="If dynamically identifying the number of PCA components, the cumulative variance to keep in determined dimensions."
        ),
    ] = None,
    clustering_implementation: Annotated[
        CLUSTERING_IMPLEMENTATION,
        typer.Option(help="Which PCA implementation to use with dimension reduction."),
    ] = "umap",
    clustering_components: Annotated[
        list[int],
        typer.Option(help="The number of components to reduce dimensions to."),
    ] = [16],
    cluster_neighbors: Annotated[
        int,
        typer.Option(
            help="The number of neighbors when running the clustering algorithm for visual projections"
        ),
    ] = 50,
    tsne: Annotated[
        bool, typer.Option(help="Whether to plot projections with t-SNE")
    ] = False,
    umap: Annotated[
        bool, typer.Option(help="Whether to plot projections with UMAP")
    ] = False,
    top_k: Annotated[int, typer.Option(help="Set top_k value")] = 10,
    whiten: Annotated[
        bool,
        typer.Option(
            help="Whether to 'whiten' the values to scale components before applying PCA"
        ),
    ] = True,
    random_state: Annotated[
        int,
        typer.Option(
            help="Random state for dimension reduction and clustering algorithms"
        ),
    ] = 42,
    prompt: Annotated[
        Optional[str],
        typer.Option(
            help="The prompt to wrap the mention in for embedding, if any. Some modern embedding models, especially LLM-based ones, expect a prompt along with the text to embed."
        ),
    ] = None,
    with_context: Annotated[
        bool,
        typer.Option(
            help="Whether to include the context of the mention when embedding"
        ),
    ] = False,
    include_labels: Annotated[
        bool,
        typer.Option(
            help="Whether to train the UMAP dimension reduction in a supervised fashion using the labels"
        ),
    ] = False,
    api_key: Annotated[
        str,
        typer.Option(
            help="API key if required by embedding endpoint. Not required if a locally or self-hosted model."
        ),
    ] = "LOCAL",
    reports_dir: Annotated[
        Path,
        typer.Option(
            help="Path to save reports in. Defaults to 'reports' and creates directory if it does not already exist."
        ),
    ] = Path("reports"),
) -> None:
    from .clustering import Clusters

    if not pca_components and not pca_cumulative_variance:
        raise ValueError(
            "pca_cumulative_variance must be set if dynamically determining components"
        )

    Clusters(
        model_id=model_id,
        base_url=base_url,
        dataset_paths=dataset_paths,
        pca_implementation=pca_implementation,
        pca_components=pca_components,
        pca_cumulative_variance=pca_cumulative_variance,
        clustering_implementation=clustering_implementation,
        clustering_components=clustering_components,
        cluster_neighbors=cluster_neighbors,
        top_k=top_k,
        whiten=whiten,
        random_state=random_state,
        api_key=api_key,
        reports_dir=reports_dir,
        tsne=tsne,
        umap=umap,
        prompt=prompt,
        with_context=with_context,
        include_labels=include_labels,
        metric=metric,
    ).cluster()


def cli() -> None:
    app()
