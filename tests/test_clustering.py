from pathlib import Path
from typing import get_args

from clustering_for_ner.clustering import Clusters
from clustering_for_ner.clustering.models import METRIC, PCA_IMPLEMENTATION


def test_smoke_check() -> None:
    clusters = Clusters(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        base_url="http://localhost:8000/v1",
        dataset_paths=[
            Path("data/conll_sharp_validated.conll"),
        ],
        whiten=True,
        with_context=False,
        pca_implementation="pca",
        pca_components=[50],
        pca_cumulative_variance=None,
        clustering_implementation="umap",
        clustering_components=[16],
        cluster_neighbors=30,
        top_k=50,
        api_key="LOCAL",
        reports_dir=Path("reports/smoke_check"),
        tsne=True,
        umap=True,
        prompt=None,
        metric="euclidean",
        random_state=42,
        include_labels=False,
    )
    clusters.cluster()


def test_umap_all_pca_impls() -> None:
    for impl in get_args(PCA_IMPLEMENTATION):
        clusters = Clusters(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            base_url="http://localhost:8000/v1",
            dataset_paths=[
                Path("data/conll_sharp_validated.conll"),
            ],
            whiten=True,
            with_context=False,
            pca_implementation=impl,
            pca_components=[50],
            pca_cumulative_variance=None,
            clustering_implementation="umap",
            clustering_components=[16],
            cluster_neighbors=30,
            top_k=10,
            api_key="LOCAL",
            reports_dir=Path("reports/all_pca_impls"),
            tsne=True,
            umap=True,
            prompt=None,
            metric="euclidean",
            random_state=42,
            include_labels=False,
        )
        clusters.cluster()


def test_umap_all_metrics() -> None:
    for metric in get_args(METRIC):
        clusters = Clusters(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            base_url="http://localhost:8000/v1",
            dataset_paths=[
                Path("data/conll_sharp_validated.conll"),
            ],
            whiten=True,
            with_context=True,
            pca_components=[50],
            pca_cumulative_variance=None,
            pca_implementation="pca",
            clustering_implementation="umap",
            clustering_components=[16],
            cluster_neighbors=100,
            top_k=10,
            api_key="LOCAL",
            reports_dir=Path("reports/all_metrics"),
            tsne=True,
            umap=True,
            prompt=None,
            metric=metric,
            random_state=42,
            include_labels=False,
        )
        clusters.cluster()


def test_find_n_pca_components() -> None:
    clusters = Clusters(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        base_url="http://localhost:8000/v1",
        dataset_paths=[
            Path("data/conll_sharp_validated.conll"),
        ],
        whiten=True,
        with_context=True,
        pca_components=None,
        pca_cumulative_variance=0.85,
        pca_implementation="pca",
        clustering_implementation="umap",
        clustering_components=[16, 32],
        cluster_neighbors=30,
        top_k=10,
        api_key="LOCAL",
        reports_dir=Path("reports/find_pca_components"),
        tsne=True,
        umap=True,
        prompt=None,
        metric="euclidean",
        random_state=42,
        include_labels=False,
    )
    clusters.cluster()
