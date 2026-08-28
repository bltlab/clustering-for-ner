"""End-to-end tests for the clustering pipeline.

Embeddings are stubbed (see conftest); everything downstream is the real
implementation. `Cluster.__call__` catches and logs every exception rather
than raising, so a run that blows up still "succeeds" — these tests
therefore assert on the artifacts a run is supposed to leave behind, which
is the only way a failure inside the pipeline can surface.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import FIXTURE_CORRECTED, FIXTURE_MENTIONS, LIVE_BASE_URL

from clustering_for_ner.clustering import Clusters
from clustering_for_ner.clustering.models import Report

# Metrics that survive the whole pipeline. The pipeline applies the chosen
# metric twice — as UMAP's `output_metric` and again in
# `silhouette_samples` — so a metric has to be valid for both. That rules
# out `hellinger` (scikit-learn rejects it), `cosine` and `correlation`
# (degenerate on a UMAP output space that is centred on the origin), and
# `braycurtis` (defined for non-negative data). See also the missing comma
# in the METRIC literal, which fuses "haversine" and "mahalanobis" into one
# unusable value.
WORKING_METRICS = ("euclidean", "manhattan", "chebyshev", "minkowski", "canberra")

# The configuration reported in the paper.
BASELINE: dict[str, Any] = {
    "model_id": "stub/stub-embed",
    "base_url": "http://embeddings.invalid/v1",
    "api_key": "LOCAL",
    "whiten": True,
    "with_context": False,
    "include_labels": False,
    "prompt": None,
    "pca_implementation": "pca",
    "pca_components": [50],
    "pca_cumulative_variance": None,
    "clustering_implementation": "umap",
    "clustering_components": [16],
    "cluster_neighbors": 30,
    "metric": "chebyshev",
    "top_k": 50,
    "tsne": False,
    "umap": False,
    "random_state": 42,
}


def run_pipeline(reports_dir: Path, fixture_path: Path, **overrides: Any) -> list[Report]:
    """Run a full clustering pass and return the deserialized reports.

    Returning parsed `Report` objects doubles as a check that the JSON a
    run writes can be read back into the model that produced it.
    """
    Clusters(
        **{**BASELINE, **overrides},
        dataset_paths=[fixture_path],
        reports_dir=reports_dir,
    ).cluster()
    return [
        Report.model_validate_json(path.read_text())
        for path in sorted(reports_dir.rglob("*.json"))
    ]


@pytest.mark.usefixtures("stub_embeddings")
def test_pipeline_writes_a_complete_report(tmp_path: Path, fixture_path: Path) -> None:
    (report,) = run_pipeline(tmp_path, fixture_path)

    assert report.mention_count == FIXTURE_MENTIONS
    assert np.isfinite(report.mean_sil) and np.isfinite(report.std_sil)
    assert -1.0 <= report.mean_sil <= 1.0
    assert 0 <= report.noise_count <= report.mention_count
    assert {s.entity_type for s in report.summary} == {"LOC", "MISC", "ORG", "PER"}
    assert len(report.mentions) == FIXTURE_MENTIONS

    # Every sibling artifact of the JSON report.
    for suffix in (".md", ".csv", "_cm.png"):
        assert Path(f"{report.path}{suffix}").is_file(), f"missing {suffix}"


@pytest.mark.usefixtures("stub_embeddings")
def test_outlier_ranking_surfaces_annotation_errors(
    tmp_path: Path, fixture_path: Path
) -> None:
    """The point of the method: mentions sitting inside the wrong entity
    type's cluster should fall to the bottom of the silhouette ranking.

    The fixture's `CORRECTED` mentions are exactly those, so a working
    pipeline concentrates them in the top-k outliers far above their 10%
    base rate."""
    (report,) = run_pipeline(tmp_path, fixture_path, top_k=FIXTURE_CORRECTED)

    outliers = report.scores["top_k_outliers"]
    assert isinstance(outliers, list)
    assert [m.silhouette for m in outliers] == sorted(m.silhouette for m in outliers)

    base_rate = FIXTURE_CORRECTED / FIXTURE_MENTIONS
    assert report.num_corrected_ratio > 5 * base_rate, (
        f"outlier ranking found {report.num_corrected} of {FIXTURE_CORRECTED} "
        f"known corrections in the top {FIXTURE_CORRECTED}; no better than chance"
    )


@pytest.mark.usefixtures("stub_embeddings")
@pytest.mark.parametrize("pca_implementation", ["pca", "truncated", "kernel"])
def test_every_pca_implementation_runs(
    tmp_path: Path, fixture_path: Path, pca_implementation: str
) -> None:
    (report,) = run_pipeline(
        tmp_path, fixture_path, pca_implementation=pca_implementation
    )
    assert report.settings.pca_implementation == pca_implementation
    assert np.isfinite(report.mean_sil)


@pytest.mark.usefixtures("stub_embeddings")
@pytest.mark.parametrize("metric", WORKING_METRICS)
def test_every_supported_metric_runs(
    tmp_path: Path, fixture_path: Path, metric: str
) -> None:
    (report,) = run_pipeline(tmp_path, fixture_path, metric=metric)
    assert report.settings.metric == metric
    assert np.isfinite(report.mean_sil)


@pytest.mark.usefixtures("stub_embeddings")
def test_pca_components_chosen_from_cumulative_variance(
    tmp_path: Path, fixture_path: Path
) -> None:
    (report,) = run_pipeline(
        tmp_path, fixture_path, pca_components=None, pca_cumulative_variance=0.85
    )
    assert report.settings.pca_components is None
    assert report.settings.determined_components is not None
    assert 0 < report.settings.determined_components <= FIXTURE_MENTIONS


@pytest.mark.usefixtures("stub_embeddings")
def test_component_sweep_produces_one_report_per_combination(
    tmp_path: Path, fixture_path: Path
) -> None:
    """`--pca-components` and `--clustering-components` are both repeatable
    and fan out over their cartesian product."""
    reports = run_pipeline(
        tmp_path, fixture_path, pca_components=[25, 50], clustering_components=[8, 16]
    )
    assert len(reports) == 4
    assert {
        (r.settings.pca_components, r.settings.clustering_components) for r in reports
    } == {(25, 8), (25, 16), (50, 8), (50, 16)}


@pytest.mark.usefixtures("stub_embeddings")
def test_projection_plots_are_written(tmp_path: Path, fixture_path: Path) -> None:
    """`--tsne`/`--umap` re-project to 2D and 3D purely for plotting,
    separately from the clustering-stage reduction. Each projection is
    drawn twice: gold labels, then HDBSCAN assignments."""
    (report,) = run_pipeline(tmp_path, fixture_path, tsne=True, umap=True)

    for projection in ("tsne", "umap"):
        for dimensions in ("2d", "3d"):
            for labelling in (projection, f"{projection}-hdbscan"):
                plot = Path(f"{report.path}_{labelling}_{dimensions}.png")
                assert plot.is_file(), f"missing plot {plot.name}"


@pytest.mark.usefixtures("stub_embeddings")
def test_tsne_clustering_backend_runs(tmp_path: Path, fixture_path: Path) -> None:
    (report,) = run_pipeline(
        tmp_path,
        fixture_path,
        clustering_implementation="tsne",
        clustering_components=[3],  # exact t-SNE is capped below 4 components
    )
    assert report.settings.clustering_implementation == "tsne"
    assert np.isfinite(report.mean_sil)


@pytest.mark.integration
@pytest.mark.skipif(
    LIVE_BASE_URL is None,
    reason="set NERCLUSTER_TEST_BASE_URL to run against a live embedding endpoint",
)
def test_against_live_embedding_endpoint(tmp_path: Path, fixture_path: Path) -> None:
    """The one test that exercises EmbeddingClient for real. Requires an
    OpenAI-compatible endpoint serving the model named below."""
    (report,) = run_pipeline(
        tmp_path,
        fixture_path,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        base_url=LIVE_BASE_URL,
    )
    assert report.mention_count == FIXTURE_MENTIONS
    assert np.isfinite(report.mean_sil)
