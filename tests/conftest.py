"""Shared test fixtures.

The clustering pipeline normally gets its vectors from a remote
OpenAI-compatible embedding endpoint. Tests stub out that one network call
and keep everything downstream real: PCA, UMAP/t-SNE (and the numba JIT
underneath), HDBSCAN, silhouette scoring, plotting and report
serialization all run exactly as they do in production. That is what makes
the suite usable as a cross-version smoke check.
"""

import os
import zlib
from pathlib import Path

import matplotlib

# Must happen before anything imports pyplot, which the clustering package
# does at module scope. The dev dependency group ships PyQt6, so without
# this matplotlib can pick an interactive backend and fail on a headless
# machine.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from clustering_for_ner.clients.embed import EmbeddingClient  # noqa: E402
from clustering_for_ner.data import Dataset, Mention  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
FIXTURE = DATA_DIR / "mini_ner.conll"

# Matches sentence-transformers/all-MiniLM-L6-v2, the model this suite was
# originally written against.
EMBEDDING_DIM = 384

# Gold facts about tests/data/mini_ner.conll, asserted in test_data.py so
# that an edited or regenerated fixture fails loudly instead of quietly
# weakening every test that leans on these counts.
FIXTURE_MENTIONS = 377
FIXTURE_CORRECTED = 37
FIXTURE_LABELS = ("LOC", "MISC", "ORG", "PER")
FIXTURE_DOCUMENTS = 6

# Set to a real endpoint (e.g. http://localhost:8000/v1) to also run the
# tests marked `integration` against a live embedding server.
LIVE_BASE_URL = os.environ.get("NERCLUSTER_TEST_BASE_URL")


def fake_embeddings(mentions: list[Mention]) -> list[NDArray[np.float32]]:
    """Deterministic stand-in for a real embedding model.

    Each entity type gets its own Gaussian cluster, spread widely enough
    that the clusters overlap rather than being trivially separable.
    Mentions carrying the gold ``CORRECTED`` flag are drawn from a
    *neighbouring* type's cluster instead of their own, which is what an
    annotation error looks like in embedding space: the point sits among
    the wrong class, so its silhouette against its gold label goes sharply
    negative. That gives the outlier ranking something real to find.

    Stable across platforms and Python versions: numpy's PCG64 stream is
    reproducible, and per-mention jitter is keyed by a CRC32 of the mention
    index rather than by iteration order.
    """
    labels = sorted({mention.label for mention in mentions})
    center_rng = np.random.default_rng(0)
    centers = {label: center_rng.normal(0.0, 1.5, EMBEDDING_DIM) for label in labels}

    embeddings: list[NDArray[np.float32]] = []
    for mention in mentions:
        if mention.corrected:
            label = labels[(labels.index(mention.label) + 1) % len(labels)]
        else:
            label = mention.label
        jitter_rng = np.random.default_rng(
            zlib.crc32(f"mention-{mention.mention_idx}".encode())
        )
        vector = centers[label] + jitter_rng.normal(0.0, 1.0, EMBEDDING_DIM)
        embeddings.append(vector.astype(np.float32))
    return embeddings


@pytest.fixture
def fixture_path() -> Path:
    """Path to the committed CoNLL fixture, resolved relative to this file
    so the suite runs from any working directory."""
    assert FIXTURE.is_file(), f"missing test fixture: {FIXTURE}"
    return FIXTURE


@pytest.fixture
def dataset(fixture_path: Path) -> Dataset:
    return Dataset.from_path(fixture_path)


@pytest.fixture
def stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the pipeline's only network call."""
    monkeypatch.setattr(
        EmbeddingClient, "embed", lambda _self, mentions: fake_embeddings(mentions)
    )


@pytest.fixture(autouse=True)
def _close_figures():
    """Plotting tests leave figures open; matplotlib warns past twenty."""
    yield
    plt.close("all")
