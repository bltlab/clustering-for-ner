"""Tests for CoNLL ingestion.

Fast and dependency-light — no dimension reduction, no numba. If the
pipeline tests break on a new Python, these narrow the cause down to
parsing versus numerics.
"""

from collections import Counter
from pathlib import Path

from conftest import (
    FIXTURE_CORRECTED,
    FIXTURE_DOCUMENTS,
    FIXTURE_LABELS,
    FIXTURE_MENTIONS,
)

from clustering_for_ner.data import Dataset


def test_fixture_matches_expected_shape(dataset: Dataset) -> None:
    mentions = dataset.all_mentions()
    assert len(mentions) == FIXTURE_MENTIONS
    assert tuple(sorted(Counter(m.label for m in mentions))) == FIXTURE_LABELS
    assert len({m.doc_idx for m in mentions}) == FIXTURE_DOCUMENTS


def test_corrected_flag_read_from_extra_column(dataset: Dataset) -> None:
    """The `CORRECTED` marker in the CoNLL extra column is the gold signal
    the outlier ranking is scored against."""
    corrected = [m for m in dataset.all_mentions() if m.corrected]
    assert len(corrected) == FIXTURE_CORRECTED
    # Corrections are spread over entity types, not confined to one.
    assert len({m.label for m in corrected}) > 1


def test_bio_decoding_yields_well_formed_mentions(dataset: Dataset) -> None:
    for mention in dataset.all_mentions():
        assert mention.tokens, "mention with no tokens"
        assert len(mention.tokens) == len(mention.labels)
        assert mention.labels[0] == f"B-{mention.label}"
        assert all(lbl == f"I-{mention.label}" for lbl in mention.labels[1:])


def test_indices_locate_the_mention_in_its_source_sequence(dataset: Dataset) -> None:
    """Every flagged outlier has to be traceable back to the source file,
    which is what the four index fields on Mention are for."""
    for mention in dataset.all_mentions():
        sequence = dataset.documents[mention.seq_idx]
        assert sequence.mentions[mention.doc_local_idx] == mention
        assert " ".join(mention.tokens) in mention.text
        assert mention.text == " ".join(sequence.tokens)


def test_mention_idx_is_a_dense_corpus_wide_ordering(dataset: Dataset) -> None:
    idxs = [m.mention_idx for m in dataset.all_mentions()]
    assert idxs == list(range(len(idxs)))


def test_from_paths_concatenates_datasets(fixture_path: Path) -> None:
    """Splits are meant to be clustered together, so multiple paths merge
    into one dataset."""
    single = Dataset.from_path(fixture_path)
    merged = Dataset.from_paths([fixture_path, fixture_path])
    assert len(merged.all_mentions()) == 2 * len(single.all_mentions())
    assert len(merged.documents) == 2 * len(single.documents)
