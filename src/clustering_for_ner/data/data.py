from itertools import chain
from pathlib import Path
from typing import override

from pydantic import BaseModel
from seqscore import conll


class Mention(BaseModel):
    """Mention object containing:
    - List of tokens in Mention
    - List of BIO-encoded labels in Mention
    - The core label of the mention e.g. "PER"
    - Indexes to map Mentions to the original dataset."""

    text: str
    tokens: list[str]
    labels: list[str]
    label: str
    doc_idx: int
    doc_local_idx: int
    seq_idx: int
    mention_idx: int
    cluster_label: int | None = None
    corrected: bool

    @override
    def __str__(self):
        """More informative string representation
        of the Mention"""
        text = " ".join(self.tokens)
        return (
            f"{text}-{self.label}-{self.cluster_label}"
            if self.cluster_label
            else f"{text}-{self.label}"
        )


class Document(BaseModel):
    """Container class for a Document within
    a CoNLL-formatted dataset. Documents
    are determined by document boundaries
    in the original file, if any.

    If there are no document boundaries,
    there will be on 'Document' in the
    parent 'Dataset' object."""

    tokens: list[str]
    mentions: list[Mention]
    sequence: tuple[tuple[str, str], ...]

    def all_tokens(self) -> list[list[str]]:
        """Returns a list of lists containing all
        tokens in this Document."""
        all_tokens: list[list[str]] = []
        [all_tokens.append(mention.tokens) for mention in self.mentions]
        return all_tokens

    def all_labels(self) -> list[str]:
        """Returns a list containing all
        labels in this Document."""
        labels: list[str] = []
        [labels.append(mention.label) for mention in self.mentions]
        return labels


class Dataset(BaseModel):
    """Container class for the Datasets"""

    documents: list[Document]

    @classmethod
    def from_path(cls, path: Path) -> "Dataset":
        """Create a 'Dataset' object from a
        CoNLL-formatted dataset at the provided path.

        Args:
            path (Path): The path to the
                CoNLL-formatted dataset.

        Returns:
            Dataset: the Dataset object
        """
        mention_encoding = conll.get_encoding("BIO")

        documents: list[Document] = []
        with open(path) as file:
            ingester = conll.CoNLLIngester(
                mention_encoding, ignore_document_boundaries=False
            )

            docs = ingester.ingest(file, str(path), None)

            doc_idx = 0
            seq_idx = 0
            mention_idx = 0
            for doc in docs:
                for seq in doc:
                    mentions: list[Mention] = []
                    for doc_local_idx, mention in enumerate(seq.mentions):
                        span = mention.span
                        text = " ".join(seq.tokens)
                        tokens = list(seq.tokens[span.start : span.end])
                        labels = list(seq.labels[span.start : span.end])
                        if any(seq.other_fields):
                            corrected: bool = (
                                True
                                if "CORRECTED"
                                == seq.other_fields[span.start : span.end][0][0]
                                else False
                            )
                        else:
                            corrected = False
                        tag = labels[0][2:]

                        mentions.append(
                            Mention(
                                text=text,
                                tokens=tokens,
                                labels=labels,
                                label=tag,
                                doc_idx=doc_idx,
                                doc_local_idx=doc_local_idx,
                                seq_idx=seq_idx,
                                mention_idx=mention_idx,
                                corrected=corrected,
                            )
                        )
                        mention_idx += 1
                    documents.append(
                        Document(
                            tokens=list(seq.tokens),
                            mentions=mentions,
                            sequence=seq.tokens_with_labels(),
                        )
                    )
                    seq_idx += 1
                doc_idx += 1
        return cls(documents=documents)

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "Dataset":
        """Create a 'Dataset' object from a list of
        Paths to CoNLL-formatted NER datasets. This
        will merge all datasets into a single object.

        Calls 'Dataset.from_path()' internally.

        Args:
            paths (list[Path]): The list of paths to the
                CoNLL-formatted datasets.

        Returns:
            Dataset: the Dataset object
        """
        datasets: list[Dataset] = [Dataset.from_path(path) for path in paths]
        documents: list[Document] = []
        [documents.extend(dataset.documents) for dataset in datasets]

        return cls(documents=documents)

    @staticmethod
    def simplify(input: Path, output: Path) -> None:
        mention_encoding = conll.get_encoding("BIO")

        simplified: list[tuple[tuple[str, str], ...]] = []

        with open(input) as file:
            ingester = conll.CoNLLIngester(mention_encoding)

            docs = ingester.ingest(file, str(input), None)
            for doc in docs:
                for seq in doc:
                    simplified.append(seq.tokens_with_labels())

        with open(output, "w") as file:
            for seq in simplified:
                [file.write(f"{pair[0]} {pair[1]}\n") for pair in seq]
                file.write("\n")

    def all_tokens(self) -> list[list[str]]:
        """Returns all tokens across all documents"""
        all_tokens: list[list[str]] = []
        [all_tokens.extend(doc.all_tokens()) for doc in self.documents]
        return all_tokens

    def all_labels(self) -> list[str]:
        """Returns all Labels across all documents"""
        all_labels: list[str] = []
        [all_labels.extend(doc.all_labels()) for doc in self.documents]
        return all_labels

    def all_mentions(self) -> list[Mention]:
        """Returns all Mentions across all documents"""
        all_mentions: list[Mention] = []
        [all_mentions.extend(doc.mentions) for doc in self.documents]
        return all_mentions

    def load_conll(self) -> tuple[list[Mention], list[str]]:
        """Returns a tuple containing all lists of all
        Mentions in the dataset, and all tokens in the dataset.
        If there are multiple underlying documents, this flattens
        the lists.

        Returns:
            tuple[list[Mention], list[str]: The tuple
                containing flattened lists of the
                Mentions and tokens in the Dataset."""
        mentions: list[Mention] = self.all_mentions()
        tokens: list[str] = list(chain.from_iterable(self.all_tokens()))
        return mentions, tokens

    def all_sequences(self) -> list[tuple[tuple[str, str], ...]]:
        """Returns a list of tuples containing a tuple of all
        sequences in the dataset.

        Returns:
            list[tuple[tuple[str, str], ...]]: The list
                of tuples containing the sequences."""
        all_sequences: list[tuple[tuple[str, str], ...]] = []
        [all_sequences.append(doc.sequence) for doc in self.documents]
        return all_sequences
