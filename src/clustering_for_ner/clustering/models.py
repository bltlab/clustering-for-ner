import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from pandas import DataFrame

# KazNERD has the most types, so this has 25 colors.
#
# Gemini generated this to try to make the palette work
# across 25 different colors, while still being
# distinguishable at lower numbers of types,
# which is the majority of corpora
COLORS = [
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#F032E6",
    "#BCF60C",
    "#FABEBE",
    "#008080",
    "#E6BEFF",
    "#9A6324",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
    "#808000",
    "#FFD8B1",
    "#000075",
    "#808080",
    "#000000",
    "#FF1493",
    "#1E90FF",
    "#32CD32",
    "#FF4500",
]

METRIC = Literal[
    # Minkowski style metrics
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    # Miscellaneous spatial metrics
    "canberra",
    "braycurtis",
    "haversine"
    # Normalized spatial metrics
    "mahalanobis",
    # Angular and correlation metrics
    "cosine",
    "correlation",
    # Other
    "hellinger",
]
PCA_IMPLEMENTATION = Literal["pca", "truncated", "kernel"]
CLUSTERING_IMPLEMENTATION = Literal["tsne", "umap"]


class Settings(BaseModel):
    dataset_paths: list[Path]
    prompt: Optional[str]
    with_context: bool
    model_id: str
    tsne: bool
    umap: bool
    metric: METRIC
    reports_dir: Path
    include_labels: bool
    top_k: int = 10
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    pca_implementation: PCA_IMPLEMENTATION
    pca_components: Optional[int]
    pca_cumulative_variance: Optional[float]
    clustering_implementation: CLUSTERING_IMPLEMENTATION
    clustering_components: int
    cluster_neighbors: int
    random_state: int = 42
    determined_components: Optional[int] = None

    @computed_field()
    @property
    def datasets(self) -> str:
        name = "-".join([dataset.stem for dataset in self.dataset_paths])
        return f"{name}"

    def path(self) -> str:
        family, model = self.model_id.split("/")
        return f"{self.reports_dir}/{self.datasets}/{family}/{model}/{self.pca_implementation}_{self.clustering_implementation}_n_pca-{self.pca_components}_n-clustering-{self.clustering_components}_cn-{self.cluster_neighbors}_metric-{self.metric}_topk-{self.top_k}_{self.id}"

    def mkdir(self) -> None:
        family, model = self.model_id.split("/")
        path = f"{self.reports_dir}/{self.datasets}/{family}/{model}"
        Path.mkdir(Path(path), exist_ok=True, parents=True)


class ReportMention(BaseModel):
    text: str
    entity_type: str
    silhouette: float
    doc_idx: int
    doc_local_idx: int
    seq_idx: int
    mention_idx: int


class Point(BaseModel):
    x: float
    y: float
    entity_type: str
    mention: str
    mention_idx: int


class Summary(BaseModel):
    entity_type: str
    count: int
    sil_mean: float
    sil_std: float


class Report(BaseModel):
    settings: Settings
    path: Path
    mean_sil: float
    std_sil: float
    ami: float
    ari: float
    noise_count: int
    mention_count: int
    num_corrected: int
    num_corrected_ratio: float
    summary: list[Summary]
    classification_report: dict
    scores: dict[str, dict[str, list[ReportMention]] | list[ReportMention]]
    mentions: list[ReportMention]
    points: Optional[dict[str, list[Point]]]

    @computed_field
    @property
    def noise_ratio(self) -> float:
        return self.noise_count / self.mention_count

    def serialize(self, mentions_df: "DataFrame") -> None:
        from sklearn.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        with open(f"{self.path}.json", "w") as file:
            file.write(self.model_dump_json())

        y_true = mentions_df["entity_type"]
        y_pred = mentions_df["hdbscan_labels"]

        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        with open(f"{self.path}.md", "w") as file:
            file.write("# Report\n")
            file.write(f"Report Time: {datetime.now()}\n")
            file.write(f"Slurm Job ID: {os.getenv('SLURM_JOB_ID')}\n")
            file.write(f"Slurm Job Name: {os.getenv('SLURM_JOB_NAME')}\n\n")

            file.write("## Settings\n")
            [file.write(f"{k}: {v}\n") for k, v in self.settings.model_dump().items()]
            file.write("\n")

            file.write("## Scores\n")
            file.write(f"mean_sil: {self.mean_sil}\n")
            file.write(f"std_sil: {self.std_sil}\n")
            file.write(f"precision: {precision}\n")
            file.write(f"recall: {recall}\n")
            file.write(f"f1: {f1}\n")
            file.write(f"ari: {self.ari}\n")
            file.write(f"ami: {self.ami}\n")
            file.write(f"noise_count: {self.noise_count}\n")
            file.write(f"mention_count: {self.mention_count}\n")
            file.write(f"noise_ratio: {self.noise_ratio}\n")
            file.write(f"num_corrected: {self.num_corrected}\n")
            file.write(f"num_corrected_ratio: {self.num_corrected_ratio}\n\n")

            file.write("## Classification Report\n")
            file.write(classification_report(y_true, y_pred))  # pyright: ignore
