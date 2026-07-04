from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, Field, computed_field
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    classification_report,
    silhouette_samples,
    silhouette_score,
)
from umap import UMAP

from ..clients import EmbeddingClient
from ..clustering.models import (
    CLUSTERING_IMPLEMENTATION,
    COLORS,
    METRIC,
    PCA_IMPLEMENTATION,
    Report,
    ReportMention,
    Settings,
    Summary,
)
from ..data import Dataset, Mention


class Clusters(BaseModel):
    model_id: str
    base_url: str
    dataset_paths: list[Path]
    pca_implementation: PCA_IMPLEMENTATION
    pca_components: Optional[list[int]]
    pca_cumulative_variance: Optional[float]
    clustering_implementation: CLUSTERING_IMPLEMENTATION
    clustering_components: list[int]
    whiten: bool
    cluster_neighbors: int
    top_k: int
    api_key: str
    reports_dir: Path
    tsne: bool
    umap: bool
    prompt: Optional[str]
    with_context: bool
    include_labels: bool
    metric: METRIC
    random_state: int
    dataset: Dataset = Field(
        default_factory=lambda data: Dataset.from_paths(paths=data["dataset_paths"])
    )

    def cluster(self) -> None:
        embeddings = self._embed()

        if self.pca_components:
            for pca_component in self.pca_components:
                [
                    Cluster(
                        settings=Settings(
                            dataset_paths=self.dataset_paths,
                            prompt=self.prompt,
                            with_context=self.with_context,
                            include_labels=self.include_labels,
                            model_id=self.model_id,
                            pca_implementation=self.pca_implementation,
                            pca_components=pca_component,
                            pca_cumulative_variance=self.pca_cumulative_variance,
                            clustering_implementation=self.clustering_implementation,
                            clustering_components=clustering_component,
                            cluster_neighbors=self.cluster_neighbors,
                            top_k=self.top_k,
                            tsne=self.tsne,
                            umap=self.umap,
                            random_state=self.random_state,
                            metric=self.metric,
                            reports_dir=self.reports_dir,
                        ),
                        mentions=self.dataset.all_mentions(),
                        reports_dir=self.reports_dir,
                        # Make sure the embeddings are copied
                        # before dimension reduction. Default
                        # for objects here is by reference,
                        # but we want a new copy for each run
                        embeddings=embeddings.copy(),
                    )()
                    for clustering_component in self.clustering_components
                ]
        else:
            [
                Cluster(
                    settings=Settings(
                        dataset_paths=self.dataset_paths,
                        prompt=self.prompt,
                        with_context=self.with_context,
                        include_labels=self.include_labels,
                        model_id=self.model_id,
                        pca_implementation=self.pca_implementation,
                        pca_components=None,
                        pca_cumulative_variance=self.pca_cumulative_variance,
                        clustering_implementation=self.clustering_implementation,
                        clustering_components=clustering_component,
                        cluster_neighbors=self.cluster_neighbors,
                        top_k=self.top_k,
                        tsne=self.tsne,
                        umap=self.umap,
                        random_state=self.random_state,
                        metric=self.metric,
                        reports_dir=self.reports_dir,
                    ),
                    mentions=self.dataset.all_mentions(),
                    reports_dir=self.reports_dir,
                    # Make sure the embeddings are copied
                    # before dimension reduction. Default
                    # for objects here is by reference,
                    # but we want a new copy for each run
                    embeddings=embeddings.copy(),
                )()
                for clustering_component in self.clustering_components
            ]

    def _embed(self) -> NDArray[np.float32]:
        return np.array(
            EmbeddingClient(
                model_id=self.model_id,
                base_url=self.base_url,
                api_key=self.api_key,
                prompt=self.prompt,
                with_context=self.with_context,
            ).embed(self.dataset.all_mentions())
        )


@dataclass
class Cluster:
    settings: Settings
    mentions: list[Mention]
    reports_dir: Path
    embeddings: NDArray[np.float32]

    @computed_field
    @property
    def mention_spans(self) -> list[str]:
        return [" ".join(mention.tokens) for mention in self.mentions]

    @computed_field
    @property
    def doc_idxs(self) -> list[int]:
        return [mention.doc_idx for mention in self.mentions]

    @computed_field
    @property
    def doc_local_idxs(self) -> list[int]:
        return [mention.doc_local_idx for mention in self.mentions]

    @computed_field
    @property
    def seq_idxs(self) -> list[int]:
        return [mention.seq_idx for mention in self.mentions]

    @computed_field
    @property
    def mention_idxs(self) -> list[int]:
        return [mention.mention_idx for mention in self.mentions]

    @computed_field
    @property
    def labels(self) -> list[str]:
        return [mention.label for mention in self.mentions]

    @computed_field
    @property
    def is_corrected(self) -> list[bool]:
        return [mention.corrected for mention in self.mentions]

    @computed_field
    @property
    def label_map(self) -> dict[str, int]:
        labels = list(sorted(set(self.labels)))
        mapping: dict[str, int] = {}

        for idx, label in enumerate(labels):
            mapping[label] = idx

        return mapping

    def __call__(self) -> str:
        logger.info(self.settings)
        self.settings.mkdir()

        try:
            self.settings.determined_components = self._decompose()
            self._cluster()
            self._report()
        except Exception as e:
            logger.error(f"Error: {e}")

        return "Completed run"

    def _decompose(self) -> Optional[int]:
        "Performs dimension reduction on the embeddings, if any."

        logger.info(
            f"{self._components_ref()} Decomposing embeddings of shape {self.embeddings.shape}"
        )
        logger.info(
            f"{self._components_ref()} Decomposing embeddings with {self.settings.pca_implementation.upper()}"
        )

        orig_dims = self.embeddings.shape[1]

        if self.settings.pca_components:
            n_components = self.settings.pca_components
        else:
            n_components = self._identify_n_pca_components()

        match self.settings.pca_implementation:
            case "pca":
                self._pca(n_components)
            case "truncated":
                self._truncated_svd(n_components)
            case "kernel":
                self._kernel_pca(n_components)

        logger.info(
            f"{self._components_ref()} Decomposed {orig_dims} -> {self.embeddings.shape[1]} dims"
        )

        return None if self.settings.pca_components else n_components

    def _pca(self, n_components: int) -> None:
        self.embeddings = PCA(
            n_components=n_components,
            random_state=self.settings.random_state,
        ).fit_transform(self.embeddings)

    def _truncated_svd(self, n_components: int) -> None:
        self.embeddings = np.array(
            TruncatedSVD(
                n_components=n_components,
                random_state=self.settings.random_state,
            ).fit_transform(self.embeddings),
            dtype=np.float32,
        )

    def _kernel_pca(self, n_components: int) -> None:
        self.embeddings = np.array(
            KernelPCA(
                n_components=n_components,
                random_state=self.settings.random_state,
            ).fit_transform(self.embeddings),
            dtype=np.float32,
        )

    def _identify_n_pca_components(self) -> int:
        pca_temp = PCA(random_state=self.settings.random_state).fit(
            self.embeddings.copy()
        )

        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = int(
            np.argmax(cumulative_variance >= self.settings.pca_cumulative_variance) + 1
        )

        logger.info(f"Optimal PCA components for 85% variance: {n_components}")

        return n_components

    def _cluster(self) -> None:
        logger.info(
            f"{self._components_ref()} Clustering embeddings of shape {self.embeddings.shape}"
        )
        logger.info(
            f"{self._components_ref()} Clustering embeddings with {self.settings.clustering_implementation.upper()}"
        )

        orig_dims = self.embeddings.shape[1]

        match self.settings.clustering_implementation:
            case "tsne":
                self._tsne_cluster()
            case "umap":
                self._umap_cluster()

        logger.info(
            f"{self._components_ref()} Clustering reduced {orig_dims} -> {self.embeddings.shape[1]} dims"
        )

    def _tsne_cluster(self) -> None:
        self.embeddings = TSNE(
            n_components=self.settings.clustering_components,
            random_state=self.settings.random_state,
            perplexity=30,
            metric=self.settings.metric,
        ).fit_transform(self.embeddings)

    def _umap_cluster(self) -> None:
        self.embeddings = np.array(
            UMAP(
                n_neighbors=self.settings.cluster_neighbors,
                min_dist=0.00,
                n_components=self.settings.clustering_components,
                random_state=self.settings.random_state,
                output_metric=self.settings.metric,
            ).fit_transform(
                self.embeddings,
                [self.label_map[label] for label in self.labels]
                if self.settings.include_labels
                else None,
            ),
            dtype=np.float32,
        )

    def _hdbscan(self) -> tuple[list[str], NDArray[Any]]:
        hdbscan_labels = np.array(
            HDBSCAN(
                min_cluster_size=25,
                min_samples=10,
                cluster_selection_method="eom",
                cluster_selection_epsilon=0.1,
                copy=True,
            ).fit_predict(self.embeddings),
            dtype=int,
        )

        df = pd.DataFrame(
            {"ground_truth": self.labels, "hdbscan_labels": hdbscan_labels}
        )
        crosstab = pd.crosstab(df["hdbscan_labels"], df["ground_truth"])

        cluster_map: dict[int, str] = {}
        for cluster_id in crosstab.index:
            if cluster_id == -1:
                cluster_map[cluster_id] = "zNoise"
            else:
                # Get the column name (true label) with the highest count for this row
                dominant_label = crosstab.loc[cluster_id].idxmax()
                cluster_map[cluster_id] = dominant_label

        df["predicted_label"] = df["hdbscan_labels"].map(cluster_map)  # pyright: ignore

        return df["predicted_label"].astype(str).tolist(), hdbscan_labels

    def _report(self) -> None:
        logger.info(
            f"{self._components_ref()} Running Cluster Eval:\n{self.settings.model_dump_json()}"
        )

        hdbscan_labels, hdbscan_idxs = self._hdbscan()
        noise_count = (hdbscan_idxs == -1).sum()

        mean_sil, std_sil, per_sil = self._compute_silhouettes()
        logger.info(
            f"{self._components_ref()} Overall silhouette: mean={mean_sil:.4f}, std={std_sil:.4f}"
        )

        ari, ami = self._compute_ari_ami(
            pd.Series(self.labels, name="true_labels"),
            pd.Series(hdbscan_idxs, name="hdbscan_labels"),
        )
        logger.info(f"{self._components_ref()} ari={ari:.4f} | ami={ami:.4f}")

        mentions_df = pd.DataFrame(
            {
                "text": self.mention_spans,
                "entity_type": self.labels,
                "corrected": self.is_corrected,
                "silhouette": per_sil,
                "doc_idx": self.doc_idxs,
                "doc_local_idx": self.doc_local_idxs,
                "seq_idx": self.seq_idxs,
                "mention_idx": self.mention_idxs,
                "hdbscan_labels": hdbscan_labels,
                "hdbscan_idxs": hdbscan_idxs,
            }
        )
        logger.info(f"{self._components_ref()} Created Mentions DataFrame")
        mentions_df["hdbscan_labels"] = mentions_df["hdbscan_labels"].str.replace(
            "zNoise", "Noise"
        )
        self._cm(mentions_df)

        mentions = [
            ReportMention(**record) for record in mentions_df.to_dict(orient="records")
        ]
        logger.info(f"{self._components_ref()} Generated mentions")

        summary = self._summarize_by_class(mentions_df)
        scores = self._pick_top_and_outliers(mentions_df)
        num_corrected = self._num_corrected(mentions_df)

        if self.settings.tsne:
            self._plot(clustering="tsne", hdbscan_labels=hdbscan_labels)
        if self.settings.umap:
            self._plot(clustering="umap", hdbscan_labels=hdbscan_labels)

        Report(
            settings=self.settings,
            path=Path(f"{self.settings.path()}"),
            mean_sil=mean_sil,
            std_sil=std_sil,
            ari=ari,
            ami=ami,
            noise_count=noise_count,
            mention_count=len(mentions_df),
            summary=summary,
            mentions=mentions,
            scores=scores,
            num_corrected=num_corrected,
            num_corrected_ratio=num_corrected / self.settings.top_k,
            points={},
            classification_report=classification_report(
                mentions_df["entity_type"],
                mentions_df["hdbscan_labels"],
                output_dict=True,
            ),  # pyright: ignore
        ).serialize(mentions_df=mentions_df)

        mentions_df.to_csv(f"{self.settings.path()}.csv", index=False)

        logger.info(
            f"{self._components_ref()} Run complete for {self.settings.model_dump_json()}"
        )

    def _compute_silhouettes(self) -> tuple[float, float, np.ndarray]:
        """
        Computes the silhouette scores for a list of embeddings
        and their
        labels: list of gold entity_type strings
        returns (mean_silhouette, std_silhouette, per-sample silhouette array)
        """
        logger.info(f"{self._components_ref()} Computing silhouettes")
        if len(set(self.labels)) < 2:
            n = len(self.labels)
            return float("nan"), float("nan"), np.full(n, np.nan)
        # per-sample scores
        sample_vals = silhouette_samples(
            self.embeddings, self.labels, metric=self.settings.metric
        )
        mean_sil = float(
            silhouette_score(self.embeddings, self.labels, metric=self.settings.metric)
        )
        std_sil = float(np.std(sample_vals, ddof=0))

        return mean_sil, std_sil, sample_vals

    def _compute_ari_ami(
        self, true_labels: pd.Series, hdbscan_labels: pd.Series
    ) -> tuple[float, float]:
        """
        Computes the adjusted rand index for the labels and their
        HDBSCAN predictions

        labels: list of gold entity_type strings
        returns (mean_silhouette, std_silhouette, per-sample silhouette array)
        """

        # TODO: Filter out noise, and record percentage of data
        # filtered as a metric
        # Balance good silh, ari, ami scores
        # with how much data is filtered out

        df = pd.concat([true_labels, hdbscan_labels], axis=1)
        filtered_df = df[df["hdbscan_labels"] != -1]

        ari = adjusted_rand_score(
            filtered_df["true_labels"], filtered_df["hdbscan_labels"]
        )
        ami = adjusted_mutual_info_score(
            filtered_df["true_labels"], filtered_df["hdbscan_labels"]
        )

        return ari, ami

    def _summarize_by_class(self, df: pd.DataFrame) -> list[Summary]:
        """
        Returns per-class summary of size, mean, and std dev
            of silhouette scores.

        Args:
            df (pd.DataFrame): DataFrame of silhouette scores

        Returns:
            pd.DataFrame: DataFrame with summaries.
        """
        logger.info(f"{self._components_ref()} Summarizing by class")
        summary_df = (
            df.groupby("entity_type")
            .silhouette.agg(["count", "mean", "std"])
            .rename(columns={"count": "count", "mean": "sil_mean", "std": "sil_std"})
            .reset_index()
        )
        return [Summary(**record) for record in summary_df.to_dict(orient="records")]

    def _pick_top_and_outliers(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, list[ReportMention]] | list[ReportMention]]:
        """
        Generates and returns for each entity type:
        {
            'core': top k by silhouette desc,
            'outliers': top k by silhouette asc
        }

        Args:
            df (pd.DataFrame): DataFrame of embedded mentions

        Returns:
            dict[str, dict[str, list[ReportMention]]: Mapping
                from each entity type to a mapping
                from 'core' and 'outliers' to their
                respective top-k mentions.
        """
        logger.info(f"{self._components_ref()} Picking top k and outliers")

        top_k = {}
        top_k["top_k_core"] = df.sort_values("silhouette", ascending=False).head(
            self.settings.top_k
        )
        top_k["top_k_outliers"] = df.sort_values("silhouette", ascending=True).head(
            self.settings.top_k
        )

        report = {}
        for cls, sub in df.groupby("entity_type"):
            sub_sorted = sub.sort_values("silhouette", ascending=False)
            report[cls] = {
                "core": sub_sorted.head(self.settings.top_k),
                "outliers": sub_sorted.tail(self.settings.top_k).sort_values(
                    "silhouette"
                ),
            }

        scores: dict[str, dict[str, list[ReportMention]] | list[ReportMention]] = {}
        for mention_type, mapping in report.items():
            scores_map: dict[str, list[ReportMention]] = {}
            scores_map["core"] = [
                ReportMention(**record)
                for record in mapping["core"].to_dict(orient="records")
            ]
            scores_map["outliers"] = [
                ReportMention(**record)
                for record in mapping["outliers"].to_dict(orient="records")
            ]
            scores[mention_type] = scores_map

        scores["top_k_core"] = [
            ReportMention(**record)
            for record in top_k["top_k_core"].to_dict(orient="records")
        ]
        scores["top_k_outliers"] = [
            ReportMention(**record)
            for record in top_k["top_k_outliers"].to_dict(orient="records")
        ]

        return scores

    def _num_corrected(self, df: pd.DataFrame) -> int:
        """
        Finds the number corrected in top-k outlier scores
        """
        logger.info(
            f"{self._components_ref()} Calculating number of corrected in top-{self.settings.top_k}"
        )

        counts = (
            df.sort_values("silhouette", ascending=True)
            .head(self.settings.top_k)
            .value_counts("corrected")
        )
        try:
            return counts[True]
        except Exception:
            return 0

    def _get_color_map(self, labels: list[str]) -> dict[str, str]:
        """Loads the color map based on the number of labels.

        Args:
            labels (list[str]): List of labels in the dataset.

        Returns:
            dict[str, str]: Mapping from each label to
                its color."""
        labels = sorted(list(set(labels)))
        return {label: COLORS[i] for i, label in enumerate(labels)}

    def _plot(
        self,
        clustering: CLUSTERING_IMPLEMENTATION,
        hdbscan_labels: list[str],
    ) -> None:
        """
        Plot embedding projections.

        Returns:
            dict[str, list[Point]]: dict mapping
        """
        logger.info(
            f"{self._components_ref()} Plotting {clustering.upper()} projections"
        )

        match clustering:
            case "umap":
                embeddings_2d = self._umap_project(n_components=2)
                embeddings_3d = self._umap_project(n_components=3)
            case "tsne":
                embeddings_2d = self._tsne_project(n_components=2)
                embeddings_3d = self._tsne_project(n_components=3)

        self._plot_2d(clustering, embeddings_2d)
        self._plot_2d(clustering, embeddings_2d, hdbscan_labels)

        self._plot_3d(clustering, embeddings_3d)
        self._plot_3d(clustering, embeddings_3d, hdbscan_labels)

    def _plot_2d(
        self,
        clustering: CLUSTERING_IMPLEMENTATION,
        embeddings: NDArray[np.float32],
        hdbscan_labels: Optional[list[str]] = None,
    ) -> None:
        plot_type = f"{clustering}-hdbscan" if hdbscan_labels else clustering
        path = Path(f"{self.settings.path()}_{plot_type}_2d.png")

        plt.figure(figsize=(8, 6))
        for label, color in self._get_color_map(
            hdbscan_labels if hdbscan_labels else self.labels
        ).items():
            idxs = [
                i
                for i, lbl in enumerate(
                    hdbscan_labels if hdbscan_labels else self.labels
                )
                if lbl == label
            ]
            if not idxs:
                continue
            xs = embeddings[idxs, 0]
            ys = embeddings[idxs, 1]

            plt.scatter(
                xs,
                ys,
                c=color,
                label=label,
                s=10,
                alpha=0.75,
                edgecolors="w",
                linewidths=0.5,
            )

        plt.title(plot_type)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(path, dpi=300)

    def _plot_3d(
        self,
        clustering: CLUSTERING_IMPLEMENTATION,
        embeddings: NDArray[np.float32],
        hdbscan_labels: Optional[list[str]] = None,
    ) -> None:
        plot_type = f"{clustering}-hdbscan" if hdbscan_labels else clustering
        path = Path(f"{self.settings.path()}_{plot_type}_3d.png")

        # embeddings projected into 3d
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")

        for label, color in self._get_color_map(
            hdbscan_labels if hdbscan_labels else self.labels
        ).items():
            idxs = [
                i
                for i, lbl in enumerate(
                    hdbscan_labels if hdbscan_labels else self.labels
                )
                if lbl == label
            ]
            if not idxs:
                continue
            xs = embeddings[idxs, 0]
            ys = embeddings[idxs, 1]
            zs = embeddings[idxs, 2]

            ax.scatter(
                xs,
                ys,
                zs,  # pyright: ignore
                c=color,
                label=label,
                s=10,
                alpha=0.75,
                edgecolors="w",
                linewidths=0.5,
            )

        ax.set_title(plot_type)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(path, dpi=300)

    def _tsne_project(self, n_components: Literal[2, 3]) -> NDArray[np.float32]:
        logger.info(
            f"{self._components_ref()} Projecting with t-SNE (n={n_components})"
        )
        return TSNE(
            n_components=n_components,
            random_state=self.settings.random_state,
            perplexity=30,
            metric=self.settings.metric,
        ).fit_transform(self.embeddings.copy())

    def _umap_project(self, n_components: Literal[2, 3]) -> NDArray[np.float32]:
        logger.info(f"{self._components_ref()} Projecting with UMAP (n={n_components})")
        return np.array(
            UMAP(
                n_neighbors=self.settings.cluster_neighbors,
                min_dist=0.25,
                n_components=n_components,
                random_state=self.settings.random_state,
                output_metric=self.settings.metric,
            ).fit_transform(
                self.embeddings.copy(),
                [self.label_map[label] for label in self.labels]
                if self.settings.include_labels
                else None,
            ),
            dtype=np.float32,
        )

    def _cm(self, mentions_df: pd.DataFrame) -> None:
        logger.info(f"{self._components_ref()} Generating Confusion Matrix")
        cm_path = Path(f"{self.settings.path()}_cm.png")

        filtered_df = mentions_df[mentions_df["hdbscan_idxs"] != -1]
        y_true = filtered_df["entity_type"]
        y_pred = filtered_df["hdbscan_labels"]
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, xticks_rotation="vertical"
        )
        plt.tight_layout()
        plt.savefig(cm_path)

    def _components_ref(self) -> str:
        return f"[{self.settings.pca_implementation.upper()} {self.settings.pca_components if self.settings.pca_components else 'FIND'} | {self.settings.clustering_implementation.upper()} {self.settings.clustering_components}]"

    # def _points(self) -> dict[str, list[Point]]:
    #     point_map: dict[str, list[Point]] = {}
    #
    #     points_df = pd.DataFrame(
    #         {
    #             "x": self.embeddings[:, 0],
    #             "y": self.embeddings[:, 1],
    #             "entity_type": self.labels,
    #             "mention": self.mention_spans,
    #             "mention_idx": self.mention_idxs,
    #         }
    #     )
    #
    #     for label in set(self.labels):
    #         filtered_df = points_df[points_df["entity_type"] == label]
    #         if not isinstance(filtered_df, pd.DataFrame):
    #             raise TypeError("Expected pd.DataFrame type for filtered object")
    #         point_map[label] = [
    #             Point(**record) for record in filtered_df.to_dict(orient="records")
    #         ]
    #     return point_map
