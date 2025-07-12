import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from tqdm import tqdm

from core.data.pca_handler import PCAHandler
from core.data.preprocessing import balance_dataset
import pyarrow.parquet as pq

from core.config import DATASET_PATH


class Plotter:
    """Visualizes statistics and PCA plots for a given dataset."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        np.random.seed(123456)
        sns.set_theme()

    def plot_count_hist(self) -> None:
        """Displays a histogram of comment counts per author."""
        author_counts = self.data["author"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.histplot(author_counts, bins=10)
        plt.xlabel("Comment count")
        plt.ylabel("User count")
        plt.title("Distribution of comment counts per author")
        plt.show()

    def plot_count_CDF(self) -> None:
        """Displays a cumulative distribution function of comment counts."""
        author_counts = self.data["author"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(author_counts)
        plt.xlabel("User's comment count threshold")
        plt.ylabel("Cumulative proportion of comments")
        plt.title("CDF of comments vs. comment count threshold")
        plt.show()

    def plot_count_threshold_sizes(self, thresholds=None) -> None:
        """Bar plot showing dataset size under different user comment thresholds."""
        if thresholds is None:
            thresholds = [25, 50, 75, 100, 125, 150]

        author_counts = self.data["author"].value_counts()
        dataset_sizes = [
            author_counts[author_counts >= t].sum() for t in thresholds
        ]

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=thresholds, y=dataset_sizes)

        for index, value in enumerate(dataset_sizes):
            ax.text(index, value, f"{value}", ha="center")

        plt.xlabel("Comment count threshold")
        plt.ylabel("Dataset size (number of comments)")
        plt.title("Dataset size vs. comment count threshold")
        plt.show()

    def plot_PCA(self) -> None:
        """Displays a 3D PCA plot using plotly."""
        categories = self.data["author"].unique()
        embedding_columns = [f"embedding_{i}" for i in range(512)]
        embeddings = self.data[embedding_columns].values

        pca = PCA(n_components=3)
        pca_embeddings = pca.fit_transform(embeddings)

        fig = go.Figure()

        for i, category in enumerate(categories):
            mask = self.data["author"] == category
            coords = pca_embeddings[mask]

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5, color=i, colorscale="Viridis", opacity=0.8
                    ),
                    name=category,
                )
            )

        fig.update_layout(
            title="3D PCA Scatter Plot of Users",
            width=1600,
            height=1000,
            scene=dict(
                xaxis=dict(title="PC 1"),
                yaxis=dict(title="PC 2"),
                zaxis=dict(title="PC 3"),
            ),
        )
        fig.show()

    def plot_PCA_incremental(
        self, transformed, transformed_ids, categories=None
    ) -> None:
        """Displays a 3D PCA plot using pre-transformed embeddings."""
        if categories is None:
            categories = self.data["author"].unique()

        id_index_map = {id_: idx for idx, id_ in enumerate(transformed_ids)}
        fig = go.Figure()

        for i, category in enumerate(tqdm(categories, desc="Rendering PCA")):
            ids = self.data.loc[self.data["author"] == category, "id"]
            indices = [id_index_map[id_] for id_ in ids if id_ in id_index_map]
            coords = transformed[indices]

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5, color=i, colorscale="Viridis", opacity=0.8
                    ),
                    name=category,
                )
            )

        fig.update_layout(
            title="3D PCA Scatter Plot of Users",
            width=1600,
            height=1000,
            scene=dict(
                xaxis=dict(title="PC 1"),
                yaxis=dict(title="PC 2"),
                zaxis=dict(title="PC 3"),
            ),
        )
        fig.show()


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = pd.read_parquet(DATASET_PATH, columns=["id", "author"])
    dataset = balance_dataset(dataset, 1000)

    print("Generating PCA embeddings...")
    pca_handler = PCAHandler(DATASET_PATH)
    transformed, transformed_ids, _ = pca_handler.generate_pca()
    categories = pca_handler.get_valid_categories()

    plotter = Plotter(dataset)

    plotter.plot_PCA_incremental(transformed, transformed_ids, categories)
