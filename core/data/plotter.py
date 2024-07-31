import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

from core.data.preprocessing import balance_dataset, prune_dataset
import pyarrow.parquet as pq

from core.config import DATASET_PATH


class Plotter:
    """Plots various parameters of the given dataset.

    Attributes:
        data: The Pandas DataFrame which contains the analysed dataset.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Inits Plotter.

        Args:
            data: The Pandas DataFrame which contains the analysed dataset
        """
        self.data = data
        np.random.seed(123456)
        sns.set_theme()

    def plot_count_hist(self) -> None:
        """Displays a histogram of counts of comments belonging to the same
        authors.

        Args:
            data: The Pandas DataFrame which contains the analysed dataset.
        """
        author_counts = self.data["author"].value_counts()
        print("Author counts")
        plt.figure(figsize=(8, 6))
        print("Plt.figure")
        sns.histplot(data=author_counts, bins=10)
        print("Past histplot")

        plt.xlabel("Comment count")
        plt.ylabel("User count")
        plt.title("Distribution of comment counts per author")

        print("About to show")

        plt.show()

    def plot_count_CDF(self) -> None:
        """Displays a CDF plot.

        Displays a cumulative distribution function plot of counts of
        comments written by the same user against their cumulative proportion.
        """
        author_counts = self.data["author"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(data=author_counts)

        plt.xlabel("User's comment count threshold")
        plt.ylabel("Cumulative proportion of comments")
        plt.title("CDF of comments vs. comment count threshold")
        plt.show()

    def plot_count_threshold_sizes(
        self, thresholds=[25, 50, 75, 100, 125, 150]
    ) -> None:
        """Displays a bar plot of dataset sizes given different comment count
        per user thresholds.
        """
        author_counts = self.data["author"].value_counts()
        dataset_sizes = []

        for threshold in thresholds:
            size = sum(count for count in author_counts if count >= threshold)
            dataset_sizes.append(size)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=thresholds, y=dataset_sizes)

        for index, value in enumerate(dataset_sizes):
            ax.text(index, value, f"{value}", ha="center")

        plt.xlabel("Comment count threshold")
        plt.ylabel("Dataset size (number of comments)")
        plt.title("Dataset size vs. comment count threshold")
        plt.show()

    def plot_PCA(self) -> None:
        """Displays a PCA plot.

        Displays a PCA (principal component analysis) plot of the dataset's
        embeddings using plotly.
        """
        categories = self.data["author"].unique()

        pca = PCA(n_components=3)

        embedding_columns = [f"embedding_{i}" for i in range(512)]
        embeddings = dataset[embedding_columns].values

        pca_embeddings = pca.fit_transform(embeddings)

        fig = go.Figure()

        for i, category in enumerate(categories):
            category_mask = self.data["author"] == category
            category_embeddings = pca_embeddings[category_mask]

            fig.add_trace(
                go.Scatter3d(
                    x=category_embeddings[:, 0],
                    y=category_embeddings[:, 1],
                    z=category_embeddings[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5, color=i, colorscale="Viridis", opacity=0.8
                    ),
                    name=category,
                )
            )

        fig.update_layout(
            autosize=False,
            title="3D Scatter Plot of Users",
            width=1600,
            height=1000,
            scene=dict(
                xaxis=dict(title="x"),
                yaxis=dict(title="y"),
                zaxis=dict(title="z"),
            ),
        )

        fig.show()

    def plot_PCA_incremental(self, data_file, batch_size=65536) -> None:
        """Displays an incremental PCA plot.

        Displays an incremental PCA (principal component analysis) plot of the
        dataset's embeddings using plotly. Use this when the dataset's size is
        too large to fit in the memory.
        """
        categories = self.data["author"].unique()
        valid_ids = self.data["id"].unique()

        ipca = IncrementalPCA(n_components=3, batch_size=batch_size)

        print("Starting PCA fitting process...")
        embedding_columns = [f"embedding_{i}" for i in range(512)]

        batch_count = sum(1 for _ in data_file.iter_batches(batch_size=batch_size))
        for batch in tqdm(
            data_file.iter_batches(batch_size=batch_size), total=batch_count, desc="Fitting the PCA"
        ):
            batch_df = batch.to_pandas()
            batch_df = batch_df[batch_df["id"].isin(valid_ids)]
            if batch_df.empty:
                continue

            batch_df[embedding_columns] = batch_df[embedding_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            embeddings = batch_df[embedding_columns].values
            ipca.partial_fit(embeddings)

        transformed = []
        transformed_ids = []

        print("PCA fitting finished. Starting PCA transformation process...")

        for batch in tqdm(
            data_file.iter_batches(batch_size=batch_size),
            total=batch_count,
            desc="Transforming the PCA",
        ):
            batch_df = batch.to_pandas()
            batch_df = batch_df[batch_df["id"].isin(valid_ids)]
            if batch_df.empty:
                continue

            batch_df[embedding_columns] = batch_df[embedding_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            embeddings = batch_df[embedding_columns].values
            pca_embeddings = ipca.transform(embeddings)

            transformed.append(pca_embeddings)
            transformed_ids.extend(batch_df["id"].values)

        transformed = np.vstack(transformed)
        transformed_ids = np.array(transformed_ids)

        explained_variance = ipca.explained_variance_ratio_
        print(f"Explained variance by component: {explained_variance}")
        print(f"Cumulative explained variance: {np.cumsum(explained_variance)}")

        print("PCA transformation process complete. Preparing the plot...")

        id_to_index = {id_: index for index, id_ in enumerate(transformed_ids)}

        fig = go.Figure()

        for i, category in enumerate(
            tqdm(
                categories, total=len(categories), desc="Constructing PCA graph"
            )
        ):
            category_mask = self.data["author"] == category
            category_ids = self.data[category_mask]["id"].values

            category_indices = [
                id_to_index[id_] for id_ in category_ids if id_ in id_to_index
            ]
            category_embeddings = transformed[category_indices]

            fig.add_trace(
                go.Scatter3d(
                    x=category_embeddings[:, 0],
                    y=category_embeddings[:, 1],
                    z=category_embeddings[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5, color=i, colorscale="Viridis", opacity=0.8
                    ),
                    name=category,
                )
            )

        fig.update_layout(
            autosize=False,
            title="3D Scatter Plot of Users",
            width=1600,
            height=1000,
            scene=dict(
                xaxis=dict(title="x"),
                yaxis=dict(title="y"),
                zaxis=dict(title="z"),
            ),
        )

        fig.show()


if __name__ == "__main__":
    print("Let's get started")
    dataset = pd.read_parquet(DATASET_PATH, columns=["id", "author"])
    print("Got the light dataset")
    dataset_pq_file = pq.ParquetFile(DATASET_PATH)
    print("Data successfully read")
    dataset = balance_dataset(dataset, 1000)
    # print("Parquet pruned")
    plotter = Plotter(dataset)
    # print("Plotter created")
    # plotter.plot_count_hist()
    # print("First plot done")
    # plotter.plot_count_CDF()
    # plotter.plot_count_threshold_sizes(
    #     [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # )
    print("Plotting PCA...")
    plotter.plot_PCA_incremental(dataset_pq_file)
