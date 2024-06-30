import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from core.data.Preprocessing import balance_dataset

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}_large.parquet.gzip"
# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


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
        pca_embeddings = pca.fit_transform(self.data["embedding"].tolist())

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


if __name__ == "__main__":
    dataset = pd.read_parquet(DATASET_PATH)
    dataset = balance_dataset(dataset, 1000)
    print("Parquet read")
    plotter = Plotter(dataset)
    print("Plotter created")
    plotter.plot_count_hist()
    print("First plot done")
    plotter.plot_count_CDF()
    plotter.plot_count_threshold_sizes([50, 100, 150, 200, 250, 300])
    plotter.plot_PCA()
