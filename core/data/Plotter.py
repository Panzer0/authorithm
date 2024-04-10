import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


SUBREDDIT_NAME = "fantasy"
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


class Embedder:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        np.random.seed(123456)
        sns.set_theme()

    def plot_count_hist(self):
        author_counts = self.data["author"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.histplot(data=author_counts)

        plt.xlabel("Comment count")
        plt.ylabel("User count")
        plt.title("Distribution of comment counts per author")

        plt.show()

    # Short for Cumulative Distribution Function
    def plot_count_CDF(self):
        author_counts = self.data["author"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(data=author_counts)

        plt.xlabel("User's comment count threshold")
        plt.ylabel("Cumulative proportion of comments")
        plt.title("CDF of comments vs. comment count threshold")
        plt.show()

    def plot_count_threshold_sizes(self):
        author_counts = self.data["author"].value_counts()
        thresholds = [25, 50, 75, 100, 125, 150]  # Example thresholds
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


if __name__ == "__main__":
    dataset = pd.read_parquet(DATASET_PATH)
    embedder = Embedder(dataset)
    embedder.plot_count_hist()
    embedder.plot_count_CDF()
    embedder.plot_count_threshold_sizes()
