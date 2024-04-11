import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
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
        plt.figure(figsize=(8, 6))
        sns.histplot(data=author_counts)

        plt.xlabel("Comment count")
        plt.ylabel("User count")
        plt.title("Distribution of comment counts per author")

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

    def plot_count_threshold_sizes(self) -> None:
        """Displays a bar plot of dataset sizes given different comment count
        per user thresholds.
        """
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
    plotter = Plotter(dataset)
    plotter.plot_count_hist()
    plotter.plot_count_CDF()
    plotter.plot_count_threshold_sizes()
