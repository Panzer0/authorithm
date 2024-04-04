import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SUBREDDIT_NAME = "fantasy"
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
DATASET_PATH = f"datasets/{DATASET_FILENAME}"



class Embedder:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        np.random.seed(123456)

    def plot_graph(self):
        author_counts = self.data["author"].value_counts()
        bins  = list(np.arange(25, 150, 5)) + [150]
        hist = author_counts.hist(edgecolor='white', bins = bins)

        plt.xlabel("Comment count")
        plt.ylabel("User count")

        hist.set_xticks(bins)
        hist.set_xticklabels(bins, rotation=45, ha='right')

        plt.show()


if __name__ == "__main__":
    dataset = pd.read_parquet(DATASET_PATH)
    embedder = Embedder(dataset)
    embedder.plot_graph()
    # print(dataset)
    # print(author_counts.sum())
    # print(author_counts[author_counts > 50].sum())