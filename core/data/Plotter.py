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

    def plot_graph(self):
        author_counts = self.data["author"].value_counts()
        author_counts_df = pd.DataFrame({"value": author_counts.index, "count": author_counts.values})
        sns.histplot(data=author_counts_df)

        plt.xlabel("Comment count")
        plt.ylabel("User count")

        plt.show()



if __name__ == "__main__":
    dataset = pd.read_parquet(DATASET_PATH)
    embedder = Embedder(dataset)
    embedder.plot_graph()
    # print(dataset)
    # print(author_counts.sum())
    # print(author_counts[author_counts > 50].sum())