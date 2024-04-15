import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from core.data.RedditDataset import RedditDataset

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


class FineTuner:
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    dataset = RedditDataset(DATASET_PATH)
    (
        embeddings_train,
        embeddings_test,
        authors_train,
        authors_test,
    ) = train_test_split(
        dataset.embeddings,
        dataset.authors,
        stratify=dataset.authors,
        random_state=42,
        test_size=0.2,
    )
