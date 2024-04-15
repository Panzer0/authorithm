from torch.utils.data import Dataset
import pandas as pd

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"

class CustomImageDataset(Dataset):
    def __init__(self, data_file = DATASET_PATH):
        dataframe = pd.read_parquet(data_file)
        self.embeddings = dataframe["embedding"]
        self.authors = dataframe["author"]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        author = self.authors[idx]
        return embedding, author