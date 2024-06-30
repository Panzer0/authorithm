from torch.utils.data import Dataset
import pandas as pd

class RedditDataset(Dataset):
    def __init__(self, data_file):
        dataframe = pd.read_parquet(data_file)
        self.embeddings = dataframe["embedding"].to_list()
        self.authors = dataframe["author"].values

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        author = self.authors[idx]
        return embedding, author
