import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class PCADataset(Dataset):
    def __init__(self, data_file):
        dataframe = pd.read_parquet(data_file)
        self.pca = np.column_stack((
            dataframe["pca_x"].values,
            dataframe["pca_y"].values,
            dataframe["pca_z"].values
        ))
        self.authors = dataframe["author"].values

    def __len__(self):
        return len(self.authors)

    def __getitem__(self, idx):
        pca = self.pca
        author = self.authors[idx]
        return pca, author
