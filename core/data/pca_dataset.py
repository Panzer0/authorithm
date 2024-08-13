from torch.utils.data import Dataset
import pandas as pd


class PCADataset(Dataset):
    def __init__(self, data_file):
        dataframe = pd.read_parquet(data_file)
        self.pca_x = dataframe["pca_x"].values
        self.pca_y = dataframe["pca_y"].values
        self.pca_z = dataframe["pca_z"].values
        self.authors = dataframe["author"].values

    def __len__(self):
        return len(self.authors)

    def __getitem__(self, idx):
        pca_x = self.pca_x[idx]
        pca_y = self.pca_y[idx]
        pca_z = self.pca_z[idx]
        author = self.authors[idx]
        return (pca_x, pca_y, pca_z), author
