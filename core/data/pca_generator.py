import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import DATASET_PATH
from core.data.preprocessing import balance_dataset
from sklearn.decomposition import IncrementalPCA
import pyarrow.parquet as pq


class PCAGenerator:
    EMBEDDING_COLUMNS = [f"embedding_{i}" for i in range(512)]

    def __init__(self, dataset_path=DATASET_PATH, batch_size=65536):
        self.data_mask = pd.read_parquet(dataset_path, columns=["id", "author"])
        self.data_mask = balance_dataset(self.data_mask, 1000)
        self.data_file = pq.ParquetFile(DATASET_PATH)
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=3, batch_size=batch_size)

    def _get_batches(self):
        return self.data_file.iter_batches(batch_size=self.batch_size)

    def _get_batch_count(self):
        return sum(1 for _ in self._get_batches())

    def parse_batch(self, batch):
        batch_df = batch.to_pandas()
        batch_df = batch_df[batch_df["id"].isin(self.get_valid_ids())]
        if batch_df.empty:
            return batch_df

        batch_df[self.EMBEDDING_COLUMNS] = batch_df[
            self.EMBEDDING_COLUMNS
        ].apply(pd.to_numeric, errors="coerce")
        return batch_df

    def fit(self):
        batch_count = self._get_batch_count()

        for batch in tqdm(
                self._get_batches(),
                total=batch_count,
                desc="Fitting the PCA",
        ):
            batch_df = self.parse_batch(batch)
            if batch_df.empty:
                continue

            embeddings = batch_df[self.EMBEDDING_COLUMNS].values
            self.ipca.partial_fit(embeddings)

    def transform(self):
        transformed = []
        transformed_ids = []
        batch_count = self._get_batch_count()

        for batch in tqdm(
                self._get_batches(),
                total=batch_count,
                desc="Transforming the PCA",
        ):
            batch_df = self.parse_batch(batch)
            if batch_df.empty:
                continue

            embeddings = batch_df[self.EMBEDDING_COLUMNS].values
            pca_embeddings = self.ipca.transform(embeddings)

            transformed.append(pca_embeddings)
            transformed_ids.extend(batch_df["id"].values)

        transformed = np.vstack(transformed)
        transformed_ids = np.array(transformed_ids)

        return transformed, transformed_ids

    def get_explained_variance(self):
        return self.ipca.explained_variance_ratio_

    def get_cumulative_explained_variance(self):
        return np.cumsum(self.get_explained_variance())

    def generate_pca(self):
        self.fit()
        return self.transform()

    def get_valid_categories(self):
        return self.data_mask["author"].unique()

    def get_valid_ids(self):
        return self.data_mask["id"].unique()
