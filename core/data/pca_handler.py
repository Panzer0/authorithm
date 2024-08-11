import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import DATASET_PATH
from core.data.preprocessing import balance_dataset
from sklearn.decomposition import IncrementalPCA
import pyarrow.parquet as pq


class PCAHandler:
    """Performs incremental PCA on a dataset.

        Attributes:
            EMBEDDING_COLUMNS: List of names of coumns under which the
                embeddings are stored in the dataset.
            data_mask: A dataframe containing a minimal subset (containing only
                comment ids and author names) of the dataset balanced around its
                author parameter.
            data_file: A ParquetFile object to handle batch reading of the
                dataset.
            batch_size: The number of samples to process in each batch.
            ipca: The IncrementalPCA object used for dimensionality reduction.
        """
    EMBEDDING_COLUMNS = [f"embedding_{i}" for i in range(512)]

    def __init__(self, dataset_path=DATASET_PATH, batch_size=65536) -> None:
        """Inits PCAGenerator.

        Args:
            dataset_path: The path to the dataset file.
            batch_size: The number of samples to process in each batch.
        """
        self.data_mask = pd.read_parquet(dataset_path, columns=["id", "author"])
        self.data_mask = balance_dataset(self.data_mask, 1000)
        self.data_file = pq.ParquetFile(DATASET_PATH)
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=3, batch_size=batch_size)

    def _get_batches(self):
        """Returns an iterator over the batches of data from self.data_file.

        Returns:
            iterator: An iterator that yields batches of data from
                self.data_file of size self.batch_size.
        """
        return self.data_file.iter_batches(batch_size=self.batch_size)

    def _get_batch_count(self) -> int:
        """Counts the number of batches in the dataset.

        Returns:
            int: The total number of batches in the dataset.
        """
        return sum(1 for _ in self._get_batches())

    def _parse_batch(self, batch) -> pd.DataFrame:
        """Parses a single batch of data.

        Converts the given batch to a dataframe consisting only of entries
        whose id parameter belongs to the normalised subset of the dataset and
        ensures its embedding columns are entirely numeric.

        Args:
        batch : A batch of data from the Parquet file.

        Returns:
            pd.DataFrame: A DataFrame containing only the valid rows with
                numeric embeddings.
        """
        batch_df = batch.to_pandas()
        batch_df = batch_df[batch_df["id"].isin(self.get_valid_ids())]
        if batch_df.empty:
            return batch_df

        batch_df[self.EMBEDDING_COLUMNS] = batch_df[
            self.EMBEDDING_COLUMNS
        ].apply(pd.to_numeric, errors="coerce")
        return batch_df

    def fit(self) -> None:
        """Fits the model on the dataset by iterating over it batch by batch."""
        batch_count = self._get_batch_count()

        for batch in tqdm(
                self._get_batches(),
                total=batch_count,
                desc="Fitting the PCA",
        ):
            batch_df = self._parse_batch(batch)
            if batch_df.empty:
                continue

            embeddings = batch_df[self.EMBEDDING_COLUMNS].values
            self.ipca.partial_fit(embeddings)

    def transform(self) -> (np.ndarray, np.ndarray):
        """
        Transforms the dataset using the fitted model by iterating over it batch
        by batch.

        Returns:
            tuple:
                - np.ndarray: An array of transformed PCA components.
                - np.ndarray: An array of corresponding IDs for the components.
        """
        transformed = []
        transformed_ids = []
        batch_count = self._get_batch_count()

        for batch in tqdm(
                self._get_batches(),
                total=batch_count,
                desc="Transforming the PCA",
        ):
            batch_df = self._parse_batch(batch)
            if batch_df.empty:
                continue

            embeddings = batch_df[self.EMBEDDING_COLUMNS].values
            pca_embeddings = self.ipca.transform(embeddings)

            transformed.append(pca_embeddings)
            transformed_ids.extend(batch_df["id"].values)

        transformed = np.vstack(transformed)
        transformed_ids = np.array(transformed_ids)

        return transformed, transformed_ids

    def get_explained_variance(self) -> np.ndarray:
        """Returns the explained variance ratio of each principal component.

        Returns:
             np.ndarray: An array containing the explained variance ratio for
             each component.
        """
        return self.ipca.explained_variance_ratio_

    def get_cumulative_explained_variance(self) -> np.ndarray:
        """
        Returns the cumulative explained variance ratio of each principal
        component.

        Returns:
            np.ndarray: An array containin the cumulative explained variance
            ratio for each component.
        """
        return np.cumsum(self.get_explained_variance())

    def generate_pca(self) -> (np.ndarray, np.ndarray):
        """Fits the model and transforms the dataset.

        Returns:
            tuple:
                - np.ndarray: An array of transformed PCA components.
                - np.ndarray: An array of corresponding IDs for the components.
        """
        self.fit()
        return self.transform()

    def get_valid_categories(self) -> np.ndarray:
        """Returns the unique authors belonging to the balanced dataset.

        Returns:
            np.ndarray: An array containing unique authors from the balanced
            dataset.
        """
        return self.data_mask["author"].unique()

    def get_valid_ids(self) -> np.ndarray:
        """Returns the ids of the comments belonging to the balanced dataset.

        Returns:
            np.ndarray: An array of ids belonging to the balanced dataset.
        """
        return self.data_mask["id"].unique()
