import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from core.config import DATASET_PATH, DATASET_PCA_PATH
from core.data.preprocessing import balance_dataset


class PCAHandler:
    """Performs incremental PCA on a dataset.

    Attributes:
        EMBEDDING_COLUMNS: List of names of columns under which the
            embeddings are stored in the dataset.
        data_mask: A dataframe containing a minimal subset (containing only
            comment IDs and author names) of the dataset, balanced around its
            author parameter.
        data_file: A ParquetFile object to handle batch reading of the dataset.
        batch_size: The number of samples to process in each batch.
        ipca: The IncrementalPCA object used for dimensionality reduction.
    """

    EMBEDDING_COLUMNS = [f"embedding_{i}" for i in range(512)]

    def __init__(
        self, dataset_path: str = DATASET_PATH, batch_size: int = 65536
    ) -> None:
        """Inits PCAHandler.

        Args:
            dataset_path: The path to the dataset file.
            batch_size: The number of samples to process in each batch.
        """
        self.data_mask = balance_dataset(
            pd.read_parquet(dataset_path, columns=["id", "author"]), 1000
        )
        self.valid_ids = set(self.data_mask["id"])
        self.data_file = pq.ParquetFile(dataset_path)
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
        whose ID parameter belongs to the normalized subset of the dataset,
        and ensures its embedding columns are entirely numeric.

        Args:
            batch: A batch of data from the Parquet file.

        Returns:
            pd.DataFrame: A DataFrame containing only the valid rows with
            numeric embeddings.
        """
        df = batch.to_pandas()
        df = df[df["id"].isin(self.valid_ids)]
        if df.empty:
            return df
        df[self.EMBEDDING_COLUMNS] = df[self.EMBEDDING_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        return df

    def fit(self) -> None:
        """Fits the model on the dataset by iterating over it batch by batch."""
        batch_count = self._get_batch_count()
        for batch in tqdm(
            self._get_batches(), total=batch_count, desc="Fitting PCA"
        ):
            df = self._parse_batch(batch)
            if not df.empty:
                self.ipca.partial_fit(df[self.EMBEDDING_COLUMNS].to_numpy())

    def transform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms the dataset using the fitted model by iterating over it batch
        by batch.

        Returns:
            tuple:
                - np.ndarray: An array of transformed PCA components.
                - np.ndarray: An array of corresponding IDs for the components.
                - np.ndarray: An array of corresponding authors for the
                    components.
        """
        transformed, ids, authors = [], [], []
        batch_count = self._get_batch_count()

        for batch in tqdm(
            self._get_batches(), total=batch_count, desc="Transforming PCA"
        ):
            df = self._parse_batch(batch)
            if df.empty:
                continue
            pca_output = self.ipca.transform(
                df[self.EMBEDDING_COLUMNS].to_numpy()
            )
            transformed.append(pca_output)
            ids.extend(df["id"].to_numpy())
            authors.extend(df["author"].to_numpy())

        return np.vstack(transformed), np.array(ids), np.array(authors)

    def generate_pca(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the model and transforms the dataset.

        Returns:
            tuple:
                - np.ndarray: An array of transformed PCA components.
                - np.ndarray: An array of corresponding IDs for the components.
                - np.ndarray: An array of corresponding authors for the
                    components.
        """
        self.fit()
        return self.transform()

    @staticmethod
    def to_parquet(
        pca: np.ndarray,
        ids: np.ndarray,
        authors: np.ndarray,
        path: str = DATASET_PCA_PATH,
    ) -> None:
        """Stores the provided products of transform() to a Parquet file.

        Args:
            pca: An array of the PCA components.
            ids: An array of the IDs for the components.
            authors: An array of corresponding authors for the components.
            path: The path of the Parquet file to write to.
        """
        df = pd.DataFrame(
            {
                "pca_x": pca[:, 0],
                "pca_y": pca[:, 1],
                "pca_z": pca[:, 2],
                "author": authors,
            },
            index=ids,
        )
        df.to_parquet(path, compression="gzip")

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
            np.ndarray: An array containing the cumulative explained variance
            ratio for each component.
        """
        return np.cumsum(self.get_explained_variance())

    def get_valid_categories(self) -> np.ndarray:
        """Returns the unique authors belonging to the balanced dataset.

        Returns:
            np.ndarray: An array containing unique authors from the balanced
            dataset.
        """
        return self.data_mask["author"].unique()

    def get_valid_ids(self) -> np.ndarray:
        """Returns the IDs of the comments belonging to the balanced dataset.

        Returns:
            np.ndarray: An array of IDs belonging to the balanced dataset.
        """
        return self.data_mask["id"].unique()


if __name__ == "__main__":
    # Save PCA to a file
    pca_handler = PCAHandler()
    pca_data = pca_handler.generate_pca()
    pca_handler.to_parquet(*pca_data)
