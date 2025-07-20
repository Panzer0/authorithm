import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import numpy as np


class LazyParquetDataset(IterableDataset):
    def __init__(self, embedding_file, pca_file, batch_size=32):
        self.embedding_file = pq.ParquetFile(embedding_file)
        self.pca_file = pq.ParquetFile(pca_file)
        self.batch_size = batch_size
        self.num_rows = self.pca_file.metadata.num_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.num_rows
        else:
            per_worker = int(np.ceil(self.num_rows / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.num_rows)

        for batch_start in range(start, end, self.batch_size):
            # Read PCA data
            pca_batch = self.pca_file.read_row_group(
                batch_start // self.batch_size
            )
            pca_ids = pca_batch["id"].to_pandas()
            pca_embeddings = torch.tensor(
                pca_batch["pca"].to_pandas().values.tolist()
            )

            # Read corresponding embedding data
            embedding_batch = self.read_embedding_batch(pca_ids)

            embeddings = torch.tensor(
                embedding_batch["embedding"].to_pandas().values.tolist()
            )
            authors = torch.tensor(
                embedding_batch["author"].to_pandas().values.tolist()
            )
            original_texts = embedding_batch["text"].to_pandas().tolist()

            yield embeddings, pca_embeddings, authors, original_texts

    def read_embedding_batch(self, pca_ids, chunk_size=100000):
        # Read the embedding file in chunks
        matching_rows = []

        for batch in self.embedding_file.iter_batches(batch_size=chunk_size):
            batch_df = batch.to_pandas()
            matched = batch_df[batch_df["id"].isin(pca_ids)]
            matching_rows.append(matched)

            if len(matching_rows) == len(pca_ids):
                break

        return pq.Table.from_pandas(pd.concat(matching_rows))


# Create DataLoader
dataset = LazyParquetDataset("embeddings.parquet", "pca_embeddings.parquet")
dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
