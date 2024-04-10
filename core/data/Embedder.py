import matplotlib, torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

SUBREDDIT_NAME = "fantasy"
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


class Embedder:
    def __init__(self, max_seq_length: int = None) -> None:
        self.model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-base-en",
            trust_remote_code=True,
        )
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

    def embed_str(self, data: str) -> torch.Tensor:
        return self.model.encode(data)

    @staticmethod
    def get_cos_sim(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
        return cos_sim(tensor_a, tensor_b)
