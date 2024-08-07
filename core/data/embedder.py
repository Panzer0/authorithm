import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch import Tensor

MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
"""Tested model names:
    jinaai/jina-embeddings-v2-small-en
    jinaai/jina-embeddings-v2-base-en
"""


class Embedder:
    """Generates embeddings for provided data using the
    jinaai/jina-embeddings-v2-base-en model.

    Attributes:
        model: An object of the jinaai/jina-embeddings-v2-base-en text
         embedding model.
    """

    def __init__(
        self, max_seq_length: int = None, model_name: str = MODEL_NAME
    ) -> None:
        """Inits Embedder.

        Args:
            data: The maximal sequence length to be accepted by the embedder.
             Inputs that exceed the limit will be truncated.
            model_name: The name of the model to use.

        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

    def embed_str(self, data: str) -> torch.Tensor:
        """Generates an embedding for the given str data.

        Args:
            data: The data to be embedded.

        Returns:
            A PyTorch tensor containing the embedding.
        """
        return self.model.encode(data, normalize_embeddings=True)

    @staticmethod
    def to_dict(embedding: torch.Tensor) -> dict:
        return {
            f"embedding_{i}": embedding[i] for i in range(embedding.shape[0])
        }

    @staticmethod
    def get_cos_sim(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tensor:
        """Evaluates the cosine similarity for the two given tensors.

        Args:
            tensor_a, tensor_b: The compared tensors.
        """
        return cos_sim(tensor_a, tensor_b)
