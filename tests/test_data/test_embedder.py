import unittest
from core.data.Embedder import Embedder


class TestEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = [
            "Bring your own lampshade, somewhere there's a party",
            "[removed]",
            "[deleted]",
            "Test comment please ignore",
        ]
        cls.expected_vector_length = 512

    def setUp(self):
        self.embedder = Embedder()

    def test_embedding_output_length(self):
        """Test if the embedding output length is as expected."""
        for text in self.data:
            vector = self.embedder.embed_str(text)
            self.assertEqual(
                len(vector),
                self.expected_vector_length,
                f"Embedding vector length for text '{text}' was not as expected: {len(vector)} != {self.expected_vector_length}",
            )

    def test_embed_empty_string(self):
        """Test if embedding an empty string returns an appropriate response."""
        vector = self.embedder.embed_str("")
        self.assertEqual(
            len(vector),
            self.expected_vector_length,
            f"Embedding vector length for empty string was not as expected: {len(vector)} != {self.expected_vector_length}",
        )

    def test_to_dict(self):
        """Test if the to_dict method converts embeddings as expected"""
        embedding = self.embedder.embed_str(self.data[0])
        embedding_dict = Embedder.to_dict(embedding)

        self.assertEqual(
            len(embedding_dict),
            self.expected_vector_length,
            f"Embedding dict's length is not as expected: {len(embedding_dict)} != {self.expected_vector_length}",
        )
        for index, value in enumerate(embedding):
            self.assertEqual(
                embedding_dict[f"embedding_{index}"],
                value,
                f"One of the values in test_to_dict are not as expected: {self.expected_vector_length} != {value}",
            )

    def test_get_cos_sim(self):
        """Tests if get_cos_sim() measures similarity as expected"""
        embedding_different = self.embedder.embed_str(self.data[0])
        embedding_similar_1 = self.embedder.embed_str(self.data[1])
        embedding_similar_2 = self.embedder.embed_str(self.data[2])

        sim_different = Embedder.get_cos_sim(
            embedding_different, embedding_similar_1
        )
        sim_similar = Embedder.get_cos_sim(
            embedding_similar_2, embedding_similar_1
        )

        self.assertGreater(
            sim_similar.item(),
            sim_different.item(),
            f"Calculated cosine similarity not as expected: {sim_similar.item()} <= {sim_different.item()}",
        )


if __name__ == "__main__":
    unittest.main()
