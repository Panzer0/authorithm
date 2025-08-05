import pytest
from core.data.embedder import Embedder


@pytest.fixture(scope="module")
def sample_data():
    return [
        "Bring your own lampshade, somewhere there's a party",
        "[removed]",
        "[deleted]",
        "Test comment please ignore",
    ]


@pytest.fixture(scope="module")
def expected_vector_length():
    return 512


@pytest.fixture
def embedder():
    return Embedder()


def test_embedding_output_length(embedder, sample_data, expected_vector_length):
    """Test if the embedding output length is as expected."""
    for text in sample_data:
        vector = embedder.embed_str(text)
        assert len(vector) == expected_vector_length, (
            f"Embedding vector length for text '{text}' was not as expected: "
            f"{len(vector)} != {expected_vector_length}"
        )


def test_embed_empty_string(embedder, expected_vector_length):
    """Test if embedding an empty string returns an appropriate response."""
    vector = embedder.embed_str("")
    assert len(vector) == expected_vector_length, (
        f"Embedding vector length for empty string was not as expected: "
        f"{len(vector)} != {expected_vector_length}"
    )


def test_to_dict(embedder, sample_data, expected_vector_length):
    """Test if the to_dict method converts embeddings as expected."""
    embedding = embedder.embed_str(sample_data[0])
    embedding_dict = Embedder.to_dict(embedding)

    assert len(embedding_dict) == expected_vector_length, (
        f"Embedding dict's length is not as expected: "
        f"{len(embedding_dict)} != {expected_vector_length}"
    )
    for index, value in enumerate(embedding):
        assert embedding_dict[f"embedding_{index}"] == value, (
            f"One of the values in test_to_dict is not as expected: "
            f"{embedding_dict[f'embedding_{index}']} != {value}"
        )


def test_get_cos_sim(embedder, sample_data):
    """Tests if get_cos_sim() measures similarity as expected."""
    embedding_different = embedder.embed_str(sample_data[0])
    embedding_similar_1 = embedder.embed_str(sample_data[1])
    embedding_similar_2 = embedder.embed_str(sample_data[2])

    sim_different = Embedder.get_cos_sim(
        embedding_different, embedding_similar_1
    )
    sim_similar = Embedder.get_cos_sim(embedding_similar_2, embedding_similar_1)

    assert sim_similar.item() > sim_different.item(), (
        f"Calculated cosine similarity not as expected: "
        f"{sim_similar.item()} <= {sim_different.item()}"
    )
