import pandas

DATASET_FILENAME = "dataset_askreddit.parquet.gzip"
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


class Embedder:
    def __init__(self, dataset_path) -> None:
        self.dataset = self.load_dataset(dataset_path)

    def load_dataset(self, path):
        return pandas.read_parquet(path)


if __name__ == "__main__":
    embedder = Embedder(DATASET_PATH)
    print(embedder.dataset)
