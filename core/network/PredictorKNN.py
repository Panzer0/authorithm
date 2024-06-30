from core.data.RedditDataset import RedditDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from core.config import DATASET_PATH


class PredictorKNN:
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    dataset = RedditDataset(DATASET_PATH)
    embed_train, embed_test, authors_train, authors_test = train_test_split(
        dataset.embeddings,
        dataset.authors,
        stratify=dataset.authors,
        random_state=42,
        test_size=0.2,
    )

    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embed_train, authors_train)

    y_pred = knn.predict(embed_test)
    accuracy = knn.score(embed_test, authors_test)
    print(f"Accuracy: {accuracy}")
