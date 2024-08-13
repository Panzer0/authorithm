from core.data.pca_dataset import PCADataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from core.config import DATASET_PCA_PATH


class PredictorKNN:
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    dataset = PCADataset(DATASET_PCA_PATH)
    pca_train, pca_test, authors_train, authors_test = train_test_split(
        dataset.pca,
        dataset.authors,
        stratify=dataset.authors,
        random_state=42,
        test_size=0.2,
    )

    k = 50
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(pca_train, authors_train)

    y_pred = knn.predict(pca_test)
    accuracy = knn.score(pca_test, authors_test)
    print(f"Accuracy: {accuracy}")
