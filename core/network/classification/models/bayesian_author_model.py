import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

from core.network.classification.base.author_model import AuthorModel


class BayesianAuthorModel(AuthorModel):
    def __init__(self, n_components=3, max_iter=1000):
        self.n_components = n_components
        self.max_iter = max_iter
        self.models = {}
        self.author_to_index = {}
        self.index_to_author = {}

    def fit(self, X, y):
        """Fit the model to training data."""
        unique_authors = sorted(y.unique())  # Sort for consistency
        self.author_to_index = {author: idx for idx, author in
                                enumerate(unique_authors)}
        self.index_to_author = {idx: author for author, idx in
                                self.author_to_index.items()}

        for author in unique_authors:
            model = BayesianGaussianMixture(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=42,
            )
            X_author = X[y == author]
            model.fit(X_author)
            self.models[author] = model

    def predict_proba(self, X):
        n_samples = len(X)
        n_authors = len(self.models)
        score_matrix = np.zeros((n_samples, n_authors))

        for author, model in self.models.items():
            author_idx = self.author_to_index[author]
            scores = model.score_samples(X)
            score_matrix[:, author_idx] = scores

        # Convert log-likelihoods to probabilities using softmax
        exp_scores = np.exp(
            score_matrix - np.max(score_matrix, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.array([self.index_to_author[idx] for idx in
                         np.argmax(probabilities, axis=1)])

    def evaluate(self, X, y, top_k=(1, 5, 10)):


        # Convert string labels to indices for sklearn compatibility
        y_encoded = np.array([self.author_to_index[author] for author in y])
        y_proba = self.predict_proba(X)

        results = {}
        for k in top_k:
            if k <= len(
                    self.models):  # Ensure k doesn't exceed number of classes
                accuracy = top_k_accuracy_score(y_encoded, y_proba, k=k)
                results[f"top_{k}_accuracy"] = round(accuracy, 3)
            else:
                results[f"top_{k}_accuracy"] = 1.0  # Perfect if k >= n_classes

        return results