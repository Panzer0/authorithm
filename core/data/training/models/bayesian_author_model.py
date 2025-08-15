import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm

from core.data.training.base.author_model import AuthorModel


class BayesianAuthorModel(AuthorModel):
    def __init__(self, n_components=3, max_iter=1000):
        self.n_components = n_components
        self.max_iter = max_iter
        self.models = {}

    def fit(self, X, y):
        for author in y.unique():
            model = BayesianGaussianMixture(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=42,
            )
            X_author = X[y == author]
            model.fit(X_author)
            self.models[author] = model

    def predict_proba(self, X):
        result = {}
        for author, model in self.models.items():
            result[author] = model.score_samples(X)
        return result

    def predict(self, X):
        scores = self.predict_proba(X)
        return [max(scores, key=lambda a: scores[a][i]) for i in range(len(X))]

    def evaluate(self, X, y, top_k=(1, 5, 10)):
        score_matrix = []
        author_list = list(self.models.keys())

        for author in tqdm(author_list, desc="Scoring authors"):
            scores = self.models[author].score_samples(X)
            score_matrix.append(scores)

        score_matrix = np.vstack(score_matrix).T
        y_true = y.tolist()
        results = {}
        for k in top_k:
            top_k_preds = np.argsort(score_matrix, axis=1)[:, -k:]
            hits = [
                y_true[i] in [author_list[idx] for idx in top_k_preds[i]]
                for i in range(len(y_true))
            ]
            results[f"top_{k}_accuracy"] = round(np.mean(hits), 3)
        return results
