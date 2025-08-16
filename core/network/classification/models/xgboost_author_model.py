import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score

from core.network.classification.base.author_model import AuthorModel


class XGBoostAuthorModel(AuthorModel):
    def __init__(self, **xgb_params):
        """Initialize XGBoost model with optional parameters.

        Args:
            **xgb_params: Additional parameters for XGBClassifier
        """
        self.label_encoder = LabelEncoder()
        self.clf = None
        self.xgb_params = xgb_params

    def fit(self, X, y):
        """Fit the model to training data."""
        y_encoded = self.label_encoder.fit_transform(y)

        default_params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        default_params.update(self.xgb_params)

        self.clf = xgb.XGBClassifier(**default_params)
        self.clf.fit(X, y_encoded)

    def predict(self, X):
        """Predict the most likely author for each sample."""
        if self.clf is None:
            raise ValueError("Model must be fitted before prediction")

        y_pred = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        """Predict class probabilities for each sample.

        Returns:
            np.ndarray: Shape (n_samples, n_authors) probability matrix
        """
        if self.clf is None:
            raise ValueError("Model must be fitted before prediction")

        return self.clf.predict_proba(X)

    def evaluate(self, X, y, top_k=(1, 5, 10)):
        """Evaluate model performance with top-k accuracy metrics.

        Args:
            X: Features
            y: True labels
            top_k: Tuple of k values for top-k accuracy

        Returns:
            dict: Evaluation metrics with standardized naming
        """
        if self.clf is None:
            raise ValueError("Model must be fitted before evaluation")

        y_encoded = self.label_encoder.transform(y)
        y_proba = self.clf.predict_proba(X)

        results = {}
        n_classes = len(self.label_encoder.classes_)

        for k in top_k:
            if k <= n_classes:
                accuracy = top_k_accuracy_score(y_encoded, y_proba, k=k)
                results[f"top_{k}_accuracy"] = round(accuracy, 3)
            else:
                results[f"top_{k}_accuracy"] = 1.0

        return results