import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from core.network.classification.base.author_model import AuthorModel


class XGBoostAuthorModel(AuthorModel):
    def __init__(self, **xgb_params):
        self.label_encoder = LabelEncoder()
        self.clf = None
        self.xgb_params = xgb_params

    def fit(self, X, y):
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
        y_pred = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def evaluate(self, X, y):
        y_encoded = self.label_encoder.transform(y)
        y_pred = self.clf.predict(X)
        y_proba = self.clf.predict_proba(X)
        return {
            "Top-1 Accuracy": round(accuracy_score(y_encoded, y_pred), 3),
            "Top-5 Accuracy": round(top_k_accuracy_score(y_encoded, y_proba, k=5), 3),
            "Top-10 Accuracy": round(top_k_accuracy_score(y_encoded, y_proba, k=10), 3),
        }
