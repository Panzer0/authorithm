import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from core.data.training.base.author_model import AuthorModel


class XGBoostAuthorModel(AuthorModel):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.clf = None

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.label_encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
        )
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
