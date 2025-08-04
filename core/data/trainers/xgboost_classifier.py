import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
import joblib

from core.config import UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.preprocessing import balance_dataset


class XGBoostClassifier:
    def __init__(
        self,
        path=UNCOMPRESSED_PATH_STYLOMETRIC,
        sample_count=1000,
        test_size=0.2,
        random_state=42,
    ):
        self.path = path
        self.sample_count = sample_count
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns = [
            "char_count",
            "avg_word_length",
            "punct_ratio",
            "uppercase_ratio",
            "readability",
            "noun_ratio",
            "verb_ratio",
            "adj_ratio",
            "adv_ratio",
            "type_token_ratio",
            # "word_count",
            "hour",
            "day_of_week",
        ]
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.clf = None

    def load_and_balance_data(self):
        self.df = pd.read_parquet(self.path)
        self.df = balance_dataset(self.df, sample_count=self.sample_count)
        print(
            f"Dataset size after balancing: {len(self.df)} comments across {self.df['author'].nunique()} authors"
        )

    def prepare_data(self):
        self.X = self.df[self.feature_columns]
        y = self.df["author"]
        self.y = self.label_encoder.fit_transform(y)

    def split_data(self):
        return train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

    def train_model(self, X_train, y_train):
        self.clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.label_encoder.classes_),
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
            verbosity=2,
        )
        print("clf initialized")
        self.clf.fit(X_train, y_train, verbose=True)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        print("Top-1 Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        if hasattr(self.clf, "predict_proba"):
            y_proba = self.clf.predict_proba(X_test)
            top5_acc = top_k_accuracy_score(y_test, y_proba, k=5)
            print(f"Top-5 Accuracy: {top5_acc:.4f}")
        else:
            print("Model does not support probability outputs.")

    def save_artifacts(
        self,
        model_path="xgb_author_classifier.pkl",
        encoder_path="label_encoder.pkl",
    ):
        joblib.dump(self.clf, model_path)
        joblib.dump(self.label_encoder, encoder_path)

    def run(self, save=True):
        self.prepare_data()
        print("Data prepared")
        X_train, X_test, y_train, y_test = self.split_data()
        print("Data split")
        self.train_model(X_train, y_train)
        print("Model trained")
        self.evaluate_model(X_test, y_test)
        print("Model evaluated")
        if save:
            self.save_artifacts()


if __name__ == "__main__":
    classifier = XGBoostClassifier()
    classifier.load_and_balance_data()

    # explorer = DataExplorer(classifier.df, classifier.feature_columns)
    # explorer.run_full_exploration()

    classifier.run(save=True)
