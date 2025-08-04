import pandas as pd
from sklearn.model_selection import train_test_split
from core.data.preprocessing import balance_dataset


class DataHandlerMixin:
    def load_data(self, path, sample_count, feature_columns):
        df = pd.read_parquet(path)
        df = balance_dataset(df, sample_count=sample_count)
        X = df[feature_columns]
        y = df["author"]
        return (
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
            df,
        )
