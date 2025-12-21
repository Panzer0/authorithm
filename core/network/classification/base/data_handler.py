import pandas as pd
from sklearn.model_selection import train_test_split

from core.data.preprocessing import balance_dataset, filter_by_feature


class DataHandlerMixin:
    def load_data(self, path, sample_count, feature_columns):
        df = pd.read_parquet(path)

        df = df.sort_values(by="id")

        df = filter_by_feature(df, "word_count", min_value=10)
        df = filter_by_feature(df, "avg_word_length", min_value=2.5)
        df = filter_by_feature(df, "readability", min_value=-5)
        df = filter_by_feature(df, "punct_ratio", max_value=0.5)
        df = balance_dataset(df, sample_count=sample_count)
        X = df[feature_columns]
        y = df["author"]
        return (
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
            df,
        )
