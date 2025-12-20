import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from core.data.preprocessing import balance_dataset, filter_by_feature


class StylometricDataLoader:
    """
    Handles loading, filtering, balancing, and splitting of stylometric data.
    """

    def __init__(
        self,
        path: str,
        feature_columns: List[str],
        filters: List[Dict] = None,
        sample_count: int = 1000,
        random_state: int = 42,
    ):
        self.path = path
        self.feature_columns = feature_columns
        self.filters = filters or []
        self.sample_count = sample_count
        self.random_state = random_state
        self._df = None

    def load_and_process(self) -> pd.DataFrame:
        """Loads data from file and applies preprocessing pipeline."""
        print(f"Loading data from {self.path}...")
        df = pd.read_parquet(self.path)

        df = df.sort_values(by="id")

        for filter_cfg in self.filters:
            feature = filter_cfg["feature"]
            kwargs = {k: v for k, v in filter_cfg.items() if k != "feature"}
            df = filter_by_feature(df, feature, **kwargs)
            print(
                f"Applied filter on '{feature}': {kwargs}. Remaining: {len(df)}"
            )

        df = balance_dataset(df, sample_count=self.sample_count)
        self._df = df
        return df

    def get_train_test_split(self, test_size=0.2) -> Tuple:
        """Returns X_train, X_test, y_train, y_test."""
        if self._df is None:
            self.load_and_process()

        X = self._df[self.feature_columns]
        y = self._df["author"]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

    @property
    def full_dataframe(self) -> pd.DataFrame:
        """Access to the full processed dataframe."""
        if self._df is None:
            self.load_and_process()
        return self._df
