from core.data.data_exploration.data_explorer import DataExplorer
from core.network.classification.base.data_handler import DataHandlerMixin


class AuthorModelTrainer(DataHandlerMixin):
    def __init__(self, model, feature_columns, path, sample_count=1000):
        self.model = model
        self.feature_columns = feature_columns
        self.path = path
        self.sample_count = sample_count

    def prepare_data(self):
        (X_train, X_test, y_train, y_test), df = self.load_data(
            path=self.path,
            sample_count=self.sample_count,
            feature_columns=self.feature_columns,
        )
        return X_train, X_test, y_train, y_test, df

    def run(self, run_exploration=False, display_mode=None):
        X_train, X_test, y_train, y_test, df = self.prepare_data()

        if run_exploration:
            print("Running data exploration...")
            explorer = (
                DataExplorer(df, self.feature_columns, display_mode)
                if display_mode
                else DataExplorer(df, self.feature_columns)
            )
            explorer.run_full_exploration()

        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

        print("Evaluating...")
        results = self.model.evaluate(X_test, y_test)
        print("Results:", results)
