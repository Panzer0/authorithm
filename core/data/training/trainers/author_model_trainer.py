from core.data.training.base.data_handler import DataHandlerMixin


class AuthorModelTrainer(DataHandlerMixin):
    def __init__(self, model, feature_columns, path, sample_count=1000):
        self.model = model
        self.feature_columns = feature_columns
        self.path = path
        self.sample_count = sample_count

    def run(self):
        (X_train, X_test, y_train, y_test), df = self.load_data(
            path=self.path,
            sample_count=self.sample_count,
            feature_columns=self.feature_columns,
        )

        print(f"Training on {len(X_train)} samples")
        self.model.fit(X_train, y_train)

        print("Evaluating...")
        results = self.model.evaluate(X_test, y_test)
        print(results)
