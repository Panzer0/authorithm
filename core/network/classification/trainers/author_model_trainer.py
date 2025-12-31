import time
from datetime import timedelta

from core.data.data_exploration.data_explorer import DataExplorer


class AuthorModelTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def run(self, run_exploration=False, display_mode=None):
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.data_loader.get_train_test_split()

        if run_exploration:
            print("Running data exploration...")
            df = self.data_loader.full_dataframe
            explorer = DataExplorer(df, self.data_loader.feature_columns)

            if display_mode:
                explorer.display_mode = display_mode

            explorer.run_full_exploration()

        start_time = (
            time.perf_counter()
        )

        self.model.fit(X_train, y_train)

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        formatted_time = str(timedelta(seconds=elapsed_seconds))

        print(
            f"Training completed in: {formatted_time} ({elapsed_seconds:.2f}s)"
        )
        print("-" * 50)

        print("Evaluating...")
        results = self.model.evaluate(X_test, y_test)
        print("Results:", results)
