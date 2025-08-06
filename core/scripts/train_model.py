import argparse
from core.config import UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.training.data_explorer import DataExplorer
from core.data.training.models.bayesian_author_model import BayesianAuthorModel
from core.data.training.models.xgboost_author_model import XGBoostAuthorModel
from core.data.training.trainers.author_model_trainer import AuthorModelTrainer

FEATURES = [
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
    "hour",
    "day_of_week",
]

MODEL_REGISTRY = {
    "xgboost": XGBoostAuthorModel,
    "bayesian": BayesianAuthorModel,
}


def main(model_name: str, run_exploration: bool = False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}"
        )

    print(f"\nUsing model: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()

    trainer = AuthorModelTrainer(model, FEATURES, UNCOMPRESSED_PATH_STYLOMETRIC)
    (X_train, X_test, y_train, y_test), df = trainer.load_data(
        trainer.path, trainer.sample_count, trainer.feature_columns
    )

    if run_exploration:
        print("Running data exploration...")
        explorer = DataExplorer(df, FEATURES)
        explorer.run_full_exploration()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print("Results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate authorship attribution models."
    )
    parser.add_argument(
        "--model", type=str, choices=MODEL_REGISTRY.keys(), default="xgboost"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        default=False,
        help="Run exploratory data analysis",
    )
    args = parser.parse_args()

    main(model_name=args.model, run_exploration=args.explore)
