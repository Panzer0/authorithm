import argparse
from core.config import UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.loaders.stylometric_data_loader import StylometricDataLoader
from core.network.classification.models.bayesian_author_model import (
    BayesianAuthorModel,
)
from core.network.classification.models.xgboost_author_model import (
    XGBoostAuthorModel,
)
from core.network.classification.trainers.author_model_trainer import (
    AuthorModelTrainer,
)


FEATURES = [
    # "char_count",
    "word_count",
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
    "markdown_ratio",
]

DATA_FILTERS = [
    {"feature": "word_count", "min_value": 10},
    {"feature": "avg_word_length", "min_value": 2.5},
    {"feature": "readability", "min_value": -5},
    {"feature": "punct_ratio", "max_value": 0.5},
]

MODEL_REGISTRY = {
    "xgboost": XGBoostAuthorModel,
    "bayesian": BayesianAuthorModel,
}


def main(
    model_name: str,
    run_exploration: bool = False,
    display_mode: str = None,
    sample_count: int = 1000,
):
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model '{model_name}'.")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    print(f"\nUsing model: {model_name}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()

    data_loader = StylometricDataLoader(
        path=UNCOMPRESSED_PATH_STYLOMETRIC,
        feature_columns=FEATURES,
        filters=DATA_FILTERS,
        sample_count=sample_count,
    )

    trainer = AuthorModelTrainer(model=model, data_loader=data_loader)

    trainer.run(run_exploration=run_exploration, display_mode=display_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate authorship attribution models."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_REGISTRY.keys(),
        default="xgboost",
        help="Type of model to use (default: xgboost).",
    )

    parser.add_argument(
        "--explore",
        action="store_true",
        default=False,
        help="Run exploratory data analysis before training.",
    )

    parser.add_argument(
        "--display_mode",
        type=str,
        default="grid",
        choices=["grid", "sequential", "save"],
        help=(
            "How exploration plots should be displayed:\n"
            "  grid       - grid layout (default)\n"
            "  sequential - show one by one\n"
            "  save       - save to ./plots/[timestamp]/ folder\n"
            "(default: grid)"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per author for dataset balancing (default: 1000).",
    )

    args = parser.parse_args()

    main(
        model_name=args.model,
        run_exploration=args.explore,
        display_mode=args.display_mode,
        sample_count=args.samples,
    )
