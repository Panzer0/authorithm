import pandas as pd

from core.config import UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.plotter import Plotter
from core.data.preprocessing import filter_by_feature, balance_dataset

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
    "markup_ratio",
]


def main():
    dataset = pd.read_parquet(UNCOMPRESSED_PATH_STYLOMETRIC)

    # todo: Code duplication with DataHandlerMixin. Centralise.
    dataset = filter_by_feature(dataset, "word_count", min_value=5)
    dataset = filter_by_feature(dataset, "avg_word_length", min_value=2.5)
    dataset = filter_by_feature(dataset, "readability", min_value=-5)
    dataset = filter_by_feature(dataset, "punct_ratio", max_value=0.5)
    dataset = filter_by_feature(dataset, "markup_ratio", max_value=1)

    dataset = balance_dataset(dataset, 1000)

    plotter = Plotter(dataset)
    plotter.plot_PCA(FEATURES)


if __name__ == "__main__":
    main()