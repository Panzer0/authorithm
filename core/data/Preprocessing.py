import pandas as pd

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"


def balance_dataset(df: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    """Balances a dataset.

    Returns a balanced version of the provided data. A balanced dataset
    is one that contains an equal amount of samples belonging to each class
    - in this case the authorship. This is achieved by omitting the classes
    that have fewer samples than the desired amount, then randomly sampling
    the same amount from the remaining ones.

    Args:
        df: The dataframe that is to be balanced.
        sample_count: The desired amount of samples of each class in the
         balanced dataset.

    Returns:
        A balanced version of the given dataframe.
    """
    return (
        df.groupby("author")
            .filter(lambda x: len(x) >= sample_count)
            .groupby("author")
            .apply(lambda x: x.sample(n=sample_count, random_state=42))
            .reset_index(level=0, drop=True)
    )

