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
        A balanced copy of the given dataframe.
    """
    balanced_df = prune_below_user_count(df, sample_count)
    balanced_df = prune_above_user_count(balanced_df, sample_count)
    return balanced_df


def prune_below_user_count(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Prunes comments from users whose comment count is below threshold.

    Removed comments from users whose comment count is below the given
    threshold. This is achieved by applying a filter to the dataframe which only
    permits comments belonging to authors that occur more than threshold times.

    Args:
        df: The dataframe that is to be pruned.
        threshold: The minimal amount of comments belonging to an author
         necessary for their comments to qualify to the resulting dataframe.

    Returns:
        A copy of the given dataframe without comments from authors who don't
         meet the threshold.
    """
    return df.groupby("author").filter(lambda x: len(x) >= threshold)


def prune_above_user_count(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Enforces a limit of comments from each author on a dataframe.

    Enforces a certain limit of comments from each author on the given
    dataframe. This is achieved by leaving users below the authors below the
    limit intact and sampling the permitted amount of comments for users that
    exceed said limit.

    Args:
        df: The dataframe that is to be pruned.
        limit: The maximal permitted amount of comments belonging to the same
         author in the resulting dataframe.

    Returns:
        A copy of the given dataframe in which no author's comment count exceeds
        the given limit.
    """
    return (
        df.groupby("author")
        .apply(lambda x: x.sample(n=min(len(x), limit), random_state=42))
        .reset_index(level=0, drop=True)
    )
