import pandas as pd


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


def prune_dataset(df: pd.DataFrame, threshold: int, limit: int) -> pd.DataFrame:
    """Prune a dataset based on user comment counts.

    This function prunes a dataset to include comments from users who have at
    least a specified number of comments (threshold) and limits the number of
    comments per user to a maximum value (limit).

    Args:
        df (pd.DataFrame): The input dataframe to be pruned.
        threshold (int): The minimum number of comments an author must have for
            their comments to be included in the result.
        limit (int): The maximum number of comments allowed per author in the
            resulting dataframe.

    Returns:
        pd.DataFrame: A copy of the input dataframe, pruned according to the
        specified threshold and limit.

    Raises:
        ValueError: If threshold is greater than limit.

    Note:
        This function calls prune_below_user_count() and
        prune_above_user_count() to perform the pruning operations.
    """
    if threshold > limit:
        raise ValueError("Threshold must not be greater than limit")
    balanced_df = prune_below_user_count(df, threshold)
    balanced_df = prune_above_user_count(balanced_df, limit)
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
