import pandas as pd

# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"
# The default dataset's filename
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}.parquet.gzip"
# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"

#
def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('author')
            .filter(lambda x: len(x) >= 50)
            .groupby('author')
            .apply(lambda x: x.sample(n=50, random_state=42))
            .reset_index(level=0, drop=True)
    )

