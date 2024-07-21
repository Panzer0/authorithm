# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"

# The default dataset's filename
# note: fantasy is stable, but small. fantasy_large is huge, but lacks the
#       embedding parameter.
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}_large_experimental.parquet.gzip"

# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"
