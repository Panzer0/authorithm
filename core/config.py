# Name of the subreddit the default dataset is derived from
SUBREDDIT_NAME = "fantasy"

## Values for the purpose of read-only methods

# The default dataset's filename
# note: fantasy is stable, but small. fantasy_large is huge, but lacks the
#       embedding parameter.
DATASET_FILENAME = f"dataset_{SUBREDDIT_NAME}_large_experimental.parquet.gzip"

# The default dataset's path
DATASET_PATH = f"datasets/{DATASET_FILENAME}"
# The default PCA-transformed dataset's path
DATASET_PCA_PATH = f"datasets/pca/{DATASET_FILENAME}"


## Processor-related values

# The .zst-compressed dataset's filename
COMPRESSED_FILENAME = f"{SUBREDDIT_NAME}_comments.zst"
# The .zst-compressed dataset's path
COMPRESSED_PATH = f"raw_data/compressed/{COMPRESSED_FILENAME}"

# The parquet-converted dataset's filename
UNCOMPRESSED_FILENAME_STYLOMETRIC = (
    f"dataset_{SUBREDDIT_NAME}_stylometric.parquet.gzip"
)
UNCOMPRESSED_FILENAME_JINA = (
    f"dataset_{SUBREDDIT_NAME}_jina.parquet.gzip"
)
# The parquet-converted dataset's path
UNCOMPRESSED_PATH_STYLOMETRIC = f"datasets/{UNCOMPRESSED_FILENAME_STYLOMETRIC}"
UNCOMPRESSED_PATH_JINA = f"datasets/{UNCOMPRESSED_FILENAME_JINA}"
