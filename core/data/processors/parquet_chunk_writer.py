import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq


class ParquetChunkWriter:
    """Writes to a parquet file in chunks.

    Attributes:
        parquet_file_path: The path of the parquet file to be written to.
        compression: The compression type of the parquet file.
        writer: The writer to write the parquet file.
    """

    def __init__(self, parquet_file_path: str, compression: str):
        """Inits ParquetChunkWriter.

        Args:
            parquet_file_path: The path of the parquet file to be written to.
            compression: The compression type of the parquet file.
        """
        self.parquet_file_path = parquet_file_path
        self.compression = compression
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.close()

    def write_chunk(self, data_chunk) -> None:
        """Writes a single data chunk

        Writes a single data chunk to the parquet file, compressing it with the
        specified compression type.

        Args:
            data_chunk: The data chunk to be written.
        """
        df = pd.DataFrame(data_chunk)
        table = pa.Table.from_pandas(df)

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.parquet_file_path,
                table.schema,
                compression=self.compression,
                use_dictionary=False,
                write_statistics=False,
            )

        self.writer.write_table(table)
