import json
import pandas as pd
from core.data.Embedder import Embedder
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import zstandard as zstd
import io
from core.config import UNCOMPRESSED_PATH, COMPRESSED_PATH


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


class PushshiftCollector:
    """Handles the management of a (.zst-compressed) Pushshift-obtained file,
    focusing primarily on migrating it to a parquet file along with newly
    generated embeddings.

    Attributes:
        embedder: An Embedder object responsible for embedding generation.
    """
    def __init__(self) -> None:
        """Inits PushshiftCollector.
        """
        self.embedder = Embedder()

    def zst_to_parquet_with_embeddings(
        self,
        zst_file_path: str,
        parquet_file_path: str,
        chunksize: int = 100000,
        compression: str = "gzip",
    ):
        try:
            total_lines = self._count_lines(zst_file_path)
            print(f"Total lines in file: {total_lines}")

            with self._open_zst_file(
                zst_file_path
            ) as text_stream, ParquetChunkWriter(
                parquet_file_path, compression
            ) as writer:
                self._process_stream(
                    text_stream, writer, chunksize, total_lines
                )

            print(
                f"\nConversion complete. Output saved to {parquet_file_path} with {compression} compression."
            )
        except FileNotFoundError:
            print(f"Error: The file {zst_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def _count_lines(self, zst_file_path: str) -> int:
        total_lines = 0
        with self._open_zst_file(zst_file_path) as text_stream:
            for _ in tqdm(text_stream, desc="Counting lines", unit=" lines"):
                total_lines += 1
        return total_lines

    def _process_stream(
        self, text_stream, writer, chunksize: int, total_lines: int
    ):
        data_chunks = []
        chunk_count = 0

        with tqdm(
            total=total_lines, desc="Processing lines", unit="line"
        ) as pbar:
            for line in text_stream:
                try:
                    data = self._process_line(line)
                    data_chunks.append(data)
                    if len(data_chunks) == chunksize:
                        writer.write_chunk(data_chunks)
                        data_chunks.clear()
                        chunk_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"Chunks": chunk_count})
                except json.JSONDecodeError:
                    pbar.write(
                        f"Warning: Skipping invalid JSON on line {pbar.n + 1}"
                    )

        if data_chunks:
            writer.write_chunk(data_chunks)

    def _open_zst_file(self, file_path):
        fh = open(file_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding="utf-8")

    def _process_line(self, line):
        data = json.loads(line)
        body = data.get("body", "")
        embedding = self.embedder.embed_str(body)
        embedding_dict = self.embedder.to_dict(embedding)
        word_count = len(body.split())

        filtered_data = {
            "id": data.get("id"),
            "author": data.get("author"),
            "body": body,
            "word_count": word_count,
        }
        filtered_data.update(embedding_dict)
        return filtered_data


if __name__ == "__main__":
    collector = PushshiftCollector()
    collector.zst_to_parquet_with_embeddings(COMPRESSED_PATH, UNCOMPRESSED_PATH)
