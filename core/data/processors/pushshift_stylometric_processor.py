import io
import json
from datetime import datetime

import zstandard as zstd
from tqdm import tqdm

from core.config import COMPRESSED_PATH, UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.embedder import Embedder
from core.data.feature_extraction.feature_extractor import FeatureExtractor
from core.data.processors.empty_comment_exception import EmptyCommentException
from core.data.processors.parquet_chunk_writer import ParquetChunkWriter
from core.data.processors.sanitizer import Sanitizer


class PushshiftStylometricProcessor:
    """Handles the data from a Pushshit-obtained file.

    Handles the management of a (.zst-compressed) Pushshift-obtained file,
    focusing primarily on migrating it to a parquet file along with newly
    generated embeddings.

    Attributes:
        embedder: An Embedder object responsible for embedding generation.
        feature_extractor: A FeatureExtractor object responsible for extraction
            of stylometric features.
    """

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.feature_extractor = FeatureExtractor()
        self.sanitizer = Sanitizer()

    def zst_to_parquet_with_embeddings(
        self,
        zst_file_path: str,
        parquet_file_path: str,
        chunksize: int = 100000,
        compression: str = "gzip",
    ) -> None:
        """Writes data from a zst file and its embeddings to a parquet file.

        Args:
            zst_file_path: The path of the zst file to be read from.
            parquet_file_path: The path of the parquet file to be written to.
            chunksize: The size of the chunks to be written.
            compression: The compression type of the parquet file.

        :raises
            FileNotFoundError: If the zst-file does not exist.
            Exception: If other issues occur.
        """
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
        """Counts the lines in a zst file.

        Counts the lines in a zst file. A progress bar is provided via the tqdm
        library.

        Args:
            zst_file_path: The path of the zst file to count lines from.
        """
        total_lines = 0
        with self._open_zst_file(zst_file_path) as text_stream:
            for _ in tqdm(text_stream, desc="Counting lines", unit=" lines"):
                total_lines += 1
        return total_lines

    def _process_stream(
        self, text_stream, writer, chunksize: int, total_lines: int
    ) -> None:
        """Processes the input stream and writes it to a parquet file in chunks.

        Processed the provided input stream, generating embeddings for each
        contained comment's body, and writes it to a parquet file in chunks.
        A progress bar is provided via the tqdm library.

        Args:
            text_stream: The input stream to be processed.
            writer: The ParquetChunkWriter to write the parquet file.
            chunksize: The size of the chunks to be written.
            total_lines: The total number of lines to be written.
        """
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
                except EmptyCommentException:
                    pass
                except json.JSONDecodeError:
                    pbar.write(
                        f"Warning: Skipping invalid JSON on line {pbar.n + 1}"
                    )

        if data_chunks:
            writer.write_chunk(data_chunks)

    def _open_zst_file(self, file_path):
        """Opens a zst file.

        returns:
         text stream of the zst file's contents
        """
        fh = open(file_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding="utf-8")

    def _extract_time_features(self, created_utc: int) -> dict:
        dt = datetime.utcfromtimestamp(created_utc)
        return {
            "hour": dt.hour,  # 0–23
            "day_of_week": dt.weekday(),  # 0=Monday, 6=Sunday
        }

    def _process_line(self, line) -> dict:
        """Processes a single line of a zst file's text stream.

        Processes a single line of a zst file's text stream, enriching it with
        a count of words in the given comment and an embedding of its body.

        Args:
            line: The line to be processed.

        :returns:
            A dictionary containing the comment's id, author name, body, word
             count and embedding, each of its values stored under a separate
             key.
        """
        data = json.loads(line)
        body = data.get("body", "")
        body, markdown_ratio = self.sanitizer.sanitize(body)

        if not body:
            raise EmptyCommentException

        created_utc = int(data.get("created_utc"))
        time_features = self._extract_time_features(created_utc)

        features = self.feature_extractor.extract(body)

        filtered_data = {
            # Core data
            "id": data.get("id"),
            "author": data.get("author"),
            "body": body,
            # Time data
            "hour": time_features["hour"],
            "day_of_week": time_features["day_of_week"],
            # Stylometric data
            "char_count": features["char_count"],
            "avg_word_length": features["avg_word_length"],
            "punct_ratio": features["punct_ratio"],
            "uppercase_ratio": features["uppercase_ratio"],
            "readability": features["readability"],
            "noun_ratio": features["noun_ratio"],
            "verb_ratio": features["verb_ratio"],
            "adj_ratio": features["adj_ratio"],
            "adv_ratio": features["adv_ratio"],
            "type_token_ratio": features["type_token_ratio"],
            "word_count": features["word_count"],
            "markdown_ratio": markdown_ratio,
        }
        return filtered_data


if __name__ == "__main__":
    collector = PushshiftStylometricProcessor()
    collector.zst_to_parquet_with_embeddings(
        COMPRESSED_PATH, UNCOMPRESSED_PATH_STYLOMETRIC
    )
