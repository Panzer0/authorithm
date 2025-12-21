import io
import json
import multiprocessing
import traceback
from datetime import datetime, timezone
from typing import Optional

import spacy
import zstandard as zstd
from tqdm import tqdm

from core.config import COMPRESSED_PATH, UNCOMPRESSED_PATH_STYLOMETRIC
from core.data.feature_extraction.feature_extractor import FeatureExtractor
from core.data.processors.parquet_chunk_writer import ParquetChunkWriter
from core.data.processors.sanitizer import Sanitizer

_worker_sanitizer: Optional[Sanitizer] = None
_worker_extractor: Optional[FeatureExtractor] = None


def worker_init():
    global _worker_sanitizer, _worker_extractor
    shared_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    _worker_extractor = FeatureExtractor(nlp_instance=shared_nlp)
    _worker_sanitizer = Sanitizer()


def process_line_task(line: str) -> Optional[dict]:
    global _worker_sanitizer, _worker_extractor

    try:
        data = json.loads(line)
        raw_body = data.get("body")

        if not raw_body:
            return None

        body, markdown_ratio = _worker_sanitizer.sanitize(raw_body)

        if not body:
            return None

        created_utc = int(data.get("created_utc"))
        dt = datetime.fromtimestamp(created_utc, timezone.utc)

        features = _worker_extractor.extract(body)

        filtered_data = {
            # Core data
            "id": data.get("id"),
            "author": data.get("author"),
            "body": body,
            # Time data
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
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

    except Exception:
        print(f"!!! WORKER ERROR processing line: {line[:50]}...")
        traceback.print_exc()
        return None


class PushshiftStylometricProcessor:
    """Handles the data from a Pushshift-obtained file.

    Handles the management of a (.zst-compressed) Pushshift-obtained file,
    focusing primarily on migrating it to a parquet file along with newly
    calculated stylometric data.
    """

    def __init__(self) -> None:
        pass

    def zst_to_parquet(
        self,
        zst_file_path: str,
        parquet_file_path: str,
        write_chunksize: int = 100000,  # RAM buffer
        process_chunksize: int = 500,  # Batch size
        num_workers: int = None,  # None = cpu_count()
        compression: str = "gzip",
    ) -> None:

        try:
            total_lines = self._count_lines(zst_file_path)
            print(f"Total lines in file: {total_lines}")

            pool = multiprocessing.Pool(
                processes=num_workers, initializer=worker_init
            )

            buffer = []
            processed_count = 0

            with self._open_zst_file(
                zst_file_path
            ) as text_stream, ParquetChunkWriter(
                parquet_file_path, compression
            ) as writer, tqdm(
                total=total_lines, desc="Processing", unit="lines"
            ) as pbar:

                results_iterator = pool.imap(
                    process_line_task, text_stream, chunksize=process_chunksize
                )

                for result in results_iterator:
                    if result is not None:
                        buffer.append(result)

                    if len(buffer) >= write_chunksize:
                        writer.write_chunk(buffer)
                        processed_count += len(buffer)
                        buffer = []
                        pbar.set_postfix({"Saved": processed_count})

                    pbar.update(1)

                if buffer:
                    writer.write_chunk(buffer)
                    processed_count += len(buffer)

            pool.close()
            pool.join()

            print(
                f"\nConversion complete. {processed_count} comments saved to {parquet_file_path}."
            )

        except FileNotFoundError:
            print(f"Error: The file {zst_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if "pool" in locals():
                pool.terminate()

    def _count_lines(self, zst_file_path: str) -> int:
        total_lines = 0
        with self._open_zst_file(zst_file_path) as text_stream:
            for _ in tqdm(text_stream, desc="Counting lines", unit=" lines"):
                total_lines += 1
        return total_lines

    def _open_zst_file(self, file_path):
        fh = open(file_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding="utf-8")


if __name__ == "__main__":
    collector = PushshiftStylometricProcessor()
    collector.zst_to_parquet(
        COMPRESSED_PATH,
        UNCOMPRESSED_PATH_STYLOMETRIC,
        write_chunksize=50000,
        process_chunksize=200,
    )
