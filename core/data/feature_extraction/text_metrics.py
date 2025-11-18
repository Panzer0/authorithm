from dataclasses import dataclass
from typing import List, Counter


@dataclass
class TextMetrics:
    """Container for basic text metrics."""

    words: List[str]
    word_lengths: List[int]
    word_count: int
    char_count: int
    letter_count: int
    pos_counts: Counter
