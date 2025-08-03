import spacy
import string
import textstat
import numpy as np
from collections import Counter
from typing import Dict


class FeatureExtractor:
    """Extracts stylometric features from a given text.

    Attributes:
        nlp: A spaCy language pipeline object.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initializes the extractor with a spaCy model."""
        # TODO: NER and parser were removed for performance reasons.
        # TODO: Confirm that I have no need for them.
        self.nlp = spacy.load(model_name, disable=["ner", "parser"])

    def extract(self, text: str) -> Dict[str, float]:
        """Extracts a stylometric feature set from the given text.

        Args:
            text: The input string (comment body).

        Returns:
            A dictionary of computed stylometric features.
        """
        doc = self.nlp(text)
        words = [token.text for token in doc if token.is_alpha]
        word_lengths = [len(w) for w in words]
        word_count = len(words)
        char_count = len(text)

        pos_counts = Counter([token.pos_ for token in doc])
        total_pos = sum(pos_counts.values())

        # Stylometric ratios
        type_token_ratio = len(set(words)) / word_count if word_count else 0
        punct_ratio = (
            sum(1 for c in text if c in string.punctuation) / char_count
            if char_count
            else 0
        )
        uppercase_ratio = (
            sum(1 for c in text if c.isupper()) / char_count
            if char_count
            else 0
        )

        noun_ratio = pos_counts.get("NOUN", 0) / total_pos if total_pos else 0
        verb_ratio = pos_counts.get("VERB", 0) / total_pos if total_pos else 0
        adj_ratio = pos_counts.get("ADJ", 0) / total_pos if total_pos else 0
        adv_ratio = pos_counts.get("ADV", 0) / total_pos if total_pos else 0

        # Readability score
        try:
            readability = textstat.flesch_reading_ease(text)
        except Exception:
            readability = 0.0

        return {
            "char_count": char_count,
            "avg_word_length": float(np.mean(word_lengths))
            if word_lengths
            else 0,
            "punct_ratio": punct_ratio,
            "uppercase_ratio": uppercase_ratio,
            "readability": readability,
            "noun_ratio": noun_ratio,
            "verb_ratio": verb_ratio,
            "adj_ratio": adj_ratio,
            "adv_ratio": adv_ratio,
            "type_token_ratio": type_token_ratio,
            "word_count": word_count,
        }
