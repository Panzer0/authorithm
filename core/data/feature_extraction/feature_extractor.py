import math
import string
from collections import Counter
from typing import Dict, List

import spacy
import textstat

from core.data.feature_extraction.text_metrics import TextMetrics


class FeatureExtractor:
    """Extracts stylometric features from text for authorship attribution."""

    DEFAULT_MODEL = "en_core_web_sm"
    DISABLED_PIPES = ["ner", "parser"]

    def __init__(self, nlp_instance=None, model_name: str = None):
        """
        Initialize the feature extractor.
        """
        if nlp_instance:
            self.nlp = nlp_instance
        else:
            model_to_load = model_name or self.DEFAULT_MODEL
            self.nlp = spacy.load(model_to_load, disable=self.DISABLED_PIPES)

    def extract(self, text: str) -> Dict[str, float]:
        """Extract stylometric features from text."""
        if not text:
            return self._get_empty_features()

        doc = self.nlp(text)
        metrics = self._compute_base_metrics(doc, text)

        features = {}
        features.update(self._compute_basic_features(metrics))
        features.update(self._compute_character_ratios(text, metrics))
        features.update(self._compute_pos_ratios(metrics))
        features["readability"] = self._compute_readability(
            text, metrics.word_count
        )

        return features

    @staticmethod
    def _compute_base_metrics(doc: spacy.tokens.Doc, text: str) -> TextMetrics:
        """Compute basic text metrics."""
        words = []
        word_lengths = []
        valid_pos_tags = []

        for token in doc:
            if not token.is_punct and not token.is_space:
                words.append(token.text.lower())
                word_lengths.append(len(token.text))
                valid_pos_tags.append(token.pos_)

        pos_counts = Counter(valid_pos_tags)
        letter_count = sum(1 for c in text if c.isalpha())

        return TextMetrics(
            words=words,
            word_lengths=word_lengths,
            word_count=len(words),
            char_count=len(text),
            letter_count=letter_count,
            pos_counts=pos_counts,
        )

    def _compute_basic_features(self, metrics: TextMetrics) -> Dict[str, float]:
        """Compute basic count and length features."""
        unique_words = len(set(metrics.words))
        total_words = metrics.word_count

        type_token_ratio = (
            unique_words / math.sqrt(total_words) if total_words > 0 else 0.0
        )

        return {
            "char_count": float(metrics.char_count),
            "word_count": float(metrics.word_count),
            "avg_word_length": self._safe_mean(metrics.word_lengths),
            "type_token_ratio": type_token_ratio,
        }

    def _compute_character_ratios(
        self, text: str, metrics: TextMetrics
    ) -> Dict[str, float]:
        """Compute character-based ratios."""
        punct_count = sum(1 for c in text if c in string.punctuation)
        uppercase_count = sum(1 for c in text if c.isupper())

        return {
            "punct_ratio": self._safe_divide(punct_count, metrics.char_count),
            "uppercase_ratio": self._safe_divide(
                uppercase_count, metrics.letter_count
            ),
        }

    def _compute_pos_ratios(self, metrics: TextMetrics) -> Dict[str, float]:
        """Compute part-of-speech ratios."""
        noun_count = metrics.pos_counts.get("NOUN", 0) + metrics.pos_counts.get(
            "PROPN", 0
        )
        verb_count = metrics.pos_counts.get("VERB", 0) + metrics.pos_counts.get(
            "AUX", 0
        )
        adj_count = metrics.pos_counts.get("ADJ", 0)
        adv_count = metrics.pos_counts.get("ADV", 0)

        return {
            "noun_ratio": self._safe_divide(noun_count, metrics.word_count),
            "verb_ratio": self._safe_divide(verb_count, metrics.word_count),
            "adj_ratio": self._safe_divide(adj_count, metrics.word_count),
            "adv_ratio": self._safe_divide(adv_count, metrics.word_count),
        }

    @staticmethod
    def _compute_readability(text: str, word_count: int) -> float:
        """Compute readability score."""
        try:
            if word_count < 5:
                return 0.0
            return float(textstat.flesch_reading_ease(text))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safely divide two numbers."""
        return float(numerator / denominator) if denominator != 0 else 0.0

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Safely compute mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _get_empty_features() -> Dict[str, float]:
        """Return feature dict with zero values for empty text."""
        return {
            "char_count": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "type_token_ratio": 0.0,
            "punct_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "readability": 0.0,
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adj_ratio": 0.0,
            "adv_ratio": 0.0,
        }
