import re
from typing import Tuple


class Sanitizer:
    """
    Comment sanitizer.
    Removes links, markdown and strips redundant whitespace.
    Computes the ratio of text that had formatting applied to it.
    """

    def __init__(self, strip_raw_links: bool = True):
        self.strip_raw_links = strip_raw_links
        self._compiled = self._compile_patterns()

    def _compile_patterns(self):
        return {
            "raw_links": re.compile(r"https?://[^\s)\]]+"),
            "horizontal_rules": re.compile(r"^\s*[*_\-]{3,}\s*$", re.MULTILINE),
            "markdown_patterns": [
                re.compile(r"\[([^]]*)]\([^)]*\)"),  # markdown link
                re.compile(r"```([\s\S]*?)```"),  # multiline code block
                re.compile(r"\*\*(.*?)\*\*"),  # bold
                re.compile(r"\*(.*?)\*"),  # italic
                re.compile(r"__(.*?)__"),  # underline
                re.compile(r"~~(.*?)~~"),  # strikethrough
                re.compile(r"`(.*?)`"),  # inline code
                re.compile(r">!(.*?)!<"),  # spoiler
                re.compile(r"^> ?(.*)", re.MULTILINE),  # blockquote
            ],
        }

    def _process_match(
        self, match: re.Match, offset: int
    ) -> Tuple[str, int, int]:
        """Cleans a single match: counts formatting, recurses, returns cleaned inner text and span."""
        start, end = match.span(1)
        global_start = offset + start
        global_end = offset + end

        if not any(
            i in self.counted_spans for i in range(global_start, global_end)
        ):
            self.formatted_char_count += end - start
            self.counted_spans.update(range(global_start, global_end))

        cleaned_inner = self._parse_markdown(match.group(1), offset + start)
        return cleaned_inner, match.start(), match.end()

    def _apply_pattern(
        self, text: str, pattern: re.Pattern, offset: int
    ) -> str:
        """Applies a single markdown pattern recursively and returns cleaned text."""
        matches = list(pattern.finditer(text))
        if not matches:
            return text

        new_text = []
        last_index = 0

        for match in matches:
            cleaned_inner, match_start, match_end = self._process_match(
                match, offset
            )
            new_text.append(text[last_index:match_start])
            new_text.append(cleaned_inner)
            last_index = match_end

        new_text.append(text[last_index:])
        return "".join(new_text)

    def _parse_markdown(self, text: str, offset: int = 0) -> str:
        for pattern in self._compiled["markdown_patterns"]:
            text = self._apply_pattern(text, pattern, offset)
        return text

    def sanitize(self, comment: str) -> Tuple[str, float]:
        self.cleaned_comment = comment
        self.formatted_char_count = 0
        self.counted_spans = set()

        if self.strip_raw_links:
            self.cleaned_comment = self._compiled["raw_links"].sub(
                "", self.cleaned_comment
            )

        self.cleaned_comment = self._parse_markdown(self.cleaned_comment)

        self.cleaned_comment = self._compiled["horizontal_rules"].sub(
            "", self.cleaned_comment
        )

        visible_len = len(self.cleaned_comment.strip())
        markdown_ratio = (
            self.formatted_char_count / visible_len if visible_len > 0 else 0.0
        )

        return self.cleaned_comment.strip(), markdown_ratio
