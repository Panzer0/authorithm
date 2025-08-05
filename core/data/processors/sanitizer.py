import re
from typing import Tuple


class Sanitizer:
    """
    Comment sanitizer.
    Removes links, markdown and strips redundant whitespace.
    Computes the ratio of text that had formatting applied to it.
    """

    def __init__(self, strip_raw_links: bool = False):
        self.strip_raw_links = strip_raw_links
        self._compiled = self._compile_patterns()

    def _compile_patterns(self):
        return {
            "raw_links": re.compile(r"https?://[^\s)\]]+"),
            "horizontal_rules": re.compile(
                r"^\s*[*_\-]{3,}\s*$", re.MULTILINE
            ),
            "markdown_patterns": [
                re.compile(r"```([\s\S]*?)```"),  # multiline code block
                re.compile(r"\*\*(.*?)\*\*"),  # bold
                re.compile(r"\*(.*?)\*"),  # italic
                re.compile(r"__(.*?)__"),  # underline
                re.compile(r"~~(.*?)~~"),  # strikethrough
                re.compile(r"`(.*?)`"),  # inline code
                re.compile(r">!(.*?)!<"),  # spoiler
                re.compile(r"\[([^]]*)]\([^)]*\)"),  # markdown link
                re.compile(r"^> ?(.*)", re.MULTILINE),  # blockquote
            ],
        }

    def sanitize(self, comment: str) -> Tuple[str, float]:
        self.cleaned_comment = comment
        self.formatted_char_count = 0

        if self.strip_raw_links:
            self.cleaned_comment = self._compiled["raw_links"].sub(
                "", self.cleaned_comment
            )

        for pattern in self._compiled["markdown_patterns"]:

            def replace(m: re.Match):
                content = m.group(1)
                self.formatted_char_count += len(content)
                return content

            self.cleaned_comment = pattern.sub(replace, self.cleaned_comment)

        self.cleaned_comment = self._compiled["horizontal_rules"].sub(
            "", self.cleaned_comment
        )

        visible_len = len(self.cleaned_comment.strip())
        if visible_len == 0:
            markup_ratio = 0.0
        else:
            markup_ratio = self.formatted_char_count / visible_len

        return self.cleaned_comment.strip(), markup_ratio
