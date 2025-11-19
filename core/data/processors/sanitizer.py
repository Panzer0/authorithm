import re
from typing import Tuple, List, Dict, Any


class Sanitizer:
    """
    Comment sanitizer.
    Removes links, markdown and strips redundant whitespace.
    Computes the ratio of text that had formatting applied to it.
    """

    def __init__(self, strip_raw_links: bool = True):
        self.strip_raw_links = strip_raw_links
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Any]:
        return {
            "raw_links": re.compile(r"https?://[^\s)\]]+"),
            "horizontal_rules": re.compile(r"^\s*[*_\-]{3,}\s*$", re.MULTILINE),
            "markdown_patterns": [
                re.compile(r"\[([^]]*)]\([^)]*\)"),  # Hyperlink
                re.compile(r"```([\s\S]*?)```"),  # Code block
                re.compile(r"\*\*(.*?)\*\*"),  # Bold
                re.compile(r"\*(.*?)\*"),  # Italic
                re.compile(r"__(.*?)__"),  # Underline
                re.compile(r"~~(.*?)~~"),  # Strikethrough
                re.compile(r"`(.*?)`"),  # Inline code
                re.compile(r">!(.*?)!<"),  # Spoiler
                re.compile(r"^> ?(.*)", re.MULTILINE),  # Blockquote
            ],
        }

    def _apply_removal(
        self, text: str, mask: List[bool], pattern: re.Pattern
    ) -> Tuple[str, List[bool]]:
        matches = list(pattern.finditer(text))
        if not matches:
            return text, mask

        new_text_parts = []
        new_mask_parts = []
        last_idx = 0

        for match in matches:
            start, end = match.span()
            new_text_parts.append(text[last_idx:start])
            new_mask_parts.append(mask[last_idx:start])
            last_idx = end

        new_text_parts.append(text[last_idx:])
        new_mask_parts.append(mask[last_idx:])

        return "".join(new_text_parts), [
            m for sub in new_mask_parts for m in sub
        ]

    def _apply_formatting(
        self, text: str, mask: List[bool], pattern: re.Pattern
    ) -> Tuple[str, List[bool]]:
        matches = list(pattern.finditer(text))
        if not matches:
            return text, mask

        new_text_parts = []
        new_mask_parts = []
        last_idx = 0

        for match in matches:
            start, end = match.span()
            new_text_parts.append(text[last_idx:start])
            new_mask_parts.append(mask[last_idx:start])

            if match.lastindex and match.lastindex >= 1:
                g_start, g_end = match.span(1)
                len_inner = g_end - g_start
                new_text_parts.append(match.group(1))
                new_mask_parts.append([True] * len_inner)

            last_idx = end

        new_text_parts.append(text[last_idx:])
        new_mask_parts.append(mask[last_idx:])

        return "".join(new_text_parts), [
            m for sub in new_mask_parts for m in sub
        ]

    def sanitize(self, comment: str) -> Tuple[str, float]:
        if not comment:
            return "", 0.0

        current_text = comment
        current_mask = [False] * len(comment)

        if self.strip_raw_links:
            current_text, current_mask = self._apply_removal(
                current_text, current_mask, self._patterns["raw_links"]
            )

        for pattern in self._patterns["markdown_patterns"]:
            current_text, current_mask = self._apply_formatting(
                current_text, current_mask, pattern
            )

        current_text, current_mask = self._apply_removal(
            current_text, current_mask, self._patterns["horizontal_rules"]
        )

        stripped_text = current_text.strip()
        if not stripped_text:
            return "", 0.0

        leading_spaces = len(current_text) - len(current_text.lstrip())
        final_mask = current_mask[
            leading_spaces : leading_spaces + len(stripped_text)
        ]

        formatted_count = sum(final_mask)
        visible_len = len(stripped_text)

        ratio = formatted_count / visible_len if visible_len > 0 else 0.0
        return stripped_text, min(ratio, 1.0)
