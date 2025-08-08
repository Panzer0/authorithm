import pytest

from core.data.processors.sanitizer import Sanitizer


@pytest.mark.parametrize(
    "text, expected_cleaned, expected_ratio",
    [
        ("**bold** text", "bold text", 4 / 9),
        ("*italic* and __underline__", "italic and underline", 15 / 20),
        ("`code` and ~~strike~~", "code and strike", 10 / 15),
        ("[label](https://url.com)", "label", 5 / 5),
        (">!secret!< is hidden", "secret is hidden", 6 / 16),
        # The newline is treated as uncovered
        ("> quoted\n> line", "quoted\nline", 10 / 11),
        ("---\nText\n---", "Text", 0.0),
        ("```def x():\n  return 1```", "def x():\n  return 1", 19 / 19),
        ("", "", 0.0),
    ],
)
def test_markdown_sanitization(text, expected_cleaned, expected_ratio):
    # Arrange
    sanitizer = Sanitizer(strip_raw_links=False)

    # Act
    cleaned, ratio = sanitizer.sanitize(text)

    # Assert
    assert cleaned == expected_cleaned
    assert round(ratio, 5) == round(expected_ratio, 5)


def test_strip_raw_links():
    # Arrange
    comment = "visit http://example.com and https://test.com"
    sanitizer = Sanitizer(strip_raw_links=True)

    # Act
    cleaned, ratio = sanitizer.sanitize(comment)

    # Assert
    assert cleaned == "visit  and"
    assert ratio == 0.0


def test_keep_raw_links():
    # Arrange
    comment = "go to http://example.com"
    sanitizer = Sanitizer(strip_raw_links=False)

    # Act
    cleaned, ratio = sanitizer.sanitize(comment)

    # Assert
    assert cleaned == comment
    assert ratio == 0.0


def test_nested_formatting():
    # Arrange
    comment = "*~~**[Nested](https://example.com)**~~* examples should work."
    sanitizer = Sanitizer(strip_raw_links=True)

    # Act
    cleaned, ratio = sanitizer.sanitize(comment)

    # Assert
    assert cleaned == "Nested examples should work."
    assert ratio == 6 / 28


def test_complex_combo():
    # Arrange
    comment = """
**bold** [link](https://foo.com) and `code`
>!spoiler!<
Visit: http://url.com
---
    """
    expected_cleaned = """bold link and code
spoiler
Visit:"""
    sanitizer = Sanitizer(strip_raw_links=True)

    # Act
    cleaned, ratio = sanitizer.sanitize(comment)

    # Assert
    print(f"Cleaned: \n{cleaned}")
    assert cleaned == expected_cleaned
    assert "bold" in cleaned
    assert "link" in cleaned
    assert "spoiler" in cleaned
    assert "http" not in cleaned
    assert "---" not in cleaned
    assert round(ratio, 3) == round((4 + 4 + 4 + 7) / len(cleaned), 3)
