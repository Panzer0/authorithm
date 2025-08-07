class EmptyCommentException(Exception):
    """
    Raised when a comment is empty and should be skipped.

    The comment does not have to be empty initially - its emptiness can be a
    result of sanitization.
    """
    pass
