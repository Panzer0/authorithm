import praw, prawcore
import pandas as pd
from core.data.Embedder import Embedder

# Default name of the subreddit to gather comments from
SUBREDDIT_NAME = "writing"
# The dataset's default filename
DATASET_FILENAME = f"{SUBREDDIT_NAME}.parquet.gzip"
# The dataset's default path
DATASET_PATH = f"datasets/dataset_{DATASET_FILENAME}"

# Default site name for praw's Reddit object
SITE_NAME = "Authorithm"
# Default user agent for praw's Reddit object
USER_AGENT = "authorithm by /u/pazur13"

# Default amount of posts to inspect in gather_comments_from_hot()
POST_LIMIT = 10
# Default limit of comments to inspect in gather_comments_from_all()
COMMENT_LIMIT = 20000

# Default limit of comments to be inspected in each given user's history
HISTORY_LIMIT = 150
# Default threshold of user's comment count in the given subreddit
COMMENT_QUOTA = 25


class RedditCollector:
    def __init__(self, subreddit_name: str, site_name: str, agent: str) -> None:
        """Inits RedditCollector.

        Args:
            subreddit_name: The name of the target subreddit.
            site_name: The name of the section in the praw.ini file from which
             the settings are to be loaded. Due to private information therein,
             praw.ini must be configured manually for each user of this program.
            agent: App's unique identifier required by Reddit API's security.
        """
        self.reddit = praw.Reddit(site_name, user_agent=agent)
        self.subreddit = self.reddit.subreddit(subreddit_name)
        self.dataset = []
        self.checked_users = set()
        self.embedder = Embedder()

    def is_valid_comment(self, comment: praw.models.Comment) -> bool:
        """Validates a comment.

        Validates whether the comment's author exists, has not been checked
        before and remains unsuspended.

        Args:
            comment: The comment to be validated.

        Returns:
            Truth value of whether the comment's author is valid for inspection.
        """
        return (
            comment.author
            and comment.author.name not in self.checked_users
            and not getattr(comment.author, "is_suspended", False)
        )

    def gather_from_user(
        self, user: praw.models.Redditor, limit: int = HISTORY_LIMIT
    ) -> list[praw.models.Comment]:
        """Gathers comments from user.

        Inspects a given amount of user's most recent comments, then returns
        a list of these that belong to the adequate subreddit.

        Args:
            user: The user whose comments are to be inspected.
            limit: The limit of comments to be inspected in user's history.

         Returns:
             A list of comments comments made by the given user in the adequate
             subreddit.
        """
        self.checked_users.add(user.name)
        return [
            comment
            for comment in user.comments.new(limit=limit)
            if comment.subreddit == self.subreddit
        ]

    def expand_dataset(
        self, comments: list[praw.models.Comment], quota: int = COMMENT_QUOTA
    ) -> None:
        """Expands dataset by a list of comments.

        Expands the dataset by a list of comments provided it meets the given
        size quota. Each dataset entry consists of the comment's ID, its
        author's name, its body and an embedding of its body.

        Args:
            comments: A list of comments to be appended to the dataset.
            quota: The minimal amount of comments for the expansion to occur.
        """
        if len(comments) >= quota:
            for comment in comments:
                self.dataset.append(
                    {
                        "id": comment.id,
                        "author": comment.author.name,
                        "body": comment.body,
                        "embedding": self.embedder.embed_str(comment.body),
                    }
                )

    def gather_comments_from_hot(self, post_limit: int = POST_LIMIT) -> None:
        """Gathers comments from hot posts and expands the dataset.

        Gathers comments from users who have left comments on a given amount of
        comments from the subreddit's "top" section. The dataset is expanded by
        the gathered comments.

        Args:
            post_limit: The maximal amount of posts to be checked.
        """
        for i, post in enumerate(self.subreddit.hot(limit=post_limit)):
            print(f"{i}/{post_limit}")
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                try:
                    if self.is_valid_comment(comment):
                        comments = self.gather_from_user(comment.author)
                        self.expand_dataset(comments)
                except prawcore.exceptions.TooManyRequests as e:
                    print(f"Exception caught: {e}")
                    break

    def gather_comments_from_all(self, limit: int = COMMENT_LIMIT) -> None:
        """Gathers comments from recent comments and expands the dataset.

        Gathers comments from users who have left one or more of the subreddit's
        most recent comments. The dataset is expanded by the gathered comments.

        Args:
            post_limit: The maximal amount of recent comments to be checked.
        """
        for i, comment in enumerate(self.subreddit.comments(limit=limit)):
            try:
                print(f"{i}/{limit}")
                if self.is_valid_comment(comment):
                    comments = self.gather_from_user(comment.author)
                    self.expand_dataset(comments)
            except prawcore.exceptions.TooManyRequests as e:
                print(f"Exception caught: {e}")
                break

    def dump_dataset(self, path: str = DATASET_PATH) -> None:
        """Stores the dataset in a file.

        Converts the dataset contained in self.dataset to a Pandas dataframe,
        then stores it to a parquet file, compressing it with gzip.

        Args:
            path: The target path for the dataset file to be created at.
        """
        df = pd.DataFrame(self.dataset)
        df.set_index("id", inplace=True)
        df.to_parquet(path, compression="gzip")


if __name__ == "__main__":
    collector = RedditCollector(SUBREDDIT_NAME, SITE_NAME, USER_AGENT)
    collector.gather_comments_from_all()
    collector.dump_dataset()
