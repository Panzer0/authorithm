import praw, prawcore
import pandas as pd
from core.embedding.Embedder import Embedder


SUBREDDIT_NAME = "fantasy"
DATASET_FILENAME = f"{SUBREDDIT_NAME}.parquet.gzip"
DATASET_PATH = f"datasets/dataset_{DATASET_FILENAME}"
POST_COUNT = 10
HISTORY_LIMIT = 150
COMMENT_QUOTA = 25
SITE_NAME = "Authorithm"
USER_AGENT = "authorithm by /u/pazur13"
COMMENT_LIMIT = 1000


class RedditCollector:
    def __init__(self, subreddit_name: str, site_name: str, agent: str) -> None:
        self.reddit = praw.Reddit(site_name, user_agent=agent)
        self.subreddit = self.reddit.subreddit(subreddit_name)
        self.dataset = []
        self.checked_users = set()
        self.embedder = Embedder()

    def is_valid_comment(self, comment: praw.models.Comment) -> bool:
        return (
            comment.author
            and comment.author.name not in self.checked_users
            and not getattr(comment.author, "is_suspended", False)
        )

    def gather_from_user(
        self,
        user: praw.models.Redditor,
        limit: int = HISTORY_LIMIT,
        quota: int = COMMENT_QUOTA,
    ) -> None:
        self.checked_users.add(user.name)

        # Find all comments in adequate subreddit
        comments = [
            comment
            for comment in user.comments.new(limit=limit)
            if comment.subreddit == self.subreddit
        ]

        # Add only if sufficient amount of comments has been found
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

    def gather_comments_from_hot(self, post_limit: int = POST_COUNT) -> None:
        for i, post in enumerate(self.subreddit.hot(limit=post_limit)):
            print(f"{i}/{post_limit}")
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                try:
                    if self.is_valid_comment(comment):
                        self.gather_from_user(comment.author)
                except prawcore.exceptions.TooManyRequests as e:
                    print(e)
                    break

    def gather_comments_from_all(self, limit: int = COMMENT_LIMIT) -> None:
        for i, comment in enumerate(self.subreddit.comments(limit=limit)):
            try:
                print(f"{i}/{limit}")
                if self.is_valid_comment(comment):
                    self.gather_from_user(comment.author)
            except prawcore.exceptions.TooManyRequests as e:
                print(f"Exception caught: {e}")
                break

    def dump_dataset(self, path: str = DATASET_PATH) -> None:
        df = pd.DataFrame(self.dataset)
        df.set_index("id", inplace=True)
        df.to_parquet(path, compression="gzip")


if __name__ == "__main__":
    collector = RedditCollector(SUBREDDIT_NAME, SITE_NAME, USER_AGENT)
    collector.gather_comments_from_all()
    collector.dump_dataset()
