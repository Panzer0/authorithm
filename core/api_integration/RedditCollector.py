import praw, prawcore, json


SUBREDDIT_NAME = "askreddit"
POST_COUNT = 3
HISTORY_LIMIT = 125
COMMENT_QUOTA = 25
SITE_NAME = "Authorithm"
USER_AGENT = "authorithm by /u/pazur13"
DEFAULT_PATH = f"datasets/dataset_{SUBREDDIT_NAME}.json"
COMMENT_LIMIT = 500


class RedditCollector:
    def __init__(self, subreddit_name, site_name, agent) -> None:
        self.reddit = praw.Reddit(site_name, user_agent=agent)
        self.subreddit = self.reddit.subreddit(subreddit_name)
        self.dataset = []
        self.checked_users = set()

    def is_valid_comment(self, comment) -> bool:
        return (
            comment.author
            and comment.author.name not in self.checked_users
            and not getattr(comment.author, "is_suspended", False)
        )

    def gather_from_user(self, user, limit=HISTORY_LIMIT, quota=COMMENT_QUOTA):
        self.checked_users.add(user.name)

        # Find all comments in adequate subreddit
        comments = [
            comment.body
            for comment in user.comments.new(limit=limit)
            if comment.subreddit == self.subreddit
        ]

        # Add only if sufficient amount of comments has been found
        if len(comments) >= quota:
            self.dataset.append({"username": user.name, "comments": comments})

    def gather_comments_from_hot(self, post_limit=POST_COUNT):
        for post in self.subreddit.hot(limit=post_limit):
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                try:
                    if self.is_valid_comment(comment):
                        self.gather_from_user(comment.author)
                except prawcore.exceptions.TooManyRequests as e:
                    print(e)
                    break

    def gather_comments_from_all(self, limit = COMMENT_LIMIT):
        for comment in self.subreddit.comments(limit=limit):
            try:
                if self.is_valid_comment(comment):
                    self.gather_from_user(comment.author)
            except prawcore.exceptions.TooManyRequests as e:
                print(e)
                break

    def dump_dataset(self, path=DEFAULT_PATH):
        with open(path, "w") as file:
            json.dump(self.dataset, file, indent=2)


if __name__ == "__main__":
    collector = RedditCollector(SUBREDDIT_NAME, SITE_NAME, USER_AGENT)
    collector.gather_comments_from_all()
    collector.dump_dataset()
