import praw, json

reddit = praw.Reddit("Authorithm", user_agent="authorithm by /u/pazur13")

SUBREDDIT_NAME = "europe"
subreddit = reddit.subreddit(SUBREDDIT_NAME)

dataset = []
checked_users = set()


def is_valid_commet(comment):
    return (
        isinstance(comment, praw.models.Comment)
        and comment.author
        and not getattr(comment.author, "is_suspended", False)
        and comment.author.name not in checked_users
    )


for post in subreddit.hot(limit=5):
    for comment in post.comments.list():
        # As opposed to praw.models.MoreComments
        if is_valid_commet(comment):
            user = comment.author
            # Find all comments in corresponding subreddit
            checked_users.add(user.name)
            user_comments = [
                comment.body
                for comment in user.comments.new(limit=200)
                if comment.subreddit.display_name == SUBREDDIT_NAME
            ]

            # Filter out users with insufficient comment count
            if len(user_comments) >= 25:
                dataset.append(
                    {"username": user.name, "comments": user_comments}
                )

with open("dataset.json", "w") as file:
    json.dump(dataset, file, indent=2)
