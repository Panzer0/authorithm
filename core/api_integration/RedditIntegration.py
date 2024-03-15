import praw

USER_SEPARATOR = "#-" * 50
COMMENT_SEPARATOR = "--" * 25

reddit = praw.Reddit("Authorithm", user_agent="authorithm by /u/pazur13")
print(reddit.read_only)

drafted_users = [post.author for post in reddit.subreddit("europe").hot(limit=10)]



for user in drafted_users:
    print(f"We've got {user.name}")
    latest_comments = reddit.redditor(user.name).comments.new(limit=5)
    print(f"{USER_SEPARATOR}\n"
          f"For user {user}:")
    print(latest_comments)
    for comment in latest_comments:
        print(COMMENT_SEPARATOR)
        print(comment.body)