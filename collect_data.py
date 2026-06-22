import requests
import csv
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def fetch_subreddit_posts(subreddit, sort="hot", limit=100):
    """Fetch post titles and selftext from a subreddit."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    posts = []
    for child in data["data"]["children"]:
        post = child["data"]
        # Combine title + selftext for standalone posts
        text = post.get("title", "") + "\n\n" + post.get("selftext", "")
        text = text.strip()
        if text and len(text) > 20:
            posts.append({
                "text": text,
                "source": f"r/{subreddit} {sort}",
                "permalink": post.get("permalink", ""),
                "author": post.get("author", ""),
            })
    return posts

def fetch_comments_for_post(permalink, limit=20):
    """Fetch top-level comments from a post."""
    url = f"https://www.reddit.com{permalink}.json?limit={limit}&depth=1"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return []
    data = resp.json()
    comments = []
    if len(data) > 1 and "data" in data[1]:
        for child in data[1]["data"]["children"]:
            if child["kind"] != "t1":
                continue
            comment = child["data"]
            body = comment.get("body", "").strip()
            if body and len(body) > 20 and len(body) < 800:
                comments.append({
                    "text": body,
                    "source": f"r/nba comment on {permalink}",
                    "permalink": permalink,
                    "author": comment.get("author", ""),
                })
    return comments

def main():
    all_posts = []

    # Fetch hot and new posts
    print("Fetching r/nba hot posts...")
    all_posts.extend(fetch_subreddit_posts("nba", "hot", limit=100))
    time.sleep(1)

    print("Fetching r/nba new posts...")
    all_posts.extend(fetch_subreddit_posts("nba", "new", limit=100))
    time.sleep(1)

    # Fetch comments from a few top posts to get more variety
    print("Fetching comments from top posts...")
    top_posts = fetch_subreddit_posts("nba", "top", limit=25)
    for post in top_posts[:10]:
        if post["permalink"]:
            comments = fetch_comments_for_post(post["permalink"], limit=15)
            all_posts.extend(comments)
            time.sleep(0.5)

    # Deduplicate by text
    seen = set()
    unique = []
    for p in all_posts:
        key = p["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"Collected {len(unique)} unique posts/comments")

    # Save to CSV
    with open("raw_posts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source", "permalink", "author"])
        writer.writeheader()
        writer.writerows(unique)

    print("Saved to raw_posts.csv")

if __name__ == "__main__":
    main()
