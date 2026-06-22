"""Generate synthetic r/nba dataset via Groq API.
Each post is generated with a specific label in the prompt to ensure
realistic, label-appropriate content.
"""

from groq import Groq
import os
import csv
import time

# Load API key
with open('groq.env') as f:
    for line in f:
        if line.strip() and '=' in line:
            key = line.split('=', 1)[1].strip()
            os.environ['GROQ_API_KEY'] = key

client = Groq(api_key=os.environ['GROQ_API_KEY'])

LABEL_CONFIG = {
    "analysis": {
        "count": 75,
        "prompt": """You are a user on r/nba writing posts that contain genuine basketball analysis.
Write a single r/nba post. The post should be a structured argument backed by specific, verifiable evidence (statistics, historical comparison, tactical observation, or video breakdown). The evidence should be central and specific — remove the opinion framing and the evidence still stands.

The post should sound like a real r/nba user wrote it. Vary the topics (different players, teams, game situations), lengths (short paragraph to 3 paragraphs), and tones (measured, passionate, data-heavy, narrative-driven).

Output ONLY the post text. No label, no explanation, no quotes around the text."""
    },
    "hot_take": {
        "count": 75,
        "prompt": """You are a user on r/nba writing bold, controversial opinions.
Write a single r/nba post. The post should be a bold, confident opinion stated with minimal supporting evidence. It may cite a stat or claim, but the stat is cherry-picked or decorative. The post asserts rather than argues. The bold framing is the point.

The post should sound like a real r/nba user wrote it. Vary the topics (different players, teams, situations), lengths (one sentence to a short paragraph), and tones (confident, dismissive, sarcastic, provocative).

Output ONLY the post text. No label, no explanation, no quotes around the text."""
    },
    "reaction": {
        "count": 75,
        "prompt": """You are a user on r/nba reacting to a specific basketball event in real time.
Write a single r/nba post. The post should be an immediate emotional response to a specific event (a play, a game, a trade, an injury, a call by the refs). Little to no argument or evidence — the post is expressing a feeling in the moment. Often uses superlatives, exclamation, or memetic language.

The post should sound like a real r/nba user wrote it. Vary the events (buzzer beaters, dunks, ejections, bad calls, trades, injuries, comebacks), lengths (one sentence to a short paragraph), and emotions (excitement, rage, disbelief, joy, despair).

Output ONLY the post text. No label, no explanation, no quotes around the text."""
    },
}


def generate_post(label, prompt):
    """Generate a single post for a given label."""
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.9,
        max_tokens=400,
    )
    text = response.choices[0].message.content.strip()
    # Clean up common formatting artifacts
    text = text.strip('"').strip("'")
    if text.startswith("Post:"):
        text = text[5:].strip()
    return text


def main():
    all_posts = []

    for label, config in LABEL_CONFIG.items():
        print(f"Generating {config['count']} {label} posts...")
        for i in range(config['count']):
            try:
                text = generate_post(label, config['prompt'])
                all_posts.append({
                    "text": text,
                    "label": label,
                    "notes": "",
                })
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{config['count']} done...")
                time.sleep(0.15)
            except Exception as e:
                print(f"  Error on {label} #{i+1}: {e}")
                time.sleep(1)

    print(f"\nGenerated {len(all_posts)} total posts")

    # Write to CSV
    with open("nba_posts_labeled.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        writer.writeheader()
        writer.writerows(all_posts)

    print("Saved to nba_posts_labeled.csv")

    # Print distribution
    from collections import Counter
    dist = Counter(p["label"] for p in all_posts)
    print("Label distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count} ({count/len(all_posts)*100:.1f}%)")


if __name__ == "__main__":
    main()
