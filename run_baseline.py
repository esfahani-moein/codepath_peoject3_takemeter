"""Run zero-shot Groq baseline on the test set.
Replicates the same 70/15/15 stratified split as the Colab notebook.
"""

from groq import Groq
import os
import time
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load API key
with open('groq.env') as f:
    for line in f:
        if line.strip() and '=' in line:
            key = line.split('=', 1)[1].strip()
            os.environ['GROQ_API_KEY'] = key

client = Groq(api_key=os.environ['GROQ_API_KEY'])

LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "reaction": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

SYSTEM_PROMPT = """You are classifying posts from r/nba, a basketball discussion community.
Assign each post to exactly one of the following categories.

analysis: A structured argument backed by specific, verifiable evidence (statistics, historical comparison, tactical observation, or video breakdown). The evidence is central — remove the opinion framing and the evidence still stands on its own.
Example: "Since the All-Star break, the Celtics' net rating with Horford on the floor is +12.4 vs -3.1 with him off. The defensive scheme switch to drop coverage is the reason — opponents are shooting 8% worse at the rim in those minutes."

hot_take: A bold, confident opinion stated with minimal or no supporting evidence. May cite a stat, but the stat is cherry-picked or decorative. The post asserts rather than argues. The bold framing is the point.
Example: "Jokic is the most overrated MVP in NBA history. One ring against a depleted Heat team doesn't make you a top-10 player all time."

reaction: An immediate emotional response to a specific event (a play, a game, a trade, an injury). Little to no argument or evidence — the post is expressing a feeling in the moment. Often uses superlatives, exclamation, or memetic language.
Example: "OH MY GOD THAT DUNK I AM NOT OKAY"

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
"""


def classify_with_groq(text):
    """Classify a single post. Returns a label string or None if unparseable."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this post:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        for label in sorted(LABEL_MAP, key=len, reverse=True):
            if raw == label or label in raw:
                return label
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def main():
    # Load dataset
    df = pd.read_csv("nba_posts_labeled.csv")
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Same split as notebook: 70/15/15, stratified
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    # Run baseline on test set
    print(f"\nRunning baseline on {len(test_df)} test examples...")
    baseline_preds = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = classify_with_groq(row["text"])
        baseline_preds.append(pred)
        if (i + 1) % 5 == 0 or i == len(test_df) - 1:
            print(f"  {i+1}/{len(test_df)} complete...")
        time.sleep(0.1)

    # Compute metrics (exclude unparseable)
    valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]
    bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
    bl_true_ids = [t for _, t in valid]

    bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
    print(f"\n🎯 Baseline accuracy: {bl_accuracy:.3f}  "
          f"(evaluated on {len(valid)}/{len(test_df)} parseable responses)")

    label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    print("\nPer-class metrics (baseline):")
    print(classification_report(bl_true_ids, bl_pred_ids,
                                target_names=label_names, zero_division=0))

    # Confusion matrix
    print("\nConfusion matrix (baseline):")
    cm = confusion_matrix(bl_true_ids, bl_pred_ids)
    print(pd.DataFrame(cm, index=[f"true_{label}" for label in label_names],
                        columns=[f"pred_{label}" for label in label_names]))

    # Save results
    results = {
        "baseline_accuracy": round(bl_accuracy, 4),
        "test_set_size": len(test_df),
        "parseable": len(valid),
        "unparseable": baseline_preds.count(None),
        "label_map": LABEL_MAP,
        "per_class_report": classification_report(
            bl_true_ids, bl_pred_ids, target_names=label_names,
            zero_division=0, output_dict=True
        ),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved baseline_results.json")

    # Save wrong predictions for analysis
    wrong = []
    for i, (pred, true_id) in enumerate(zip(baseline_preds, test_df["label_id"])):
        if pred is not None and LABEL_MAP[pred] != true_id:
            wrong.append({
                "text": test_df.iloc[i]["text"],
                "true": ID_TO_LABEL[true_id],
                "predicted": pred,
            })

    with open("baseline_wrong.json", "w") as f:
        json.dump(wrong, f, indent=2)
    print(f"✅ Saved {len(wrong)} wrong predictions to baseline_wrong.json")


if __name__ == "__main__":
    main()
