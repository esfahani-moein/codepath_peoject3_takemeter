# Colab Fine-Tuning Instructions

## Step 1: Open the Notebook
1. Go to the TakeMeter starter notebook link (from the project page)
2. Click **File → Save a copy in Drive**
3. In your copy, go to **Runtime → Change runtime type → T4 GPU**, then click Save

## Step 2: Upload Your Data
In Section 1 of the notebook, run the upload cell and select `nba_posts_labeled.csv` from this repo.

## Step 3: Update the Label Map
In Section 1, replace the `LABEL_MAP` with:

```python
LABEL_MAP = {
    "analysis":  0,
    "hot_take":  1,
    "reaction":  2,
}
```

## Step 4: Run Sections 1 and 2
These load the data, split it (70/15/15), and tokenize. Confirm the split sizes look reasonable (~157 train, ~34 val, ~34 test).

## Step 5: Skip Section 5 (Baseline) — Already Done Locally
We already ran the Groq baseline locally (see `baseline_results.json`). Running it again in Colab risks hitting the same rate limit.

**Instead:** After running Sections 1, 2, 3, and 4, manually create the baseline variable before Section 6 by adding this cell:

```python
# Pre-computed baseline from local run
bl_accuracy = 1.0
```

Then run Section 6.

## Step 6: Run Section 5 (Optional — Only if you want to re-run baseline in Colab)
If you want to run the baseline in Colab anyway:
1. Add your Groq API key via Colab Secrets (🔑 icon in left sidebar)
2. Replace the `SYSTEM_PROMPT` in Section 5 with:

```python
SYSTEM_PROMPT = """
You are classifying posts from r/nba, a basketball discussion community.
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
```

## Step 7: Run Sections 3, 4, and 6
- Section 3: Fine-tunes DistilBERT (~5–15 min on T4)
- Section 4: Evaluates on test set, generates confusion matrix
- Section 6: Compares baseline vs. fine-tuned

## Step 8: Download Outputs
From the Files panel (📁) on the left:
- Download `evaluation_results.json`
- Download `confusion_matrix.png`

Commit both to this repo.

## Step 9: Return Here
Once you have the fine-tuned results, come back and I'll help you write the README evaluation report (Milestone 6).
