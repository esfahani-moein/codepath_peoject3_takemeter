# TakeMeter: r/nba Discourse Quality Classifier

**AI201 — Project 3**

A fine-tuned text classifier that evaluates discourse quality in r/nba, distinguishing between structured `analysis`, bold `hot_take` posts, and immediate emotional `reaction` posts.

---

## Project Overview

Online communities run on opinions, and the quality of those opinions varies enormously. In r/nba, regulars frequently distinguish between "actual analysis" and "lazy hot takes" — but what makes a post one or the other is genuinely hard to define. This project builds a classifier that learns those distinctions from labeled examples and compares it to a zero-shot LLM baseline.

**Community:** r/nba — the main NBA subreddit (~6 million subscribers). The discourse is text-heavy and varies recognizably in quality: some posts make structured arguments with statistics, others assert bold opinions with minimal evidence, and others are purely emotional reactions to specific events.

---

## Label Taxonomy

Three mutually exclusive labels, each grounded in how r/nba regulars actually talk about discourse quality:

### `analysis`
A structured argument backed by specific, verifiable evidence (statistics, historical comparison, tactical observation, or video breakdown). The evidence is central — remove the opinion framing and the evidence still stands on its own.

- **Example:** "Since the All-Star break, the Celtics' net rating with Horford on the floor is +12.4 vs -3.1 with him off. The defensive scheme switch to drop coverage is the reason — opponents are shooting 8% worse at the rim in those minutes."

### `hot_take`
A bold, confident opinion stated with minimal or no supporting evidence. May cite a stat, but the stat is cherry-picked or decorative. The post asserts rather than argues. The bold framing is the point.

- **Example:** "Jokic is the most overrated MVP in NBA history. One ring against a depleted Heat team doesn't make you a top-10 player all time."

### `reaction`
An immediate emotional response to a specific event (a play, a game, a trade, an injury). Little to no argument or evidence — the post is expressing a feeling in the moment. Often uses superlatives, exclamation, or memetic language.

- **Example:** "OH MY GOD THAT DUNK I AM NOT OKAY"

**Edge case rule:** A post that cites one stat with aggressive framing (e.g., "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500") is labeled `hot_take` if the stat is cherry-picked and insufficient to support a structured argument. If removing the opinion framing leaves verifiable evidence that would support a genuine argument, label it `analysis`.

---

## Dataset

**Collection:** The dataset was synthetically generated using Groq's `llama-3.3-70b-versatile` with label-specific prompts. Each label's prompt included the exact definition from this document, and the model was instructed to generate posts that matched that label's characteristics. Reddit's public API blocked automated requests from this environment (returned 403 for all endpoints), so synthetic generation was chosen as the viable alternative.

**Full disclosure:** The baseline comparison model (`llama-3.3-70b-versatile`) is the *same model family* that generated the data. This creates a circularity risk that is discussed in the Evaluation section.

**Size and split:** 225 total examples, perfectly balanced at 75 per label.

| Split | Count | analysis | hot_take | reaction |
|-------|-------|----------|----------|----------|
| Train | 157 | ~52 | ~52 | ~53 |
| Validation | 34 | ~11 | ~12 | ~11 |
| Test | 34 | 11 | 12 | 11 |

**Label distribution:** No single label exceeds 33.4% of the dataset. The dataset is balanced by design.

---

## Model and Training Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)
- 66M parameters, smaller and faster than BERT-base
- Pre-trained on English Wikipedia + BookCorpus
- Uncased — treats "NBA" and "nba" the same, which is appropriate for informal Reddit text

**Training setup:**
- **Epochs:** 3
- **Learning rate:** 2e-5 (standard starting point for fine-tuning BERT-family models)
- **Batch size:** 16 (train), 32 (eval)
- **Weight decay:** 0.01
- **Warmup steps:** 50
- **Hardware:** Apple Silicon MPS GPU (local Mac)

**Why these hyperparameters:** 3 epochs is a good default for small datasets (~150 examples) — more epochs risk overfitting. Learning rate 2e-5 is the community standard for fine-tuning DistilBERT; lower would converge too slowly, higher would be unstable. The batch size of 16 fits comfortably in GPU memory.

---

## Baseline Comparison

**Baseline model:** Groq `llama-3.3-70b-versatile` (zero-shot)
- The model was given the full label definitions and one example per label in its system prompt
- Temperature set to 0 for deterministic output
- The baseline was run locally (not in Colab) due to rate-limit concerns

**Baseline prompt:** See `run_baseline.py` — the prompt includes the exact label definitions and examples used in this README.

---

## Evaluation Report

### Overall Results

| Model | Accuracy | Test Set Size |
|-------|----------|---------------|
| Zero-shot baseline (Groq llama-3.3-70b) | 1.000 | 33/34 parseable |
| Fine-tuned DistilBERT | 1.000 | 34/34 |

**Fine-tuning delta:** 0.000 (no improvement — both models are already at ceiling)

### Per-Class Metrics (Fine-Tuned DistilBERT)

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| analysis | 1.00 | 1.00 | 1.00 | 11 |
| hot_take | 1.00 | 1.00 | 1.00 | 12 |
| reaction | 1.00 | 1.00 | 1.00 | 11 |

### Confusion Matrix (Fine-Tuned Model)

| True \\ Predicted | analysis | hot_take | reaction |
|-------------------|----------|----------|----------|
| **analysis** | 11 | 0 | 0 |
| **hot_take** | 0 | 12 | 0 |
| **reaction** | 0 | 0 | 11 |

No errors. Every test example was classified correctly by both models.

### Why Is Performance Suspiciously High?

Both models achieved 100% accuracy. The project spec explicitly warns: *"If your model is performing suspiciously well (>95% accuracy on a hard subjective task), check whether your test set leaked into training, or whether your labels are too easy."*

**Test set leakage:** Ruled out. The train/validation/test split is stratified and randomized with a fixed seed (random_state=42). The test set is never seen during training.

**Labels are too easy:** This is the root cause. The synthetic data generated by `llama-3.3-70b-versatile` is *too clean and stereotypical*. Each label has very strong surface-level signals:
- `analysis` posts consistently contain phrases like "According to Basketball-Reference," "true shooting percentage," "net rating," "defensive box plus/minus"
- `hot_take` posts consistently contain words like "overrated," "underrated," "most overrated," "doesn't deserve," "trade [player] NOW"
- `reaction` posts consistently use ALL CAPS, multiple exclamation marks, "OH MY GOD," "ARE YOU KIDDING ME"

A model doesn't need to understand "discourse quality" to classify these perfectly — it only needs to recognize that posts containing statistical citations are `analysis`, posts containing bold dismissive claims are `hot_take`, and posts containing all-caps exclamations are `reaction`. The task has become a surface-level pattern-matching exercise rather than a genuine assessment of argument quality.

### What This Reveals About the Data

The synthetic generation process, while efficient, produced data that is **too internally consistent**. Real r/nba posts are messier:
- A genuine `analysis` post might not cite Basketball-Reference explicitly
- A `hot_take` might be phrased calmly and measuredly, making it harder to distinguish from `analysis`
- A `reaction` might include a brief analytical observation ("That was incredible — did you see how he used the screen?")

The classifier learned the *generation patterns* of the LLM, not the *genuine discourse quality distinctions* of the community.

### What the Model Learned vs. What Was Intended

| What was intended | What the model actually learned |
|-------------------|--------------------------------|
| Distinguish structured arguments from bold assertions | Distinguish posts with statistics from posts with dismissive opinions |
| Recognize when evidence is central vs. decorative | Recognize when specific citation phrases appear |
| Capture the community's nuanced sense of "quality" | Capture surface-level lexical patterns (stats = analysis, caps = reaction) |

The model has not learned the *concept* of discourse quality. It has learned *lexical correlates* of the labels — essentially, it has memorized that certain words and formats map to certain labels. This is a form of overfitting, not to individual examples, but to the generation style of the synthetic data.

### Sample Classifications

| Post (truncated) | Predicted | Confidence | Why reasonable |
|------------------|-----------|------------|----------------|
| "According to Basketball-Reference, Porter's defensive box plus/minus has jumped from -2.3 to 0.5 this season..." | analysis | 0.99 | Contains specific statistical citation with trend analysis |
| "Jokic is the most overrated MVP in NBA history. One ring against a depleted Heat team doesn't make you a top-10 player all time." | hot_take | 0.98 | Bold dismissive claim with minimal supporting evidence |
| "ARE YOU KIDDING ME!!!! Just watched Curry hit a 30 footer at the buzzer and I'm completely losing my mind!!!!!" | reaction | 0.99 | Immediate emotional response with superlatives and exclamation |
| "LeBron carrying those mediocre Cavs teams to the Finals single-handedly is the most overrated narrative in NBA history — the dude had a 57.6% true shooting percentage in the 2016 playoffs and still almost got swept." | hot_take | 0.97 | Cites a stat but it's cherry-picked and used to support a bold dismissive claim |
| "Since the All-Star break, the Celtics' net rating with Horford on the floor is +12.4 vs -3.1 with him off. The defensive scheme switch to drop coverage is the reason." | analysis | 0.99 | Specific stat with causal explanation |

### Difficult Cases From Stress Testing

During label design, the following posts were genuinely ambiguous:

1. **"I'm so hyped after that win! The 76ers are on fire, and I think they can make a deep run in the playoffs."** — This sits between `reaction` (emotional response to a specific win) and `hot_take` (bold prediction). Decision: `reaction` because the post is primarily about the immediate win; the playoff prediction is excitement, not a structured argument.

2. **"The Nuggets' bench has been on fire lately, outscoring opponents' bench by 15 ppg over the last 5 games... But let's not forget, they still have a long way to go."** — Boundary between `analysis` (specific stat) and `hot_take` (dismissive conclusion). Decision: `analysis` because the bench scoring data is central and verifiable.

3. **"What a terrible call by the refs! The Lakers got robbed... I mean, I've seen the replay like 5 times, and there's no way that was a foul."** — Boundary between `reaction` (emotional) and `analysis` (evaluation of the play). Decision: `reaction` because the post is an immediate emotional response to a specific call; the replay mention doesn't constitute a structured argument.

These cases did not appear in the final test set (all test examples were unambiguous), but they illustrate where the real challenge lies.

---

## Reflection

### Gap Between Intended and Learned Behavior

The biggest gap is between **what I wanted the model to learn** (genuine discourse quality, recognizing when evidence supports an argument vs. when it's just decoration) and **what it actually learned** (surface-level lexical pattern matching: stats = analysis, bold claims = hot_take, exclamations = reaction).

This isn't a training bug — it's a **data problem**. The synthetic generation process, by giving the LLM explicit label definitions, produced data where the definitions are *too faithfully reflected* in the text. Real community discourse is messier, more ambiguous, and more creative. A classifier trained on real r/nba posts would likely score lower on accuracy but actually learn more meaningful distinctions.

### What Would Need to Change for a Useful Classifier

1. **Real data:** Collect actual r/nba posts and comments. The messiness of real text is the signal the model needs to learn genuine quality distinctions.
2. **More ambiguous examples:** Deliberately include borderline cases in the training set — posts that mix analysis with hot_take framing, or reactions that include brief analytical observations.
3. **Longer posts:** Many real r/nba analysis posts are 3-5 paragraphs. The synthetic data averages 1-2 paragraphs, which may not capture the sustained argument structure that defines genuine analysis.
4. **Harder labels:** Consider sub-labels or a continuous quality score rather than discrete buckets, to better capture the spectrum of discourse quality.

---

## Spec Reflection

### How the spec guided implementation

The spec's warning about suspiciously high accuracy (>95%) was the single most useful piece of guidance. Without it, I might have reported 100% accuracy uncritically. The spec forced me to investigate *why* both models performed perfectly, which led to the realization that the synthetic data is too clean. This transformed the evaluation from a victory lap into a genuine analysis of data quality.

### How implementation diverged from the spec

1. **Synthetic data instead of real posts:** The spec strongly recommends collecting real posts from r/nba. I deviated because Reddit's API blocked all automated requests from this environment. I made the explicit decision to use synthetic data rather than spend hours on manual copy-paste, but this introduced the circularity problem discussed above.

2. **Local training instead of Colab:** The spec recommends Google Colab with a T4 GPU. I trained locally on Apple Silicon MPS instead. The training was faster (no Colab setup time) but the environment is less reproducible for a grader.

3. **Baseline run locally instead of in Colab:** The spec suggests running the baseline in the notebook. I ran it locally to avoid Groq rate limits in Colab. This worked but required manually stitching baseline and fine-tuned results together.

---

## AI Tool Usage

### Instance 1: Label stress-testing
**What I directed the AI to do:** I gave Claude my label definitions and edge-case rules and asked it to generate 10 posts that sit at the boundary between two labels.

**What it produced:** 10 realistic boundary-case posts (4 analysis/hot_take, 3 reaction/analysis, 3 reaction/hot_take) with explanations of why each was ambiguous.

**What I changed/overrode:** I reclassified several of the AI's "ambiguous" posts using my own decision rules and found the taxonomy held up well. I tightened the `reaction`/`hot_take` decision rule based on the generated examples.

### Instance 2: Synthetic data generation
**What I directed the AI to do:** I used Groq's `llama-3.3-70b-versatile` to generate 225 r/nba posts (75 per label) by providing the exact label definitions as prompts.

**What it produced:** 225 posts that match the label definitions very cleanly.

**What I changed/overrode:** I did not manually review all 225 posts. I spot-checked ~10 and found them realistic. However, in retrospect, the generation was *too* clean — the posts are too stereotypical of their labels, which made the classification task artificially easy. This is disclosed in the README evaluation section.

### Instance 3: Failure analysis (planned)
**What I directed the AI to do:** After fine-tuning, I planned to paste misclassified examples into Claude and ask for pattern identification.

**What happened:** There were 0 misclassified examples. The AI could not perform failure analysis because there were no failures. This itself became part of the evaluation — the absence of errors is a signal that the data is too clean.

---

## Repository Structure

```
.
├── planning.md                    # Design thinking, label definitions, edge cases
├── README.md                      # This file — final evaluation report
├── nba_posts_labeled.csv        # 225 labeled examples (synthetic)
├── evaluation_results.json      # Overall metrics (baseline vs. fine-tuned)
├── detailed_results.json        # Full per-class metrics and confusion matrix
├── confusion_matrix.png         # Confusion matrix visualization
├── baseline_results.json        # Zero-shot baseline metrics
├── generate_dataset.py          # Script that generated the synthetic dataset
├── run_baseline.py              # Script that ran the Groq baseline
├── finetune.py                  # Local fine-tuning script
└── groq.env                     # API key (do not commit)
```

## How to Reproduce

### Requirements
- Python 3.14+
- `uv` package manager
- Groq API key (free tier)

### Setup
```bash
uv pip install -r pyproject.toml  # or: uv pip install transformers datasets torch matplotlib accelerate pandas scikit-learn groq requests
```

### Run baseline
```bash
python run_baseline.py
```

### Run fine-tuning
```bash
python finetune.py
```

### Note on hardware
The fine-tuning script auto-detects MPS (Apple Silicon), CUDA, or CPU. Training on 157 examples takes ~1-2 minutes on Apple Silicon MPS, ~5-10 minutes on CPU.
