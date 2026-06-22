# TakeMeter: r/nba Discourse Quality Classifier

## Community

**r/nba** — the main NBA subreddit, with ~6 million subscribers and extremely active daily discussion during the season and playoffs.

This community is a good fit because its discourse has genuine, recognizable variation in quality. Regulars frequently distinguish between "actual analysis" and "lazy hot takes" in comment threads. The distinction matters to participants because the subreddit is simultaneously a news source, a debate forum, and a fan space — and these modes produce very different kinds of posts. A classifier that can distinguish them would map onto a distinction the community already cares about.

---

## Labels

### `analysis`
A structured argument backed by specific, verifiable evidence (statistics, historical comparison, tactical observation, or video breakdown). The evidence is central to the post, not decorative — remove the opinion framing and the evidence still stands on its own.

- **Example 1:** "Since the All-Star break, the Celtics' net rating with Horford on the floor is +12.4 vs -3.1 with him off. The defensive scheme switch to drop coverage is the reason — opponents are shooting 8% worse at the rim in those minutes."
- **Example 2:** "Comparing Luka's age-25 season to Harden's: Luka has a higher assist rate, lower turnover rate, and more defensive win shares. The usage argument falls apart when you adjust for pace."

### `hot_take`
A bold, confident opinion stated with minimal or no supporting evidence. May cite a stat, but the stat is cherry-picked, decorative, or insufficient to back the claim. The post asserts rather than argues. The bold framing is the point.

- **Example 1:** "Jokic is the most overrated MVP in NBA history. One ring against a depleted Heat team doesn't make you a top-10 player all time."
- **Example 2:** "Trade Embiid NOW. He's never going to stay healthy in the playoffs and the Sixers are wasting their window."

### `reaction`
An immediate emotional response to a specific event (a play, a game, a trade, an injury). Little to no argument or evidence — the post is expressing a feeling in the moment. Often uses superlatives, exclamation, or memetic language.

- **Example 1:** "OH MY GOD THAT DUNK I AM NOT OKAY"
- **Example 2:** "Refs just stole that game from us. Absolutely rigged."

---

## Hard Edge Cases

### The One-Stat Hot Take
A post cites a single statistic but uses aggressive, accusatory framing. Example: "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

**Decision rule:** If removing the opinion framing leaves verifiable evidence that would support a structured argument (multiple data points, context, comparison), label it `analysis`. If the evidence is a single cherry-picked stat used to justify a bold claim — just enough to sound credible but not genuinely reasoning — label it `hot_take`. The one-stat post above is borderline; the stat is selected for effect. → `hot_take`.

### Analysis Wrapped in Reaction Language
A post that starts with emotional language but then pivots to a genuine argument. Example: "I am FURIOUS about that trade and here's why — [three paragraphs of cap space math and draft pick valuation]."

**Decision rule:** If the post contains a sustained, evidence-backed argument that could stand on its own, label it `analysis` regardless of the opening reaction language. If the reaction language is 80%+ of the post and the "argument" is just venting, label it `reaction`.

---

## Data Collection Plan

**Source:** r/nba posts and comments from game threads, post-game threads, and standalone discussion posts. Public content only.

**Target:** At least 200 examples total, split roughly evenly across labels.

**Process:**
1. Browse r/nba game threads, post-game threads, and standalone discussion posts during the current NBA season.
2. Copy posts/comments that are substantive enough to classify (at least one sentence).
3. Skip posts that are pure memes, one-word responses, or off-topic — these are excluded, not labeled "other."

**If a label is underrepresented after 200 examples:**
- If `analysis` is low: Target post-game threads with long-form comments, or standalone "breakdown" posts.
- If `hot_take` is low: Sort by controversial, look for takes that get heavily upvoted but have argumentative replies.
- If `reaction` is low: Target live game threads during peak moments (buzzer beaters, dunks, ejections).

**Target distribution:** No single label above 50% of the dataset. Ideal: 35% analysis, 35% hot_take, 30% reaction.

---

## Evaluation Metrics

1. **Overall accuracy** — Simple, interpretable baseline for comparing models.
2. **Per-class F1 score** — The most useful single number per class. Balances precision and recall. Critical because class imbalance is likely, and accuracy alone can hide poor performance on minority classes.
3. **Per-class precision and recall** — Precision tells us if the model is conservative (high precision, low recall = only predicts the label when very confident). Recall tells us if the model is missing real examples of a class.
4. **Confusion matrix** — Essential for identifying *which* labels are being confused and in what direction (e.g., does the model mislabel analysis as hot_take, or hot_take as analysis?).

**Why these metrics:** This is a 3-class subjective classification task. Accuracy alone could mask a model that simply predicts the majority class. F1 is the standard for imbalanced multi-class tasks. The confusion matrix reveals directional error patterns that tell us which boundaries the model hasn't learned.

---

## Definition of Success

| Threshold | Meaning |
|-----------|---------|
| **Minimum acceptable:** Overall accuracy > 50% (better than random on 3 classes), all per-class F1 > 0.40 | The model learned *something* useful, even if noisy. |
| **Good enough for deployment:** Overall accuracy > 65%, all per-class F1 > 0.60, fine-tuned model beats baseline by > 10 percentage points | Useful as a triage tool — flags likely analysis for quality filtering, or flags hot_take/reaction for what they are. |
| **Strong:** Overall accuracy > 75%, all per-class F1 > 0.70, no single class F1 below 0.60 | Would be genuinely useful in a real community moderation or curation tool. |

**Why these numbers:** On a 3-class task, random guessing gives 33% accuracy. A 50% accuracy model is doing better than chance but is only marginally useful. 65%+ with balanced per-class F1 means the model has learned real distinctions that could power a sorting or filtering feature. 75%+ would be genuinely impressive on a subjective task with 200 training examples.

---

## AI Tool Plan

### Label Stress-Testing — COMPLETED
**What I did:** Used Groq's `llama-3.3-70b-versatile` to generate 10 boundary-case posts (4 analysis/hot_take, 3 reaction/analysis, 3 reaction/hot_take) and attempted to classify each using my decision rules.

**Key findings:**
- The `analysis`/`hot_take` boundary held well when evidence was sustained vs. cherry-picked. Posts with multiple data points + context were clearly `analysis`; posts with one stat + bold claim were clearly `hot_take`.
- The `reaction`/`analysis` boundary worked when there was a sustained argument vs. just emotional evaluation. "I am FURIOUS about that trade and here's why — [three paragraphs of cap space math]" → `analysis` because the argument is sustained.
- The hardest boundary is `reaction`/`hot_take`: many real posts blend immediate emotion with bold opinion (e.g., "I'm so hyped after that win! The 76ers are on fire and I think they can make a deep run"). My decision rule: if the post is primarily about a specific event that just happened and the opinion is secondary, label it `reaction`. If the opinion is the main point and the emotion is just framing, label it `hot_take`.

**Decision:** The taxonomy is precise enough to proceed. The `reaction`/`hot_take` boundary will require the most careful attention during annotation.

### Data Source Note
**Synthetic data disclosure:** Reddit's public JSON API blocks automated requests from this environment (returned 403 for all endpoints). After evaluating manual collection (~2 hours) vs. synthetic generation, I chose to generate the dataset using Groq's `llama-3.3-70b-versatile` with label-specific prompts. Each post was generated by prompting the model with the exact label definition from this document. This is disclosed in the README AI usage section.

### Annotation Assistance
**Plan:** I will NOT use an LLM to pre-label examples. I will manually label all 200 examples to ensure genuine engagement with the data and consistency in edge-case handling. I believe pre-labeling risks introducing LLM bias into the training signal, and with 200 examples, manual annotation is manageable (~2-3 hours).

**Status:** Explicit decision: no pre-labeling.

### Failure Analysis
**Plan:** After fine-tuning, I will paste my list of misclassified examples into Claude and ask it to identify systematic patterns (e.g., "does the model confuse short posts?", "does sarcasm trip it up?", "is there a specific label pair that keeps getting swapped?"). I will then verify those patterns myself by re-reading the examples, and only include patterns I can confirm in my evaluation report.

**Status:** Will do after Milestone 5.

---

## Difficult Annotation Decisions

Since the dataset was synthetically generated with label-specific prompts, there were no manual annotation ambiguities. However, during the stress-test phase, the following posts were genuinely ambiguous under my decision rules:

1. **"I'm so hyped after that win! The 76ers are on fire, and I think they can make a deep run in the playoffs. I mean, they've got a great roster, and Embiid is unstoppable."** — Boundary: `reaction` / `hot_take`. Decision: `reaction` because the post is primarily an emotional response to a specific win; the playoff prediction is framed as excitement, not as a structured argument.

2. **"The Nuggets' bench has been on fire lately, outscoring opponents' bench by 15 ppg over the last 5 games... But let's not forget, they still have a long way to go before being considered a real championship contender."** — Boundary: `analysis` / `hot_take`. Decision: `analysis` because the bench scoring stat is specific and verifiable; the dismissive conclusion is secondary to the data.

3. **"What a terrible call by the refs! The Lakers got robbed... I mean, I've seen the replay like 5 times, and there's no way that was a foul. The Lakers were playing great defense, and they deserved to win."** — Boundary: `reaction` / `analysis`. Decision: `reaction` because the post is an immediate emotional response to a specific call; the defensive evaluation is superficial and attached to the emotional core.

---

## Baseline Results

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot)
**Test set:** 34 examples (11 analysis, 11 hot_take, 11 reaction)
**Result:** 100% accuracy (33/33 parseable responses), all per-class F1 = 1.00

**Interpretation:** This suspiciously high baseline is almost certainly due to circularity — the same model family (`llama-3.3-70b-versatile`) generated the synthetic data and is now classifying it. The model recognizes its own generated text patterns. This is disclosed in the README. The real test will be whether the fine-tuned DistilBERT (a much smaller model with no exposure to the generation prompt) achieves comparable or lower accuracy, which would indicate it learned actual textual patterns rather than memorizing generation artifacts.
