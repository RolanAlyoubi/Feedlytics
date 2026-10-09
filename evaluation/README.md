# Evaluation

> **Status: no Claude evaluation has been run.** This project has no Anthropic API credits.
> Every result in this folder comes from the non-AI methods (rating rule, VADER + keyword
> baseline), which run locally with no API key.

How accurate is the labelling? This folder holds the answers. Every number here was measured
by [`scripts/evaluate_labels.py`](../scripts/evaluate_labels.py), never typed in by hand.

| File | Contents |
|---|---|
| `RESULTS_<profile>.md` | Summary tables, outcome-flag scores, confusion matrices |
| `results/<profile>_<method>.json` | Full metrics per method |
| `labeling/<profile>_to_label.csv` | Hand-labelling sample (fill in the `true_*` columns) |
| `labeling/<profile>_GUIDE.md` | Allowed values and labelling rules for that profile |

## Methods compared

1. **Rating rule:** sentiment from the rating alone (no text). Only when the data has ratings.
2. **Baseline:** VADER sentiment + the profile's keyword rules. No AI.
3. **AI:** Claude with the profile's taxonomy (`--ai`; needs `ANTHROPIC_API_KEY`, asks before
   spending). **Not run in this project.** Without API access, `--ai` exits with a clear message.

## Real-data workflow (apparel demo)

No API access is needed for any step in this section.

### The sample and the guide

- `labeling/apparel_ecommerce_to_label.csv` is a blind template: 200 reviews with their title and text and empty `true_*` columns. It is git-ignored because it contains third-party review text. Labelling has been completed locally; the labelled CSV and the reports derived from it are excluded from the public repository.
- The rating is hidden from this file, so it cannot influence the labels.
- The sample over-represents low ratings (100 low-rated, 50 mid-rated, 50 high-rated), so there are enough complaints to measure.
- These same 200 reviews were excluded when the baseline was calibrated.
- [labeling/apparel_ecommerce_GUIDE.md](labeling/apparel_ecommerce_GUIDE.md) explains how to label sentiment, the main topic, complaints, praise and outcome flags, and lists every allowed value.

Label columns:

| Column | Content |
|---|---|
| `true_sentiment` | `Positive`, `Neutral` or `Negative`: the tone of the text |
| `true_topic` | The main topic: the main complaint's topic if there is one, otherwise the main praise |
| `true_issues` | Every complaint, separated by `;`, main complaint first; empty if none |
| `true_praise_topics` | Every praised topic, separated by `;`; empty if none |
| `true_mentions_return`, `true_mentions_repurchase` | `true` or `false` |

### Steps

Step 1. Copy the template and label the copy. Both files stay git-ignored.

```bash
cp evaluation/labeling/apparel_ecommerce_to_label.csv evaluation/labeling/apparel_ecommerce_labeled.csv
```

Step 2. Check progress and catch invalid values at any time. This writes nothing, and rows without `true_sentiment` are skipped.

```bash
python -m scripts.evaluate_labels --data data/raw/reviews.csv --truth evaluation/labeling/apparel_ecommerce_labeled.csv --check-only
```

Step 3. When labelling is done, evaluate the rating rule and the calibrated baseline against the labels.

```bash
python -m scripts.evaluate_labels --data data/raw/reviews.csv --truth evaluation/labeling/apparel_ecommerce_labeled.csv
```

- The evaluation refuses to start if any label is outside the profile's lists, for example `negative` instead of `Negative`, and suggests the fix.
- Files saved by Excel as "CSV UTF-8" are read correctly.
- `make_labeling_sample` never overwrites a file that contains labels. `--force` re-exports an unlabelled template with the same 200 review ids.

### Metrics produced

Step 3 writes `RESULTS_apparel_ecommerce.md` and `results/apparel_ecommerce_*.json` locally. These files are intentionally excluded from the public repository (git-ignored) because the dataset's redistribution terms have not been verified. They contain these metrics:

- Sentiment: accuracy with a 95% Wilson interval, macro-F1, per-class precision, recall and F1, and a confusion matrix, for both the rating rule and the baseline.
- Main topic: accuracy and macro-F1.
- Complaint detection (does the review contain any complaint?): accuracy, precision, recall with an interval, and F1.
- Issues (multi-label): micro and macro precision, recall and F1, exact-set match, and a per-issue table. Also main issue found or exact, and the false-complaint rate.
- Complaint topics and praised topics (multi-label): micro and macro precision, recall and F1.
- Outcome flags: accuracy, precision and recall.

### Population weighting

Plain accuracies describe the sample's 100 / 50 / 50 rating mix. The report also gives population-weighted accuracies, which re-weight the results to the dataset's real rating mix. Each review's weight is its rating group's size in the dataset divided by that group's size among the evaluated reviews.

### No rating leakage

- The labeller never sees the rating.
- Text methods are scored only on their text-only labels (`sentiment_ai` and the complaint and praise aspects), never on the rating-based primary `sentiment` column.
- The calibrated baseline was tuned on data that excludes these 200 reviews.
- The rating rule is a separate comparison row, scored only on reviews that have a usable rating.

The AI path (`--ai` and `scripts.check_ai_setup`) needs Anthropic API access and is not used in this project.

## Baseline calibration (apparel)

`python -m scripts.calibrate_baseline --data data/raw/reviews.csv --exclude evaluation/labeling/apparel_ecommerce_to_label.csv`
tunes the non-AI baseline's sentiment cut-offs against **ratings**. It excludes the hand-label
set, tunes on one hash-based half and reports on the other.
Results are kept local: the calibration report is excluded from the public repository because the dataset's redistribution terms have not been verified. Regenerate it with the command above.
The calibrated baseline is used only for reviews **without** a rating. Its numbers are
agreement with ratings, not accuracy.

## Caveats

- Synthetic-demo results are optimistic: the reviews are template-based, and the baseline
  keywords were written by the same person who wrote the templates.
- 200 hand labels give confidence intervals of roughly ±7 percentage points; the Wilson
  interval is reported with every accuracy.
