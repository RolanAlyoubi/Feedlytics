# Labelling evaluation — profile `food_delivery_demo`
_Generated 2026-10-09 by `python -m scripts.evaluate_labels` (data `sample_reviews_SYNTHETIC.csv`, 301 reviews, seed 7)._

> **Measured on SYNTHETIC, template-based reviews.** Real reviews are messier, so
> real-world accuracy will be lower. The keyword baseline was written by the same
> person who wrote the templates, which also flatters the baseline.

Evaluated reviews by rating group: {'Positive': 152, 'Negative': 83, 'Neutral': 61, 'Unrated': 5}. Accuracy columns are on this sample; *population-weighted* values re-weight them to the dataset's real rating mix.

## Summary

| method | n | sentiment_accuracy_% | sentiment_95%_CI | sentiment_macro_F1 | topic_accuracy_% | issue_found_% | primary_issue_exact_% | false_complaint_% | complaint_detection_F1 | issue_micro_F1 | praise_micro_F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rating rule (no text) | 296 | 85.8 | [81.4, 89.3] | 0.814 |  |  |  |  |  |  |  |
| Baseline: VADER + keywords | 301 | 69.4 | [64.0, 74.4] | 0.635 | 73.4 | 81.1 | 78.0 | 11.2 | 0.829 | 0.82 | 0.779 |

- **issue_found_%** — of reviews with a true complaint, how often the main true issue is among the predicted complaint issues.
- **false_complaint_%** — of reviews with no complaint, how often a complaint was predicted anyway.
- **complaint_detection_F1** — does the review contain any complaint (yes/no).
- **issue_micro_F1 / praise_micro_F1** — every (review, issue) or (review, praised topic) pair.
- 95% CI uses the Wilson score interval.

**Claude (AI) was not evaluated in this run.** It needs Anthropic API access (`--ai` with `ANTHROPIC_API_KEY`), which this project currently does not have. Only the non-AI methods above were measured.

## Population-weighted accuracy (%)

| method | sentiment_accuracy | topic_accuracy | complaint_detection_accuracy |
|---|---|---|---|
| Rating rule (no text) | 85.6 |  |  |
| Baseline: VADER + keywords | 69.2 | 73.2 | 85.2 |

## Rating rule (no text) — sentiment precision / recall / F1

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| Positive | 0.934 | 0.986 | 0.959 | 144 |
| Neutral | 0.623 | 0.731 | 0.673 | 52 |
| Negative | 0.892 | 0.74 | 0.809 | 100 |

## Baseline: VADER + keywords — sentiment precision / recall / F1

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| Positive | 0.79 | 0.769 | 0.779 | 147 |
| Neutral | 0.299 | 0.377 | 0.333 | 53 |
| Negative | 0.835 | 0.752 | 0.792 | 101 |

## Complaint detection (does the review contain any complaint?)

| method | n | accuracy | accuracy_ci95 | precision | recall | recall_ci95 | f1 | true_positives_in_truth | predicted_positives |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: VADER + keywords | 301 | 85.4 | [80.9, 88.9] | 0.849 | 0.811 | [73.5, 86.8] | 0.829 | 132 | 126 |

## Issue detection (multi-label)

| method | reviews | true labels | predicted | micro P | micro R | micro F1 | macro F1 | exact set % |
|---|---|---|---|---|---|---|---|---|
| Baseline: VADER + keywords | 301 | 155 | 150 | 0.833 | 0.806 | 0.82 | 0.849 | 82.4 |

## Complaint topics (multi-label, coarser)

| method | reviews | true labels | predicted | micro P | micro R | micro F1 | macro F1 | exact set % |
|---|---|---|---|---|---|---|---|---|
| Baseline: VADER + keywords | 301 | 152 | 148 | 0.831 | 0.809 | 0.82 | 0.821 | 82.7 |

## Praised topics (multi-label)

| method | reviews | true labels | predicted | micro P | micro R | micro F1 | macro F1 | exact set % |
|---|---|---|---|---|---|---|---|---|
| Baseline: VADER + keywords | 301 | 201 | 161 | 0.876 | 0.701 | 0.779 | 0.639 | 78.1 |

### Baseline: VADER + keywords — per issue (sorted by how often it truly occurs)

| label | support | predicted | tp | precision | recall | f1 |
|---|---|---|---|---|---|---|
| Late delivery | 34 | 39 | 25 | 0.641 | 0.735 | 0.685 |
| Missing items | 20 | 20 | 20 | 1.0 | 1.0 | 1.0 |
| Cold food | 17 | 11 | 11 | 1.0 | 0.647 | 0.786 |
| Poor packaging | 10 | 6 | 6 | 1.0 | 0.6 | 0.75 |
| Unresponsive support | 10 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| Hidden or high fees | 8 | 8 | 8 | 1.0 | 1.0 | 1.0 |
| Refund problems | 7 | 7 | 7 | 1.0 | 1.0 | 1.0 |
| Spilled or leaking | 7 | 7 | 7 | 1.0 | 1.0 | 1.0 |
| Courier behaviour | 6 | 6 | 6 | 1.0 | 1.0 | 1.0 |
| App crashes or bugs | 5 | 5 | 5 | 1.0 | 1.0 | 1.0 |
| Poor taste | 5 | 4 | 4 | 1.0 | 0.8 | 0.889 |
| Promo not applied | 5 | 5 | 5 | 1.0 | 1.0 | 1.0 |
| Inaccurate tracking | 4 | 12 | 4 | 0.333 | 1.0 | 0.5 |
| Payment failure | 4 | 4 | 4 | 1.0 | 1.0 | 1.0 |
| Too expensive | 4 | 4 | 4 | 1.0 | 1.0 | 1.0 |
| Order not delivered | 3 | 3 | 3 | 1.0 | 1.0 | 1.0 |
| Undercooked or stale food | 3 | 3 | 3 | 1.0 | 1.0 | 1.0 |
| Wrong items | 3 | 6 | 3 | 0.5 | 1.0 | 0.667 |

## Rating rule (no text) — sentiment confusion matrix (rows = truth)

| truth | Negative | Neutral | Positive |
|---|---|---|---|
| Positive | 2 | 0 | 142 |
| Neutral | 7 | 38 | 7 |
| Negative | 74 | 23 | 3 |

## Baseline: VADER + keywords — sentiment confusion matrix (rows = truth)

| truth | Negative | Neutral | Positive |
|---|---|---|---|
| Positive | 6 | 28 | 113 |
| Neutral | 9 | 20 | 24 |
| Negative | 76 | 19 | 6 |
