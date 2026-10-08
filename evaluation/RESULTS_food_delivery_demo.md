# Labelling evaluation — profile `food_delivery_demo`
_Generated 2026-10-08 by `python -m scripts.evaluate_labels` (data `sample_reviews_SYNTHETIC.csv`, 301 reviews, seed 7)._

> **Measured on SYNTHETIC, template-based reviews.** Real reviews are messier, so
> real-world accuracy will be lower. The keyword baseline was written by the same
> person who wrote the templates, which also flatters the baseline.

## Summary

| method | n | sentiment_accuracy_% | sentiment_95%_CI | sentiment_macro_F1 | topic_accuracy_% | issue_found_% | primary_issue_exact_% | false_complaint_% |
|---|---|---|---|---|---|---|---|---|
| Rating rule (no text) | 301 | 84.4 | [79.9, 88.0] | 0.807 |  |  |  |  |
| Baseline: VADER + keywords | 301 | 69.4 | [64.0, 74.4] | 0.635 | 73.4 | 81.1 | 78.0 | 11.2 |

- **issue_found_%** — of reviews with a true complaint, how often the true issue is among the predicted complaint issues.
- **false_complaint_%** — of reviews with no complaint, how often a complaint was predicted anyway.
- 95% CI uses the Wilson score interval.

**Claude (AI) was not evaluated in this run.** It needs Anthropic API access (`--ai` with `ANTHROPIC_API_KEY`), which this project currently does not have. Only the non-AI methods above were measured.

## Rating rule (no text) — sentiment confusion matrix (rows = truth)

| truth | (none) | Negative | Neutral | Positive |
|---|---|---|---|---|
| Positive | 3 | 2 | 0 | 142 |
| Neutral | 1 | 7 | 38 | 7 |
| Negative | 1 | 74 | 23 | 3 |

## Baseline: VADER + keywords — sentiment confusion matrix (rows = truth)

| truth | Negative | Neutral | Positive |
|---|---|---|---|
| Positive | 6 | 28 | 113 |
| Neutral | 9 | 20 | 24 |
| Negative | 76 | 19 | 6 |
