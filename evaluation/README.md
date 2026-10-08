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

```bash
python -m scripts.make_labeling_sample --data data/raw/reviews.csv --n 200
# label evaluation/labeling/apparel_ecommerce_to_label.csv using the GUIDE,
# save it as evaluation/labeling/apparel_ecommerce_labeled.csv, then check (free):
python -m scripts.check_ai_setup             # key, model, label values, cost estimate
python -m scripts.check_ai_setup --smoke     # optional: ONE paid call (~$0.01) to confirm the API works
python -m scripts.evaluate_labels --data data/raw/reviews.csv \
    --truth evaluation/labeling/apparel_ecommerce_labeled.csv --ai
```

The sample over-represents low ratings (100 low-rated, 50 mid-rated, 50 high-rated), so
complaint detection has enough examples. Accuracy is reported **on this sample mix**, not the
dataset's natural mix. Label from the text, not the rating.

The evaluation refuses to start (before any API call) if the labels contain values outside
the profile's lists, e.g. `negative` instead of `Negative`, and suggests the fix.
Files saved by Excel as "CSV UTF-8" are read correctly.

## Caveats

- Synthetic-demo results are optimistic: the reviews are template-based, and the baseline
  keywords were written by the same person who wrote the templates.
- 200 hand labels give confidence intervals of roughly ±7 percentage points; the Wilson
  interval is reported with every accuracy.
