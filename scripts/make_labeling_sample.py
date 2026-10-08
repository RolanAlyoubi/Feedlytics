"""Create a sample of real reviews for hand labelling (ground truth for evaluation).

The sample over-represents low ratings (same allocation as the AI sample), so
complaints get enough examples. It writes:
  evaluation/labeling/<profile>_to_label.csv   fill in the true_* columns
  evaluation/labeling/<profile>_GUIDE.md       allowed values and labelling rules

Label in a spreadsheet, save as <profile>_labeled.csv, then run:
  python -m scripts.evaluate_labels --data <data.csv> \\
      --truth evaluation/labeling/<profile>_labeled.csv

Usage:
  python -m scripts.make_labeling_sample --data data/raw/reviews.csv --n 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from app.ai_analysis import sample_for_ai
from app.data_processing import DataValidationError, process_reviews
from app.profiles import load_profile

OUT_DIR = Path("evaluation/labeling")


def guide_text(profile) -> str:
    lines = [
        f"# Hand-labelling guide — profile `{profile.name}`", "",
        "Label each review from its **text only** (ignore the rating column when judging sentiment).", "",
        "| Column | Allowed values |", "|---|---|",
        "| `true_sentiment` | Positive, Neutral, Negative — the overall tone of the text |",
        "| `true_topic` | the main topic (list below) |",
        "| `true_issue` | the main complaint's issue (list below); **leave empty** if there is no complaint |",
    ]
    for flag, definition in profile.taxonomy.flags.items():
        lines.append(f"| `true_{flag}` | true / false — {definition} |")
    lines += ["", "## Topics and issues", ""]
    for topic, issues in profile.taxonomy.topics.items():
        lines.append(f"**{topic}**")
        lines += [f"- {issue} — {definition}" for issue, definition in issues.items()]
        lines.append("")
    lines += [
        "## Tips",
        "- A review that praises something but whose main point is a complaint is Negative.",
        "- If a review was cut off mid-sentence, label only what is there.",
        "- Label in one sitting if possible and keep notes on hard cases; consistency matters more than speed.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create a hand-labelling sample.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    try:
        df, report = process_reviews(args.data, profile=args.profile)
    except DataValidationError as exc:
        print(f"Cannot read {args.data}: {exc}", file=sys.stderr)
        return 1
    profile = load_profile(report.profile_name)
    picked = sample_for_ai(df, max_rows=args.n, seed=args.seed).index
    cols = [c for c in ("review_id", "rating", "review_title", "review_text") if c in df.columns]
    out = df.loc[picked, cols].sample(frac=1, random_state=args.seed)  # shuffle so ratings are mixed
    out["true_sentiment"] = ""
    out["true_topic"] = ""
    out["true_issue"] = ""
    for flag in profile.taxonomy.flags:
        out[f"true_{flag}"] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{profile.name}_to_label.csv"
    guide_path = args.out_dir / f"{profile.name}_GUIDE.md"
    out.to_csv(csv_path, index=False)
    guide_path.write_text(guide_text(profile), encoding="utf-8")
    print(f"Wrote {len(out)} reviews to {csv_path}\nLabelling guide: {guide_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
