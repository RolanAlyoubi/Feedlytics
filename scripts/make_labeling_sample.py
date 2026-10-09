"""Create a BLIND sample of real reviews for hand labelling (ground truth for evaluation).

The labelling file contains no rating: the labeller judges the text alone, so
the rating cannot leak into the ground truth (which would flatter the rating
rule and penalise text methods unfairly).

The sample over-represents low ratings (same allocation as the AI sample), so
complaints get enough examples. The evaluation re-weights results back to the
dataset's real rating mix.

Writes (both in evaluation/labeling/):
  <profile>_to_label.csv   review_id, title, text + empty true_* columns (git-ignored)
  <profile>_GUIDE.md       how to label, with every allowed value (committed)

Safety:
  - An existing labelling file is never overwritten unless --force is given,
    and never if it already contains any labels.
  - With --force, the review ids of the existing file are re-used, so the same
    reviews stay excluded from baseline calibration.

Usage:
  python -m scripts.make_labeling_sample --data data/raw/reviews.csv --n 200
  python -m scripts.make_labeling_sample --data data/raw/reviews.csv --force   # re-export, same ids
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.ai_analysis import sample_for_ai
from app.data_processing import DataValidationError, process_reviews
from app.evaluation import LIST_SEPARATOR
from app.profiles import Profile, load_profile

OUT_DIR = Path("evaluation/labeling")
TEXT_COLUMNS = ("review_id", "review_title", "review_text")


def label_columns(profile: Profile) -> List[str]:
    return (["true_sentiment", "true_topic", "true_issues", "true_praise_topics"]
            + [f"true_{flag}" for flag in profile.taxonomy.flags])


def guide_text(profile: Profile) -> str:
    sep = LIST_SEPARATOR
    lines = [
        f"# Hand-labelling guide — profile `{profile.name}`",
        "",
        f"You are labelling {profile.domain_description}. Your labels are the **ground truth** used to",
        "measure how accurate the labelling methods are, so consistency matters more than speed.",
        "",
        "## Workflow",
        "",
        f"1. Copy `{profile.name}_to_label.csv` to **`{profile.name}_labeled.csv`** in this folder and",
        "   label the copy (the original stays as a clean template).",
        "2. Open it in a spreadsheet. Fill in the `true_*` columns for each row; leave every other column",
        "   unchanged. Save as **CSV** (comma-separated; \"CSV UTF-8\" is fine).",
        "3. Check your progress and catch typos at any time (no API, no cost):",
        f"   `python -m scripts.evaluate_labels --data data/raw/reviews.csv --truth evaluation/labeling/{profile.name}_labeled.csv --check-only`",
        "4. You can stop part-way: rows with an empty `true_sentiment` are skipped.",
        "",
        "**The rating is hidden on purpose.** Judge the text (title + review) only.",
        "",
        "## Columns",
        "",
        "| Column | What to enter | Allowed values |",
        "|---|---|---|",
        "| `true_sentiment` | The overall tone of the text | `Positive`, `Neutral`, `Negative` |",
        "| `true_topic` | The single topic the review is mainly about: its main complaint if it has one, otherwise its main praise | one topic name (list below) |",
        f"| `true_issues` | **Every** complaint the text mentions, most important first, separated by `{sep}` — leave **empty** if there is no complaint | issue names (list below) |",
        f"| `true_praise_topics` | Every topic the customer praises, separated by `{sep}` — leave empty if none | topic names (list below) |",
    ]
    for flag, definition in profile.taxonomy.flags.items():
        lines.append(f"| `true_{flag}` | {definition} | `true` or `false` |")
    lines += [
        "",
        "Values must match the lists below exactly (capitals and spelling). The check command",
        "suggests a fix for near-misses such as `negative` → `Negative`.",
        "",
        "## Sentiment rules",
        "",
        "- **Negative** — the main point is a complaint or disappointment, even if something is praised",
        "  (\"Love the colour, but it fell apart after one wash and I'm returning it\").",
        "- **Positive** — mostly happy; a small reservation does not change that",
        "  (\"Beautiful dress, runs a little long but I'll keep it\").",
        "- **Neutral** — mixed with no clear winner, indifferent, or purely factual",
        "  (\"It's okay. Fits as expected.\").",
        "",
        "## Complaint (issue) rules",
        "",
        "- List **every** distinct complaint, not just the main one. Put the main complaint first.",
        "- A complaint counts even inside a positive review (\"Love it, but it runs small\" → `Runs small`).",
        "- Neutral sizing *advice* is still a complaint about fit if it says the size is off",
        "  (\"order a size down\" → `Runs large`). Pure facts (\"I'm 5'4 and ordered a S\") are not.",
        "- Use `Other` only for a real complaint that fits no issue below.",
        "- Each issue belongs to one topic; the main issue's topic should normally equal `true_topic`.",
        "",
        "## Praise rules",
        "",
        "- List the topics the customer explicitly praises (\"so soft\" → `Fabric & Comfort`,",
        "  \"flattering\" → `Style & Design`). General enthusiasm with no topic (\"Love it!\") → leave empty.",
        "",
        "## Outcome flags",
        "",
    ]
    for flag, definition in profile.taxonomy.flags.items():
        lines.append(f"- `true_{flag}`: `true` if {definition}; otherwise `false`.")
    lines += [
        "",
        "## Truncated reviews",
        "",
        "Some reviews were cut off by the data source mid-sentence. Label only what is there; do not",
        "guess how the sentence ended.",
        "",
        "## Topics and issues",
        "",
    ]
    for topic, issues in profile.taxonomy.topics.items():
        lines.append(f"**{topic}**")
        lines += [f"- `{issue}` — {definition}" for issue, definition in issues.items()]
        lines.append("")
    lines += [
        "## Tips",
        "",
        "- Label in a few sittings and keep a short note of hard cases and how you decided them.",
        "- Expect roughly 30–60 seconds per review (about 2–3 hours for 200).",
        "- Don't look up the rating, and don't change `review_id`, the title or the text.",
    ]
    return "\n".join(lines) + "\n"


def has_labels(path: Path) -> bool:
    existing = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    true_cols = [c for c in existing.columns if c.startswith("true_")]
    return bool(true_cols) and bool((existing[true_cols].apply(lambda c: c.str.strip()) != "").any().any())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create a blind hand-labelling sample.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="re-export an existing (unlabelled) sample, re-using its review ids")
    args = parser.parse_args(argv)

    try:
        df, report = process_reviews(args.data, profile=args.profile)
    except DataValidationError as exc:
        print(f"Cannot read {args.data}: {exc}", file=sys.stderr)
        return 1
    profile = load_profile(report.profile_name)
    csv_path = args.out_dir / f"{profile.name}_to_label.csv"
    guide_path = args.out_dir / f"{profile.name}_GUIDE.md"

    if csv_path.exists():
        if not args.force:
            print(f"{csv_path} already exists; not overwriting. Use --force to re-export it with the "
                  "same review ids.", file=sys.stderr)
            return 1
        if has_labels(csv_path):
            print(f"{csv_path} already contains labels; refusing to overwrite it.", file=sys.stderr)
            return 1
        ids = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")["review_id"].tolist()
        missing = sorted(set(ids) - set(df["review_id"]))
        if missing:
            print(f"Existing sample ids not found in the data: {missing[:5]}", file=sys.stderr)
            return 1
        out = df.set_index("review_id").loc[ids].reset_index()  # same ids, same order
        how = "re-used the existing review ids"
    else:
        picked = sample_for_ai(df, max_rows=args.n, seed=args.seed).index
        out = df.loc[picked].sample(frac=1, random_state=args.seed)  # shuffle so strata are mixed
        how = f"new sample (seed {args.seed})"

    out = out[[c for c in TEXT_COLUMNS if c in out.columns]].copy()
    for col in label_columns(profile):
        out[col] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    guide_path.write_text(guide_text(profile), encoding="utf-8")
    print(f"Wrote {len(out)} reviews to {csv_path} ({how}; rating hidden)\nLabelling guide: {guide_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
