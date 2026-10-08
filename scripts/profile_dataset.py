"""Print a data-quality report and basic analytics for any feedback CSV.

Sections adapt to the data: rating, trend and driver sections appear only
when the dataset has those columns, and breakdowns follow the profile's roles.

Usage:
  python -m scripts.profile_dataset                               # synthetic food demo
  python -m scripts.profile_dataset data/raw/reviews.csv          # profile auto-detected
  python -m scripts.profile_dataset my.csv --profile generic
  python -m scripts.profile_dataset --check-ground-truth          # synthetic demo only

--check-ground-truth runs the priority ranking on the generator's TRUE issue
labels. It is a sanity check that the analysis recovers the patterns built
into the synthetic data — not a finding about real customers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app import analytics
from app.data_processing import DataValidationError, process_reviews
from app.evaluation import rating_outcome_agreement
from app.profiles import load_profile

DEFAULT_FILE = Path("data/sample_reviews_SYNTHETIC.csv")
GROUND_TRUTH_FILE = Path("data/sample_reviews_SYNTHETIC_ground_truth.csv")
MAX_GROUP_ROWS = 15


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(df: pd.DataFrame, empty_message: str = "(not available for this dataset)") -> None:
    print(df.to_string(index=False) if not df.empty else empty_message)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a customer-feedback CSV.")
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--profile", default=None, help="profile name (default: auto-detect)")
    parser.add_argument("--check-ground-truth", action="store_true")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    try:
        df, report = process_reviews(args.csv, profile=args.profile)
    except DataValidationError as exc:
        print(f"Cannot analyse {args.csv}: {exc}", file=sys.stderr)
        return 1
    profile = load_profile(report.profile_name)
    roles, scale = report.roles, profile.rating_scale
    n = 0

    def next_section(title: str) -> None:
        nonlocal n
        n += 1
        section(f"{n}. {title}")

    print(f"Profile: {profile.display_name} — {report.profile_reason}")
    for note in profile.notes:
        print(f"NOTE: {note}")

    next_section("DATA QUALITY")
    print(f"Rows received: {report.rows_received:,}   rows retained: {report.rows_retained:,}   "
          f"rows removed: {report.rows_removed:,}")
    if report.column_mapping:
        print(f"Columns renamed: {report.column_mapping}")
    if report.missing_optional_columns:
        print(f"Optional columns not present: {report.missing_optional_columns}")
    show(report.summary_table(), "No problems found.")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for note in report.notes:
        print(f"INFO: {note}")
    print(f"Roles: entity={roles.entity!r}, segments={[c for c, _ in roles.segments]}, "
          f"numeric={[s.column for s in roles.numeric]}, outcomes={[o.column for o in roles.outcomes]}")

    next_section("OVERVIEW (sentiment here is derived from the rating)" if report.has_rating
                 else "OVERVIEW (no rating column — sentiment needs text labelling)")
    kpis = analytics.overview_kpis(df, scale=scale, dimensions=[c for c, _ in roles.dimensions])
    for key, value in kpis.items():
        print(f"  {key:<22} {value}")

    if report.has_rating:
        next_section("RATING DISTRIBUTION")
        show(analytics.rating_distribution(df, scale))

    if roles.has_date:
        next_section("MONTHLY TREND")
        show(analytics.reviews_over_time(df, scale=scale))

    outcome_cols = [o.column for o in roles.outcomes]
    for col, label in roles.dimensions:
        next_section(f"BY {label.upper()}"
                     + (f" (largest {MAX_GROUP_ROWS} of {df[col].nunique():,})" if df[col].nunique() > MAX_GROUP_ROWS else ""))
        summary = analytics.group_summary(df, col, scale=scale, outcome_cols=outcome_cols)
        show(summary.head(MAX_GROUP_ROWS))
        if not summary.empty and not summary["reliable"].all():
            print(f"({(~summary['reliable']).sum():,} groups have fewer than 15 reviews: read with caution)")

    for spec in roles.numeric:
        if not report.has_rating:
            break
        next_section(f"{spec.label.upper()} vs RATING")
        show(analytics.band_summary(df, spec, scale))
        corr = analytics.correlation_with_rating(df, spec.column)
        if corr:
            print(f"Spearman rho = {corr['rho']} (n={corr['n']:,}): {corr['strength']} "
                  f"{corr['direction']} association. Association is not causation.")

    for outcome in roles.outcomes:
        agreement = rating_outcome_agreement(df, outcome.column, scale.negative_max, scale.positive_min)
        if agreement:
            next_section(f"RATING vs {outcome.label.upper()}")
            print(agreement["table"].to_string())
            print(f"Agreement on clearly positive/negative ratings: {agreement['agreement_pct']}% "
                  f"(n={agreement['n']:,})")

    if report.has_rating:
        next_section("GROUPS MOST ASSOCIATED WITH LOW RATINGS")
        show(analytics.low_rating_drivers(df, roles.dimensions, scale=scale).head(10))

    if roles.has_date and roles.entity:
        next_section(f"NEGATIVE-REVIEW RATE: LAST 3 MONTHS vs PREVIOUS 3, BY {roles.entity_label.upper()}")
        show(analytics.recent_change(df, roles.entity))

    if args.check_ground_truth:
        next_section("GENERATOR SANITY CHECK — priority on TRUE issue labels (synthetic only)")
        truth = pd.read_csv(GROUND_TRUTH_FILE, dtype=str)
        merged = df.merge(truth[["review_id", "true_issue"]], on="review_id", how="left")
        issues = merged[merged["true_issue"].notna() & (merged["true_issue"] != "")]
        show(analytics.priority_table(issues, "true_issue", total_reviews=len(df), scale=scale))
        print("\nTrue issue 'App crashes or bugs' by month (built-in spike in Sep–Oct 2025):")
        app = issues[issues["true_issue"] == "App crashes or bugs"].groupby("month").size()
        print(app.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
