"""Evaluate labelling methods against ground truth and write a report.

Methods compared:
  rating    sentiment from the rating (scale from the profile) — no text used
  baseline  VADER sentiment + the profile's keyword rules — no AI
  ai        Claude (only with --ai; needs ANTHROPIC_API_KEY and costs money).
            Without --ai nothing here touches the API, the SDK or any key.

Ground truth:
  - synthetic food-delivery demo: the generator's ground-truth file (default)
  - real data: a hand-labelled CSV made with scripts/make_labeling_sample.py

Usage:
  python -m scripts.evaluate_labels                                  # synthetic demo
  python -m scripts.evaluate_labels --ai                             # also Claude (asks first)
  python -m scripts.evaluate_labels --data data/raw/reviews.csv \\
      --truth evaluation/labeling/apparel_ecommerce_labeled.csv

Outputs: evaluation/RESULTS_<profile>.md and evaluation/results/<profile>_<method>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.ai_analysis import (AILabeller, BaselineLabeller, LabelCache, default_cache_path,
                             estimate_ai_cost, label_reviews)
from app.data_processing import DataValidationError, process_reviews
from app.evaluation import (classification_metrics, evaluate_labels, population_weights,
                            rating_outcome_agreement, summary_row, validate_truth, weighted_accuracy)
from app.llm_client import ClaudeJsonLLM, LLMError, configured_model, has_credentials
from app.profiles import load_profile

DEFAULT_DATA = Path("data/sample_reviews_SYNTHETIC.csv")
DEFAULT_TRUTH = Path("data/sample_reviews_SYNTHETIC_ground_truth.csv")
OUT_DIR = Path("evaluation")


def stratified_sample(df: pd.DataFrame, truth: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Reviews with text and ground truth. If the truth file is small (e.g. hand
    labels), use all of it; otherwise sample keeping each true sentiment's share."""
    pool = df[df["has_text"]].merge(truth[["review_id", "true_sentiment"]], on="review_id")
    if len(pool) <= n:
        return df[df["review_id"].isin(pool["review_id"])]
    frac = n / len(pool)
    picked = pool.groupby("true_sentiment", group_keys=False)[["review_id"]].apply(
        lambda g: g.sample(frac=frac, random_state=seed))
    return df[df["review_id"].isin(picked["review_id"])]


def to_jsonable(result: Dict) -> Dict:
    out = {}
    for key, value in result.items():
        if isinstance(value, dict):
            out[key] = to_jsonable(value)
        elif isinstance(value, pd.DataFrame):
            out[key] = json.loads(value.to_json())
        else:
            out[key] = value
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate review labelling against ground truth.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--profile", default=None, help="profile name (default: detect from data)")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ai", action="store_true", help="also evaluate Claude (uses the API)")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--check-only", action="store_true",
                        help="only validate the labels and show progress (writes nothing)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.ai and not has_credentials():
        print("--ai was requested, but no Anthropic API access is configured "
              "(ANTHROPIC_API_KEY is missing or still the placeholder).\n"
              "Run without --ai to evaluate the non-AI methods (rating rule and baseline).",
              file=sys.stderr)
        return 2

    try:
        df, report = process_reviews(args.data, profile=args.profile)
    except DataValidationError as exc:
        print(f"Cannot read {args.data}: {exc}", file=sys.stderr)
        return 1
    profile = load_profile(report.profile_name)
    if not args.truth.exists():
        print(f"No ground-truth file at {args.truth}. For real data, create one with:\n"
              f"  python -m scripts.make_labeling_sample --data {args.data}", file=sys.stderr)
        return 1
    # utf-8-sig: spreadsheets often save CSV files with an invisible BOM marker.
    truth = pd.read_csv(args.truth, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    check = validate_truth(truth, {t: list(i) for t, i in profile.taxonomy.topics.items()},
                           list(profile.taxonomy.flags), known_ids=df["review_id"])
    if args.check_only:
        return print_progress(truth, check, df)
    if check["errors"]:
        print(f"The ground-truth file has {len(check['errors'])} problem(s); fix them before evaluating:",
              file=sys.stderr)
        for line in check["errors"][:30]:
            print(f"  - {line}", file=sys.stderr)
        return 1
    for line in check["warnings"][:10]:
        print(f"Note: {line}")
    truth = truth[truth["true_sentiment"].str.strip() != ""]  # skip rows not yet labelled
    if truth.empty:
        print("No labelled rows yet (true_sentiment is empty everywhere).", file=sys.stderr)
        return 1
    sample = stratified_sample(df, truth, args.sample, args.seed)
    mix = sample.merge(truth, on="review_id")["true_sentiment"].value_counts().to_dict()
    print(f"Profile {profile.name}; evaluating on {len(sample)} reviews (true sentiment mix: {mix})")

    # Population weights: the evaluation sample may over-represent some ratings
    # (the apparel hand-label sample is 50% low-rated); re-weight to the real mix.
    eval_strata = sample.set_index("review_id")["sentiment_from_rating"].fillna("Unrated")
    pop_strata = df.loc[df["has_text"], "sentiment_from_rating"].fillna("Unrated")
    weights = population_weights(eval_strata, pop_strata)
    composition = eval_strata.value_counts().to_dict()

    results: Dict[str, Dict] = {}
    topics = profile.taxonomy.topic_names
    issue_to_topic = profile.taxonomy.issue_to_topic

    # 1. Rating only (no text) — only if the data has ratings.
    if report.has_rating:
        rated = sample.merge(truth, on="review_id")
        rated = rated[rated["sentiment_from_rating"].notna()]
        rating_sent = classification_metrics(rated["true_sentiment"], rated["sentiment_from_rating"],
                                             ("Positive", "Neutral", "Negative"))
        correct = (rated["true_sentiment"] == rated["sentiment_from_rating"]).set_axis(rated["review_id"])
        results["Rating rule (no text)"] = {
            "n_reviews": rating_sent["n"], "sentiment": rating_sent,
            "population_weighted": {"sentiment_accuracy": weighted_accuracy(correct, weights)}}

    # 2. Baseline (VADER + keywords), text only; calibrated if the profile defines it.
    baseline_name = "Baseline: VADER + keywords" + (" (calibrated)" if profile.baseline_sentiment else "")
    labelled, _ = label_reviews(sample, BaselineLabeller(profile))
    results[baseline_name] = evaluate_labels(labelled, truth, topics, issue_to_topic, weights)

    # 3. Claude.
    ai_note = ("**Claude (AI) was not evaluated in this run.** It needs Anthropic API access "
               "(`--ai` with `ANTHROPIC_API_KEY`), which this project currently does not have. "
               "Only the non-AI methods above were measured.")
    if args.ai:  # credentials were checked at start-up
        model = configured_model()
        est = estimate_ai_cost(sample.loc[sample["has_text"], "analysis_text"].tolist(), model)
        print(f"AI labelling: {est['reviews']} reviews in {est['batches']} calls to {model}; "
              f"estimated cost ≤ ${est['cost_usd']:.2f} (cached labels are free).")
        if args.yes or confirm("Proceed? [y/N] "):
            labeller = AILabeller(ClaudeJsonLLM(model), profile)
            cache = LabelCache(default_cache_path(model, profile))
            try:
                labelled_ai, run = label_reviews(
                    sample, labeller, cache=cache, max_rows=len(sample),
                    progress=lambda d, t: print(f"  labelled {d}/{t}", end="\r"))
            except LLMError as exc:
                print(f"AI evaluation failed: {exc}")
            else:
                print()
                ai_rows = labelled_ai["label_source"].isin(["ai", "cache"])
                results[f"AI: {model}"] = evaluate_labels(labelled_ai[ai_rows], truth, topics,
                                                          issue_to_topic, weights)
                ai_note = (f"AI run: model `{model}`, {run.from_ai} reviews labelled by the API, "
                           f"{run.from_cache} from cache, {run.fallback_rows} fell back to the "
                           f"baseline (excluded from AI scores). Tokens: {run.usage.input_tokens:,} in / "
                           f"{run.usage.output_tokens:,} out; cost ≈ ${run.cost_usd or 0:.2f}.")
                if run.errors:
                    ai_note += f" Errors: {sorted(set(run.errors))}."

    agreements = {o.label: rating_outcome_agreement(df, o.column, profile.rating_scale.negative_max,
                                                    profile.rating_scale.positive_min)
                  for o in report.roles.outcomes}
    write_report(args, profile.name, results, ai_note, sample, agreements, composition)
    return 0


def print_progress(truth: pd.DataFrame, check: Dict, df: pd.DataFrame) -> int:
    """--check-only: label progress and problems; writes nothing."""
    labelled = truth[truth["true_sentiment"].astype(str).str.strip() != ""]
    print(f"Labelled: {check['labelled']} of {check['rows']} reviews")
    if not labelled.empty:
        print(f"  sentiment: {labelled['true_sentiment'].value_counts().to_dict()}")
        issue_col = "true_issues" if "true_issues" in truth.columns else "true_issue"
        with_issue = (labelled[issue_col].astype(str).str.strip() != "").sum()
        print(f"  with at least one complaint: {with_issue}")
    for line in check["errors"][:30]:
        print(f"  error: {line}")
    for line in check["warnings"][:15]:
        print(f"  warning: {line}")
    if check["errors"]:
        print(f"{len(check['errors'])} error(s) to fix before evaluating.")
        return 1
    print("No errors." + (" Ready to evaluate." if check["labelled"] else ""))
    return 0


def _per_class_table(result: Dict) -> pd.DataFrame:
    rows = result["sentiment"]["per_class"]
    return pd.DataFrame(rows)[["label", "precision", "recall", "f1", "support"]]


def _multilabel_row(name: str, m: Optional[Dict]) -> Optional[Dict]:
    if not m:
        return None
    return {"method": name, "reviews": m["n_reviews"], "true labels": m["true_labels"],
            "predicted": m["predicted_labels"], "micro P": m["micro_precision"],
            "micro R": m["micro_recall"], "micro F1": m["micro_f1"], "macro F1": m["macro_f1"],
            "exact set %": m["exact_match_pct"]}


def write_report(args, profile_name: str, results: Dict[str, Dict], ai_note: str,
                 sample: pd.DataFrame, agreements: Dict[str, Optional[Dict]],
                 composition: Optional[Dict] = None) -> None:
    out_dir = args.out_dir
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        (out_dir / "results" / f"{profile_name}_{slug}.json").write_text(
            json.dumps(to_jsonable(result), indent=2, default=str))

    table = pd.DataFrame([summary_row(n, r) if "primary_topic" in r else
                          {"method": n, "n": r["n_reviews"],
                           "sentiment_accuracy_%": r["sentiment"]["accuracy"],
                           "sentiment_95%_CI": r["sentiment"]["accuracy_ci95"],
                           "sentiment_macro_F1": r["sentiment"]["macro_f1"]}
                          for n, r in results.items()])
    print("\n" + table.to_string(index=False))

    synthetic = "SYNTHETIC" in args.data.name.upper()
    sections = [
        f"# Labelling evaluation — profile `{profile_name}`",
        f"_Generated {date.today().isoformat()} by `python -m scripts.evaluate_labels` "
        f"(data `{args.data.name}`, {len(sample)} reviews, seed {args.seed})._",
        "",
    ]
    if synthetic:
        sections += [
            "> **Measured on SYNTHETIC, template-based reviews.** Real reviews are messier, so",
            "> real-world accuracy will be lower. The keyword baseline was written by the same",
            "> person who wrote the templates, which also flatters the baseline.",
            "",
        ]
    else:
        sections += [
            "> **Ground truth = hand labels made blind to the rating** (the labelling file hides it).",
            "> Text methods are scored on their text-only labels; the rating never enters their score.",
            "",
        ]
    if composition:
        sections += [f"Evaluated reviews by rating group: {composition}. Accuracy columns are on this "
                     "sample; *population-weighted* values re-weight them to the dataset's real rating mix.", ""]
    sections += [
        "## Summary", "", markdown_table(table), "",
        "- **issue_found_%** — of reviews with a true complaint, how often the main true issue is among the predicted complaint issues.",
        "- **false_complaint_%** — of reviews with no complaint, how often a complaint was predicted anyway.",
        "- **complaint_detection_F1** — does the review contain any complaint (yes/no).",
        "- **issue_micro_F1 / praise_micro_F1** — every (review, issue) or (review, praised topic) pair.",
        "- 95% CI uses the Wilson score interval.", "", ai_note, "",
    ]
    weighted = [{"method": n, **r["population_weighted"]} for n, r in results.items()
                if r.get("population_weighted")]
    if weighted:
        sections += ["## Population-weighted accuracy (%)", "", markdown_table(pd.DataFrame(weighted)), ""]
    for name, result in results.items():
        sections += [f"## {name} — sentiment precision / recall / F1", "",
                     markdown_table(_per_class_table(result)), ""]
    detection = [{"method": n, **{k: v for k, v in r["complaint_detection"].items()}}
                 for n, r in results.items() if r.get("complaint_detection")]
    if detection:
        sections += ["## Complaint detection (does the review contain any complaint?)", "",
                     markdown_table(pd.DataFrame(detection)), ""]
    for key, title in (("issues", "Issue detection (multi-label)"),
                       ("complaint_topics", "Complaint topics (multi-label, coarser)"),
                       ("praise_topics", "Praised topics (multi-label)")):
        rows = [r for r in (_multilabel_row(n, res.get(key)) for n, res in results.items()) if r]
        if rows:
            sections += [f"## {title}", "", markdown_table(pd.DataFrame(rows)), ""]
    for name, result in results.items():
        per_issue = (result.get("issues") or {}).get("per_label")
        if per_issue:
            sections += [f"### {name} — per issue (sorted by how often it truly occurs)", "",
                         markdown_table(pd.DataFrame(per_issue)), ""]
    for name, result in results.items():
        flags = result.get("flags") or {}
        if flags:
            rows = pd.DataFrame([{"flag": f, **m} for f, m in flags.items()])
            sections += [f"### {name} — outcome flags", "", markdown_table(rows), ""]
    for label, agreement in agreements.items():
        if agreement:
            sections += [
                f"## Rating vs '{label}' (sanity check, no hand labels needed)", "",
                f"Agreement on clearly positive/negative ratings: **{agreement['agreement_pct']}%** "
                f"(95% CI {agreement['agreement_ci95']}, n={agreement['n']:,}). "
                f"Positive rating but not {label.lower()}: {agreement['positive_but_no']:,}; "
                f"negative rating but {label.lower()}: {agreement['negative_but_yes']:,}.", "",
            ]
    for name, result in results.items():
        conf = result["sentiment"]["confusion"]
        sections += [f"## {name} — sentiment confusion matrix (rows = truth)", "",
                     markdown_table(conf.reset_index().rename(columns={"true": "truth"})), ""]
    path = out_dir / f"RESULTS_{profile_name}.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    sys.exit(main())
