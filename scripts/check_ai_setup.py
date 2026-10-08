"""Check that everything is ready for a paid Claude evaluation — without calling the API.

What it checks (free, offline):
  1. Python and SDK versions
  2. The API key: where it comes from (.env or the shell), whether it still holds
     the placeholder, whether it looks like an Anthropic key. The key is never printed.
  3. The model that will be used
  4. The hand-labelled ground-truth file: columns, allowed values, review ids
  5. How many labels are already cached (free to re-use) and the cost estimate

Optional smoke test (PAID, about one cent): one API call labelling three short
made-up reviews, to confirm the request is accepted before the full evaluation.
It asks for confirmation unless --yes is given.

Usage:
  python -m scripts.check_ai_setup
  python -m scripts.check_ai_setup --smoke
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_DATA = Path("data/raw/reviews.csv")
DEFAULT_TRUTH = Path("evaluation/labeling/apparel_ecommerce_labeled.csv")
from app.llm_client import PLACEHOLDER_KEYS as PLACEHOLDERS  # one definition, shared
SMOKE_REVIEWS = [  # invented examples, so no third-party text is sent for the smoke test
    "Runs small, I had to size up and will return this one.",
    "Beautiful soft fabric. I bought it in two colours!",
    "It's okay, nothing special.",
]


def status(ok: Optional[bool], text: str) -> None:
    mark = {True: "OK  ", False: "FIX ", None: "INFO"}[ok]
    print(f"[{mark}] {text}")


def describe_key(value: Optional[str]) -> str:
    """Describe a key without revealing it."""
    if value is None:
        return "not set"
    if value.strip() in PLACEHOLDERS:
        return "still the placeholder from .env.example"
    shape = "looks like an Anthropic API key" if value.startswith("sk-ant-") and len(value) > 40 \
        else "does not look like an Anthropic API key (expected to start with 'sk-ant-')"
    return f"set, {len(value)} characters, {shape}"


def check_key() -> bool:
    from dotenv import dotenv_values

    shell_key = os.environ.get("ANTHROPIC_API_KEY")
    env_path = Path(".env")
    file_key = dotenv_values(env_path).get("ANTHROPIC_API_KEY") if env_path.exists() else None

    status(env_path.exists() or None, f".env file: {'found' if env_path.exists() else 'not found'}")
    status(None, f"ANTHROPIC_API_KEY in .env: {describe_key(file_key)}")
    status(None, f"ANTHROPIC_API_KEY in your shell environment: {describe_key(shell_key)}")

    used = shell_key if shell_key is not None else file_key
    source = "shell environment" if shell_key is not None else ".env"
    if shell_key is not None and file_key is not None and shell_key != file_key:
        status(None, "Both are set and differ: the SHELL value wins (.env does not override it).")
    usable = used is not None and used.strip() not in PLACEHOLDERS
    status(usable, f"Key that will be used: from {source} — {describe_key(used)}" if used is not None
           else "No API key found. Copy .env.example to .env and paste your key.")
    if os.environ.get("ANTHROPIC_AUTH_TOKEN") and not usable:
        status(None, "ANTHROPIC_AUTH_TOKEN is set and will be used instead of an API key.")
        usable = True
    return usable


def check_truth(truth_path: Path, data_path: Path):
    from app.data_processing import DataValidationError, process_reviews
    from app.evaluation import validate_truth
    from app.profiles import load_profile

    if not data_path.exists():
        status(False, f"Dataset not found: {data_path}")
        return None, None, False
    try:
        df, report = process_reviews(data_path)
    except DataValidationError as exc:
        status(False, f"Dataset could not be read: {exc}")
        return None, None, False
    profile = load_profile(report.profile_name)
    status(True, f"Dataset: {data_path} ({report.rows_retained:,} reviews, profile '{profile.name}')")

    if not truth_path.exists():
        status(False, f"Ground-truth file not found: {truth_path}\n"
                      f"       Label evaluation/labeling/{profile.name}_to_label.csv and save it as that name.")
        return df, profile, False
    truth = pd.read_csv(truth_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if truth.shape[1] == 1 and ";" in truth.columns[0]:
        status(False, "The file uses ';' separators. Save it as a comma-separated CSV.")
        return df, profile, False
    result = validate_truth(truth, {t: list(i) for t, i in profile.taxonomy.topics.items()},
                            list(profile.taxonomy.flags), known_ids=df["review_id"])
    ok = not result["errors"] and result["labelled"] > 0
    status(ok, f"Ground truth: {result['labelled']} of {result['rows']} rows labelled, "
               f"{len(result['errors'])} error(s), {len(result['warnings'])} warning(s)")
    for line in result["errors"][:20]:
        print(f"       error: {line}")
    for line in result["warnings"][:10]:
        print(f"       warning: {line}")
    if result["labelled"] and result["labelled"] < 150:
        status(None, f"Only {result['labelled']} labelled reviews: accuracy intervals will be wide.")
    return df, profile, ok


def estimate(df, profile, truth_path: Path, model: str) -> None:
    from app.ai_analysis import LabelCache, default_cache_path, estimate_ai_cost

    if df is None or not truth_path.exists():
        return
    truth = pd.read_csv(truth_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ids = set(truth.loc[truth["true_sentiment"].str.strip() != "", "review_id"])
    texts = df.loc[df["review_id"].isin(ids) & df["has_text"], "analysis_text"].tolist()
    cache = LabelCache(default_cache_path(model, profile))
    todo = [t for t in texts if cache.get(LabelCache.key(model, t, profile)) is None]
    est = estimate_ai_cost(todo, model)
    status(None, f"{len(texts) - len(todo)} of {len(texts)} reviews already cached (free). "
                 f"To label: {est['reviews']} in {est['batches']} API calls, "
                 f"estimated ≤ ${est['cost_usd']:.2f} with {model}")


def smoke_test(model: str, assume_yes: bool) -> bool:
    from app.ai_analysis import AILabeller
    from app.llm_client import ClaudeJsonLLM, LLMError
    from app.profiles import load_profile

    print(f"\nSmoke test: 1 API call to {model} with 3 invented reviews (about $0.01).")
    if not assume_yes:
        try:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Skipped.")
                return False
        except EOFError:
            print("Skipped (no confirmation).")
            return False
    try:
        labeller = AILabeller(ClaudeJsonLLM(model), load_profile("apparel_ecommerce"))
        labels = labeller.label_batch(SMOKE_REVIEWS)
    except LLMError as exc:
        status(False, f"Smoke test failed: {exc}")
        return False
    for text, label in zip(SMOKE_REVIEWS, labels):
        if label is None:
            print(f"  (no label returned) {text}")
            continue
        issues = [a["issue"] for a in label["aspects"] if a["polarity"] == "negative"]
        print(f"  {label['sentiment']:<8} topic={label['primary_topic']:<18} issues={issues} "
              f"flags={label['flags']}  <- {text}")
    usage = labeller.usage
    cost = usage.cost_usd(model)
    status(all(labels), f"API call accepted. Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out"
                        + (f", cost ≈ ${cost:.4f}" if cost is not None else ""))
    return all(labels)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check the AI setup without spending credits.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--smoke", action="store_true", help="also make ONE paid API call (~$0.01)")
    parser.add_argument("--yes", action="store_true", help="skip the smoke-test confirmation")
    args = parser.parse_args(argv)

    from app.llm_client import configured_model

    print("Feedlytics AI setup check (no API calls unless --smoke)")
    print("The AI path is optional: everything else in Feedlytics works without it.\n")
    status(sys.version_info >= (3, 9), f"Python {platform.python_version()}")
    try:
        import anthropic
        status(True, f"anthropic SDK {anthropic.__version__}")
    except ImportError:
        status(False, "anthropic SDK not installed (only needed for the AI path: pip install anthropic)")
    key_ok = check_key()
    model = configured_model()
    status(None, f"Model: {model}" + (" (set by FEEDLYTICS_MODEL)" if os.getenv("FEEDLYTICS_MODEL") else " (default)"))
    df, profile, truth_ok = check_truth(args.truth, args.data)
    estimate(df, profile, args.truth, model)

    smoke_ok = None
    if args.smoke:
        smoke_ok = smoke_test(model, args.yes) if key_ok else False
        if not key_ok:
            status(False, "Smoke test skipped: no usable API key.")

    print()
    if key_ok and truth_ok:
        print("Ready. Run the evaluation with:")
        print(f"  .venv/bin/python -m scripts.evaluate_labels --data {args.data} --truth {args.truth} --ai")
    else:
        print("Not ready yet: fix the items marked FIX above, then run this check again.")
    return 0 if key_ok and truth_ok and smoke_ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
