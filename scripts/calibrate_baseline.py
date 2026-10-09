"""Calibrate the non-AI baseline's sentiment cut-offs for one profile (no API, no AI).

The baseline scores each review with VADER and detects complaints with the
profile's keywords. This script searches for cut-offs (and a per-complaint
penalty) that make the baseline's 3-class sentiment agree best with the
rating-based sentiment:

    score = VADER compound − penalty × complaints detected
    score ≤ negative_max → Negative;  score ≥ positive_min → Positive;  else Neutral

Protocol
  - The hand-labelled evaluation reviews are excluded before anything else.
  - The rest is split by a fixed hash of review_id into 'tune' and 'test' halves.
  - The grid is searched on 'tune' only (objective: macro-F1, ties by accuracy).
  - Results are reported on 'test' only, next to the current (default) baseline.

The target is the star rating, so the outputs are *agreement with ratings*,
not accuracy. Real accuracy needs the hand labels.

This script does not change any profile; it prints the block to add.

Usage:
  python -m scripts.calibrate_baseline --data data/raw/reviews.csv \\
      --exclude evaluation/labeling/apparel_ecommerce_to_label.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app import config
from app.ai_analysis import BaselineLabeller, label_reviews
from app.data_processing import process_reviews
from app.evaluation import (calibrate_sentiment, calibration_grid, classification_metrics,
                            sentiment_from_score, tuning_split)
from app.profiles import BaselineSentiment, load_profile

NEGATIVE_MAX = [round(-0.20 + 0.10 * i, 2) for i in range(10)]   # -0.20 … 0.70
POSITIVE_MIN = [round(0.30 + 0.05 * i, 2) for i in range(14)]    #  0.30 … 0.95
PENALTIES = [0.0, 0.1, 0.2, 0.3, 0.4]
OUT_DIR = Path("evaluation")


def describe(truth: pd.Series, pred: pd.Series) -> Dict:
    m = classification_metrics(truth, pred, config.SENTIMENT_LABELS)
    positive = truth == "Positive"
    return {
        "n": m["n"],
        "accuracy": m["accuracy"],
        "accuracy_ci95": m["accuracy_ci95"],
        "macro_f1": m["macro_f1"],
        "recall": {c["label"]: c["recall"] for c in m["per_class"]},
        "precision": {c["label"]: c["precision"] for c in m["per_class"]},
        "positive_rated_as_neutral_pct": round(100 * (pred[positive] == "Neutral").mean(), 1),
        "positive_rated_as_negative_pct": round(100 * (pred[positive] == "Negative").mean(), 1),
        "negative_rated_as_positive_pct": round(100 * (pred[truth == "Negative"] == "Positive").mean(), 1),
        "confusion": m["confusion"],
    }


def md_table(rows: List[List], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the baseline sentiment rule (no API).")
    parser.add_argument("--data", type=Path, default=Path("data/raw/reviews.csv"))
    parser.add_argument("--profile", default="apparel_ecommerce")
    parser.add_argument("--exclude", type=Path, required=True,
                        help="CSV with a review_id column to exclude (the hand-label set)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not args.exclude.exists():
        print(f"Refusing to run: exclusion file {args.exclude} not found. The hand-labelled set must "
              "be excluded from tuning.", file=sys.stderr)
        return 1
    excluded_ids = set(pd.read_csv(args.exclude, dtype=str, encoding="utf-8-sig")["review_id"])

    profile = load_profile(args.profile)
    df, _ = process_reviews(args.data, profile=profile)
    df = df.assign(split=tuning_split(df["review_id"], excluded_ids))
    n_excluded_found = int((df["split"] == "excluded").sum())

    # Current (default-rule) baseline on everything except the excluded set.
    default_labeller = BaselineLabeller(profile, calibration=None)
    usable = df[df["split"] != "excluded"]
    labelled, _ = label_reviews(usable, default_labeller)
    data = labelled[labelled["label_source"].eq("baseline") & labelled["sentiment_from_rating"].notna()].copy()
    data["compound"] = data["analysis_text"].map(lambda t: default_labeller._vader.polarity_scores(t)["compound"])
    data["complaints"] = data["aspects_ai"].map(lambda a: sum(x["polarity"] == "negative" for x in a))
    assert not data["review_id"].isin(excluded_ids).any(), "excluded reviews leaked into the data"

    tune, test = data[data["split"] == "tune"], data[data["split"] == "test"]
    grid = calibration_grid(NEGATIVE_MAX, POSITIVE_MIN, PENALTIES)
    ranking = calibrate_sentiment(tune["compound"], tune["complaints"], tune["sentiment_from_rating"], grid)
    best = ranking.iloc[0]
    chosen = BaselineSentiment(float(best["negative_max"]), float(best["positive_min"]),
                               float(best["complaint_penalty"]))

    def calibrated(d: pd.DataFrame) -> pd.Series:
        return sentiment_from_score(d["compound"] - chosen.complaint_penalty * d["complaints"],
                                    chosen.negative_max, chosen.positive_min)

    # Consistency check: the real labeller with this calibration gives the same answers.
    check = test.sample(min(500, len(test)), random_state=0)
    real = BaselineLabeller(profile, calibration=chosen)
    real_pred = check["analysis_text"].map(lambda t: real.label_one(t)["sentiment"])
    mismatches = int((real_pred != calibrated(check)).sum())

    results = {
        "tune": {"current": describe(tune["sentiment_from_rating"], tune["sentiment_ai"]),
                 "calibrated": describe(tune["sentiment_from_rating"], calibrated(tune))},
        "test": {"current": describe(test["sentiment_from_rating"], test["sentiment_ai"]),
                 "calibrated": describe(test["sentiment_from_rating"], calibrated(test))},
    }
    meta = {
        "profile": profile.name, "data": args.data.name, "generated": date.today().isoformat(),
        "objective": "macro-F1 (3 classes) vs rating-based sentiment on the tune split; ties by accuracy",
        "grid": {"negative_max": NEGATIVE_MAX, "positive_min": POSITIVE_MIN, "complaint_penalty": PENALTIES,
                 "points": len(grid)},
        "split": {"tune": int(len(tune)), "test": int(len(test)), "excluded_rows": n_excluded_found,
                  "excluded_ids_in_file": len(excluded_ids)},
        "chosen": {"negative_max": chosen.negative_max, "positive_min": chosen.positive_min,
                   "complaint_penalty": chosen.complaint_penalty},
        "consistency_check": {"reviews": int(len(check)), "mismatches": mismatches},
        "top10_on_tune": ranking.head(10).to_dict("records"),
    }
    write_outputs(args.out_dir, profile.name, meta, results)
    print_summary(meta, results)
    return 0 if mismatches == 0 else 1


def jsonable(results: Dict) -> Dict:
    return {split: {name: {k: (json.loads(v.to_json()) if isinstance(v, pd.DataFrame) else v)
                           for k, v in r.items()} for name, r in parts.items()}
            for split, parts in results.items()}


def write_outputs(out_dir: Path, profile: str, meta: Dict, results: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"BASELINE_CALIBRATION_{profile}.json").write_text(
        json.dumps({"meta": meta, "results": jsonable(results)}, indent=2, default=str))
    rows = []
    for split in ("tune", "test"):
        for name in ("current", "calibrated"):
            r = results[split][name]
            rows.append([split, name, r["n"], r["accuracy"], r["accuracy_ci95"], r["macro_f1"],
                         r["recall"]["Negative"], r["recall"]["Neutral"], r["recall"]["Positive"],
                         r["positive_rated_as_neutral_pct"]])
    c = meta["chosen"]
    text = [
        f"# Baseline sentiment calibration — profile `{profile}`",
        f"_Generated {meta['generated']} by `python -m scripts.calibrate_baseline` on `{meta['data']}`._",
        "",
        "> **Non-AI baseline only.** The target is the **star rating**, so these numbers measure",
        "> *agreement with ratings*, not accuracy. The hand-labelled evaluation set "
        f"({meta['split']['excluded_rows']} reviews) was excluded before splitting.",
        "",
        f"Rule: `score = VADER compound − {c['complaint_penalty']} × complaints`; "
        f"Negative if score ≤ {c['negative_max']}, Positive if ≥ {c['positive_min']}, else Neutral.",
        f"Searched {meta['grid']['points']} combinations on the tune split ({meta['split']['tune']:,} reviews); "
        f"reported on the held-out split ({meta['split']['test']:,} reviews). Objective: {meta['objective']}.",
        "",
        md_table(rows, ["split", "rule", "n", "accuracy %", "95% CI", "macro-F1", "recall Neg",
                        "recall Neu", "recall Pos", "4–5★ labelled Neutral %"]),
        "",
        f"Consistency check: the real labeller reproduced the search's predictions on "
        f"{meta['consistency_check']['reviews']} held-out reviews with "
        f"{meta['consistency_check']['mismatches']} mismatches.",
    ]
    (out_dir / f"BASELINE_CALIBRATION_{profile}.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def print_summary(meta: Dict, results: Dict) -> None:
    print(json.dumps({k: meta[k] for k in ("split", "chosen", "consistency_check")}, indent=2))
    print("top 5 on tune:")
    print(pd.DataFrame(meta["top10_on_tune"]).head(5).to_string(index=False))
    for split in ("tune", "test"):
        for name in ("current", "calibrated"):
            r = results[split][name]
            print(f"\n[{split} / {name}] n={r['n']} accuracy={r['accuracy']}% CI={r['accuracy_ci95']} "
                  f"macro-F1={r['macro_f1']}")
            print(f"  recall {r['recall']}  precision {r['precision']}")
            print(f"  4–5★ labelled Neutral: {r['positive_rated_as_neutral_pct']}%  "
                  f"4–5★ labelled Negative: {r['positive_rated_as_negative_pct']}%  "
                  f"1–2★ labelled Positive: {r['negative_rated_as_positive_pct']}%")
            print("  confusion (rows = rating-based, cols = baseline):")
            print("  " + r["confusion"].to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    sys.exit(main())
