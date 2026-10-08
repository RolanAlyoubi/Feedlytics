"""Measure how accurate a labeller is against ground-truth labels.

Ground truth comes from the synthetic generator, or from a hand-labelled CSV
for real data (see evaluation/README.md). Metrics are implemented with pandas
so the calculations are visible and need no extra dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import pandas as pd

from app import config


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% confidence interval for a proportion (Wilson score method).

    Better than the simple ±1.96·SE interval for small samples or values near 0/100%.
    Returned in percent.
    """
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def classification_metrics(y_true: pd.Series, y_pred: pd.Series,
                           labels: Optional[Sequence[str]] = None) -> Dict:
    """Accuracy (with 95% CI), macro-F1, per-class scores and a confusion matrix."""
    data = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna(subset=["true"])
    data["pred"] = data["pred"].fillna("(none)")
    n = len(data)
    labels = list(labels) if labels is not None else sorted(set(data["true"]) | set(data["pred"]))
    correct = int((data["true"] == data["pred"]).sum())
    per_class = []
    for label in labels:
        tp = int(((data["true"] == label) & (data["pred"] == label)).sum())
        fp = int(((data["true"] != label) & (data["pred"] == label)).sum())
        fn = int(((data["true"] == label) & (data["pred"] != label)).sum())
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "precision": round(precision, 3),
                          "recall": round(recall, 3), "f1": round(f1, 3), "support": support})
    present = [c for c in per_class if c["support"] > 0]
    macro_f1 = sum(c["f1"] for c in present) / len(present) if present else 0.0
    confusion = pd.crosstab(data["true"], data["pred"]).reindex(
        index=[l for l in labels if l in set(data["true"])], fill_value=0)
    return {
        "n": n,
        "accuracy": round(100 * correct / n, 1) if n else None,
        "accuracy_ci95": wilson_interval(correct, n),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "confusion": confusion,
    }


def _negative_issues(aspects) -> List[str]:
    return [a["issue"] for a in aspects or [] if a.get("polarity") == "negative"]


def evaluate_labels(labelled: pd.DataFrame, truth: pd.DataFrame,
                    topics: Optional[Sequence[str]] = None) -> Dict:
    """Compare predicted labels with ground truth, joined on review_id.

    ``truth`` columns: review_id, true_sentiment, true_topic, true_issue
    (empty true_issue = the review contains no complaint), and optionally
    true_<flag> for each outcome flag (values true/false/1/0/yes/no).
    ``topics``: the profile's topic list (fixes the order of per-class scores).
    """
    merged = labelled.merge(truth, on="review_id", how="inner")
    merged = merged[merged["has_text"] & merged["sentiment_ai"].notna()]
    issue_true = merged["true_issue"].fillna("").astype(str)
    has_issue = issue_true != ""
    predicted_sets = merged["aspects_ai"].map(_negative_issues)
    predicted_primary = merged["issue_ai"].fillna("")

    found = sum(t in p for t, p in zip(issue_true[has_issue], predicted_sets[has_issue]))
    exact = int((predicted_primary[has_issue] == issue_true[has_issue]).sum())
    no_issue = ~has_issue
    false_alarms = int(predicted_sets[no_issue].map(bool).sum())

    return {
        "n_reviews": int(len(merged)),
        "sentiment": classification_metrics(merged["true_sentiment"], merged["sentiment_ai"],
                                            config.SENTIMENT_LABELS),
        "primary_topic": classification_metrics(merged["true_topic"].replace("", None),
                                                merged["primary_topic_ai"], topics),
        "issue": {
            "reviews_with_complaint": int(has_issue.sum()),
            "true_issue_found_pct": round(100 * found / has_issue.sum(), 1) if has_issue.any() else None,
            "true_issue_found_ci95": wilson_interval(found, int(has_issue.sum())),
            "primary_issue_exact_pct": round(100 * exact / has_issue.sum(), 1) if has_issue.any() else None,
            "reviews_without_complaint": int(no_issue.sum()),
            "false_complaint_pct": round(100 * false_alarms / no_issue.sum(), 1) if no_issue.any() else None,
        },
        "flags": {col[len("true_"):]: flag_metrics(merged[col], merged.get(f"flag_{col[len('true_'):]}"))
                  for col in truth.columns
                  if col.startswith("true_") and col not in ("true_sentiment", "true_topic",
                                                             "true_issue", "true_aspects")
                  and f"flag_{col[len('true_'):]}" in merged.columns},
    }


_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in _TRUE_VALUES else False if text in _FALSE_VALUES else None


def flag_metrics(truth: pd.Series, predicted: Optional[pd.Series]) -> Dict:
    """Accuracy, precision and recall for a yes/no outcome flag."""
    data = pd.DataFrame({"t": truth.map(_to_bool), "p": predicted.map(_to_bool)}).dropna()
    t, p = data["t"].astype(bool), data["p"].astype(bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    n = len(data)
    return {
        "n": n,
        "accuracy": round(100 * int((t == p).sum()) / n, 1) if n else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "true_rate_pct": round(100 * t.mean(), 1) if n else None,
    }


def rating_outcome_agreement(df: pd.DataFrame, outcome_col: str,
                             negative_max: float = config.NEGATIVE_MAX_RATING,
                             positive_min: float = config.POSITIVE_MIN_RATING) -> Optional[Dict]:
    """How often a yes/no outcome (e.g. Recommended) agrees with the rating.

    A sanity check that needs no hand labels: positive ratings should mostly
    come with "yes", negative ratings with "no". Neutral ratings are excluded.
    """
    if outcome_col not in df.columns or "rating" not in df.columns:
        return None
    data = df[["rating", outcome_col]].dropna()
    pos, neg = data["rating"] >= positive_min, data["rating"] <= negative_max
    considered = pos | neg
    if not considered.any():
        return None
    agree = (pos & (data[outcome_col] == 1)) | (neg & (data[outcome_col] == 0))
    table = pd.crosstab(data["rating"], data[outcome_col])
    return {
        "n": int(considered.sum()),
        "agreement_pct": round(100 * agree[considered].mean(), 1),
        "agreement_ci95": wilson_interval(int(agree[considered].sum()), int(considered.sum())),
        "positive_but_no": int((pos & (data[outcome_col] == 0)).sum()),
        "negative_but_yes": int((neg & (data[outcome_col] == 1)).sum()),
        "table": table,
    }


def summary_row(name: str, result: Dict) -> Dict:
    """One line per method for the comparison table."""
    issue = result.get("issue") or {}
    return {
        "method": name,
        "n": result["n_reviews"],
        "sentiment_accuracy_%": result["sentiment"]["accuracy"],
        "sentiment_95%_CI": result["sentiment"]["accuracy_ci95"],
        "sentiment_macro_F1": result["sentiment"]["macro_f1"],
        "topic_accuracy_%": result["primary_topic"]["accuracy"] if result.get("primary_topic") else None,
        "issue_found_%": issue.get("true_issue_found_pct"),
        "primary_issue_exact_%": issue.get("primary_issue_exact_pct"),
        "false_complaint_%": issue.get("false_complaint_pct"),
    }


TRUTH_COLUMNS = ("review_id", "true_sentiment", "true_topic", "true_issue")


def validate_truth(truth: pd.DataFrame, topics: Dict[str, Sequence[str]],
                   flags: Sequence[str] = (), known_ids: Optional[Sequence[str]] = None) -> Dict:
    """Check a hand-labelled file before it is used (and before any API spend).

    Invalid values would otherwise be silently scored as model errors.
    ``topics``: profile taxonomy {topic: [issues]}. Returns counts, errors
    (must fix) and warnings (worth a look), each naming the review_id.
    """
    errors: List[str] = []
    warnings: List[str] = []
    missing = [c for c in TRUTH_COLUMNS if c not in truth.columns]
    if missing:
        return {"rows": len(truth), "labelled": 0, "warnings": [],
                "errors": [f"missing column(s): {', '.join(missing)} — columns found: "
                           f"{', '.join(map(str, truth.columns))}"]}

    issue_to_topic = {issue: topic for topic, issues in topics.items() for issue in issues}
    sentiments = list(config.SENTIMENT_LABELS)
    data = truth.fillna("").astype(str).apply(lambda col: col.str.strip())
    labelled = data["true_sentiment"] != ""

    def check(value: str, allowed: Sequence[str], column: str, rid: str) -> None:
        if value in allowed:
            return
        near = [a for a in allowed if a.lower() == value.lower()]
        hint = f" (did you mean '{near[0]}'?)" if near else ""
        errors.append(f"review {rid}: {column} '{value}' is not an allowed value{hint}")

    for _, row in data[labelled].iterrows():
        rid = row["review_id"]
        check(row["true_sentiment"], sentiments, "true_sentiment", rid)
        if row["true_topic"]:
            check(row["true_topic"], list(topics), "true_topic", rid)
        else:
            errors.append(f"review {rid}: true_topic is empty")
        issue = row["true_issue"]
        if issue:
            check(issue, list(issue_to_topic), "true_issue", rid)
            if issue in issue_to_topic and row["true_topic"] in topics \
                    and issue_to_topic[issue] != row["true_topic"]:
                warnings.append(f"review {rid}: issue '{issue}' belongs to topic "
                                f"'{issue_to_topic[issue]}', not '{row['true_topic']}'")
        elif row["true_sentiment"] == "Negative":
            warnings.append(f"review {rid}: Negative but no true_issue (is there really no complaint?)")
        for flag in flags:
            col = f"true_{flag}"
            if col in data.columns and row[col] and row[col].lower() not in _TRUE_VALUES | _FALSE_VALUES:
                errors.append(f"review {rid}: {col} '{row[col]}' must be true or false")

    partial = ~labelled & (data[["true_topic", "true_issue"]] != "").any(axis=1)
    for rid in data.loc[partial, "review_id"]:
        warnings.append(f"review {rid}: has a topic/issue but no true_sentiment, so it will be skipped")
    dupes = data.loc[data["review_id"].duplicated(), "review_id"].unique()
    errors += [f"review {rid}: appears more than once" for rid in dupes]
    if known_ids is not None:
        # Not an error: cleaning can legitimately remove rows (duplicates, empty rows).
        unknown = sorted(set(data.loc[labelled, "review_id"]) - set(map(str, known_ids)))
        warnings += [f"review {rid}: not in the cleaned dataset (removed during cleaning, "
                     f"or a typo) — it will be skipped" for rid in unknown[:20]]
    return {"rows": len(truth), "labelled": int(labelled.sum()), "errors": errors, "warnings": warnings}
