"""Measure how accurate a labeller is against ground-truth labels.

Ground truth comes from the synthetic generator, or from a hand-labelled CSV
for real data (see evaluation/README.md). Metrics are implemented with pandas
so the calculations are visible and need no extra dependencies.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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


def _positive_topics(aspects) -> List[str]:
    return [a["topic"] for a in aspects or [] if a.get("polarity") == "positive"]


# ---------------------------------------------------------------------------
# Ground truth: one normalised form for hand labels and synthetic labels
# ---------------------------------------------------------------------------

LIST_SEPARATOR = ";"


def parse_label_list(cell) -> List[str]:
    """'Runs small; Thin or see-through' -> ['Runs small', 'Thin or see-through'].

    Order is kept (the first item is the main complaint); blanks and repeats are dropped.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    items = [part.strip() for part in str(cell).split(LIST_SEPARATOR)]
    return list(dict.fromkeys(i for i in items if i))


def _parse_aspect_string(cell) -> Tuple[List[str], List[str]]:
    """Synthetic ground truth: 'Topic:negative:Issue|Topic:positive:' -> (issues, praise topics)."""
    issues, praise = [], []
    for part in str(cell or "").split("|"):
        bits = part.split(":")
        if len(bits) >= 2 and bits[1] == "negative" and len(bits) >= 3 and bits[2]:
            issues.append(bits[2])
        elif len(bits) >= 2 and bits[1] == "positive":
            praise.append(bits[0])
    return list(dict.fromkeys(issues)), list(dict.fromkeys(praise))


def prepare_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """Add normalised columns to a ground-truth table, whatever its schema:

    - hand labels:  true_issues ("A; B", main complaint first) and true_praise_topics
    - legacy:       true_issue (single main complaint)
    - synthetic:    true_aspects ("Topic:negative:Issue|Topic:positive:")

    Adds ``_issues`` (list), ``_main_issue`` (str, "" if no complaint) and
    ``_praise`` (list, or None when the schema has no praise information).
    """
    out = truth.copy()
    if "true_issues" in out.columns:
        out["_issues"] = out["true_issues"].map(parse_label_list)
    elif "true_aspects" in out.columns:
        out["_issues"] = out["true_aspects"].map(lambda c: _parse_aspect_string(c)[0])
    elif "true_issue" in out.columns:
        out["_issues"] = out["true_issue"].map(lambda c: parse_label_list(c)[:1])
    else:
        out["_issues"] = [[] for _ in range(len(out))]
    if "true_issue" in out.columns and "true_issues" not in out.columns:
        out["_main_issue"] = out["true_issue"].fillna("").astype(str).str.strip()
    else:
        out["_main_issue"] = out["_issues"].map(lambda items: items[0] if items else "")
    if "true_praise_topics" in out.columns:
        out["_praise"] = out["true_praise_topics"].map(parse_label_list)
    elif "true_aspects" in out.columns:
        out["_praise"] = out["true_aspects"].map(lambda c: _parse_aspect_string(c)[1])
    else:
        out["_praise"] = None
    return out


# ---------------------------------------------------------------------------
# Metrics for yes/no and multi-label questions
# ---------------------------------------------------------------------------

def binary_metrics(truth: pd.Series, pred: pd.Series) -> Dict:
    """Accuracy (with 95% CI), precision, recall and F1 for a yes/no question."""
    t, p = truth.astype(bool), pred.astype(bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    n = len(t)
    correct = int((t == p).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))
    return {
        "n": n,
        "accuracy": round(100 * correct / n, 1) if n else None,
        "accuracy_ci95": wilson_interval(correct, n),
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "recall_ci95": wilson_interval(tp, tp + fn),
        "f1": round(f1, 3) if f1 is not None else None,
        "true_positives_in_truth": int(t.sum()),
        "predicted_positives": int(p.sum()),
    }


def multilabel_metrics(true_sets: Sequence[Iterable[str]], pred_sets: Sequence[Iterable[str]],
                       labels: Optional[Sequence[str]] = None) -> Dict:
    """Precision/recall/F1 when each review can have several labels (e.g. issues).

    micro = pooled over all (review, label) pairs; macro = mean F1 over labels
    that occur in the truth. ``exact_match_pct`` = reviews whose label set is
    exactly right (including correctly predicting no labels).
    """
    true_sets = [set(t) for t in true_sets]
    pred_sets = [set(p) for p in pred_sets]
    all_labels = list(labels) if labels is not None else sorted(set().union(*true_sets, *pred_sets))
    per_label = []
    tp_all = fp_all = fn_all = 0
    for label in all_labels:
        tp = sum(label in t and label in p for t, p in zip(true_sets, pred_sets))
        fp = sum(label not in t and label in p for t, p in zip(true_sets, pred_sets))
        fn = sum(label in t and label not in p for t, p in zip(true_sets, pred_sets))
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if tp + fp + fn:
            per_label.append({"label": label, "support": tp + fn, "predicted": tp + fp, "tp": tp,
                              "precision": round(precision, 3), "recall": round(recall, 3),
                              "f1": round(f1, 3)})
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    supported = [row["f1"] for row in per_label if row["support"] > 0]
    exact = sum(t == p for t, p in zip(true_sets, pred_sets))
    n = len(true_sets)
    return {
        "n_reviews": n,
        "true_labels": tp_all + fn_all,
        "predicted_labels": tp_all + fp_all,
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_f1": round(sum(supported) / len(supported), 3) if supported else None,
        "exact_match_pct": round(100 * exact / n, 1) if n else None,
        "per_label": sorted(per_label, key=lambda r: (-r["support"], r["label"])),
    }


def weighted_accuracy(correct: pd.Series, weights: pd.Series) -> Optional[float]:
    """Accuracy re-weighted to the population (e.g. undo over-sampling of low ratings)."""
    w = weights.reindex(correct.index).fillna(0.0)
    return round(100 * float((w * correct.astype(float)).sum() / w.sum()), 1) if w.sum() else None


def population_weights(eval_strata: pd.Series, population_strata: pd.Series) -> pd.Series:
    """weight = (stratum size in the dataset) / (stratum size among evaluated reviews)."""
    pop = population_strata.value_counts()
    sample = eval_strata.value_counts()
    return eval_strata.map(lambda s: pop.get(s, 0) / sample[s])


# ---------------------------------------------------------------------------
# Evaluate a labeller against ground truth
# ---------------------------------------------------------------------------

def evaluate_labels(labelled: pd.DataFrame, truth: pd.DataFrame,
                    topics: Optional[Sequence[str]] = None,
                    issue_to_topic: Optional[Dict[str, str]] = None,
                    weights: Optional[pd.Series] = None) -> Dict:
    """Compare a labeller's TEXT labels with ground truth, joined on review_id.

    Only text-only predictions are scored (``sentiment_ai``, ``aspects_ai``,
    ``primary_topic_ai``, ``flag_*``). The primary ``sentiment`` column, which
    uses the rating when available, is deliberately ignored, so ratings never
    leak into the score.

    ``truth``: see ``prepare_truth`` for the accepted schemas.
    ``issue_to_topic``: profile mapping; enables complaint-topic metrics.
    ``weights``: optional per-review population weights (index = review_id)
    for population-weighted accuracies.
    """
    prepared = prepare_truth(truth)
    merged = labelled.merge(prepared, on="review_id", how="inner")
    merged = merged[merged["has_text"] & merged["sentiment_ai"].notna()].reset_index(drop=True)
    true_issues = merged["_issues"]
    pred_issues = merged["aspects_ai"].map(_negative_issues)
    issue_true = merged["_main_issue"]
    has_issue = issue_true != ""
    predicted_primary = merged["issue_ai"].fillna("")

    found = sum(t in p for t, p in zip(issue_true[has_issue], pred_issues[has_issue]))
    exact = int((predicted_primary[has_issue] == issue_true[has_issue]).sum())
    no_issue = ~has_issue
    false_alarms = int(pred_issues[no_issue].map(bool).sum())

    result = {
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
        "complaint_detection": binary_metrics(true_issues.map(bool), pred_issues.map(bool)),
        "issues": multilabel_metrics(true_issues, pred_issues),
        "flags": {col[len("true_"):]: flag_metrics(merged[col], merged.get(f"flag_{col[len('true_'):]}"))
                  for col in truth.columns
                  if col.startswith("true_") and col not in _TRUTH_LABEL_COLUMNS
                  and f"flag_{col[len('true_'):]}" in merged.columns},
    }
    if issue_to_topic is not None:
        def topic_of(issue: str) -> str:
            return issue_to_topic.get(issue, "Other")
        result["complaint_topics"] = multilabel_metrics(
            true_issues.map(lambda items: {topic_of(i) for i in items}),
            pred_issues.map(lambda items: {topic_of(i) for i in items}))
    if merged["_praise"].notna().all():
        result["praise_topics"] = multilabel_metrics(merged["_praise"],
                                                     merged["aspects_ai"].map(_positive_topics))
    if weights is not None:
        w = merged["review_id"].map(weights)
        result["population_weighted"] = {
            "sentiment_accuracy": weighted_accuracy(merged["true_sentiment"] == merged["sentiment_ai"], w),
            "topic_accuracy": weighted_accuracy(merged["true_topic"] == merged["primary_topic_ai"], w),
            "complaint_detection_accuracy": weighted_accuracy(true_issues.map(bool) == pred_issues.map(bool), w),
        }
    return result


_TRUTH_LABEL_COLUMNS = ("true_sentiment", "true_topic", "true_issue", "true_issues",
                        "true_praise_topics", "true_aspects")


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
        "complaint_detection_F1": (result.get("complaint_detection") or {}).get("f1"),
        "issue_micro_F1": (result.get("issues") or {}).get("micro_f1"),
        "praise_micro_F1": (result.get("praise_topics") or {}).get("micro_f1"),
    }


TRUTH_COLUMNS = ("review_id", "true_sentiment", "true_topic")
ISSUE_COLUMNS = ("true_issues", "true_issue")  # hand-label schema, legacy single-issue schema


def validate_truth(truth: pd.DataFrame, topics: Dict[str, Sequence[str]],
                   flags: Sequence[str] = (), known_ids: Optional[Sequence[str]] = None) -> Dict:
    """Check a hand-labelled file before it is used (and before any API spend).

    Invalid values would otherwise be silently scored as model errors.
    ``topics``: profile taxonomy {topic: [issues]}. Accepts the hand-label
    schema (true_issues / true_praise_topics, ';'-separated) and the legacy
    single true_issue column. Returns counts, errors (must fix) and warnings
    (worth a look), each naming the review_id.
    """
    errors: List[str] = []
    warnings: List[str] = []
    missing = [c for c in TRUTH_COLUMNS if c not in truth.columns]
    if not any(c in truth.columns for c in ISSUE_COLUMNS):
        missing.append(" or ".join(ISSUE_COLUMNS))
    if missing:
        return {"rows": len(truth), "labelled": 0, "warnings": [],
                "errors": [f"missing column(s): {', '.join(missing)} — columns found: "
                           f"{', '.join(map(str, truth.columns))}"]}

    issue_col = "true_issues" if "true_issues" in truth.columns else "true_issue"
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
        issues = parse_label_list(row[issue_col])
        if issue_col == "true_issue" and len(issues) > 1:
            errors.append(f"review {rid}: true_issue holds one issue; use true_issues for several")
        for issue in issues:
            check(issue, list(issue_to_topic), issue_col, rid)
        main = issues[0] if issues else ""
        if main in issue_to_topic and row["true_topic"] in topics \
                and issue_to_topic[main] != row["true_topic"] and main != "Other":
            warnings.append(f"review {rid}: issue '{main}' belongs to topic "
                            f"'{issue_to_topic[main]}', not '{row['true_topic']}'")
        if not issues and row["true_sentiment"] == "Negative":
            warnings.append(f"review {rid}: Negative but no {issue_col} (is there really no complaint?)")
        if "true_praise_topics" in data.columns:
            for topic in parse_label_list(row["true_praise_topics"]):
                check(topic, list(topics), "true_praise_topics", rid)
        for flag in flags:
            col = f"true_{flag}"
            if col in data.columns and row[col] and row[col].lower() not in _TRUE_VALUES | _FALSE_VALUES:
                errors.append(f"review {rid}: {col} '{row[col]}' must be true or false")

    label_cols = [c for c in ("true_topic", issue_col, "true_praise_topics") if c in data.columns]
    partial = ~labelled & (data[label_cols] != "").any(axis=1)
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


# ---------------------------------------------------------------------------
# Tuning / held-out split and baseline sentiment calibration
# ---------------------------------------------------------------------------

def tuning_split(review_ids: pd.Series, exclude_ids: Iterable[str] = ()) -> pd.Series:
    """Fixed, reproducible split: 'tune' / 'test' / 'excluded'.

    Each review goes to 'tune' if the SHA-256 hash of its id is even, otherwise
    'test'. ``exclude_ids`` (e.g. the hand-labelled evaluation set) are removed
    first and never used for tuning or held-out testing.
    """
    excluded = set(map(str, exclude_ids))
    def assign(rid: str) -> str:
        if rid in excluded:
            return "excluded"
        return "tune" if int(hashlib.sha256(rid.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"
    return review_ids.astype(str).map(assign)


def sentiment_from_score(score: pd.Series, negative_max: float, positive_min: float) -> pd.Series:
    return pd.Series(np.where(score <= negative_max, "Negative",
                              np.where(score >= positive_min, "Positive", "Neutral")), index=score.index)


def calibration_grid(negative_max: Sequence[float], positive_min: Sequence[float],
                     penalties: Sequence[float]) -> List[Dict[str, float]]:
    """All combinations with positive_min > negative_max."""
    return [{"negative_max": round(float(n), 2), "positive_min": round(float(p), 2),
             "complaint_penalty": round(float(k), 2)}
            for n, p, k in itertools.product(negative_max, positive_min, penalties) if p > n]


def calibrate_sentiment(compound: pd.Series, n_complaints: pd.Series, target: pd.Series,
                        grid: Sequence[Dict[str, float]]) -> pd.DataFrame:
    """Score every grid point: score = compound − penalty × complaints, then cut-offs.

    Objective: macro-F1 (3 classes) against ``target``; ties broken by accuracy.
    Returns one row per grid point, best first.
    """
    rows = []
    for params in grid:
        score = compound - params["complaint_penalty"] * n_complaints
        pred = sentiment_from_score(score, params["negative_max"], params["positive_min"])
        m = classification_metrics(target, pred, config.SENTIMENT_LABELS)
        rows.append({**params, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
    return (pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
            .reset_index(drop=True))
