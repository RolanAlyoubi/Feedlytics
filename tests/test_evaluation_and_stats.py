"""Evaluation metrics, multiple-testing correction and weighted trend tests."""

import numpy as np
import pandas as pd
import pytest

from app import analytics
from app.data_processing import add_derived_columns
from app.evaluation import (classification_metrics, evaluate_labels, flag_metrics,
                            rating_outcome_agreement, wilson_interval)


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(40.4, abs=0.1) and high == pytest.approx(59.6, abs=0.1)
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(100.0) and low > 65  # never exceeds 100%


def test_classification_metrics_by_hand():
    truth = pd.Series(["Positive", "Positive", "Negative", "Negative", "Neutral"])
    pred = pd.Series(["Positive", "Negative", "Negative", "Negative", None])
    m = classification_metrics(truth, pred, ["Positive", "Neutral", "Negative"])
    assert m["n"] == 5 and m["accuracy"] == 60.0
    per = {c["label"]: c for c in m["per_class"]}
    assert per["Negative"]["precision"] == pytest.approx(0.667, abs=0.001)
    assert per["Negative"]["recall"] == 1.0
    assert per["Neutral"]["recall"] == 0.0
    assert m["confusion"].loc["Neutral", "(none)"] == 1


def test_flag_metrics():
    m = flag_metrics(pd.Series(["true", "false", "yes", "0"]), pd.Series([True, True, True, False]))
    assert m["n"] == 4 and m["accuracy"] == 75.0
    assert m["precision"] == pytest.approx(0.667, abs=0.001) and m["recall"] == 1.0


def test_evaluate_labels_issue_and_flag_metrics():
    labelled = pd.DataFrame({
        "review_id": ["1", "2", "3"],
        "has_text": True,
        "sentiment_ai": ["Negative", "Positive", "Negative"],
        "primary_topic_ai": ["Fit & Sizing", "Style & Design", "Fit & Sizing"],
        "issue_ai": ["Runs small", None, "Runs large"],
        "aspects_ai": [[{"topic": "Fit & Sizing", "polarity": "negative", "issue": "Runs small"}],
                       [],
                       [{"topic": "Fit & Sizing", "polarity": "negative", "issue": "Runs large"},
                        {"topic": "Fabric & Comfort", "polarity": "negative", "issue": "Thin or see-through"}]],
        "flag_mentions_return": [True, False, False],
    })
    truth = pd.DataFrame({
        "review_id": ["1", "2", "3"],
        "true_sentiment": ["Negative", "Positive", "Negative"],
        "true_topic": ["Fit & Sizing", "Style & Design", "Fabric & Comfort"],
        "true_issue": ["Runs small", "", "Thin or see-through"],
        "true_mentions_return": ["true", "false", "true"],
    })
    result = evaluate_labels(labelled, truth)
    assert result["sentiment"]["accuracy"] == 100.0
    assert result["primary_topic"]["accuracy"] == pytest.approx(66.7)
    assert result["issue"]["true_issue_found_pct"] == 100.0      # both true issues were found
    assert result["issue"]["primary_issue_exact_pct"] == 50.0    # but only one as the main issue
    assert result["issue"]["false_complaint_pct"] == 0.0
    assert result["flags"]["mentions_return"]["recall"] == 0.5


def test_rating_outcome_agreement():
    df = pd.DataFrame({"rating": [5, 5, 4, 1, 2, 3], "recommended": [1, 1, 0, 0, 1, 1]})
    out = rating_outcome_agreement(df, "recommended")
    assert out["n"] == 5  # the 3-star review is excluded
    assert out["agreement_pct"] == 60.0
    assert out["positive_but_no"] == 1 and out["negative_but_yes"] == 1
    assert rating_outcome_agreement(df, "missing") is None


def test_benjamini_hochberg_matches_reference():
    p = pd.Series([0.01, 0.04, 0.03, 0.005])
    q = analytics.benjamini_hochberg(p)
    # Reference values computed by hand: sorted p * m / rank, then cumulative minimum from the top.
    assert q.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert analytics.benjamini_hochberg(pd.Series([], dtype=float)).empty


def test_low_rating_drivers_flag_chance_lifts():
    rows = ([(1, "big_bad")] * 60 + [(5, "big_bad")] * 40      # 60% low, n=100: real
            + [(1, "tiny")] * 2 + [(5, "tiny")] * 13            # 13% low, n=15: noise
            + [(5, "rest")] * 900 + [(1, "rest")] * 100)
    df = pd.DataFrame(rows, columns=["rating", "shop"]).assign(review_text="t", has_text=True)
    df = add_derived_columns(df)
    out = analytics.low_rating_drivers(df, ["shop"]).set_index("group")
    assert out.loc["big_bad", "significant"]
    assert not out.loc["tiny", "significant"]
    assert {"p_value", "q_value"} <= set(out.columns)


def test_recent_change_with_weights_uses_effective_sample_size():
    rows = []
    for month, neg in [("2025-01", 5), ("2025-02", 5), ("2025-03", 5),
                       ("2025-04", 15), ("2025-05", 15), ("2025-06", 15)]:
        rows += [(1, f"{month}-10", "A")] * neg + [(5, f"{month}-10", "A")] * (50 - neg)
    df = pd.DataFrame(rows, columns=["rating", "date", "shop"])
    df["date"] = pd.to_datetime(df["date"])
    df = add_derived_columns(df.assign(review_text="t", has_text=True))
    plain = analytics.recent_change(df, "shop").iloc[0]
    # Unequal weights shrink the effective sample size, so the same change is less certain.
    rng = np.random.RandomState(0)
    weighted_df = df.assign(sample_weight=rng.choice([1.0, 20.0], size=len(df)))
    weighted = analytics.recent_change(weighted_df, "shop", weight_col="sample_weight").iloc[0]
    assert plain["direction"] == "Increasing"
    assert weighted["p_value"] > plain["p_value"]
