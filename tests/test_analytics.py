"""Tests for descriptive analytics, using small hand-checkable datasets."""

import numpy as np
import pandas as pd
import pytest

from app import analytics
from app.data_processing import add_derived_columns
from app.profiles import load_profile


def make_df(rows, **extra_cols):
    """rows: list of (rating, date, restaurant). Builds a cleaned-style frame."""
    df = pd.DataFrame(rows, columns=["rating", "date", "restaurant"])
    df["rating"] = df["rating"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["review_text"] = "text"
    df["has_text"] = True
    for name, values in extra_cols.items():
        df[name] = values
    return add_derived_columns(df)


@pytest.fixture
def small_df():
    return make_df([
        (5, "2025-01-10", "A"), (4, "2025-01-12", "A"), (1, "2025-01-15", "B"),
        (2, "2025-03-01", "B"), (3, "2025-03-02", "A"), (np.nan, "2025-03-05", "B"),
    ])


def test_overview_kpis(small_df):
    k = analytics.overview_kpis(small_df, dimensions=["restaurant"])
    assert k["total_reviews"] == 6
    assert k["rated_reviews"] == 5
    assert k["avg_rating"] == 3.0
    assert (k["pct_positive"], k["pct_neutral"], k["pct_negative"]) == (40.0, 20.0, 40.0)
    assert str(k["date_min"]) == "2025-01-10" and k["n_restaurant"] == 2


def test_overview_kpis_on_empty_frame(small_df):
    k = analytics.overview_kpis(small_df.iloc[0:0])
    assert k["total_reviews"] == 0 and k["avg_rating"] is None and k["pct_negative"] is None


def test_rating_distribution_includes_zero_counts(small_df):
    dist = analytics.rating_distribution(small_df)
    assert dist["rating"].tolist() == [1, 2, 3, 4, 5]
    assert dist["reviews"].tolist() == [1, 1, 1, 1, 1]
    assert dist["pct"].sum() == pytest.approx(100)


def test_sentiment_distribution_fixed_order(small_df):
    dist = analytics.sentiment_distribution(small_df)
    assert dist["sentiment"].tolist() == ["Positive", "Neutral", "Negative"]
    assert dist["reviews"].tolist() == [2, 1, 2]


def test_reviews_over_time_fills_empty_months(small_df):
    monthly = analytics.reviews_over_time(small_df, min_count=2)
    assert monthly["month"].dt.month.tolist() == [1, 2, 3]
    assert monthly["reviews"].tolist() == [3, 0, 3]
    assert monthly.loc[0, "pct_negative"] == pytest.approx(33.3)
    assert pd.isna(monthly.loc[1, "pct_negative"])
    assert monthly["reliable"].tolist() == [True, False, True]


def test_reviews_over_time_without_dates():
    df = make_df([(5, None, "A")]).drop(columns=["date", "month"])
    assert analytics.reviews_over_time(df).empty


def test_sentiment_over_time_shares_sum_to_100(small_df):
    out = analytics.sentiment_over_time(small_df)
    sums = out.groupby("month")["pct"].sum()
    assert np.allclose(sums, 100, atol=0.2)


def test_group_summary(small_df):
    out = analytics.group_summary(small_df, "restaurant", min_count=3).set_index("group")
    assert out.loc["A", "reviews"] == 3
    assert out.loc["A", "avg_rating"] == 4.0
    assert out.loc["B", "avg_rating"] == 1.5
    assert out.loc["B", "pct_negative"] == 100.0  # 2 of 2 rated reviews
    assert out.loc["A", "reliable"] and out.loc["B", "reliable"]


def test_group_summary_missing_column(small_df):
    assert analytics.group_summary(small_df, "category").empty


def test_delivery_time_bands_and_correlation():
    rows = [(5, "2025-01-01", "A")] * 20 + [(1, "2025-01-01", "A")] * 20
    times = [20] * 20 + [80] * 20
    df = make_df(rows, delivery_time=times)
    spec = next(n for n in load_profile("food_delivery_demo").numeric if n.column == "delivery_time")
    bands = analytics.band_summary(df, spec).set_index("band")
    assert bands.loc["≤30 min", "avg_rating"] == 5.0
    assert bands.loc["61–90 min", "pct_low_rating"] == 100.0
    assert bands.loc["46–60 min", "reviews"] == 0
    corr = analytics.correlation_with_rating(df, "delivery_time")
    assert corr["rho"] == pytest.approx(-1.0)
    assert corr["strength"] == "strong" and corr["direction"] == "negative"


def test_correlation_needs_enough_data(small_df):
    df = small_df.assign(delivery_time=[10, 20, 30, 40, 50, 60])
    assert analytics.correlation_with_rating(df, "delivery_time") is None  # n < 15


def test_numeric_band_summary_quantiles():
    rows = [(r, "2025-01-01", "A") for r in [1, 2, 3, 4, 5] * 8]
    df = make_df(rows, order_value=list(range(10, 50)))
    out = analytics.numeric_band_summary(df, "order_value", quantiles=4)
    assert len(out) == 4 and out["reviews"].sum() == 40


def test_low_rating_drivers_lift():
    rows = [(1, "2025-01-01", "Bad")] * 20 + [(5, "2025-01-01", "Good")] * 20
    out = analytics.low_rating_drivers(make_df(rows), dimensions=("restaurant",))
    top = out.iloc[0]
    assert top["group"] == "Bad" and top["lift"] == 2.0  # 100% vs 50% overall


def test_recent_change_detects_real_increase_only():
    rows = []
    # Restaurant A: 10% negative before, 60% negative recently (clear change).
    # Restaurant B: 30% -> 33% (small change, not significant).
    for month, neg_a, neg_b in [("2025-01", 4, 12), ("2025-02", 4, 12), ("2025-03", 4, 12),
                                ("2025-04", 24, 13), ("2025-05", 24, 13), ("2025-06", 24, 13)]:
        rows += [(1, f"{month}-10", "A")] * neg_a + [(5, f"{month}-10", "A")] * (40 - neg_a)
        rows += [(1, f"{month}-10", "B")] * neg_b + [(5, f"{month}-10", "B")] * (40 - neg_b)
    out = analytics.recent_change(make_df(rows), "restaurant", months=3).set_index("group")
    assert out.loc["A", "direction"] == "Increasing"
    assert out.loc["A", "change_pp"] == pytest.approx(50.0)
    assert out.loc["B", "direction"] == "No clear change"


def test_recent_change_small_groups_not_reported():
    rows = [(1, "2025-01-10", "A"), (5, "2025-04-10", "A")]
    out = analytics.recent_change(make_df(rows), "restaurant", months=3)
    assert out.loc[0, "direction"] == "Not enough data"


def test_two_proportion_p_value():
    assert analytics.two_proportion_p_value(10, 1000, 10, 1000) == 1.0
    assert analytics.two_proportion_p_value(10, 1000, 20, 1000) < 0.001
    assert analytics.two_proportion_p_value(10, 0, 20, 100) is None


def test_priority_table_ranks_frequent_severe_issues_first():
    rows, issues = [], []
    rows += [(1, "2025-01-01", "x")] * 40; issues += ["Late delivery"] * 40      # frequent, severe
    rows += [(2, "2025-01-01", "x")] * 20; issues += ["Cold food"] * 20          # medium
    rows += [(3, "2025-01-01", "x")] * 20; issues += ["Promo not applied"] * 20  # mild
    rows += [(1, "2025-01-01", "x")] * 5;  issues += ["Rare issue"] * 5          # too few
    df = make_df(rows, issue=issues)
    out = analytics.priority_table(df, "issue", total_reviews=200)
    assert out["group"].tolist()[0] == "Late delivery"
    levels = dict(zip(out["group"], out["priority"]))
    assert levels["Late delivery"] == "High"
    assert levels["Promo not applied"] == "Low"  # neutral 3★ -> severity 0
    assert levels["Rare issue"] == "Insufficient data"
    assert out.loc[out["group"] == "Late delivery", "frequency_pct"].item() == 20.0
    assert out["group"].tolist()[-1] == "Rare issue"


def test_priority_table_empty_input(small_df):
    assert analytics.priority_table(small_df, "issue").empty
