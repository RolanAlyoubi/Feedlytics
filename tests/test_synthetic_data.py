"""End-to-end checks: the generator is reproducible, messy, and cleanable."""

import pandas as pd
import pytest

from app import analytics
from app.config import SENTIMENT_LABELS
from app.profiles import load_profile
from app.data_processing import process_reviews
from scripts.generate_synthetic_data import (
    PUBLIC_COLUMNS,
    add_data_quality_problems,
    generate_clean,
    main,
)


@pytest.fixture(scope="module")
def clean():
    return generate_clean(seed=123)


@pytest.fixture(scope="module")
def messy(clean):
    return add_data_quality_problems(clean[PUBLIC_COLUMNS], seed=123)


def test_generator_is_reproducible(clean):
    again = generate_clean(seed=123)
    pd.testing.assert_frame_equal(clean, again)


def test_clean_data_is_well_formed(clean):
    assert 2500 < len(clean) < 4000
    assert clean["review_id"].is_unique
    assert clean["rating"].between(1, 5).all()
    assert set(clean["_true_sentiment"]) <= set(SENTIMENT_LABELS)
    issues = set(clean["_true_issue"]) - {""}
    assert issues <= set(load_profile("food_delivery_demo").taxonomy.issue_to_topic)


def test_messy_data_contains_problems(messy):
    assert messy.duplicated().sum() > 0
    assert (messy["review_text"].isin(["", "N/A", "   "])).sum() > 0
    assert messy["rating"].astype(str).isin(["6", "0", "10", "-1", "abc", ""]).sum() > 0
    assert messy["date"].astype(str).str.contains(",").sum() > 0  # "March 5, 2025" style


def test_cleaning_recovers_original_reviews(clean, messy):
    df, report = process_reviews(messy, reference_date="2026-10-08")
    # Injected duplicates are removed; at most a handful of near-identical rows differ.
    assert abs(len(df) - len(clean)) <= 5
    assert report.rows_received - report.rows_removed == len(df)
    assert df["rating"].dropna().between(1, 5).all()
    assert df["restaurant"].nunique() == clean["restaurant"].nunique()


def test_built_in_delivery_pattern_is_detectable(messy):
    df, _ = process_reviews(messy, reference_date="2026-10-08")
    corr = analytics.correlation_with_rating(df, "delivery_time")
    assert corr["direction"] == "negative"
    spec = next(n for n in load_profile("food_delivery_demo").numeric if n.column == "delivery_time")
    bands = analytics.band_summary(df, spec).set_index("band")
    assert bands.loc["61–90 min", "avg_rating"] < bands.loc["≤30 min", "avg_rating"]


def test_cli_writes_both_files(tmp_path):
    main(["--seed", "1", "--out-dir", str(tmp_path)])
    reviews = pd.read_csv(tmp_path / "sample_reviews_SYNTHETIC.csv", dtype=str)
    truth = pd.read_csv(tmp_path / "sample_reviews_SYNTHETIC_ground_truth.csv", dtype=str)
    assert list(reviews.columns) == PUBLIC_COLUMNS
    assert {"true_sentiment", "true_topic", "true_issue"} <= set(truth.columns)
