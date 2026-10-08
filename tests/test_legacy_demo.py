"""Guards for the legacy synthetic food-delivery demo: it must stay exactly as it was."""

import io
from pathlib import Path

import pytest

from app import analytics
from app.data_processing import process_reviews
from scripts.generate_synthetic_data import PUBLIC_COLUMNS, add_data_quality_problems, generate_clean

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_reviews_SYNTHETIC.csv"


def test_generator_reproduces_committed_file_byte_for_byte():
    clean = generate_clean(42)
    messy = add_data_quality_problems(clean[PUBLIC_COLUMNS], 42)
    buffer = io.StringIO()
    messy.to_csv(buffer, index=False)
    assert buffer.getvalue() == DATA.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def food():
    return process_reviews(DATA, reference_date="2026-10-08")


def test_food_demo_uses_its_profile(food):
    _, report = food
    assert report.profile_name == "food_delivery_demo"
    assert report.roles.entity == "restaurant"


def test_food_demo_results_unchanged_by_refactor(food):
    """Numbers recorded before the domain-neutral refactor (Phase 1 report)."""
    df, report = food
    assert (report.rows_received, report.rows_retained) == (3193, 3102)
    assert report.removed == {
        "Exact duplicate row": 62,
        "Re-submitted review (same text, rating, restaurant and date)": 29,
    }
    kpis = analytics.overview_kpis(df)
    assert kpis["avg_rating"] == 3.43
    assert (kpis["pct_positive"], kpis["pct_neutral"], kpis["pct_negative"]) == (51.0, 21.2, 27.8)
    corr = analytics.correlation_with_rating(df, "delivery_time")
    assert corr["rho"] == -0.248
