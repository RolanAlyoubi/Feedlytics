"""End-to-end pipeline: consistent sentiment sources, weights, tables and filtering.

All review texts are invented; no API calls (an in-memory fake model is used).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import analytics
from app.data_processing import process_reviews
from app.pipeline import (LABEL_COLUMNS, PipelineError, _add_weights, run_analysis)
from scripts import run_analysis as run_analysis_cli
from tests.test_ai_analysis import FakeLLM

HEADER = ",Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name,Class Name"
ROOT = Path(__file__).resolve().parent.parent


def apparel_csv(rows):
    """rows: (text, rating or '', department) -> CSV bytes in the Kaggle layout."""
    lines = [HEADER] + [f'{i},{100 + i % 7},35,,"{text}",{rating},1,0,General,{dept},Knits'
                        for i, (text, rating, dept) in enumerate(rows)]
    return "\n".join(lines).encode("utf-8")


def population(n_neg=100, n_neu=100, n_pos=800):
    """A known population: every 1★ review says 'runs small' (10% of reviews).
    Department A holds 80 of the 100 complaints."""
    rows = []
    rows += [(f"It runs small, sadly #{i}", 1, "A" if i < 80 else "B") for i in range(n_neg)]
    rows += [(f"It is okay #{i}", 3, "B") for i in range(n_neu)]
    rows += [(f"Lovely and soft #{i}", 5, "A" if i < 200 else "B") for i in range(n_pos)]
    return apparel_csv(rows)


@pytest.fixture(scope="module")
def ai_result():
    return run_analysis(population(), labelling="ai", llm=FakeLLM(), max_ai_rows=200, seed=1)


# ---------------------------------------------------------------------------
# Rated data, metadata, determinism
# ---------------------------------------------------------------------------

def test_rated_dataset_uses_rating_everywhere_and_tables_carry_metadata():
    result = run_analysis(apparel_csv([("Love it", 5, "A"), ("Bad fit, runs small", 1, "B"),
                                       ("Fine", 3, "A")]))
    assert result.profile.name == "apparel_ecommerce" and result.method.kind == "baseline"
    assert not result.method.is_ai
    assert result.kpis["sentiment_sources"] == {"rating": 3, "text_ai": 0, "text_baseline": 0, "none": 0}
    assert result.table("rating_distribution").available
    for table in result.tables.values():
        assert table.basis in {"all_reviews", "rated_reviews", "text_labelled"}
        assert isinstance(table.weighted, bool) and table.n_reviews >= 0
        assert table.available or table.reason
    assert not result.table("trend").available and "date" in result.table("trend").reason


def test_same_seed_gives_identical_results():
    a = run_analysis(population(), labelling="ai", llm=FakeLLM(), max_ai_rows=200, seed=3)
    b = run_analysis(population(), labelling="ai", llm=FakeLLM(), max_ai_rows=200, seed=3)
    assert a.data["in_label_sample"].equals(b.data["in_label_sample"])
    for name, table in a.tables.items():
        if table.available:
            pd.testing.assert_frame_equal(table.df, b.tables[name].df)


def test_dataframe_input_and_profile_detection():
    df = pd.read_csv(pd.io.common.BytesIO(population(5, 5, 5)), dtype=str)
    result = run_analysis(df)
    assert result.profile.name == "apparel_ecommerce" and len(result.data) == 15


# ---------------------------------------------------------------------------
# Without a rating; partly rated; labelling="none"
# ---------------------------------------------------------------------------

NO_RATING = b"""feedback,product
"The app crashes all the time, terrible and frustrating.",App
"Helpful and friendly support, great service.",Web
"Way too expensive, awful value.",App
"""


def test_without_rating_uses_text_labels_and_marks_rating_tables_unavailable():
    result = run_analysis(NO_RATING)
    assert result.profile.name == "generic"
    assert result.kpis["sentiment_sources"]["text_baseline"] == 3
    assert result.kpis["rating"] is None
    for name in ("rating_distribution", "low_rating_drivers"):
        assert not result.table(name).available and "rating" in result.table(name).reason
    priority = result.table("issue_priority").df
    assert not priority.empty and priority["avg_rating"].isna().all() and (priority["severity"] > 0).any()


def test_partly_rated_rows_follow_the_weight_rules():
    result = run_analysis(apparel_csv([("Love it", 5, "A"), ("It runs small", "", "A"),
                                       ("", 2, "B")]))
    d = result.data.set_index("review_id")
    assert d.loc["0", "sentiment_source"] == "rating" and d.loc["0", "sentiment_weight"] == 1.0
    assert d.loc["1", "sentiment_source"] == "text_baseline"
    assert d.loc["1", "sentiment_weight"] == d.loc["1", "sample_weight"] == 1.0
    assert d.loc["2", "sentiment_source"] == "rating" and not d.loc["2", "in_label_sample"]
    assert np.isnan(d.loc["2", "sample_weight"])


def test_labelling_none_has_no_text_labels():
    result = run_analysis(population(5, 5, 5), labelling="none")
    assert not any(col in result.data.columns for col in LABEL_COLUMNS)
    assert result.method.kind == "none"
    assert result.aspects.empty
    for name in ("topics", "issue_priority", "praise", "flags", "sentiment_text"):
        assert not result.table(name).available
    assert set(result.data["sentiment_source"].dropna()) == {"rating"}


def test_unknown_labelling_mode_is_rejected():
    with pytest.raises(PipelineError, match="labelling must be one of"):
        run_analysis(NO_RATING, labelling="magic")


# ---------------------------------------------------------------------------
# AI mode: sampling, weights, no silent fallback
# ---------------------------------------------------------------------------

def test_ai_without_credentials_fails_clearly():
    with pytest.raises(PipelineError, match="no Anthropic API access"):
        run_analysis(NO_RATING, labelling="ai")


def test_ai_sample_size_and_weights(ai_result):
    data = ai_result.data
    assert ai_result.method.is_ai and ai_result.method.sampled
    assert int(data["in_label_sample"].sum()) == 200
    assert data.loc[data["in_label_sample"], "sample_weight"].sum() == pytest.approx(1000)
    assert data.loc[~data["in_label_sample"], "sample_weight"].isna().all()
    assert ai_result.sampling["weight_sum"] == pytest.approx(1000)
    assert ai_result.sampling["strata"]["Negative"]["in_sample"] == 100


def test_weighted_issue_frequency_recovers_the_population_share(ai_result):
    priority = ai_result.table("issue_priority")
    assert priority.weighted and priority.basis == "text_labelled"
    row = priority.df.set_index("group").loc["Runs small"]
    assert row["frequency_pct"] == pytest.approx(10.0)   # true population share
    assert row["reviews"] == 100                          # raw count shown alongside
    topics = ai_result.table("topics").df.set_index("topic")
    assert topics.loc["Fit & Sizing", "est_pct_of_reviews_complaint"] == pytest.approx(10.0)


def test_primary_sentiment_matches_ratings_exactly_in_ai_mode(ai_result):
    primary = ai_result.table("sentiment_primary").df.set_index("sentiment")
    assert primary.loc["Negative", "pct"] == 10.0 and primary.loc["Positive", "pct"] == 80.0
    assert ai_result.kpis["sentiment_sources"]["rating"] == 1000


def test_text_sentiment_table_is_weighted_not_raw_sample_shares(ai_result):
    text = ai_result.table("sentiment_text")
    assert text.weighted and text.n_reviews == 200
    df = text.df.set_index("sentiment")
    assert df.loc["Negative", "reviews"] == 100           # raw: half the sample
    assert df.loc["Negative", "pct"] == pytest.approx(10.0)  # estimate: 10% of the population


def test_api_failure_rows_are_baseline_not_ai():
    unrated = apparel_csv([(f"It runs small #{i}", "", "A") for i in range(6)])
    result = run_analysis(unrated, labelling="ai", llm=FakeLLM(fail_calls={1}, fatal=True))
    assert result.method.mixed and result.method.fallback_rows == 6 and result.method.ai_rows == 0
    assert set(result.data["sentiment_source"]) == {"text_baseline"}
    assert "fell back" in result.table("issue_priority").note


def test_weight_sum_check_fails_fast():
    df = pd.DataFrame({"has_text": [True, True], "label_source": ["ai", "ai"],
                       "sample_weight": [1.0, 5.0], "sentiment_source": ["text_ai", "text_ai"]})
    with pytest.raises(PipelineError, match="Sampling weights sum to 6"):
        _add_weights(df)


# ---------------------------------------------------------------------------
# Filtering keeps the original sampling design
# ---------------------------------------------------------------------------

def test_filtering_does_not_resample_or_change_weights(ai_result):
    sub = ai_result.filtered({"department": "A"})
    original = ai_result.data.set_index("review_id")
    kept = sub.data.set_index("review_id")
    # Same rows as a plain filter, same sample membership, identical weights.
    assert set(kept.index) == set(original.index[original["department"] == "A"])
    pd.testing.assert_series_equal(kept["in_label_sample"], original.loc[kept.index, "in_label_sample"])
    pd.testing.assert_series_equal(kept["sample_weight"], original.loc[kept.index, "sample_weight"])
    pd.testing.assert_series_equal(kept["sentiment_weight"], original.loc[kept.index, "sentiment_weight"])
    # Weights are not re-normalised to the subset size.
    labelled = kept[kept["in_label_sample"]]
    assert labelled["sample_weight"].sum() != pytest.approx(len(kept[kept["has_text"]])) or \
        labelled["sample_weight"].nunique() == 1
    # The filtered percentage is exactly the original-weight estimate for department A.
    complaints = labelled["issue_ai"] == "Runs small"
    expected = round(100 * labelled.loc[complaints, "sample_weight"].sum() / labelled["sample_weight"].sum(), 1)
    got = sub.table("issue_priority").df.set_index("group").loc["Runs small", "frequency_pct"]
    assert got == pytest.approx(expected)
    naive = round(100 * complaints.mean(), 1)               # treating the sample as representative
    assert got != pytest.approx(naive)
    assert sub.table("issue_priority").df.set_index("group").loc["Runs small", "reviews"] == complaints.sum()
    assert sub.filters == {"department": "A"}


def test_filter_by_mask_callable_and_bad_column(ai_result):
    by_mask = ai_result.filtered(ai_result.data["rating"] == 5)
    by_callable = ai_result.filtered(lambda d: d["rating"] == 5)
    assert len(by_mask.data) == len(by_callable.data) == 800
    assert by_mask.kpis["primary"]["pct_positive"] == 100.0
    with pytest.raises(PipelineError, match="no such column"):
        ai_result.filtered({"nope": 1})


def test_original_result_is_unchanged_by_filtering(ai_result):
    before = ai_result.table("issue_priority").df.copy()
    ai_result.filtered({"department": "B"})
    pd.testing.assert_frame_equal(before, ai_result.table("issue_priority").df)


# ---------------------------------------------------------------------------
# Legacy synthetic demo: numbers reproduced
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def food():
    return run_analysis(ROOT / "data" / "sample_reviews_SYNTHETIC.csv", reference_date="2026-10-08")


def test_legacy_food_headline_numbers_reproduced(food):
    rating = food.kpis["rating"]
    assert rating["avg_rating"] == 3.43
    assert (rating["pct_positive"], rating["pct_neutral"], rating["pct_negative"]) == (51.0, 21.2, 27.8)
    assert len(food.data) == 3102


def test_legacy_food_trend_numbers_reproduced(food):
    trend = food.table("trend_rating")
    df, _ = process_reviews(ROOT / "data" / "sample_reviews_SYNTHETIC.csv", reference_date="2026-10-08")
    pd.testing.assert_frame_equal(trend.df, analytics.reviews_over_time(df))
    first = trend.df.iloc[0]
    assert (first["reviews"], first["avg_rating"], first["pct_negative"]) == (140, 3.53, 25.0)
    assert len(trend.df) == 18
    assert food.table("trend").available and food.table("recent_change").available


def test_primary_trend_differs_only_by_unrated_rows(food):
    """The primary trend also counts the 36 unrated reviews (text baseline), so it can
    differ slightly from the rating-only trend; months stay identical."""
    assert food.kpis["sentiment_sources"]["text_baseline"] == 36
    assert food.table("trend").df["month"].equals(food.table("trend_rating").df["month"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_aggregate_outputs_only(tmp_path):
    data = tmp_path / "reviews.csv"
    data.write_bytes(population(10, 10, 10))
    assert run_analysis_cli.main([str(data), "--out-dir", str(tmp_path / "out")]) == 0
    out = tmp_path / "out" / "apparel_ecommerce"
    assert (out / "summary.json").exists() and any((out / "tables").iterdir())
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert "runs small, sadly" not in text and "Lovely and soft" not in text


def test_cli_ai_without_access_exits_2(tmp_path, capsys):
    data = tmp_path / "reviews.csv"
    data.write_bytes(population(2, 2, 2))
    assert run_analysis_cli.main([str(data), "--labelling", "ai", "--out-dir", str(tmp_path / "o")]) == 2
    assert "no Anthropic API access" in capsys.readouterr().err
