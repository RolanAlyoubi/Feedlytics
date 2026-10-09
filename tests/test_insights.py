"""Rule-based insights: every item is traced to a table, small samples are flagged,
and nothing is invented. All data is invented; no API calls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from app import config
from app.insights import generate_insights
from app.pipeline import LabelMethod, Table, run_analysis
from tests.test_pipeline import population

BASELINE = LabelMethod(kind="baseline", is_ai=False, calibrated_baseline=True)
ISSUE_COLUMNS = ["group", "reviews", "frequency_pct", "avg_rating", "pct_negative",
                 "severity", "priority_index", "priority"]
DRIVER_COLUMNS = ["dimension", "group", "reviews", "pct_low_rating", "lift", "p_value", "q_value",
                  "significant"]


def fake_result(**tables):
    """A stand-in AnalysisResult: generate_insights only reads .tables and .method."""
    return SimpleNamespace(method=BASELINE, tables=tables)


def table(name, rows, columns, n_reviews=500, weighted=False, available=True):
    return Table(name, pd.DataFrame(rows, columns=columns), "text_labelled", weighted, None,
                 n_reviews, available=available)


def test_pipeline_insights_trace_to_tables_and_say_baseline():
    result = run_analysis(population())
    items = generate_insights(result)
    sources = {i.source_table for i in items}
    assert {"issue_priority", "low_rating_drivers", "praise", "flags"} <= sources
    for item in items:
        assert result.tables[item.source_table].available
        expected = "Observed star ratings" if item.source_table == "low_rating_drivers" else "Non-AI baseline"
        assert item.basis.startswith(expected)
    issue = result.tables["issue_priority"].df.iloc[0]
    first = items[0]
    assert first.title == f"Complaint: {issue['group']}"
    assert f"{issue['frequency_pct']:g}% of reviews" in first.detail
    segment = next(i for i in items if i.source_table == "low_rating_drivers")
    assert "association, not a cause" in segment.detail


def test_top_issues_are_sorted_by_priority_not_table_order():
    rows = [["Low one", 40, 2.0, 3.5, 50.0, 0.1, 10.0, "Low"],
            ["Small one", 5, 1.0, np.nan, 80.0, 0.5, np.nan, "Insufficient data"],
            ["Medium one", 60, 5.0, 3.0, 60.0, 0.2, 40.0, "Medium"],
            ["High one", 90, 9.0, 2.0, 90.0, 0.6, 100.0, "High"]]
    items = generate_insights(fake_result(issue_priority=table("issue_priority", rows, ISSUE_COLUMNS)))
    findings = [i.title for i in items if i.kind == "finding"]
    assert findings == ["Complaint: High one", "Complaint: Medium one", "Complaint: Low one"]
    recommendations = [i for i in items if i.kind == "recommendation"]
    assert [i.title for i in recommendations] == ["Prioritise: High one"]
    detail = recommendations[0].detail
    assert "index 100 of 100" in detail and "9% of reviews" in detail and "severity 0.6" in detail
    assert "confirm the cause before acting" in detail
    assert "caused by" not in detail and "because" not in detail


def test_empty_missing_and_unavailable_tables_give_no_items():
    empty = fake_result(issue_priority=table("issue_priority", [], ISSUE_COLUMNS),
                        low_rating_drivers=table("low_rating_drivers", [], DRIVER_COLUMNS))
    assert generate_insights(empty) == []
    unavailable = fake_result(issue_priority=table("issue_priority", [], ISSUE_COLUMNS, available=False),
                              praise=table("praise", [], ["topic", "reviews", "est_pct_of_reviews"],
                                           available=False))
    assert generate_insights(unavailable) == []
    assert generate_insights(fake_result()) == []


def test_labelling_none_still_gives_rating_based_findings_only():
    result = run_analysis(population(), labelling="none")
    sources = {i.source_table for i in generate_insights(result)}
    assert sources <= {"low_rating_drivers"}


def test_small_samples_are_flagged_in_the_visible_text():
    rows = [["Rare issue", 5, 1.0, np.nan, 80.0, 0.5, np.nan, "Insufficient data"]]
    praise = table("praise", [["Fit & Sizing", 3, 30.0]], ["topic", "reviews", "est_pct_of_reviews"],
                   n_reviews=10)
    items = generate_insights(fake_result(issue_priority=table("issue_priority", rows, ISSUE_COLUMNS),
                                          praise=praise))
    assert [i.kind for i in items] == ["finding", "finding"]  # no recommendation for a small issue
    for item in items:
        assert item.small_sample
        assert f"fewer than {config.MIN_GROUP_SIZE} reviews" in item.detail
    assert "average rating" not in items[0].detail  # missing rating is omitted, not shown as nan


def test_missing_and_invalid_values_are_shown_safely():
    drivers = table("low_rating_drivers",
                    [["department", "A", np.nan, 30.0, 2.5, np.nan, np.nan, True],
                     ["department", "B", 200, 40.0, "bad", 0.01, 0.02, True],
                     ["department", "C", 200, 5.0, 0.5, 0.01, 0.02, True],
                     ["department", "D", 200, 50.0, 3.0, 0.2, 0.3, False]], DRIVER_COLUMNS)
    items = generate_insights(fake_result(low_rating_drivers=drivers))
    assert [i.title for i in items] == ["Low ratings concentrated in department = A"]  # B invalid, C below 1, D not significant
    detail = items[0].detail
    assert "q = n/a" in detail and "unknown number of reviews" in detail and "nan" not in detail.lower()
    assert items[0].small_sample


def test_return_rate_comes_from_flags_table():
    flags = table("flags", [["mentions_return", 120, 24.0, "def"], ["mentions_repurchase", 3, 0.6, "def"]],
                  ["flag", "reviews", "est_pct_of_reviews", "definition"], weighted=True)
    items = generate_insights(fake_result(flags=flags))
    assert len(items) == 1 and items[0].source_table == "flags"
    assert items[0].detail.startswith("An estimated 24% of reviews mention returning the item (120 reviews).")
    assert not items[0].small_sample
