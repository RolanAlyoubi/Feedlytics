"""Tests for loading, validation and cleaning."""

import io

import numpy as np
import pandas as pd
import pytest

from app import config
from app.data_processing import (
    CleaningReport,
    DataValidationError,
    clean_text,
    load_csv,
    normalize_header,
    parse_number,
    parse_rating,
    process_reviews,
    sentiment_from_rating,
    standardize_columns,
)

REF_DATE = "2026-06-30"


def csv_bytes(text: str) -> bytes:
    return text.strip().encode("utf-8")


def run(text: str):
    return process_reviews(csv_bytes(text), reference_date=REF_DATE)


# ---------------------------------------------------------------------------
# Loading errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [b"", b"   \n  \n"])
def test_empty_file_is_rejected(content):
    with pytest.raises(DataValidationError, match="empty"):
        load_csv(content)


def test_headers_without_rows_are_rejected():
    with pytest.raises(DataValidationError, match="no data rows"):
        load_csv(b"review_text,rating\n")


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_csv(tmp_path / "nope.csv")


def test_too_many_rows_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 2)
    with pytest.raises(DataValidationError, match="limit"):
        load_csv(b"review_text,rating\na,1\nb,2\nc,3\n")


def test_reads_file_like_objects_and_non_utf8():
    data = "review_text,rating\nCaf\xe9 was great,5\n".encode("cp1252")
    df = load_csv(io.BytesIO(data))
    assert df.loc[0, "review_text"] == "Café was great"


def test_utf8_bom_is_handled():
    df = load_csv("﻿review_text,rating\nok,3\n".encode("utf-8"))
    assert list(df.columns) == ["review_text", "rating"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_missing_required_columns_lists_what_is_needed():
    with pytest.raises(DataValidationError) as err:
        run("comment_title,stars_given\nhi,5")
    message = str(err.value)
    assert "review_text" in message and "rating" in message
    assert "comment_title" in message  # shows what was found


def test_synonym_headers_are_mapped():
    df, report = run("Comment,Stars,Review Date,Cuisine\nGreat pizza,5,2025-01-01,Pizza")
    assert {"review_text", "rating", "date", "category"} <= set(df.columns)
    assert report.column_mapping["Stars"] == "rating"


def test_synonym_does_not_override_existing_canonical_column():
    report = CleaningReport()
    raw = pd.DataFrame({"review_text": ["a"], "comment": ["b"], "rating": ["5"]})
    out = standardize_columns(raw, report)
    assert out.loc[0, "review_text"] == "a"
    assert "comment" in report.extra_columns


def test_normalize_header():
    assert normalize_header("  Review Text ") == "review_text"
    assert normalize_header("Delivery-Time (min)") == "delivery_time_min"


def test_all_text_empty_is_rejected():
    with pytest.raises(DataValidationError, match="empty in every row"):
        run("review_text,rating\n,5\n  ,4")


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("4", 4.0), ("4.0", 4.0), ("4/5", 4.0), ("4 stars", 4.0), ("four", 4.0),
    ("Five Stars", 5.0), (3, 3.0), ("1 star", 1.0),
])
def test_parse_rating_valid(raw, expected):
    assert parse_rating(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "", "N/A", None, "4/10", "great"])
def test_parse_rating_invalid(raw):
    assert np.isnan(parse_rating(raw))


@pytest.mark.parametrize("raw, expected", [
    ("35 min", 35.0), ("$1,234.50", 1234.5), ("SAR 40", 40.0), ("-5", -5.0), (12, 12.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_parse_number_missing():
    assert np.isnan(parse_number("")) and np.isnan(parse_number("n/a"))


def test_clean_text():
    assert clean_text("  Great<br>food &amp; fast  ") == "Great food & fast"
    assert clean_text("N/A") is None
    assert clean_text("<p> </p>") is None


def test_sentiment_from_rating_boundaries():
    assert [sentiment_from_rating(r) for r in (1, 2, 3, 4, 5)] == \
        ["Negative", "Negative", "Neutral", "Positive", "Positive"]
    assert sentiment_from_rating(np.nan) is None


# ---------------------------------------------------------------------------
# Cleaning behaviour
# ---------------------------------------------------------------------------

def test_invalid_ratings_are_cleared_but_rows_kept():
    df, report = run("""
review_text,rating
Great food here,5
Was fine I guess,6
Bad bad bad,abc
Lovely meal,4.5
""")
    assert len(df) == 4
    assert df.loc[0, "rating"] == 5.0
    assert df["rating"].iloc[1:3].isna().all()
    assert report.flagged["Invalid rating (cleared)"] == 2
    assert report.flagged["Fractional rating rounded to whole star"] == 1
    assert df.loc[3, "rating"] == 5.0  # 4.5 rounds half up


def test_rows_without_text_and_rating_are_removed():
    df, report = run("review_text,rating\nGood,5\n,\nN/A,abc\n,3")
    assert len(df) == 2
    assert report.removed["No review text and no valid rating"] == 2
    assert df["has_text"].tolist() == [True, False]


def test_mixed_date_formats_and_invalid_dates():
    df, report = run("""
review_text,rating,date
a,5,2025-03-05
b,4,"March 5, 2025"
c,3,05 Mar 2025
d,2,2025-03-05 18:30:00
e,1,not a date
f,1,2031-01-01
""")
    assert (df["date"].iloc[:4] == pd.Timestamp("2025-03-05")).all()
    assert df["date"].iloc[4:].isna().all()
    assert report.flagged["Unreadable date (cleared)"] == 1
    assert report.flagged["Future or implausible date (cleared)"] == 1
    assert df.loc[0, "month"] == pd.Timestamp("2025-03-01")


def test_exact_duplicate_rows_removed():
    df, report = run("review_id,review_text,rating\n1,Nice,5\n1,Nice,5\n2,Nice,5")
    assert len(df) == 2
    assert report.removed["Exact duplicate row"] == 1


def test_resubmitted_reviews_removed_but_short_common_texts_kept():
    df, report = run("""
review_id,review_text,rating,restaurant,date
1,"The burger was cold and the fries were soggy.",2,Burger Barn,2025-01-01
2,"the burger was cold, and the fries were soggy!",2,Burger Barn,2025-01-01
3,"The burger was cold and the fries were soggy.",2,Burger Barn,2025-02-10
4,Great food!,5,Burger Barn,2025-01-01
5,Great food!,5,Burger Barn,2025-01-01
""")
    # id 2 is a re-submission of id 1; id 3 is a different day; short texts are not merged.
    assert df["review_id"].tolist() == ["1", "3", "4", "5"]
    assert report.removed["Re-submitted review (same text, rating, restaurant and date)"] == 1


def test_restaurant_spelling_variants_are_merged():
    df, report = run("""
review_text,rating,restaurant
a,5,Burger Barn
b,4,Burger Barn
c,3,  burger barn
d,2,BURGER BARN
""")
    assert df["restaurant"].unique().tolist() == ["Burger Barn"]
    assert report.flagged["Restaurant name variant merged"] == 2


def test_numeric_fields_parsed_and_implausible_values_cleared():
    df, report = run("""
review_text,rating,delivery_time,order_value
a,5,35 min,$45.50
b,4,999,0
c,3,-5,SAR 12
d,2,,
""")
    assert df["delivery_time"].tolist()[0] == 35.0
    assert df["delivery_time"].iloc[1:].isna().all()
    assert df["order_value"].tolist()[0] == 45.5 and df.loc[2, "order_value"] == 12.0
    assert report.flagged["Implausible delivery time (cleared)"] == 2
    assert report.flagged["Implausible order value (cleared)"] == 1


def test_review_ids_generated_and_repeats_made_unique():
    df, report = run("review_id,review_text,rating\n,first,5\nA1,second,4\nA1,third,3")
    assert df.loc[0, "review_id"].startswith("gen_")
    assert df["review_id"].is_unique
    assert report.flagged["Missing review_id (generated)"] == 1
    assert report.flagged["Repeated review_id with different content (suffixed)"] == 1


def test_report_row_counts_add_up():
    df, report = run("review_text,rating\nGood,5\nGood,5\n,\nOk food overall,3")
    assert report.rows_received == 4
    assert report.rows_received - report.rows_removed == report.rows_retained == len(df)


def test_optional_columns_absent_is_fine():
    df, report = run("review_text,rating\nGood,5")
    assert "date" not in df.columns
    assert "date" in report.missing_optional_columns


def test_warning_when_no_valid_ratings():
    df, report = run("review_text,rating\nGood,x\nBad,y")
    assert any("No valid 1–5 ratings" in w for w in report.warnings)
