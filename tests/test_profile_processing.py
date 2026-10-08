"""Cleaning with profiles: a Kaggle-style apparel file, and datasets without ratings.

Uses small in-memory fixtures shaped like the real files — the raw Kaggle data
is not needed to run the tests.
"""

import pandas as pd
import pytest

from app import analytics
from app.ai_analysis import BaselineLabeller, explode_aspects, label_reviews
from app.data_processing import DataValidationError, process_reviews
from app.profiles import load_profile

APPAREL_CSV = """,Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name,Class Name
0,767,33,,Absolutely wonderful - silky and sexy and comfortable,4,1,0,Initmates,Intimate,Intimates
1,1080,34,,"Love this dress! it's sooo pretty and fits perfectly.",5,1,4,General,Dresses,Dresses
2,1077,60,Some major design flaws,"I had such high hopes for this dress. it runs small and the zipper is cheap.",3,0,0,General,Dresses,Dresses
3,1049,50,My favorite buy!,"I love, love, love this jumpsuit.",5,1,0,General Petite,Bottoms,Pants
4,847,47,Flattering shirt,,5,1,6,General,Tops,Blouses
5,847,47,Flattering shirt,,5,1,6,General,Tops,Blouses
6,1077,24,,,2,0,0,General,Dresses,Dresses
7,1095,29,Not for me,"Way too sheer, you can see right through it. Will be returning.",1,0,12,General,Dresses,Dresses
8,1095,121,,"Bought it in two colors, love it.",5,1,0,General,Dresses,Dresses
9,1095,45,,"Nice dress overall",4,yes,0,General,Dresses,Dresses
"""


@pytest.fixture(scope="module")
def apparel():
    return process_reviews(APPAREL_CSV.encode("utf-8"))


def test_apparel_profile_is_detected(apparel):
    _, report = apparel
    assert report.profile_name == "apparel_ecommerce"


def test_apparel_columns_mapped_to_roles(apparel):
    df, report = apparel
    assert {"review_id", "review_title", "review_text", "rating", "product_id",
            "department", "product_class", "division", "recommended", "age_band"} <= set(df.columns)
    assert report.roles.entity == "product_id" and report.roles.entity_label == "Product"
    assert [c for c, _ in report.roles.segments] == ["department", "product_class", "division", "age_band"]
    assert "date" in report.missing_optional_columns


def test_row_number_column_becomes_review_id_and_is_ignored_for_duplicates(apparel):
    df, report = apparel
    # Rows 4 and 5 are identical apart from the row number -> one is removed.
    assert report.removed["Exact duplicate row (ignoring the row-number column)"] == 1
    assert "5" not in set(df["review_id"]) and "4" in set(df["review_id"])
    assert "review_id" in report.row_number_columns


def test_title_combined_with_text_and_title_only_rows_kept(apparel):
    df, report = apparel
    row = df.set_index("review_id").loc["2"]
    assert row["analysis_text"].startswith("Some major design flaws. I had such high hopes")
    title_only = df.set_index("review_id").loc["4"]
    assert title_only["analysis_text"] == "Flattering shirt" and title_only["has_text"]
    assert report.flagged["Title used as text (review text missing)"] == 1


def test_rows_with_only_a_rating_are_kept(apparel):
    df, _ = apparel
    row = df.set_index("review_id").loc["6"]
    assert not row["has_text"] and row["rating"] == 2


def test_value_alias_and_bands(apparel):
    df, report = apparel
    assert "Initmates" not in set(df["division"]) and "Intimates" in set(df["division"])
    assert report.flagged["Division value corrected"] == 1
    by_id = df.set_index("review_id")
    assert by_id.loc["0", "age_band"] == "30–39"
    assert by_id.loc["7", "age_band"] == "Under 30"
    assert by_id.loc["2", "age_band"] == "60+"
    assert by_id.loc["7", "helpful_votes_band"] == "10+"
    assert pd.isna(by_id.loc["8", "age"])  # 121 is outside the valid range
    assert report.flagged["Implausible age (cleared)"] == 1


def test_outcome_parsed_as_yes_no(apparel):
    df, report = apparel
    by_id = df.set_index("review_id")
    assert by_id.loc["0", "recommended"] == 1.0 and by_id.loc["7", "recommended"] == 0.0
    assert by_id.loc["9", "recommended"] == 1.0  # "yes" is accepted too


def test_group_summary_reports_outcome_rate(apparel):
    df, report = apparel
    out = analytics.group_summary(df, "department", min_count=1,
                                  outcome_cols=[o.column for o in report.roles.outcomes])
    assert "pct_recommended" in out.columns
    dresses = out.set_index("group").loc["Dresses"]
    assert dresses["reviews"] == 6
    assert dresses["pct_recommended"] == pytest.approx(100 * 3 / 6, abs=0.1)


def test_explicit_profile_overrides_detection():
    _, report = process_reviews(APPAREL_CSV.encode("utf-8"), profile="generic")
    assert report.profile_name == "generic" and report.profile_reason == "selected explicitly"


def test_source_truncation_is_flagged():
    long_cut = "x" * 480 + " and then the zipper br"
    rows = ["review_text,rating"] + [f'"{long_cut} {i:02d}",3' for i in range(25)] + ['"Short and complete.",5']
    df, report = process_reviews("\n".join(rows).encode("utf-8"))
    assert df["text_truncated"].sum() == 25
    assert any("cut off by the source" in k for k in report.flagged)


def test_truncation_not_flagged_for_normal_lengths():
    rows = ["review_text,rating"] + [f'"Review number {i} is fine.",4' for i in range(30)]
    df, report = process_reviews("\n".join(rows).encode("utf-8"))
    assert not df["text_truncated"].any()


# ---------------------------------------------------------------------------
# Datasets without ratings
# ---------------------------------------------------------------------------

NO_RATING_CSV = """feedback,product,channel,submitted_at
"The app keeps crashing when I log in. Terrible and frustrating.",Mobile app,iOS,2026-01-03
"Support was helpful and friendly.",Website,Web,2026-01-04
"Way too expensive, awful value for what you get.",Mobile app,Android,2026-02-10
"Easy to use, love it.",Website,Web,2026-02-11
,Website,Web,2026-02-12
"The app keeps crashing when I log in. Terrible and frustrating.",Mobile app,iPad,2026-01-03
"""


@pytest.fixture(scope="module")
def no_rating():
    return process_reviews(NO_RATING_CSV.encode("utf-8"))


def test_rating_is_optional(no_rating):
    df, report = no_rating
    assert report.profile_name == "generic"
    assert not report.has_rating
    assert df["rating"].isna().all() and df["sentiment_from_rating"].isna().all()
    assert any("No rating column" in note for note in report.notes)
    assert report.roles.entity == "product" and report.roles.has_date


def test_no_rating_removes_textless_rows_and_resubmissions(no_rating):
    df, report = no_rating
    assert report.removed["No review text"] == 1
    assert report.removed["Re-submitted review (same text, product and date)"] == 1
    assert len(df) == 4


def test_missing_text_column_error_mentions_optional_rating():
    with pytest.raises(DataValidationError) as err:
        process_reviews(b"stars,product\n5,x")
    assert "review_text" in str(err.value) and "rating" in str(err.value)


def test_core_analysis_works_without_rating(no_rating):
    df, report = no_rating
    labelled, run = label_reviews(df, BaselineLabeller(load_profile(report.profile_name)))
    assert run.from_baseline == 4
    kpis = analytics.overview_kpis(labelled, sentiment_col="sentiment_ai")
    assert kpis["avg_rating"] is None and kpis["rated_reviews"] == 0
    assert kpis["pct_negative"] is not None
    summary = analytics.group_summary(labelled, "product", sentiment_col="sentiment_ai", min_count=1)
    assert summary["avg_rating"].isna().all() and summary["pct_negative"].notna().all()
    assert analytics.rating_distribution(labelled)["reviews"].sum() == 0
    assert analytics.low_rating_drivers(labelled, ["product"]).empty
    assert analytics.correlation_with_rating(labelled, "rating") is None
    aspects = explode_aspects(labelled)
    issues = aspects[aspects["polarity"] == "negative"]
    priority = analytics.priority_table(issues, "issue", sentiment_col="sentiment_ai",
                                        total_reviews=len(labelled), min_count=1)
    assert not priority.empty
    assert priority["avg_rating"].isna().all()
    assert (priority["severity"] > 0).any()  # severity falls back to the negative share
