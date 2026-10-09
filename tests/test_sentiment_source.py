"""Primary sentiment: rating first, then the text label (calibrated baseline for apparel).

All review texts here are invented.
"""

import pandas as pd
import pytest

from app.ai_analysis import AILabeller, BaselineLabeller, label_reviews
from app.data_processing import process_reviews
from app.profiles import BaselineSentiment, load_profile
from tests.test_ai_analysis import FakeLLM

APPAREL = load_profile("apparel_ecommerce")
# Invented texts on which the calibrated and default rules disagree.
DEFAULT_NEUTRAL_CALIBRATED_POSITIVE = "Gorgeous dress and so flattering, I love it. It runs small so size up."
DEFAULT_POSITIVE_CALIBRATED_NEUTRAL = "The top is nice enough."
HEADER = ",Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name,Class Name"


def apparel(rows):
    """rows: (text, rating or '') -> cleaned apparel DataFrame."""
    lines = [HEADER] + [f'{i},{100 + i},35,,"{text}",{rating},1,0,General,Tops,Knits'
                        for i, (text, rating) in enumerate(rows)]
    df, report = process_reviews("\n".join(lines).encode("utf-8"))
    assert report.profile_name == "apparel_ecommerce"
    return df


def test_apparel_profile_carries_the_approved_calibration():
    assert APPAREL.baseline_sentiment == BaselineSentiment(negative_max=-0.1, positive_min=0.65,
                                                           complaint_penalty=0.1)


# 1. Rated review -> rating-based sentiment ----------------------------------

def test_rated_review_uses_the_rating():
    df = apparel([(DEFAULT_NEUTRAL_CALIBRATED_POSITIVE, 1), ("Terrible quality, it fell apart.", 5)])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert out["sentiment"].tolist() == ["Negative", "Positive"]          # from the 1★ and 5★ ratings
    assert out["sentiment_source"].tolist() == ["rating", "rating"]
    assert out.loc[0, "sentiment_ai"] == "Positive"                      # text label kept separately


def test_rating_scale_comes_from_the_profile():
    df = apparel([("Okay.", 3), ("Okay.", 2), ("Okay.", 4)])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert out["sentiment"].tolist() == ["Neutral", "Negative", "Positive"]


# 2. Unrated review -> calibrated text baseline (apparel) --------------------

def test_unrated_apparel_review_uses_the_calibrated_baseline():
    df = apparel([(DEFAULT_NEUTRAL_CALIBRATED_POSITIVE, ""), (DEFAULT_POSITIVE_CALIBRATED_NEUTRAL, "")])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert out["sentiment_source"].tolist() == ["text_baseline", "text_baseline"]
    assert out["sentiment"].tolist() == ["Positive", "Neutral"]          # calibrated rule
    default = BaselineLabeller(APPAREL, calibration=None)  # explicitly uncalibrated
    assert [default.label_one(t)["sentiment"] for t in df["analysis_text"]] == ["Neutral", "Positive"]


def test_mixed_rated_and_unrated_rows():
    df = apparel([(DEFAULT_POSITIVE_CALIBRATED_NEUTRAL, 5), (DEFAULT_POSITIVE_CALIBRATED_NEUTRAL, "")])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert out["sentiment"].tolist() == ["Positive", "Neutral"]
    assert out["sentiment_source"].tolist() == ["rating", "text_baseline"]


def test_unrated_review_with_ai_label_is_marked_text_ai():
    df = apparel([("It runs small, will return it.", "")])
    out, _ = label_reviews(df, AILabeller(FakeLLM(), APPAREL))
    assert out.loc[0, "sentiment_source"] == "text_ai" and out.loc[0, "sentiment"] == "Negative"


def test_no_rating_and_no_text_leaves_sentiment_empty():
    df = apparel([("Fine.", 4), ("", "")])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert len(out) == 1  # the empty row is removed during cleaning
    no_text = apparel([("Fine.", 4), ("", 3)])
    out2, _ = label_reviews(no_text, BaselineLabeller(APPAREL))
    row = out2[~out2["has_text"]].iloc[0]
    assert row["sentiment"] == "Neutral" and row["sentiment_source"] == "rating"


# 3. Generic and food behaviour unchanged ------------------------------------

@pytest.mark.parametrize("name", ["generic", "food_delivery_demo"])
def test_generic_and_food_have_no_calibration_and_use_the_default_rule(name):
    profile = load_profile(name)
    assert profile.baseline_sentiment is None
    labeller = BaselineLabeller(profile)
    assert labeller.calibration is None
    for text in (DEFAULT_NEUTRAL_CALIBRATED_POSITIVE, DEFAULT_POSITIVE_CALIBRATED_NEUTRAL,
                 "The food was great. My order was late again."):
        label = labeller.label_one(text)
        expected = labeller.sentiment(text)
        if expected == "Positive" and any(a["polarity"] == "negative" for a in label["aspects"]):
            expected = "Neutral"  # the default downgrade rule
        assert label["sentiment"] == expected


def test_food_unrated_rows_use_the_default_text_baseline():
    df, report = process_reviews(b"review_text,rating,restaurant\nThe top is nice enough.,,Burger Barn\n")
    assert report.profile_name == "food_delivery_demo"
    out, _ = label_reviews(df, BaselineLabeller(load_profile(report.profile_name)))
    assert out.loc[0, "sentiment_source"] == "text_baseline"
    assert out.loc[0, "sentiment"] == "Positive"  # default rule (calibrated apparel rule would say Neutral)


# 4. Complaint/topic detection independent of the sentiment source -----------

def test_aspects_do_not_depend_on_rating_or_calibration():
    text = "Love the colour. It runs small. The fabric is see-through."
    rated = apparel([(text, 1)])
    unrated = apparel([(text, "")])
    out_rated, _ = label_reviews(rated, BaselineLabeller(APPAREL))
    out_unrated, _ = label_reviews(unrated, BaselineLabeller(APPAREL))
    uncalibrated = BaselineLabeller(APPAREL, calibration=None)
    for col in ("aspects_ai", "issue_ai", "primary_topic_ai", "flag_mentions_return"):
        assert out_rated.loc[0, col] == out_unrated.loc[0, col]
    assert out_rated.loc[0, "aspects_ai"] == uncalibrated.label_one(text)["aspects"]
    assert out_rated.loc[0, "sentiment_source"] == "rating"
    assert out_unrated.loc[0, "sentiment_source"] == "text_baseline"


def test_evaluation_still_scores_the_text_label_only():
    """sentiment_ai must stay text-only so ratings never leak into accuracy figures."""
    df = apparel([(DEFAULT_NEUTRAL_CALIBRATED_POSITIVE, 1)])
    out, _ = label_reviews(df, BaselineLabeller(APPAREL))
    assert out.loc[0, "sentiment_ai"] == "Positive" and out.loc[0, "sentiment"] == "Negative"
