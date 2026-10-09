"""Baseline sentiment calibration: split, search, profile block, labeller, script."""

import json

import pandas as pd
import pytest

from app.ai_analysis import BaselineLabeller
from app.evaluation import (calibrate_sentiment, calibration_grid, sentiment_from_score,
                            tuning_split)
from app.profiles import BaselineSentiment, ProfileError, load_profile, parse_profile
from scripts import calibrate_baseline

APPAREL = load_profile("apparel_ecommerce")


def test_tuning_split_is_deterministic_balanced_and_excludes():
    ids = pd.Series([str(i) for i in range(2000)])
    split = tuning_split(ids, exclude_ids=["5", "6", "7"])
    assert split.equals(tuning_split(ids, exclude_ids=["5", "6", "7"]))
    assert (split[ids.isin(["5", "6", "7"])] == "excluded").all()
    assert set(split) == {"tune", "test", "excluded"}
    assert 900 < (split == "tune").sum() < 1100
    # Excluding ids never moves any other review between tune and test.
    assert (split[~ids.isin(["5", "6", "7"])] == tuning_split(ids)[~ids.isin(["5", "6", "7"])]).all()


def test_grid_respects_ordering_and_matches_planned_size():
    grid = calibration_grid(calibrate_baseline.NEGATIVE_MAX, calibrate_baseline.POSITIVE_MIN,
                            calibrate_baseline.PENALTIES)
    assert len(grid) == 575
    assert all(g["positive_min"] > g["negative_max"] for g in grid)
    assert calibrate_baseline.NEGATIVE_MAX[0] == -0.2 and calibrate_baseline.NEGATIVE_MAX[-1] == 0.7
    assert calibrate_baseline.POSITIVE_MIN[0] == 0.3 and calibrate_baseline.POSITIVE_MIN[-1] == 0.95


def test_search_finds_the_separating_thresholds():
    compound = pd.Series([-0.9, -0.5, 0.1, 0.2, 0.8, 0.9])
    complaints = pd.Series([0, 0, 0, 0, 0, 0])
    target = pd.Series(["Negative", "Negative", "Neutral", "Neutral", "Positive", "Positive"])
    grid = calibration_grid([-0.3, 0.0], [0.5, 0.95], [0.0])
    best = calibrate_sentiment(compound, complaints, target, grid).iloc[0]
    assert (best["negative_max"], best["positive_min"]) == (-0.3, 0.5)
    assert best["accuracy"] == 100.0 and best["macro_f1"] == 1.0


def test_penalty_turns_polite_complaints_negative():
    compound = pd.Series([0.6, 0.6])
    complaints = pd.Series([3, 0])
    pred = sentiment_from_score(compound - 0.4 * complaints, negative_max=-0.1, positive_min=0.65)
    assert pred.tolist() == ["Negative", "Neutral"]


def test_profile_baseline_sentiment_block():
    raw = {"name": "t", "taxonomy": {"Other": {"Other": "x"}},
           "baseline_sentiment": {"negative_max": -0.1, "positive_min": 0.65, "complaint_penalty": 0.1}}
    assert parse_profile(raw).baseline_sentiment == BaselineSentiment(-0.1, 0.65, 0.1)
    for bad, message in [({"negative_max": 0.7, "positive_min": 0.3}, "below"),
                         ({"negative_max": 0, "positive_min": 0.5, "complaint_penalty": -1}, "≥ 0"),
                         ({"positive_min": 0.5}, "numeric")]:
        with pytest.raises(ProfileError, match=message):
            parse_profile({**raw, "baseline_sentiment": bad})


def test_only_apparel_has_a_calibration():
    for name in ("generic", "food_delivery_demo"):
        assert load_profile(name).baseline_sentiment is None
    assert load_profile("apparel_ecommerce").baseline_sentiment == BaselineSentiment(-0.1, 0.65, 0.1)


def test_default_rule_is_unchanged_without_calibration():
    labeller = BaselineLabeller(load_profile("food_delivery_demo"))
    label = labeller.label_one("The food was great. My order was late again.")
    assert label["sentiment"] == "Neutral"  # complaint + positive wording -> Neutral (default rule)


def test_calibrated_rule_counts_complaints_on_the_final_label():
    calibration = BaselineSentiment(negative_max=-0.1, positive_min=0.65, complaint_penalty=0.1)
    labeller = BaselineLabeller(APPAREL, calibration=calibration)
    text = "Love the colour. It runs small. The fabric is see-through."
    label = labeller.label_one(text)
    complaints = sum(a["polarity"] == "negative" for a in label["aspects"])
    score = labeller._vader.polarity_scores(text)["compound"] - 0.1 * complaints
    expected = sentiment_from_score(pd.Series([score]), -0.1, 0.65).iloc[0]
    assert label["sentiment"] == expected
    assert BaselineLabeller(APPAREL).label_one(text)["aspects"] == label["aspects"]  # only sentiment changes


APPAREL_ROWS = ",Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name,Class Name"


def small_apparel_csv(path, n=120):
    texts = {1: "Terrible, it runs small and the fabric is cheap. Returning it.",
             3: "It is okay, nothing special.",
             5: "Love it, beautiful and so soft!"}
    lines = [APPAREL_ROWS]
    for i in range(n):
        rating = (1, 3, 5)[i % 3]
        lines.append(f'{i},{100 + i},35,,"{texts[rating]} #{i}",{rating},1,0,General,Tops,Knits')
    path.write_text("\n".join(lines))


def test_script_refuses_without_exclusion_file(tmp_path, capsys):
    data = tmp_path / "reviews.csv"
    small_apparel_csv(data)
    code = calibrate_baseline.main(["--data", str(data), "--exclude", str(tmp_path / "missing.csv"),
                                    "--out-dir", str(tmp_path / "out")])
    assert code == 1 and "Refusing to run" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_script_excludes_hand_labels_and_writes_aggregates_only(tmp_path):
    data = tmp_path / "reviews.csv"
    small_apparel_csv(data)
    exclude = tmp_path / "hand.csv"
    pd.DataFrame({"review_id": ["0", "1", "2", "3"]}).to_csv(exclude, index=False)
    out = tmp_path / "out"
    assert calibrate_baseline.main(["--data", str(data), "--exclude", str(exclude), "--out-dir", str(out)]) == 0
    result = json.loads((out / "BASELINE_CALIBRATION_apparel_ecommerce.json").read_text())
    split = result["meta"]["split"]
    assert split["excluded_rows"] == 4 and split["tune"] + split["test"] == 116
    assert result["meta"]["consistency_check"]["mismatches"] == 0
    report = (out / "BASELINE_CALIBRATION_apparel_ecommerce.md").read_text()
    assert "Terrible" not in report and "Love it" not in report  # no review text written
    assert "agreement with ratings" in report


def test_explicit_none_means_uncalibrated_even_when_profile_is_calibrated():
    assert BaselineLabeller(APPAREL).calibration == BaselineSentiment(-0.1, 0.65, 0.1)
    assert BaselineLabeller(APPAREL, calibration=None).calibration is None
    other = BaselineSentiment(0.0, 0.5, 0.2)
    assert BaselineLabeller(APPAREL, calibration=other).calibration == other
