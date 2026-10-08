"""Pre-flight checks for a paid evaluation run. None of these tests call the API."""

from pathlib import Path

import pandas as pd
import pytest

from app.evaluation import validate_truth
from app.profiles import load_profile
from scripts import evaluate_labels
from scripts.check_ai_setup import describe_key

APPAREL = load_profile("apparel_ecommerce")
TOPICS = {t: list(i) for t, i in APPAREL.taxonomy.topics.items()}
FLAGS = list(APPAREL.taxonomy.flags)


def truth(rows):
    cols = ["review_id", "true_sentiment", "true_topic", "true_issue",
            "true_mentions_return", "true_mentions_repurchase"]
    return pd.DataFrame(rows, columns=cols)


def test_valid_labels_pass():
    result = validate_truth(truth([
        ["1", "Negative", "Fit & Sizing", "Runs small", "true", "false"],
        ["2", "Positive", "Fabric & Comfort", "", "false", "yes"],
        ["3", "", "", "", "", ""],  # not labelled yet: ignored
    ]), TOPICS, FLAGS, known_ids=["1", "2", "3"])
    assert result["errors"] == [] and result["labelled"] == 2 and result["rows"] == 3


def test_invalid_values_are_errors_with_hints():
    result = validate_truth(truth([
        ["1", "negative", "Fit & Sizing", "Runs small", "", ""],
        ["2", "Negative", "Fit and sizing", "runs small", "maybe", ""],
        ["3", "Neutral", "", "", "", ""],
    ]), TOPICS, FLAGS)
    text = "\n".join(result["errors"])
    assert "true_sentiment 'negative'" in text and "did you mean 'Negative'" in text
    assert "true_topic 'Fit and sizing'" in text
    assert "did you mean 'Runs small'" in text
    assert "true_mentions_return 'maybe'" in text
    assert "review 3: true_topic is empty" in text


def test_warnings_for_inconsistent_but_valid_labels():
    result = validate_truth(truth([
        ["1", "Negative", "Style & Design", "Runs small", "", ""],   # issue from another topic
        ["2", "Negative", "Fit & Sizing", "", "", ""],               # negative, no issue
        ["3", "", "Fit & Sizing", "Runs small", "", ""],             # partly labelled
    ]), TOPICS, FLAGS)
    assert result["errors"] == []
    text = "\n".join(result["warnings"])
    assert "belongs to topic 'Fit & Sizing'" in text
    assert "Negative but no true_issue" in text
    assert "will be skipped" in text


def test_duplicate_unknown_ids_and_missing_columns():
    result = validate_truth(truth([
        ["1", "Positive", "Other", "", "", ""],
        ["1", "Positive", "Other", "", "", ""],
        ["999", "Positive", "Other", "", "", ""],
    ]), TOPICS, FLAGS, known_ids=["1"])
    assert "review 1: appears more than once" in "\n".join(result["errors"])
    assert "review 999: not in the cleaned dataset" in "\n".join(result["warnings"])
    missing = validate_truth(pd.DataFrame({"id": ["1"], "label": ["x"]}), TOPICS)
    assert "missing column(s)" in missing["errors"][0]


@pytest.mark.parametrize("value, expected", [
    (None, "not set"),
    ("your-api-key-here", "placeholder"),
    ("sk-ant-" + "x" * 60, "looks like an Anthropic API key"),
    ("abc123", "does not look like"),
])
def test_key_description_never_reveals_the_key(value, expected):
    described = describe_key(value)
    assert expected in described
    if value and len(value) > 10:
        assert value not in described and value[7:20] not in described


@pytest.fixture
def synthetic_truth(tmp_path):
    source = pd.read_csv("data/sample_reviews_SYNTHETIC_ground_truth.csv", dtype=str, keep_default_na=False)
    return source.sample(40, random_state=0)


def test_evaluation_reads_spreadsheet_bom_files(tmp_path, synthetic_truth):
    path = tmp_path / "labeled.csv"
    synthetic_truth.to_csv(path, index=False, encoding="utf-8-sig")  # what Excel "CSV UTF-8" writes
    code = evaluate_labels.main(["--truth", str(path), "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / "RESULTS_food_delivery_demo.md").exists()


def test_evaluation_stops_on_invalid_labels(tmp_path, synthetic_truth, capsys):
    bad = synthetic_truth.copy()
    bad.iloc[0, bad.columns.get_loc("true_sentiment")] = "negativ"
    path = tmp_path / "labeled.csv"
    bad.to_csv(path, index=False)
    code = evaluate_labels.main(["--truth", str(path), "--out-dir", str(tmp_path / "out")])
    assert code == 1
    assert "negativ" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
