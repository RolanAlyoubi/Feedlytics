"""Hand-label evaluation: blind sample, label schema, metrics, no rating leakage.

All review texts are invented; nothing here touches the real labelling CSV.
"""

import pandas as pd
import pytest

from app.ai_analysis import BaselineLabeller, label_reviews
from app.data_processing import process_reviews
from app.evaluation import (binary_metrics, evaluate_labels, multilabel_metrics, parse_label_list,
                            population_weights, prepare_truth, validate_truth, weighted_accuracy)
from app.profiles import load_profile
from scripts import evaluate_labels as evaluate_cli
from scripts import make_labeling_sample

APPAREL = load_profile("apparel_ecommerce")
TOPICS = {t: list(i) for t, i in APPAREL.taxonomy.topics.items()}
FLAGS = list(APPAREL.taxonomy.flags)
HEADER = ",Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name,Class Name"


def apparel_file(path, n=30):
    texts = {1: "Terrible. It runs small and the fabric is see-through. Returning it.",
             3: "It is okay, nothing special.",
             5: "Love it, beautiful and so soft!"}
    lines = [HEADER] + [f'{i},{100 + i},35,Title {i},"{texts[(1, 3, 5)[i % 3]]} #{i}",{(1, 3, 5)[i % 3]},1,0,General,Tops,Knits'
                        for i in range(n)]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Label parsing and truth schemas
# ---------------------------------------------------------------------------

def test_parse_label_list():
    assert parse_label_list(" Runs small ;Thin or see-through; ;Runs small") == ["Runs small", "Thin or see-through"]
    assert parse_label_list("") == [] and parse_label_list(None) == [] and parse_label_list(float("nan")) == []


def test_prepare_truth_hand_label_schema():
    t = prepare_truth(pd.DataFrame({"review_id": ["1", "2"], "true_sentiment": ["Negative", "Positive"],
                                    "true_topic": ["Fit & Sizing", "Style & Design"],
                                    "true_issues": ["Runs small; Wrong length", ""],
                                    "true_praise_topics": ["", "Style & Design; Fabric & Comfort"]}))
    assert t["_issues"].tolist() == [["Runs small", "Wrong length"], []]
    assert t["_main_issue"].tolist() == ["Runs small", ""]
    assert t["_praise"].tolist() == [[], ["Style & Design", "Fabric & Comfort"]]


def test_prepare_truth_legacy_and_synthetic_schemas():
    legacy = prepare_truth(pd.DataFrame({"review_id": ["1"], "true_issue": ["Cold food"]}))
    assert legacy["_issues"].tolist() == [["Cold food"]] and legacy["_praise"].isna().all()
    synthetic = prepare_truth(pd.DataFrame({"review_id": ["1"], "true_issue": ["Late delivery"],
                                            "true_aspects": ["Delivery:negative:Late delivery|Food Quality:positive:"]}))
    assert synthetic["_issues"].tolist() == [["Late delivery"]]
    assert synthetic["_praise"].tolist() == [["Food Quality"]]


# ---------------------------------------------------------------------------
# Metrics, checked by hand
# ---------------------------------------------------------------------------

def test_binary_metrics_by_hand():
    m = binary_metrics(pd.Series([True, True, False, False]), pd.Series([True, False, True, False]))
    assert (m["accuracy"], m["precision"], m["recall"], m["f1"]) == (50.0, 0.5, 0.5, 0.5)
    assert m["true_positives_in_truth"] == 2 and m["predicted_positives"] == 2


def test_multilabel_metrics_by_hand():
    truth = [{"A", "B"}, set(), {"C"}]
    pred = [{"A"}, {"B"}, {"C"}]
    m = multilabel_metrics(truth, pred)
    # pairs: TP = A, C (2); FP = B in review 2 (1); FN = B in review 1 (1)
    assert (m["micro_precision"], m["micro_recall"], m["micro_f1"]) == (0.667, 0.667, 0.667)
    per = {r["label"]: r for r in m["per_label"]}
    assert per["A"]["f1"] == 1.0 and per["B"]["f1"] == 0.0 and per["C"]["f1"] == 1.0
    assert m["macro_f1"] == pytest.approx(0.667, abs=0.001)
    assert m["exact_match_pct"] == pytest.approx(33.3)
    assert m["true_labels"] == 3 and m["predicted_labels"] == 3


def test_population_weights_undo_oversampling():
    eval_strata = pd.Series(["Negative"] * 50 + ["Positive"] * 50, index=[str(i) for i in range(100)])
    population = pd.Series(["Negative"] * 100 + ["Positive"] * 900)
    w = population_weights(eval_strata, population)
    assert w.iloc[0] == 2.0 and w.iloc[-1] == 18.0
    correct = pd.Series([False] * 50 + [True] * 50, index=eval_strata.index)  # right only on positives
    assert correct.mean() == 0.5
    assert weighted_accuracy(correct, w) == 90.0  # positives are 90% of the population


# ---------------------------------------------------------------------------
# Evaluation: new metrics, and no rating leakage
# ---------------------------------------------------------------------------

def labelled_frame():
    return pd.DataFrame({
        "review_id": ["1", "2", "3"],
        "has_text": True,
        "sentiment_ai": ["Negative", "Positive", "Positive"],
        "sentiment": ["Positive", "Negative", "Negative"],   # rating-based: must be ignored
        "primary_topic_ai": ["Fit & Sizing", "Style & Design", "Fabric & Comfort"],
        "issue_ai": ["Runs small", None, None],
        "aspects_ai": [[{"topic": "Fit & Sizing", "polarity": "negative", "issue": "Runs small"}],
                       [{"topic": "Style & Design", "polarity": "positive", "issue": None}],
                       [{"topic": "Fabric & Comfort", "polarity": "positive", "issue": None}]],
    })


def hand_truth():
    return pd.DataFrame({
        "review_id": ["1", "2", "3"],
        "true_sentiment": ["Negative", "Positive", "Negative"],
        "true_topic": ["Fit & Sizing", "Style & Design", "Fabric & Comfort"],
        "true_issues": ["Runs small; Thin or see-through", "", "Thin or see-through"],
        "true_praise_topics": ["", "Style & Design", "Fabric & Comfort"],
    })


def test_evaluation_reports_issue_complaint_and_praise_metrics():
    r = evaluate_labels(labelled_frame(), hand_truth(), APPAREL.taxonomy.topic_names,
                        APPAREL.taxonomy.issue_to_topic)
    assert r["complaint_detection"]["recall"] == 0.5 and r["complaint_detection"]["precision"] == 1.0
    assert r["issues"]["micro_recall"] == pytest.approx(0.333, abs=0.001)
    assert r["issues"]["micro_precision"] == 1.0
    assert r["complaint_topics"]["micro_recall"] == pytest.approx(0.333, abs=0.001)
    assert r["praise_topics"]["micro_f1"] == 1.0
    assert r["primary_topic"]["accuracy"] == 100.0


def test_rating_based_sentiment_never_enters_the_score():
    """sentiment (rating-based) contradicts the truth on every row; sentiment_ai does not."""
    labelled = labelled_frame()
    r = evaluate_labels(labelled, hand_truth())
    assert r["sentiment"]["accuracy"] == pytest.approx(66.7)  # from sentiment_ai only
    labelled["sentiment"] = ["Negative", "Positive", "Negative"]  # now matching the truth
    assert evaluate_labels(labelled, hand_truth())["sentiment"]["accuracy"] == pytest.approx(66.7)


def test_population_weighted_block_appears_with_weights():
    w = pd.Series({"1": 1.0, "2": 10.0, "3": 1.0})
    r = evaluate_labels(labelled_frame(), hand_truth(), weights=w)
    # correct on 1 and 2: (1 + 10) / 12
    assert r["population_weighted"]["sentiment_accuracy"] == pytest.approx(91.7)


# ---------------------------------------------------------------------------
# Validation of the hand-label schema
# ---------------------------------------------------------------------------

def test_validate_hand_label_schema():
    truth = pd.DataFrame({
        "review_id": ["1", "2", "3", "4"],
        "true_sentiment": ["Negative", "Positive", "Negative", ""],
        "true_topic": ["Fit & Sizing", "Style & Design", "Style & Design", ""],
        "true_issues": ["Runs small; runs large", "", "Runs small", ""],
        "true_praise_topics": ["", "style & design", "", "Fit & Sizing"],
        "true_mentions_return": ["true", "false", "", ""],
        "true_mentions_repurchase": ["", "", "", ""],
    })
    result = validate_truth(truth, TOPICS, FLAGS, known_ids=["1", "2", "3", "4"])
    errors, warnings = "\n".join(result["errors"]), "\n".join(result["warnings"])
    assert "true_issues 'runs large'" in errors and "did you mean 'Runs large'" in errors
    assert "true_praise_topics 'style & design'" in errors
    assert "issue 'Runs small' belongs to topic 'Fit & Sizing', not 'Style & Design'" in warnings
    assert "review 4: has a topic/issue but no true_sentiment" in warnings
    assert result["labelled"] == 3


def test_legacy_single_issue_column_rejects_lists():
    truth = pd.DataFrame({"review_id": ["1"], "true_sentiment": ["Negative"], "true_topic": ["Fit & Sizing"],
                          "true_issue": ["Runs small; Wrong length"]})
    assert "use true_issues for several" in "\n".join(validate_truth(truth, TOPICS)["errors"])


def test_missing_issue_column_is_reported():
    truth = pd.DataFrame({"review_id": ["1"], "true_sentiment": ["Negative"], "true_topic": ["Other"]})
    assert "true_issues or true_issue" in validate_truth(truth, TOPICS)["errors"][0]


# ---------------------------------------------------------------------------
# Blind sample creation and the guide
# ---------------------------------------------------------------------------

def test_sample_is_blind_and_protected(tmp_path):
    data = apparel_file(tmp_path / "reviews.csv")
    out = tmp_path / "labeling"
    assert make_labeling_sample.main(["--data", str(data), "--n", "12", "--out-dir", str(out)]) == 0
    sample = pd.read_csv(out / "apparel_ecommerce_to_label.csv", dtype=str, keep_default_na=False)
    assert "rating" not in sample.columns
    assert list(sample.columns) == ["review_id", "review_title", "review_text", "true_sentiment", "true_topic",
                                    "true_issues", "true_praise_topics", "true_mentions_return",
                                    "true_mentions_repurchase"]
    assert len(sample) == 12
    # No overwrite without --force.
    assert make_labeling_sample.main(["--data", str(data), "--n", "12", "--out-dir", str(out)]) == 1
    # --force re-uses exactly the same ids in the same order.
    assert make_labeling_sample.main(["--data", str(data), "--force", "--out-dir", str(out)]) == 0
    again = pd.read_csv(out / "apparel_ecommerce_to_label.csv", dtype=str)
    assert again["review_id"].tolist() == sample["review_id"].tolist()
    # Never overwrite a file that contains labels, even with --force.
    sample.loc[0, "true_sentiment"] = "Positive"
    sample.to_csv(out / "apparel_ecommerce_to_label.csv", index=False)
    assert make_labeling_sample.main(["--data", str(data), "--force", "--out-dir", str(out)]) == 1
    kept = pd.read_csv(out / "apparel_ecommerce_to_label.csv", dtype=str, keep_default_na=False)
    assert kept.loc[0, "true_sentiment"] == "Positive"


def test_guide_covers_every_column_and_allowed_value():
    guide = make_labeling_sample.guide_text(APPAREL)
    for column in ("true_sentiment", "true_topic", "true_issues", "true_praise_topics",
                   "true_mentions_return", "true_mentions_repurchase"):
        assert f"`{column}`" in guide
    for issue in APPAREL.taxonomy.issue_names:
        assert f"`{issue}`" in guide
    assert "rating is hidden" in guide.lower() and "--check-only" in guide
    assert "apparel_ecommerce_labeled.csv" in guide


# ---------------------------------------------------------------------------
# CLI: check-only and a full evaluation with the hand-label schema
# ---------------------------------------------------------------------------

def labelled_copy(tmp_path, data):
    out = tmp_path / "labeling"
    make_labeling_sample.main(["--data", str(data), "--n", "12", "--out-dir", str(out)])
    truth = pd.read_csv(out / "apparel_ecommerce_to_label.csv", dtype=str, keep_default_na=False)
    is_negative = truth["review_text"].str.startswith("Terrible")
    truth["true_sentiment"] = is_negative.map({True: "Negative", False: "Positive"})
    truth["true_topic"] = is_negative.map({True: "Fit & Sizing", False: "Fabric & Comfort"})
    truth["true_issues"] = is_negative.map({True: "Runs small; Thin or see-through", False: ""})
    truth["true_praise_topics"] = is_negative.map({True: "", False: "Fabric & Comfort"})
    truth["true_mentions_return"] = is_negative.map({True: "true", False: "false"})
    truth["true_mentions_repurchase"] = "false"
    path = out / "apparel_ecommerce_labeled.csv"
    truth.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_check_only_writes_nothing_and_reports_problems(tmp_path, capsys):
    data = apparel_file(tmp_path / "reviews.csv")
    truth = labelled_copy(tmp_path, data)
    out_dir = tmp_path / "eval"
    args = ["--data", str(data), "--truth", str(truth), "--check-only", "--out-dir", str(out_dir)]
    assert evaluate_cli.main(args) == 0
    assert "Labelled: 12 of 12" in capsys.readouterr().out
    bad = pd.read_csv(truth, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    bad.loc[0, "true_sentiment"] = "positive"
    bad.to_csv(truth, index=False)
    assert evaluate_cli.main(args) == 1
    assert "did you mean 'Positive'" in capsys.readouterr().out
    assert not out_dir.exists()


def test_full_evaluation_with_hand_labels(tmp_path):
    data = apparel_file(tmp_path / "reviews.csv")
    truth = labelled_copy(tmp_path, data)
    out_dir = tmp_path / "eval"
    assert evaluate_cli.main(["--data", str(data), "--truth", str(truth), "--out-dir", str(out_dir)]) == 0
    report = (out_dir / "RESULTS_apparel_ecommerce.md").read_text()
    for section in ("blind to the rating", "Population-weighted accuracy", "sentiment precision / recall / F1",
                    "Complaint detection", "Issue detection (multi-label)", "Complaint topics",
                    "Praised topics", "per issue", "outcome flags", "Baseline: VADER + keywords (calibrated)",
                    "Claude (AI) was not evaluated"):
        assert section in report
    assert "Terrible" not in report and "Love it" not in report  # no review text in the report


def test_rating_rule_is_scored_only_on_rated_reviews(tmp_path):
    data = tmp_path / "reviews.csv"
    data.write_text("\n".join([HEADER, '0,1,35,,"Awful, it runs small.",1,0,0,General,Tops,Knits',
                               '1,2,35,,"Lovely and soft.",,1,0,General,Tops,Knits']))
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"review_id": ["0", "1"], "true_sentiment": ["Negative", "Positive"],
                  "true_topic": ["Fit & Sizing", "Fabric & Comfort"], "true_issues": ["Runs small", ""],
                  "true_praise_topics": ["", "Fabric & Comfort"]}).to_csv(truth, index=False)
    out_dir = tmp_path / "eval"
    assert evaluate_cli.main(["--data", str(data), "--truth", str(truth), "--out-dir", str(out_dir)]) == 0
    import json
    rating = json.loads((out_dir / "results" / "apparel_ecommerce_rating_rule__no_text.json").read_text())
    assert rating["n_reviews"] == 1 and rating["sentiment"]["accuracy"] == 100.0
