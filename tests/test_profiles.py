"""Profiles: bundled files are valid, broken ones are rejected clearly, detection works."""

import copy
import json

import pytest

from app.profiles import (ProfileError, available_profiles, detect_profile, load_profile,
                          parse_profile)

KAGGLE_HEADERS = ["", "Clothing ID", "Age", "Title", "Review Text", "Rating", "Recommended IND",
                  "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]


@pytest.fixture
def minimal():
    return {
        "name": "test",
        "taxonomy": {"Product": {"Broken": "it broke"}, "Other": {"Other": "anything else"}},
    }


def test_bundled_profiles_load():
    assert {"generic", "apparel_ecommerce", "food_delivery_demo"} <= set(available_profiles())
    for name in available_profiles():
        profile = load_profile(name)
        assert profile.taxonomy.issue_to_topic["Other"] == "Other"
        assert len(profile.fingerprint) == 12


def test_apparel_taxonomy_matches_approved_design():
    profile = load_profile("apparel_ecommerce")
    assert profile.taxonomy.topic_names == [
        "Fit & Sizing", "Fabric & Comfort", "Quality & Durability", "Style & Design",
        "Appearance vs Listing", "Price & Value", "Order & Service", "Other"]
    assert len(profile.taxonomy.issue_names) == 25  # 24 complaint issues + Other
    assert set(profile.taxonomy.flags) == {"mentions_return", "mentions_repurchase"}


def test_generic_is_default():
    assert load_profile().name == "generic"
    assert load_profile(None).name == "generic"


def test_unknown_profile_lists_available():
    with pytest.raises(ProfileError, match="Available"):
        load_profile("does_not_exist")


def test_load_by_path(tmp_path, minimal):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(minimal))
    assert load_profile(path).name == "test"


def test_invalid_json_is_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ProfileError, match="not valid JSON"):
        load_profile(path)


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.pop("taxonomy"), "missing required key 'taxonomy'"),
    (lambda p: p["taxonomy"]["Product"].update({"Other": "dup"}), "appears in both"),
    (lambda p: p["taxonomy"].pop("Other"), "must include topic 'Other'"),
    (lambda p: p["taxonomy"].update({"Empty": {}}), "has no issues"),
    (lambda p: p.update({"rating_scale": {"min": 1, "max": 5, "negative_max": 4, "positive_min": 3}}),
     "rating_scale"),
    (lambda p: p.update({"numeric": [{"column": "x", "bins": [0, 1, 2], "band_labels": ["a"]}]}),
     "band_labels"),
    (lambda p: p.update({"numeric": [{"column": "x", "bins": [0, 2, 1], "band_labels": ["a", "b"]}]}),
     "increasing"),
    (lambda p: p.update({"baseline_keywords": {"issues": {"Nope": ["x"]}}}), "not in the taxonomy"),
    (lambda p: p.update({"outcome_flags": {"Bad Name": "x"}}), "snake_case"),
    (lambda p: p.update({"segments": [{"label": "no column"}]}), "'column' or 'candidates'"),
])
def test_broken_profiles_are_rejected(minimal, mutate, message):
    broken = copy.deepcopy(minimal)
    mutate(broken)
    with pytest.raises(ProfileError, match=message):
        parse_profile(broken)


def test_detects_apparel_from_kaggle_headers():
    profile, reason = detect_profile(KAGGLE_HEADERS)
    assert profile.name == "apparel_ecommerce"
    assert "clothing_id" in reason


def test_detects_food_demo_from_its_columns():
    profile, _ = detect_profile(["review_text", "rating", "restaurant", "delivery_time"])
    assert profile.name == "food_delivery_demo"


@pytest.mark.parametrize("headers", [
    ["review_text", "rating"],
    ["comment", "score", "date", "category"],  # generic words only
])
def test_falls_back_to_generic(headers):
    profile, reason = detect_profile(headers)
    assert profile.name == "generic"
    assert "generic" in reason


def test_tie_falls_back_to_generic():
    profile, reason = detect_profile(["restaurant", "clothing_id"])
    assert profile.name == "generic" and "equally" in reason


def test_generic_roles_use_candidate_columns():
    roles = load_profile("generic").resolve_roles(["review_text", "product", "channel", "region"])
    assert roles.entity == "product" and roles.entity_label == "Product"
    assert [c for c, _ in roles.segments] == ["channel", "region"]
    assert not roles.has_rating


def test_rating_scale_sentiment():
    scale = load_profile("apparel_ecommerce").rating_scale
    assert [scale.sentiment(r) for r in (1, 2, 3, 4, 5)] == \
        ["Negative", "Negative", "Neutral", "Positive", "Positive"]
    assert scale.sentiment(float("nan")) is None
