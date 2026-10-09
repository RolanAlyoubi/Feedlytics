"""Offline smoke tests for the Streamlit dashboard (Streamlit's AppTest; no browser,
no network, no AI). Uses the synthetic demo file and invented CSV bytes."""

from streamlit.testing.v1 import AppTest


def _page_without_apparel_file():
    from pathlib import Path
    from app.dashboard import main
    main(apparel=Path("/nonexistent/reviews.csv"))  # never reads the real dataset in tests


def _invalid_upload():
    from app.dashboard import render
    render("upload:invalid", b"name,score\nx,1\ny,2\n")  # no review text column


def _invented_apparel_upload():
    from app.dashboard import render
    header = ",Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count," \
             "Division Name,Department Name,Class Name"
    rows = [f'{i},{100 + i % 5},35,,"{text} #{i}",{rating},1,0,General,{dept},Knits'  # unique: no re-submissions
            for i, (text, rating, dept) in enumerate(
                [("It runs small, sadly", 1, "A")] * 30 + [("Lovely and soft", 5, "B")] * 30)]
    render("upload:invented", "\n".join([header] + rows).encode("utf-8"))


def test_opens_on_apparel_and_explains_when_the_file_is_missing():
    at = AppTest.from_function(_page_without_apparel_file, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.radio[0].value.startswith("Apparel reviews")
    assert "apparel dataset was not found" in at.warning[0].value
    assert len(at.metric) == 0 and not at.error  # no silent fallback to the food-delivery demo


def _page_with_synthetic_file():
    from app.dashboard import ROOT, main
    # The synthetic file stands in for the apparel dataset, so the full page renders offline.
    main(apparel=ROOT / "data" / "sample_reviews_SYNTHETIC.csv")


def test_sidebar_offers_only_apparel_and_upload():
    at = AppTest.from_function(_page_without_apparel_file, default_timeout=60).run()
    assert at.sidebar.radio[0].options == ["Apparel reviews (Kaggle, local file)", "Upload a CSV"]


def test_full_page_renders():
    at = AppTest.from_function(_page_with_synthetic_file, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("Feedlytics")
    assert "non-AI baseline" in at.info[0].value
    assert len(at.metric) == 5
    assert "Insights" in [h.value for h in at.header]
    assert not at.error


def test_invalid_csv_shows_an_error_instead_of_crashing():
    at = AppTest.from_function(_invalid_upload, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 1 and at.error[0].value.startswith("Cannot analyse this file")
    assert len(at.metric) == 0


def test_uploaded_bytes_render_and_segment_filter_updates_the_page():
    at = AppTest.from_function(_invented_apparel_upload, default_timeout=60).run()
    assert not at.exception and not at.error
    assert at.metric[0].value == "60"
    at.sidebar.selectbox[0].set_value("Department").run()
    at.sidebar.multiselect[0].set_value(["A"]).run()
    assert not at.exception
    assert at.metric[0].value == "30"
    assert any("filtered" in c.value for c in at.caption)
