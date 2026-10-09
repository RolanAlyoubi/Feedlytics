"""Feedlytics dashboard: one page over the existing pipeline (no AI, no new analytics).

    streamlit run app/dashboard.py --browser.gatherUsageStats false

Every number comes from ``run_analysis()`` tables (non-AI baseline labelling)
and ``app.insights``. Uploaded files are analysed in this local process only;
nothing is saved or sent anywhere.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run app/dashboard.py` puts app/ first, not the repo root
    sys.path.insert(0, str(ROOT))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.data_processing import DataValidationError  # noqa: E402
from app.insights import generate_insights  # noqa: E402
from app.pipeline import AnalysisResult, PipelineError, Table, run_analysis  # noqa: E402
from app.profiles import ProfileError  # noqa: E402

APPAREL = ROOT / "data" / "raw" / "reviews.csv"
CAVEAT = (
    "**Method: non-AI baseline, not AI.** Topics, issues, praise and flags come from VADER sentiment "
    "plus keyword rules. On 200 hand-labelled apparel reviews this baseline reached roughly 42% text-"
    "sentiment accuracy and 50% main-topic accuracy (`evaluation/RESULTS_apparel_ecommerce.md`), so "
    "treat label-based figures as indicative. Overall sentiment uses the star rating whenever a review "
    "has one. No AI service is called and uploaded data stays on this machine."
)


@st.cache_data(max_entries=4, show_spinner=False)
def analyse(key: str, _source) -> AnalysisResult:
    """Baseline analysis cached per dataset ``key`` (``_source`` is not hashed); each rerun gets a copy."""
    return run_analysis(_source, labelling="baseline")


def load(key: str, source) -> Tuple[Optional[AnalysisResult], Optional[str]]:
    """(result, None), or (None, user-facing error) for files the pipeline rejects."""
    try:
        return analyse(key, source), None
    except (DataValidationError, PipelineError, ProfileError) as exc:
        return None, f"Cannot analyse this file: {exc}"


def choose_source(apparel: Path = APPAREL):
    """Returns (cache key, source), or (None, None) when there is nothing to analyse yet.
    Opens on the apparel reviews; if that file is missing, says so and offers the upload."""
    options = {"Apparel reviews (Kaggle, local file)": apparel, "Upload a CSV": None}
    choice = st.sidebar.radio("Data", list(options))
    path = options[choice]
    if path is not None and path.exists():
        return f"{path}:{path.stat().st_mtime_ns}", str(path)
    if path is not None:
        st.warning(f"The apparel dataset was not found at `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`. "
                   "Download the Kaggle 'Women's E-Commerce Clothing Reviews' CSV to that path, "
                   "or upload a CSV in the sidebar.")
    upload = st.sidebar.file_uploader("CSV file (a review text column is required)", type="csv")
    if upload is None:
        return None, None
    data = upload.getvalue()
    return "upload:" + hashlib.sha256(data).hexdigest(), data


def apply_filter(result: AnalysisResult) -> AnalysisResult:
    """One segment filter; .filtered() recomputes every table and KPI for the subset."""
    segments = [(col, label) for col, label in result.roles.segments if col in result.data.columns]
    if not segments:
        return result
    labels = {label: col for col, label in segments}
    label = st.sidebar.selectbox("Filter by segment", ["All reviews"] + list(labels))
    if label == "All reviews":
        return result
    col = labels[label]
    values = sorted(result.data[col].dropna().unique(), key=str)
    chosen = st.sidebar.multiselect(label, values, format_func=str)
    return result.filtered({col: chosen}) if chosen else result


def usable(table: Optional[Table]) -> bool:
    """Show a short reason and return False when a table cannot be displayed."""
    if table is None or not table.available:
        st.caption(f"Not available: {table.reason if table is not None else 'not computed'}")
        return False
    if table.df.empty:
        st.caption("No rows for this selection.")
        return False
    return True


def show_table(table: Optional[Table], caption: str = "") -> bool:
    if not usable(table):
        return False
    st.dataframe(table.df, hide_index=True)
    notes = [caption, table.note, f"{table.n_reviews:,} reviews" + (", weighted" if table.weighted else "")]
    st.caption(" · ".join(n for n in notes if n))
    return True


def bar_chart(table: Optional[Table], x: str, y: str, y_label: str) -> None:
    if usable(table):
        st.plotly_chart(px.bar(table.df, x=x, y=y, labels={y: y_label}))
        if table.note:
            st.caption(table.note)


def pct(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def render(key: Optional[str], source) -> None:
    """Everything below the title for one data source."""
    if key is None:
        st.write("Upload a CSV to analyse it. Only a review text column is required.")
        return
    with st.spinner("Analysing reviews…"):
        result, error = load(key, source)
    if error:
        st.error(error)
        return
    view = apply_filter(result)
    filter_text = "; ".join(f"{k} = {', '.join(map(str, v))}" for k, v in view.filters.items())
    st.caption(f"Profile: {result.profile.display_name} · labels: non-AI baseline"
               + (f" · **filtered:** {filter_text} ({len(view.data):,} of {len(result.data):,} reviews)"
                  if view.filters else ""))

    k = view.kpis["primary"]
    cols = st.columns(5)
    cols[0].metric("Reviews", f"{k['total_reviews']:,}")
    cols[1].metric("Average rating", "n/a" if k["avg_rating"] is None else f"{k['avg_rating']:.2f}")
    cols[2].metric("Negative (overall sentiment)", pct(k["pct_negative"]))
    cols[3].metric("Low ratings", pct(k["pct_low_rating"]) if k["rated_reviews"] else "n/a")
    cols[4].metric("Text-labelled reviews", f"{view.kpis['labelled_reviews']:,}")

    st.header("Insights")
    st.caption("Rule-based: every sentence is filled from one table below. Associations, not causes. "
               "⚠️ marks small samples.")
    insights = generate_insights(view)
    if not insights:
        st.write("No findings for this selection.")
    for item in insights:
        icon = "➜" if item.kind == "recommendation" else "•"
        warning = " ⚠️" if item.small_sample else ""
        st.markdown(f"{icon} **{item.title}**{warning}  \n{item.detail}")
        st.caption(f"Source: `{item.source_table}` · {item.basis}")

    st.header("Ratings and sentiment")
    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        bar_chart(view.tables.get("rating_distribution"), "rating", "reviews", "Reviews")
    with right:
        st.subheader("Overall sentiment")
        bar_chart(view.tables.get("sentiment_primary"), "sentiment", "pct", "% of reviews")

    st.header("Issue priority")
    priority = view.tables.get("issue_priority")
    if show_table(priority, "Priority = frequency × severity, relative to the top issue"):
        chart = priority.df.dropna(subset=["frequency_pct", "severity"])
        if not chart.empty:
            st.plotly_chart(px.scatter(
                chart, x="frequency_pct", y="severity", size="reviews", color="priority", hover_name="group",
                labels={"frequency_pct": "Frequency (% of reviews)", "severity": "Severity (0–1)"},
                title="Priority matrix"))

    left, right = st.columns(2)
    with left:
        st.header("Praise")
        show_table(view.tables.get("praise"))
    with right:
        st.header("Flags")
        show_table(view.tables.get("flags"))

    with st.expander("Data quality (whole file, before any segment filter)"):
        report = result.report
        st.write(f"{report.rows_received:,} rows received, {report.rows_retained:,} kept "
                 f"({report.rows_removed:,} removed).")
        st.dataframe(report.summary_table(), hide_index=True)
        for message in report.warnings + report.notes:
            st.caption(message)


def main(apparel: Path = APPAREL) -> None:
    st.set_page_config(page_title="Feedlytics", layout="wide")
    st.title("Feedlytics: customer feedback insights")
    st.info(CAVEAT)
    render(*choose_source(apparel))


if __name__ == "__main__":  # `streamlit run` executes the file as __main__
    main()
