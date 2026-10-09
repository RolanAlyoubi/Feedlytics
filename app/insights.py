"""Rule-based findings and recommendations, built only from AnalysisResult tables.

No AI and no new statistics: every sentence is a template filled with values
that already exist in one named table, so each item can be traced back to its
source. Findings describe associations in the data, never causes.

    from app.insights import generate_insights
    for item in generate_insights(result):
        print(item.kind, item.title, item.detail, item.source_table)

Small samples: an item is flagged when the statistic rests on fewer than
``config.MIN_GROUP_SIZE`` reviews: the group's reviews for issues and segments,
the labelled reviews for praise and return shares. A missing count counts as small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from app import config
from app.pipeline import AnalysisResult, Table

TOP_N = 3
RETURN_FLAG = "mentions_return"
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Insufficient data": 3}


@dataclass
class Insight:
    kind: str            # "finding" | "recommendation"
    title: str
    detail: str
    source_table: str    # the AnalysisResult table every number comes from
    basis: str           # what the numbers are based on (text labels or star ratings)
    small_sample: bool = False


def label_basis(result: AnalysisResult) -> str:
    """How the text labels behind issue, praise and flag findings were made."""
    method = result.method
    if method.is_ai:
        note = f"AI labels ({method.model})"
        if method.mixed:
            note += f"; {method.fallback_rows} rows fell back to the non-AI baseline"
        return note
    return "Non-AI baseline labels (VADER + keywords), not AI"


def generate_insights(result: AnalysisResult) -> List[Insight]:
    """Findings (issues, low-rating segments, praise, returns), then recommendations.
    Tables that are missing, unavailable or empty are skipped."""
    issues = _sorted_issues(result)
    return (_top_issues(result, issues) + _low_rating_segments(result) + _praise(result)
            + _return_rate(result) + _recommendations(result, issues))


# ---------------------------------------------------------------------------

def _table(result: AnalysisResult, name: str) -> Optional[Table]:
    table = result.tables.get(name)
    return table if table is not None and table.available and not table.df.empty else None


def _number(value) -> Optional[float]:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _fmt(value) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:g}"


def _count(value) -> str:
    number = _number(value)
    return "unknown number of" if number is None else str(int(number))


def _is_small(count) -> bool:
    number = _number(count)
    return number is None or number < config.MIN_GROUP_SIZE


def _with_warning(detail: str, small: bool) -> str:
    if not small:
        return detail
    return (detail + f" Small sample (fewer than {config.MIN_GROUP_SIZE} reviews): "
            "treat as indicative only.")


def _share(table: Table) -> str:
    return "an estimated " if table.weighted else ""


def _sorted_issues(result: AnalysisResult) -> Optional[pd.DataFrame]:
    """issue_priority ordered by level, then index (highest first); missing values last."""
    table = _table(result, "issue_priority")
    if table is None:
        return None
    df = table.df.copy()
    df["_level"] = df["priority"].map(PRIORITY_ORDER).fillna(len(PRIORITY_ORDER))
    df["_index"] = pd.to_numeric(df["priority_index"], errors="coerce").fillna(-1)
    return df.sort_values(["_level", "_index"], ascending=[True, False], kind="stable")


def _top_issues(result: AnalysisResult, issues: Optional[pd.DataFrame]) -> List[Insight]:
    if issues is None:
        return []
    table = result.tables["issue_priority"]
    items = []
    for _, row in issues.head(TOP_N).iterrows():
        small = _is_small(row["reviews"])
        rating = "" if _number(row["avg_rating"]) is None else f", average rating {_fmt(row['avg_rating'])}"
        detail = (f"Mentioned in {_share(table)}{_fmt(row['frequency_pct'])}% of reviews "
                  f"({_count(row['reviews'])} reviews); {_fmt(row['pct_negative'])}% negative{rating}. "
                  f"Priority: {row['priority']}.")
        items.append(Insight("finding", f"Complaint: {row['group']}", _with_warning(detail, small),
                             "issue_priority", label_basis(result), small))
    return items


def _low_rating_segments(result: AnalysisResult) -> List[Insight]:
    table = _table(result, "low_rating_drivers")
    if table is None:
        return []
    df = table.df.copy()
    df["_lift"] = pd.to_numeric(df["lift"], errors="coerce")
    df = df[df["significant"].eq(True) & (df["_lift"] > 1)].sort_values("_lift", ascending=False)
    items = []
    for _, row in df.head(TOP_N).iterrows():
        small = _is_small(row["reviews"])
        q = _number(row["q_value"])
        # q-values are rounded to 4 decimals upstream, so 0 means "below 0.0001", not zero.
        q_text = "= n/a" if q is None else "< 0.0001" if q < 0.0001 else f"= {q:.3g}"
        detail = (f"{_fmt(row['pct_low_rating'])}% low ratings, {_fmt(row['lift'])}x the overall rate "
                  f"({_count(row['reviews'])} reviews; significant after multiple-testing correction, "
                  f"q {q_text}). This is an association, not a cause.")
        items.append(Insight("finding", f"Low ratings concentrated in {row['dimension']} = {row['group']}",
                             _with_warning(detail, small), "low_rating_drivers",
                             "Observed star ratings (no text labels)", small))
    return items


def _praise(result: AnalysisResult) -> List[Insight]:
    table = _table(result, "praise")
    if table is None:
        return []
    small = _is_small(table.n_reviews)
    df = table.df.assign(_pct=pd.to_numeric(table.df["est_pct_of_reviews"], errors="coerce"))
    items = []
    for _, row in df.sort_values("_pct", ascending=False).head(TOP_N).iterrows():
        detail = (f"Praised in {_share(table)}{_fmt(row['est_pct_of_reviews'])}% of reviews "
                  f"({_count(row['reviews'])} reviews).")
        items.append(Insight("finding", f"Praised: {row['topic']}", _with_warning(detail, small),
                             "praise", label_basis(result), small))
    return items


def _return_rate(result: AnalysisResult) -> List[Insight]:
    table = _table(result, "flags")
    if table is None:
        return []
    rows = table.df[table.df["flag"] == RETURN_FLAG]
    if rows.empty:
        return []
    row, small = rows.iloc[0], _is_small(table.n_reviews)
    detail = (f"{_share(table).capitalize()}{_fmt(row['est_pct_of_reviews'])}% of reviews mention "
              f"returning the item ({_count(row['reviews'])} reviews).")
    return [Insight("finding", "Returns mentioned", _with_warning(detail, small), "flags",
                    label_basis(result), small)]


def _recommendations(result: AnalysisResult, issues: Optional[pd.DataFrame]) -> List[Insight]:
    if issues is None:
        return []
    table = result.tables["issue_priority"]
    items = []
    for _, row in issues[issues["priority"] == "High"].iterrows():
        small = _is_small(row["reviews"])
        detail = (f"High priority (index {_fmt(row['priority_index'])} of 100): mentioned in "
                  f"{_share(table)}{_fmt(row['frequency_pct'])}% of reviews with severity "
                  f"{_fmt(row['severity'])}. Read a sample of these reviews to confirm the cause "
                  "before acting.")
        items.append(Insight("recommendation", f"Prioritise: {row['group']}", _with_warning(detail, small),
                             "issue_priority", label_basis(result), small))
    return items
