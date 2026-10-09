"""Descriptive analytics on cleaned feedback data (pandas only — no AI).

All functions take the cleaned DataFrame produced by
``data_processing.process_reviews`` and return plain DataFrames or dicts, so
they can be tested, printed in a notebook, or charted by the dashboard.
Functions return empty results (never crash) when a filter leaves no rows or
an optional column (rating, date, a segment) is absent.

Weights: when AI labels come from a sample that over-represents low ratings,
pass ``weight_col="sample_weight"``. Percentages and averages then estimate
the whole dataset rather than the sample. Review counts stay raw, so the
sample size behind every number is still visible.

Percentages are on a 0–100 scale and rounded to one decimal place.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app import config
from app.profiles import NumericSpec, RatingScale

DEFAULT_SENTIMENT_COL = "sentiment_from_rating"
DEFAULT_SCALE = RatingScale()


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return round(100.0 * numerator / denominator, 1) if denominator else None


def _pct_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (100.0 * numerator / denominator.replace(0, np.nan)).round(1)


def has_column(df: pd.DataFrame, col: str) -> bool:
    """True if the column exists and contains at least one value."""
    return col in df.columns and df[col].notna().any()


def _weights(df: pd.DataFrame, weight_col: Optional[str]) -> pd.Series:
    if weight_col and weight_col in df.columns:
        return df[weight_col].astype(float).fillna(0.0)
    return pd.Series(1.0, index=df.index)


def _column(df: pd.DataFrame, col: str, dtype=object) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index, dtype=dtype)


def _weighted_parts(df: pd.DataFrame, sentiment_col: str, weight_col: Optional[str],
                    scale: RatingScale, outcome_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Per-row weighted indicator columns; summing them by group gives every metric."""
    w = _weights(df, weight_col)
    rating = _column(df, "rating", float).astype(float)
    sentiment = _column(df, sentiment_col)
    parts = pd.DataFrame({
        "n": 1,
        "w": w,
        "w_rated": w * rating.notna(),
        "w_rating": w * rating.fillna(0.0),
        "w_low": w * (rating <= scale.negative_max),
        "w_sent": w * sentiment.notna(),
        "w_neg": w * (sentiment == "Negative"),
        "n_sent": sentiment.notna().astype(int),
        "w2_sent": (w ** 2) * sentiment.notna(),
    }, index=df.index)
    for col in outcome_cols:
        values = _column(df, col, float)
        parts[f"w_{col}_valid"] = w * values.notna()
        parts[f"w_{col}_yes"] = w * (values == 1)
    return parts


def _finish(sums: pd.DataFrame, outcome_cols: Sequence[str] = ()) -> pd.DataFrame:
    out = pd.DataFrame(index=sums.index)
    out["reviews"] = sums["n"].astype(int)
    out["avg_rating"] = (sums["w_rating"] / sums["w_rated"].replace(0, np.nan)).round(2)
    out["pct_negative"] = _pct_series(sums["w_neg"], sums["w_sent"])
    out["pct_low_rating"] = _pct_series(sums["w_low"], sums["w_rated"])
    for col in outcome_cols:
        out[f"pct_{col}"] = _pct_series(sums[f"w_{col}_yes"], sums[f"w_{col}_valid"])
    return out


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def overview_kpis(df: pd.DataFrame, sentiment_col: str = DEFAULT_SENTIMENT_COL,
                  weight_col: Optional[str] = None,
                  scale: RatingScale = DEFAULT_SCALE,
                  dimensions: Iterable[str] = ()) -> Dict:
    """Headline numbers for the dashboard's KPI row.

    ``dimensions`` (e.g. the entity column) adds distinct counts such as n_product_id.
    """
    parts = _weighted_parts(df, sentiment_col, weight_col, scale)
    total = parts.sum()
    rated = _column(df, "rating", float).notna()
    kpis = {
        "total_reviews": int(len(df)),
        "reviews_with_text": int(df["has_text"].sum()) if "has_text" in df.columns else None,
        "rated_reviews": int(rated.sum()),
        "avg_rating": (round(float(total["w_rating"] / total["w_rated"]), 2)
                       if total.get("w_rated", 0) else None),
        "pct_positive": _pct((parts["w"] * (_column(df, sentiment_col) == "Positive")).sum(), total["w_sent"]),
        "pct_neutral": _pct((parts["w"] * (_column(df, sentiment_col) == "Neutral")).sum(), total["w_sent"]),
        "pct_negative": _pct(total["w_neg"], total["w_sent"]),
        "pct_low_rating": _pct(total["w_low"], total["w_rated"]),
        "weighted": bool(weight_col and weight_col in df.columns),
        "date_min": None,
        "date_max": None,
    }
    if has_column(df, "date"):
        kpis["date_min"] = df["date"].min().date()
        kpis["date_max"] = df["date"].max().date()
    kpis.update(dimension_counts(df, dimensions))
    return kpis


def dimension_counts(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    """Number of distinct values per dimension, e.g. {'n_product_id': 1206}."""
    return {f"n_{c}": int(df[c].nunique()) for c in columns if has_column(df, c)}


def rating_distribution(df: pd.DataFrame, scale: RatingScale = DEFAULT_SCALE) -> pd.DataFrame:
    """Count and share of each rating value, including values with zero reviews."""
    stars = range(int(scale.min), int(scale.max) + 1)
    ratings = _column(df, "rating", float).dropna().astype(int)
    counts = ratings.value_counts().reindex(stars, fill_value=0)
    total = counts.sum()
    return pd.DataFrame({
        "rating": list(stars),
        "reviews": counts.values,
        "pct": [_pct(c, total) for c in counts.values],
    })


def sentiment_distribution(df: pd.DataFrame, sentiment_col: str = DEFAULT_SENTIMENT_COL,
                           weight_col: Optional[str] = None) -> pd.DataFrame:
    """Reviews (raw count) and share (weighted if weights are given) per sentiment."""
    sentiment = _column(df, sentiment_col)
    w = _weights(df, weight_col)
    counts = sentiment.value_counts().reindex(config.SENTIMENT_LABELS, fill_value=0)
    weighted = w.groupby(sentiment).sum().reindex(config.SENTIMENT_LABELS, fill_value=0)
    total = weighted.sum()
    return pd.DataFrame({
        "sentiment": list(config.SENTIMENT_LABELS),
        "reviews": counts.values.astype(int),
        "pct": [_pct(c, total) for c in weighted.values],
    })


# ---------------------------------------------------------------------------
# Time trends (need a date column)
# ---------------------------------------------------------------------------

def reviews_over_time(
    df: pd.DataFrame,
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
    min_count: int = config.MIN_GROUP_SIZE,
    weight_col: Optional[str] = None,
    scale: RatingScale = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Monthly volume, average rating and share of negative reviews."""
    columns = ["month", "reviews", "avg_rating", "pct_negative", "reliable"]
    if not has_column(df, "month"):
        return pd.DataFrame(columns=columns)
    dated = df.dropna(subset=["month"])
    parts = _weighted_parts(dated, sentiment_col, weight_col, scale)
    monthly = _finish(parts.groupby(dated["month"]).sum())
    # Include months with no reviews so gaps are visible rather than hidden.
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_range).rename_axis("month").reset_index()
    monthly["reviews"] = monthly["reviews"].fillna(0).astype(int)
    monthly["reliable"] = monthly["reviews"] >= min_count
    return monthly[columns]


def sentiment_over_time(df: pd.DataFrame, sentiment_col: str = DEFAULT_SENTIMENT_COL,
                        weight_col: Optional[str] = None) -> pd.DataFrame:
    """Monthly share of each sentiment (long format, for stacked charts)."""
    if not has_column(df, "month"):
        return pd.DataFrame(columns=["month", "sentiment", "reviews", "pct"])
    dated = df.dropna(subset=["month", sentiment_col])
    keys = [dated["month"], dated[sentiment_col]]
    counts = dated.groupby(keys).size().unstack(fill_value=0)
    weighted = _weights(dated, weight_col).groupby(keys).sum().unstack(fill_value=0)
    counts = counts.reindex(columns=config.SENTIMENT_LABELS, fill_value=0)
    weighted = weighted.reindex(columns=config.SENTIMENT_LABELS, fill_value=0)
    shares = weighted.div(weighted.sum(axis=1), axis=0).mul(100).round(1)
    out = counts.stack().rename("reviews").to_frame().join(shares.stack().rename("pct"))
    return out.rename_axis(["month", "sentiment"]).reset_index()


def recent_change(
    df: pd.DataFrame,
    group_col: str,
    months: int = 3,
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
    min_count: int = config.MIN_GROUP_SIZE,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """Is the negative-review rate of each group rising or falling?

    Compares the last ``months`` complete months in the data with the
    ``months`` before that. Change is in percentage points (pp). A direction
    is only reported when the change passes a two-proportion z-test. With
    weights, the test uses the Kish effective sample size.
    """
    columns = ["group", "previous_reviews", "previous_pct_negative", "recent_reviews",
               "recent_pct_negative", "change_pp", "p_value", "direction", "reliable"]
    if not (has_column(df, "month") and has_column(df, group_col)):
        return pd.DataFrame(columns=columns)

    last_month = df["month"].max()
    recent_start = last_month - pd.DateOffset(months=months - 1)
    previous_start = recent_start - pd.DateOffset(months=months)
    windows = {
        "recent": df[df["month"].between(recent_start, last_month)],
        "previous": df[df["month"].between(previous_start, recent_start - pd.DateOffset(days=1))],
    }
    stats = {}
    for name, window in windows.items():
        window = window.dropna(subset=[group_col])
        sums = _weighted_parts(window, sentiment_col, weight_col, DEFAULT_SCALE).groupby(window[group_col]).sum()
        stats[f"{name}_reviews"] = sums["n_sent"]
        stats[f"{name}_pct_negative"] = _pct_series(sums["w_neg"], sums["w_sent"])
        stats[f"{name}_n_eff"] = sums["w_sent"] ** 2 / sums["w2_sent"].replace(0, np.nan)

    out = pd.DataFrame(stats).rename_axis("group").reset_index()
    for col in ("recent_reviews", "previous_reviews"):
        out[col] = out[col].fillna(0).astype(int)
    out["change_pp"] = (out["recent_pct_negative"] - out["previous_pct_negative"]).round(1)
    out["p_value"] = [
        two_proportion_p_value(p1, n1, p2, n2)
        for p1, n1, p2, n2 in zip(out["previous_pct_negative"], out["previous_n_eff"],
                                  out["recent_pct_negative"], out["recent_n_eff"])
    ]
    out["reliable"] = (out["recent_reviews"] >= min_count) & (out["previous_reviews"] >= min_count)
    # Only call a trend when the change is both meaningful in size and
    # unlikely to be random variation between two small samples.
    real_change = out["reliable"] & (out["p_value"] < config.TREND_SIGNIFICANCE_LEVEL)
    out["direction"] = np.select(
        [~out["reliable"] | out["change_pp"].isna(),
         real_change & (out["change_pp"] >= config.TREND_THRESHOLD_PP),
         real_change & (out["change_pp"] <= -config.TREND_THRESHOLD_PP)],
        ["Not enough data", "Increasing", "Decreasing"],
        default="No clear change",
    )
    return out[columns].sort_values("change_pp", ascending=False, na_position="last").reset_index(drop=True)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """False-discovery-rate adjusted p-values (q-values) for many simultaneous tests."""
    p = p_values.astype(float)
    m = len(p)
    if m == 0:
        return p
    order = p.sort_values().index
    ranked = p[order] * m / np.arange(1, m + 1)
    q = ranked[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index).round(4)


def two_proportion_p_value(pct1: float, n1: float, pct2: float, n2: float) -> Optional[float]:
    """Two-sided p-value of a two-proportion z-test (normal approximation).

    Answers: if the true negative rate had not changed, how likely is a
    difference at least this large? Small values (< 0.05) suggest a real change.
    """
    if not n1 or not n2 or pd.isna(n1) or pd.isna(n2) or pd.isna(pct1) or pd.isna(pct2):
        return None
    p1, p2 = pct1 / 100, pct2 / 100
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p2 - p1) / se
    return round(math.erfc(z / math.sqrt(2)), 4)


# ---------------------------------------------------------------------------
# Group comparisons and drivers of low ratings
# ---------------------------------------------------------------------------

def group_summary(
    df: pd.DataFrame,
    by: str,
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
    min_count: int = config.MIN_GROUP_SIZE,
    weight_col: Optional[str] = None,
    scale: RatingScale = DEFAULT_SCALE,
    outcome_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Reviews, average rating, negative share (and outcome rates) per value of ``by``.

    ``reliable`` is False for groups smaller than ``min_count``; their numbers
    should be read with caution. ``outcome_cols`` adds e.g. ``pct_recommended``.
    """
    outcome_cols = [c for c in outcome_cols if c in df.columns]
    columns = (["group", "reviews", "pct_of_reviews", "avg_rating", "pct_negative", "pct_low_rating"]
               + [f"pct_{c}" for c in outcome_cols] + ["reliable"])
    if not has_column(df, by):
        return pd.DataFrame(columns=columns)
    data = df.dropna(subset=[by])
    sums = _weighted_parts(data, sentiment_col, weight_col, scale, outcome_cols).groupby(data[by]).sum()
    out = _finish(sums, outcome_cols)
    out["pct_of_reviews"] = _pct_series(sums["w"], pd.Series(sums["w"].sum(), index=sums.index))
    out["reliable"] = out["reviews"] >= min_count
    out = out.rename_axis("group").reset_index()
    return out[columns].sort_values("reviews", ascending=False).reset_index(drop=True)


def low_rating_drivers(
    df: pd.DataFrame,
    dimensions: Iterable[Union[str, Tuple[str, str]]] = (),
    min_count: int = config.MIN_GROUP_SIZE,
    scale: RatingScale = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Which groups have a higher share of low ratings than the data overall?

    ``dimensions``: column names (or (column, label) pairs) — normally the
    entity and segments from the profile's resolved roles.
    ``lift`` = group low-rating share / overall low-rating share.
    A lift of 1.5 means the group gets 50% more low ratings than average.
    ``significant`` = the group's share differs from the rest of the data in a
    two-proportion z-test, after a Benjamini–Hochberg correction because many
    groups are tested at once (``q_value`` < 0.05). Small groups can show
    large lifts by chance, so read lifts without significance with caution.
    ``reviews`` and the ``min_count`` rule count only reviews with a valid
    rating, so group sizes match the shares and the test.
    This shows association, not cause. Needs a rating column.
    """
    columns = ["dimension", "group", "reviews", "pct_low_rating", "lift", "p_value", "q_value",
               "significant"]
    rated = _column(df, "rating", float).dropna()
    rated_df = df.loc[rated.index]
    overall = (rated <= scale.negative_max).mean() * 100 if len(rated) else 0
    total_low, total_rated = (rated <= scale.negative_max).sum(), len(rated)
    frames: List[pd.DataFrame] = []
    for dim in dimensions:
        col = dim[0] if isinstance(dim, tuple) else dim
        summary = group_summary(rated_df, col, min_count=min_count, scale=scale)
        summary = summary[summary["reliable"]]
        if summary.empty:
            continue
        frames.append(summary.assign(dimension=col))
    if not frames or not overall:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out["lift"] = (out["pct_low_rating"] / overall).round(2)
    group_low = out["pct_low_rating"] / 100 * out["reviews"]
    rest_n = total_rated - out["reviews"]
    rest_pct = 100 * (total_low - group_low) / rest_n.replace(0, np.nan)
    out["p_value"] = [two_proportion_p_value(p1, n1, p2, n2) for p1, n1, p2, n2 in
                      zip(out["pct_low_rating"], out["reviews"], rest_pct, rest_n)]
    out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")
    out["q_value"] = benjamini_hochberg(out["p_value"].fillna(1.0))
    out["significant"] = out["q_value"] < config.TREND_SIGNIFICANCE_LEVEL
    return out[columns].sort_values("lift", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Numeric relationships (any numeric driver defined by the profile)
# ---------------------------------------------------------------------------

def numeric_band_summary(
    df: pd.DataFrame,
    col: str,
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    quantiles: int = 4,
    min_count: int = config.MIN_GROUP_SIZE,
    right: bool = True,
    scale: RatingScale = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Average rating and low-rating share across bands of a numeric column.

    Uses fixed ``bins`` if given, otherwise equal-sized quantile bands.
    """
    columns = ["band", "reviews", "avg_rating", "pct_low_rating", "reliable"]
    if not (has_column(df, col) and has_column(df, "rating")):
        return pd.DataFrame(columns=columns)
    data = df.dropna(subset=[col, "rating"])
    if bins is not None:
        bands = pd.cut(data[col], bins=list(bins), labels=list(labels) if labels else None,
                       include_lowest=True, right=right)
    else:
        bands = pd.qcut(data[col], q=quantiles, duplicates="drop")
        names = [f"{iv.left:,.0f}–{iv.right:,.0f}" for iv in bands.cat.categories]
        if len(set(names)) == len(names):  # keep interval text if rounding collides
            bands = bands.cat.rename_categories(names)
    grouped = data.groupby(bands, observed=False)["rating"]
    out = pd.DataFrame({
        "reviews": grouped.size(),
        "avg_rating": grouped.mean().round(2),
        "pct_low_rating": grouped.apply(
            lambda s: _pct((s <= scale.negative_max).sum(), len(s))),
    }).rename_axis("band").reset_index()
    out["band"] = out["band"].astype(str)
    out["reliable"] = out["reviews"] >= min_count
    return out[columns]


def band_summary(df: pd.DataFrame, spec: NumericSpec,
                 scale: RatingScale = DEFAULT_SCALE) -> pd.DataFrame:
    """numeric_band_summary using a profile's band definition (or quartiles)."""
    return numeric_band_summary(df, spec.column, bins=spec.bins, labels=spec.band_labels,
                                right=spec.closed == "right", scale=scale)


def correlation_with_rating(df: pd.DataFrame, col: str,
                            min_count: int = config.MIN_GROUP_SIZE) -> Optional[Dict]:
    """Spearman rank correlation between ``col`` and rating.

    Spearman is used because ratings are ordinal and the relationship need
    not be linear. Returns None if there is too little data or no rating.
    """
    if not (has_column(df, col) and has_column(df, "rating")):
        return None
    data = df[[col, "rating"]].dropna()
    if len(data) < min_count or data[col].nunique() < 2 or data["rating"].nunique() < 2:
        return None
    # Spearman = Pearson correlation of the ranks (avoids a scipy dependency).
    ranks = data.rank(method="average")
    rho = float(ranks[col].corr(ranks["rating"]))
    size = abs(rho)
    strength = ("negligible" if size < 0.1 else "weak" if size < 0.3
                else "moderate" if size < 0.5 else "strong")
    direction = "negative" if rho < 0 else "positive"
    return {"variable": col, "rho": round(rho, 3), "n": int(len(data)),
            "strength": strength, "direction": direction, "method": "spearman"}


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

def priority_table(
    df: pd.DataFrame,
    group_col: str,
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
    total_reviews: Optional[int] = None,
    min_count: int = config.MIN_GROUP_SIZE,
    weight_col: Optional[str] = None,
    total_weight: Optional[float] = None,
    scale: RatingScale = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Rank groups (e.g. complaint issues) by how much they hurt satisfaction.

    frequency  = reviews mentioning the group / total reviews
                 (weighted: estimated share of all reviews)
    severity   = share negative × rating gap, where rating gap = (max − avg) /
                 (max − min). Without ratings, the gap is 1 and severity is
                 just the share negative.                     (0 = mild, 1 = severe)
    score      = frequency × severity
    index      = score as a % of the highest score (0–100)
    level      = High ≥ 60, Medium ≥ 25, else Low; "Insufficient data" if
                 fewer than ``min_count`` reviews.

    ``total_reviews`` / ``total_weight`` give the true denominator when ``df``
    has one row per mention (a review can mention several issues).

    Note: "impact" here is a proxy built from frequency and severity. The data
    contains no revenue or churn figures, so business impact is not measured.
    """
    columns = ["group", "reviews", "frequency_pct", "avg_rating", "pct_negative",
               "severity", "priority_index", "priority"]
    summary = group_summary(df, group_col, sentiment_col=sentiment_col, min_count=min_count,
                            weight_col=weight_col, scale=scale)
    if summary.empty:
        return pd.DataFrame(columns=columns)
    out = summary.copy()
    if weight_col and weight_col in df.columns:
        data = df.dropna(subset=[group_col])
        mentions = _weights(data, weight_col).groupby(data[group_col]).sum()
        denominator = total_weight or _weights(df, weight_col).sum()
        out["frequency_pct"] = (100.0 * out["group"].map(mentions) / denominator).round(1)
    else:
        out["frequency_pct"] = (100.0 * out["reviews"] / (total_reviews or len(df))).round(1)
    rating_gap = ((scale.max - out["avg_rating"]) / (scale.max - scale.min)).clip(0, 1)
    out["severity"] = (out["pct_negative"].fillna(0) / 100 * rating_gap.fillna(1.0)).round(3)
    score = out["frequency_pct"] / 100 * out["severity"]
    eligible = out["reliable"]
    top = score[eligible].max() if eligible.any() else 0
    out["priority_index"] = (100 * score / top).round(0) if top else 0.0
    out["priority"] = np.select(
        [~eligible,
         out["priority_index"] >= config.PRIORITY_HIGH_MIN_INDEX,
         out["priority_index"] >= config.PRIORITY_MEDIUM_MIN_INDEX],
        ["Insufficient data", "High", "Medium"],
        default="Low",
    )
    out.loc[~eligible, "priority_index"] = np.nan
    order = {"High": 0, "Medium": 1, "Low": 2, "Insufficient data": 3}
    out = out.sort_values(["priority", "priority_index"],
                          key=lambda s: s.map(order) if s.name == "priority" else -s.fillna(-1))
    return out[columns].reset_index(drop=True)
