"""End-to-end analysis: one call from a raw dataset to every analytics table.

    result = run_analysis("data/raw/reviews.csv")             # baseline labels, no API
    result = run_analysis(df, labelling="ai")                 # Claude on a weighted sample
    subset = result.filtered({"department": "Dresses"})       # same rules, same weights

Why a single entry point: the rules that make the numbers trustworthy are
applied here once, so callers (scripts, the future dashboard) cannot mix them up:

* Three sentiment columns are kept apart and never overwritten:
    sentiment_from_rating   rating only (profile scale)
    sentiment_ai            text only (label_source says ai / cache / baseline / baseline_fallback)
    sentiment + sentiment_source   the primary value: rating when usable, else text
* Two weight columns, each with one job:
    sample_weight     text-label analytics; set only on labelled rows; sums to the
                      number of reviews with text (checked, fail-fast)
    sentiment_weight  primary sentiment; 1 for rating-sourced rows (ratings cover
                      everyone), the sample weight for text-sourced rows, blank otherwise
* Every table is computed here with the right weights and carries metadata
  (basis, weighted, sentiment column, raw review count). No table exposes
  unweighted shares of an over-sampled sample.
* ``filtered()`` keeps the original sampling design: rows keep their original
  weights; the subset is never re-sampled or re-normalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from app import analytics, config
from app.ai_analysis import (AILabeller, BaselineLabeller, LabelCache, LabelRun, LABELLED_SOURCES,
                             default_cache_path, explode_aspects, label_reviews, with_dimensions)
from app.data_processing import CleaningReport, CsvSource, process_reviews
from app.evaluation import rating_outcome_agreement
from app.llm_client import ClaudeJsonLLM, JsonLLM, has_credentials
from app.profiles import Profile, Roles, load_profile

LABELLING_MODES = ("baseline", "ai", "none")
LABEL_COLUMNS = ("sentiment_ai", "primary_topic_ai", "issue_ai", "confidence_ai", "aspects_ai",
                 "label_source")


class PipelineError(RuntimeError):
    """A user-facing problem with the requested analysis (e.g. AI requested without access)."""


@dataclass
class Table:
    """One analytics result plus the facts needed to read it correctly."""

    name: str
    df: pd.DataFrame
    basis: str               # "all_reviews" | "rated_reviews" | "text_labelled"
    weighted: bool           # percentages are weighted estimates of the population
    sentiment_col: Optional[str]
    n_reviews: int           # raw number of reviews behind the table
    available: bool = True
    reason: str = ""         # why the table is unavailable
    note: str = ""


@dataclass
class LabelMethod:
    kind: str                # "baseline" | "ai" | "none"
    is_ai: bool
    calibrated_baseline: bool
    model: Optional[str] = None
    ai_rows: int = 0         # labelled by the API in this run
    cache_rows: int = 0      # AI labels re-used from the cache
    baseline_rows: int = 0   # labelled by the non-AI baseline (by design)
    fallback_rows: int = 0   # AI was requested but the baseline was used (API failure)
    sampled: bool = False

    @property
    def mixed(self) -> bool:
        """An AI run in which some rows fell back to the baseline."""
        return self.is_ai and self.fallback_rows > 0


@dataclass
class AnalysisResult:
    data: pd.DataFrame
    aspects: pd.DataFrame
    report: CleaningReport
    profile: Profile
    roles: Roles
    method: LabelMethod
    label_run: Optional[LabelRun]
    sampling: Dict
    kpis: Dict
    tables: Dict[str, Table]
    filters: Dict = field(default_factory=dict)

    def table(self, name: str) -> Table:
        return self.tables[name]

    def filtered(self, criteria: Union[Mapping[str, object], pd.Series, Callable[[pd.DataFrame], pd.Series]],
                 ) -> "AnalysisResult":
        """Re-compute every table on a subset of rows, keeping the original design.

        ``criteria``: {column: value or list of values}, a boolean mask aligned
        with ``data``, or a function returning one. Rows keep their original
        ``sample_weight`` / ``sentiment_weight``: the subset is not re-sampled
        or re-normalised, so percentages stay estimates of the filtered
        population with raw counts shown alongside.
        """
        mask = _mask(self.data, criteria)
        data = self.data.loc[mask].copy()
        aspects = self.aspects[self.aspects["review_id"].isin(data["review_id"])].copy()
        kpis, tables = build_tables(data, aspects, self.profile, self.roles, self.method)
        description = dict(criteria) if isinstance(criteria, Mapping) else {"mask": int(mask.sum())}
        return replace(self, data=data, aspects=aspects, kpis=kpis, tables=tables,
                       filters={**self.filters, **description})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analysis(
    source: CsvSource,
    *,
    profile: Union[str, Path, Profile, None] = None,
    labelling: str = "baseline",
    llm: Optional[JsonLLM] = None,
    max_ai_rows: int = config.MAX_AI_ROWS,
    seed: int = 0,
    cache_path: Optional[Path] = None,
    reference_date: Optional[pd.Timestamp] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AnalysisResult:
    """Clean, label and analyse a feedback dataset in one consistent pass.

    labelling:
      "baseline"  non-AI labels for every review with text (default, no API)
      "ai"        Claude on a weighted sample of at most ``max_ai_rows`` reviews;
                  needs credentials (or an injected ``llm`` for tests), otherwise
                  raises PipelineError — it never silently falls back to the baseline
      "none"      cleaning and rating analytics only; no text labels
    """
    if labelling not in LABELLING_MODES:
        raise PipelineError(f"labelling must be one of {LABELLING_MODES}, not {labelling!r}")
    if labelling == "ai" and llm is None and not has_credentials():
        raise PipelineError("AI labelling was requested, but no Anthropic API access is configured "
                            "(ANTHROPIC_API_KEY is missing or still the placeholder). "
                            "Use labelling='baseline' for the non-AI analysis.")

    df, report = process_reviews(source, reference_date=reference_date, profile=profile)
    chosen = load_profile(report.profile_name) if profile is None or not isinstance(profile, Profile) else profile
    roles = report.roles

    label_run: Optional[LabelRun] = None
    if labelling == "none":
        data = _without_text_labels(df)
        method = LabelMethod(kind="none", is_ai=False,
                             calibrated_baseline=chosen.baseline_sentiment is not None)
    elif labelling == "baseline":
        data, label_run = label_reviews(df, BaselineLabeller(chosen), seed=seed)
        method = LabelMethod(kind="baseline", is_ai=False,
                             calibrated_baseline=chosen.baseline_sentiment is not None,
                             baseline_rows=label_run.from_baseline)
    else:
        model_llm = llm or ClaudeJsonLLM()
        cache = LabelCache(cache_path if cache_path is not None
                           else (None if llm is not None else default_cache_path(model_llm.model, chosen)))
        data, label_run = label_reviews(df, AILabeller(model_llm, chosen), cache=cache,
                                        max_rows=max_ai_rows, progress=progress, seed=seed)
        method = LabelMethod(kind="ai", is_ai=True,
                             calibrated_baseline=chosen.baseline_sentiment is not None,
                             model=model_llm.model, ai_rows=label_run.from_ai,
                             cache_rows=label_run.from_cache, fallback_rows=label_run.fallback_rows,
                             sampled=label_run.sampled)

    data = _add_weights(data)
    sampling = _sampling_summary(data, label_run)
    aspects = _aspects(data, roles)
    kpis, tables = build_tables(data, aspects, chosen, roles, method)
    return AnalysisResult(data=data, aspects=aspects, report=report, profile=chosen, roles=roles,
                          method=method, label_run=label_run, sampling=sampling, kpis=kpis,
                          tables=tables)


# ---------------------------------------------------------------------------
# Labels and weights
# ---------------------------------------------------------------------------

def _without_text_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rated = out["sentiment_from_rating"].notna()
    out["sentiment"] = out["sentiment_from_rating"]
    out["sentiment_source"] = pd.Series("rating", index=out.index, dtype=object).where(rated, None)
    out["sample_weight"] = np.nan
    return out


def _add_weights(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    labelled = out["label_source"].isin(LABELLED_SOURCES) if "label_source" in out.columns \
        else pd.Series(False, index=out.index)
    out["in_label_sample"] = labelled
    out["sample_weight"] = out["sample_weight"].where(labelled)
    source = out["sentiment_source"]
    out["sentiment_weight"] = np.where(source == "rating", 1.0,
                                       np.where(source.isin(["text_ai", "text_baseline"]),
                                                out["sample_weight"], np.nan))
    if labelled.any():
        expected = int(out["has_text"].sum())
        total = float(out.loc[labelled, "sample_weight"].sum())
        if abs(total - expected) > 1e-6 * max(1, expected):
            raise PipelineError(f"Sampling weights sum to {total:.3f}, expected {expected} "
                                "(reviews with text). Refusing to produce weighted estimates.")
    return out


def _sampling_summary(data: pd.DataFrame, run: Optional[LabelRun]) -> Dict:
    labelled = data["in_label_sample"]
    return {
        "labelled_reviews": int(labelled.sum()),
        "reviews_with_text": int(data["has_text"].sum()),
        "sampled": bool(run.sampled) if run else False,
        "strata": run.stratum_sizes if run else {},
        "weight_sum": round(float(data.loc[labelled, "sample_weight"].sum()), 6) if labelled.any() else 0.0,
        "allocation": dict(config.AI_SAMPLE_ALLOCATION) if run and run.sampled else None,
    }


def _aspects(data: pd.DataFrame, roles: Roles) -> pd.DataFrame:
    if not data["in_label_sample"].any():
        return pd.DataFrame(columns=["review_id", "topic", "polarity", "issue", "sample_weight"])
    aspects = explode_aspects(data)
    return with_dimensions(aspects, data, [c for c, _ in roles.dimensions])


def _mask(data: pd.DataFrame, criteria) -> pd.Series:
    if callable(criteria):
        criteria = criteria(data)
    if isinstance(criteria, pd.Series):
        return criteria.reindex(data.index).fillna(False).astype(bool)
    mask = pd.Series(True, index=data.index)
    for column, value in criteria.items():
        if column not in data.columns:
            raise PipelineError(f"Cannot filter on '{column}': no such column.")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        mask &= data[column].isin(values)
    return mask


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def build_tables(data: pd.DataFrame, aspects: pd.DataFrame, profile: Profile, roles: Roles,
                 method: LabelMethod):
    """All analytics for ``data`` using the weights already stored on each row."""
    scale = profile.rating_scale
    has_rating = roles.has_rating and data["rating"].notna().any()
    has_date = roles.has_date and "month" in data.columns and data["month"].notna().any()
    labelled = data[data["in_label_sample"]]
    text_available = method.kind != "none" and not labelled.empty
    dims = [c for c, _ in roles.dimensions if c in data.columns]
    outcomes = [o.column for o in roles.outcomes if o.column in data.columns]
    rated_n = int(data["rating"].notna().sum())
    tables: Dict[str, Table] = {}

    def add(name, df, basis, weighted, sentiment_col, n, available=True, reason="", note=""):
        tables[name] = Table(name, df if df is not None else pd.DataFrame(), basis, weighted,
                             sentiment_col, int(n), available, reason, note)

    def unavailable(name, basis, reason):
        add(name, None, basis, False, None, 0, available=False, reason=reason)

    no_rating = "The dataset has no usable rating column."
    no_text = "No text labels (labelling='none')." if method.kind == "none" else "No reviews were labelled."

    # --- headline numbers ---------------------------------------------------
    primary_weighted = _is_weighted(data, "sentiment_weight")
    kpis = {
        "primary": analytics.overview_kpis(data, "sentiment", "sentiment_weight", scale,
                                           dimensions=[c for c, _ in roles.dimensions]),
        "rating": analytics.overview_kpis(data, "sentiment_from_rating", None, scale) if has_rating else None,
        "text": (analytics.overview_kpis(labelled, "sentiment_ai", "sample_weight", scale)
                 if text_available else None),
        "sentiment_sources": _source_counts(data),
        "labelled_reviews": int(len(labelled)),
        "estimated_reviews": round(float(data["sentiment_weight"].fillna(0).sum()), 1),
        "method": method.kind,
        "is_ai": method.is_ai,
    }
    kpis["primary"]["weighted"] = primary_weighted

    # --- sentiment ------------------------------------------------------------
    if has_rating:
        add("rating_distribution", analytics.rating_distribution(data, scale), "rated_reviews",
            False, "sentiment_from_rating", rated_n)
    else:
        unavailable("rating_distribution", "rated_reviews", no_rating)
    add("sentiment_primary", analytics.sentiment_distribution(data, "sentiment", "sentiment_weight"),
        "all_reviews", primary_weighted, "sentiment", int(data["sentiment"].notna().sum()),
        note="Rating where available, otherwise the text label (see sentiment_sources).")
    add("sentiment_sources", _source_table(data), "all_reviews", False, "sentiment_source", len(data))
    if text_available:
        add("sentiment_text", analytics.sentiment_distribution(labelled, "sentiment_ai", "sample_weight"),
            "text_labelled", _is_weighted(labelled, "sample_weight"), "sentiment_ai", len(labelled),
            note="Text-only labels" + (" (AI)" if method.is_ai else " (non-AI baseline)") + ".")
    else:
        unavailable("sentiment_text", "text_labelled", no_text)

    # --- time -----------------------------------------------------------------
    if has_date:
        add("trend", analytics.reviews_over_time(data, "sentiment", weight_col="sentiment_weight", scale=scale),
            "all_reviews", primary_weighted, "sentiment", int(data["month"].notna().sum()))
        add("trend_rating", analytics.reviews_over_time(data, "sentiment_from_rating", scale=scale),
            "rated_reviews", False, "sentiment_from_rating", rated_n)
        add("sentiment_trend", analytics.sentiment_over_time(data, "sentiment", "sentiment_weight"),
            "all_reviews", primary_weighted, "sentiment", int(data["month"].notna().sum()))
        if roles.entity:
            add("recent_change", analytics.recent_change(data, roles.entity, sentiment_col="sentiment",
                                                         weight_col="sentiment_weight"),
                "all_reviews", primary_weighted, "sentiment", len(data))
        else:
            unavailable("recent_change", "all_reviews", "No entity column to compare.")
    else:
        for name in ("trend", "trend_rating", "sentiment_trend", "recent_change"):
            unavailable(name, "all_reviews", "The dataset has no usable date column.")

    # --- breakdowns by entity / segment -----------------------------------------
    for col in dims:
        add(f"breakdown:{col}", _breakdown(data, col, scale, outcomes), "all_reviews", primary_weighted,
            "sentiment", int(data[col].notna().sum()),
            note=f"Observed columns use every review; pct_negative uses the primary sentiment.")

    # --- rating relationships -----------------------------------------------------
    for spec in roles.numeric:
        if has_rating:
            add(f"numeric:{spec.column}", analytics.band_summary(data, spec, scale), "rated_reviews",
                False, "sentiment_from_rating", rated_n,
                note=str(analytics.correlation_with_rating(data, spec.column)))
        else:
            unavailable(f"numeric:{spec.column}", "rated_reviews", no_rating)
    if has_rating:
        add("low_rating_drivers", analytics.low_rating_drivers(data, dims, scale=scale), "rated_reviews",
            False, "sentiment_from_rating", rated_n)
        for col in outcomes:
            agreement = rating_outcome_agreement(data, col, scale.negative_max, scale.positive_min)
            if agreement:
                add(f"outcome_agreement:{col}", agreement["table"].reset_index(), "rated_reviews", False,
                    "sentiment_from_rating", agreement["n"],
                    note=f"agreement {agreement['agreement_pct']}% (CI {agreement['agreement_ci95']})")
    else:
        unavailable("low_rating_drivers", "rated_reviews", no_rating)

    # --- text labels: topics, issues, praise, flags --------------------------------
    if text_available:
        total_weight = float(labelled["sample_weight"].sum())
        weighted = _is_weighted(labelled, "sample_weight")
        label_note = "AI labels" if method.is_ai else "Non-AI baseline labels (not AI)"
        if method.mixed:
            label_note += f"; {method.fallback_rows} rows fell back to the baseline"
        negative = _per_review(aspects, "negative", "issue")
        positive = _per_review(aspects, "positive", "topic")
        add("topics", _topic_table(aspects, total_weight, len(labelled)), "text_labelled", weighted,
            "sentiment", len(labelled), note=label_note)
        add("issue_priority", analytics.priority_table(negative, "issue", sentiment_col="sentiment",
                                                       weight_col="sample_weight", total_weight=total_weight,
                                                       scale=scale),
            "text_labelled", weighted, "sentiment", len(labelled), note=label_note)
        add("praise", _mention_table(positive, "topic", total_weight), "text_labelled", weighted,
            "sentiment", len(labelled), note=label_note)
        add("flags", _flag_table(labelled, profile), "text_labelled", weighted, None, len(labelled),
            note=label_note)
        for col, _ in roles.segments:
            if col in aspects.columns:
                add(f"issues_by:{col}", _issues_by(negative, labelled, col), "text_labelled", weighted,
                    None, int(labelled[col].notna().sum()), note=label_note)
    else:
        for name in ("topics", "issue_priority", "praise", "flags"):
            unavailable(name, "text_labelled", no_text)

    return kpis, tables


def _is_weighted(df: pd.DataFrame, col: str) -> bool:
    w = df[col].dropna()
    return bool(len(w)) and not np.allclose(w, 1.0)


def _source_counts(data: pd.DataFrame) -> Dict[str, int]:
    counts = data["sentiment_source"].fillna("none").value_counts()
    return {k: int(counts.get(k, 0)) for k in ("rating", "text_ai", "text_baseline", "none")}


def _source_table(data: pd.DataFrame) -> pd.DataFrame:
    counts = _source_counts(data)
    total = sum(counts.values())
    return pd.DataFrame({"sentiment_source": list(counts), "reviews": list(counts.values()),
                         "pct_of_rows": [round(100 * v / total, 1) if total else None for v in counts.values()]})


def _breakdown(data: pd.DataFrame, col: str, scale, outcomes: Sequence[str]) -> pd.DataFrame:
    """Observed metrics from every review; negative share from the weighted primary sentiment."""
    observed = analytics.group_summary(data, col, sentiment_col="sentiment_from_rating", scale=scale,
                                       outcome_cols=outcomes)
    primary = analytics.group_summary(data, col, sentiment_col="sentiment", weight_col="sentiment_weight",
                                      scale=scale)
    observed = observed.drop(columns=["pct_negative"])
    merged = observed.merge(primary[["group", "pct_negative"]], on="group", how="left")
    order = ["group", "reviews", "pct_of_reviews", "avg_rating", "pct_negative", "pct_low_rating"]
    return merged[order + [c for c in merged.columns if c not in order]]


def _per_review(aspects: pd.DataFrame, polarity: str, key: str) -> pd.DataFrame:
    """One row per (review, key): a review mentioning two issues of a topic counts once per topic."""
    rows = aspects[aspects["polarity"] == polarity]
    return rows.drop_duplicates(subset=["review_id", key])


def _mention_table(rows: pd.DataFrame, key: str, total_weight: float) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=[key, "reviews", "est_pct_of_reviews"])
    grouped = rows.groupby(key)
    out = pd.DataFrame({"reviews": grouped.size(),
                        "est_pct_of_reviews": (100 * grouped["sample_weight"].sum() / total_weight).round(1)})
    return out.rename_axis(key).reset_index().sort_values("est_pct_of_reviews", ascending=False,
                                                          ignore_index=True)


def _topic_table(aspects: pd.DataFrame, total_weight: float, n: int) -> pd.DataFrame:
    complaints = _mention_table(_per_review(aspects, "negative", "topic"), "topic", total_weight)
    praise = _mention_table(_per_review(aspects, "positive", "topic"), "topic", total_weight)
    out = complaints.merge(praise, on="topic", how="outer", suffixes=("_complaint", "_praise"))
    for col in ("reviews_complaint", "reviews_praise"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ("est_pct_of_reviews_complaint", "est_pct_of_reviews_praise"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out.sort_values("est_pct_of_reviews_complaint", ascending=False, ignore_index=True)


def _flag_table(labelled: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    rows = []
    w = labelled["sample_weight"]
    for flag, definition in profile.taxonomy.flags.items():
        col = f"flag_{flag}"
        if col not in labelled.columns:
            continue
        values = labelled[col].map(lambda v: v is True)
        rows.append({"flag": flag, "reviews": int(values.sum()),
                     "est_pct_of_reviews": round(100 * float((w * values).sum() / w.sum()), 1) if w.sum() else None,
                     "definition": definition})
    return pd.DataFrame(rows, columns=["flag", "reviews", "est_pct_of_reviews", "definition"])


def _issues_by(negative: pd.DataFrame, labelled: pd.DataFrame, col: str, top: int = 10) -> pd.DataFrame:
    """Estimated share of each segment's reviews that mention each of the top issues."""
    columns = [col, "issue", "reviews", "est_pct_of_segment"]
    if negative.empty or col not in negative.columns:
        return pd.DataFrame(columns=columns)
    top_issues = (negative.groupby("issue")["sample_weight"].sum().sort_values(ascending=False)
                  .head(top).index)
    segment_weight = labelled.groupby(col)["sample_weight"].sum()
    rows = negative[negative["issue"].isin(top_issues)].dropna(subset=[col])
    grouped = rows.groupby([col, "issue"])
    out = pd.DataFrame({"reviews": grouped.size(), "weight": grouped["sample_weight"].sum()}).reset_index()
    out["est_pct_of_segment"] = (100 * out["weight"] / out[col].map(segment_weight)).round(1)
    return out[columns].sort_values([col, "est_pct_of_segment"], ascending=[True, False], ignore_index=True)
