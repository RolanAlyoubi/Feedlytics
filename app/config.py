"""Central configuration for Feedlytics (domain-neutral).

Every threshold used by the cleaning, analytics and AI code lives here so the
rules are visible in one place. Anything specific to a business domain —
topic taxonomy, column names, numeric bands, keywords — belongs in a profile
under profiles/ instead (see app/profiles.py).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core schema
# ---------------------------------------------------------------------------

# Only the review text is required. Rating, date, title and id are optional
# core roles; entity, segments, numeric drivers and outcomes come from profiles.
REQUIRED_COLUMNS = ("review_text",)
CORE_OPTIONAL_COLUMNS = ("review_id", "review_title", "rating", "date")
CORE_COLUMNS = ("review_id", "review_text", "review_title", "rating", "date")

# Alternative header names for the core roles (headers are normalised first:
# lower-case, non-alphanumerics -> "_"). Domain-specific synonyms live in profiles.
CORE_COLUMN_SYNONYMS = {
    "review_id": ["id", "reviewid", "review_number", "feedback_id", "response_id"],
    "review_text": [
        "review", "text", "comment", "comments", "feedback", "review_body",
        "review_content", "content", "body", "message", "verbatim",
    ],
    "review_title": ["title", "summary", "headline", "subject", "review_summary"],
    "rating": ["stars", "star_rating", "score", "rating_value", "review_rating", "overall_rating"],
    "date": ["review_date", "created_at", "timestamp", "datetime", "date_time", "submitted_at"],
}

# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

# Default rating scale (a profile can define its own, e.g. 1–10).
RATING_MIN = 1
RATING_MAX = 5
NEGATIVE_MAX_RATING = 2  # 1-2 stars  -> Negative
POSITIVE_MIN_RATING = 4  # 4-5 stars  -> Positive, 3 -> Neutral

EARLIEST_VALID_DATE = "2000-01-01"
DATE_DAYFIRST = False  # ambiguous dates like 03/04/2025 are read as month-first

# Texts shorter than this are not used for duplicate detection, because short
# reviews such as "Great product!" are legitimately written by many customers.
MIN_DUPLICATE_TEXT_LENGTH = 20

# Source truncation check: if at least this share of texts sit within
# TRUNCATION_WINDOW characters of the longest text, the source probably cut
# reviews off at a fixed length.
TRUNCATION_MIN_SHARE = 0.01
TRUNCATION_WINDOW = 10

# Strings that mean "no value" in hand-made CSV files.
MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "?"}

# Hard limit to keep the app responsive. Larger files should be sampled first.
MAX_ROWS = 200_000

# ---------------------------------------------------------------------------
# Analytics rules
# ---------------------------------------------------------------------------

# Groups with fewer reviews than this are reported but marked as unreliable.
MIN_GROUP_SIZE = 15

# Change in percentage points needed before a trend is called up or down.
TREND_THRESHOLD_PP = 2.0
# ...and the change must be statistically significant at this level.
TREND_SIGNIFICANCE_LEVEL = 0.05

# Priority levels are relative to the highest-scoring group (0-100 index).
PRIORITY_HIGH_MIN_INDEX = 60
PRIORITY_MEDIUM_MIN_INDEX = 25

SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")

# ---------------------------------------------------------------------------
# AI settings. Model and limits can be overridden in .env.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5-5"
LABEL_EFFORT = "low"        # classification is simple; low effort keeps cost down
INSIGHT_EFFORT = "medium"   # writing findings needs a little more reasoning
LABEL_BATCH_SIZE = 25       # reviews per API call
MAX_AI_ROWS = 2000          # larger datasets are sampled before AI labelling
MAX_REVIEW_CHARS = 2000     # longer reviews are shortened (and counted) before labelling
MAX_ASPECTS = 3
LABEL_PROMPT_VERSION = "v2"    # change when the prompt changes, so the cache refreshes
INSIGHT_PROMPT_VERSION = "v1"
CACHE_DIR = "data/cache"

# When a dataset is larger than MAX_AI_ROWS, the AI sample over-represents
# low ratings so complaints are well covered. Shares of the *rated* sample:
AI_SAMPLE_ALLOCATION = {"Negative": 0.50, "Neutral": 0.25, "Positive": 0.25}
# Every sampled review gets weight = (stratum size in data) / (stratum size in
# sample), so AI-based percentages estimate the whole dataset, not the sample.

# USD per million tokens, used only for the cost estimate shown before running.
MODEL_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# VADER thresholds for the non-AI baseline (the values its authors recommend).
VADER_POSITIVE_MIN = 0.05
VADER_NEGATIVE_MAX = -0.05
