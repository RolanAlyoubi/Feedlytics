"""Loading, validation and cleaning of customer-feedback CSV files.

Pipeline:  load_csv -> (choose profile) -> standardize_columns -> validate_schema
           -> clean_reviews

Only a review-text column is required. Rating, date, title and id are optional
core roles; everything domain-specific (entity, segments, numeric drivers,
outcomes, value fixes) comes from the selected profile in profiles/.

Every step records what it changed in a ``CleaningReport`` so the dashboard
can show users exactly what happened to their data. Nothing here uses AI.
"""

from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app import config
from app.profiles import (Profile, RatingScale, Roles, detect_profile, load_profile,
                          normalize_header)

CsvSource = Union[str, Path, io.IOBase, bytes, pd.DataFrame]
_ROW_NUMBER_HEADER = re.compile(r"^(unnamed_\d+|index|)$")

__all__ = ["CleaningReport", "DataValidationError", "process_reviews", "load_csv",
           "standardize_columns", "validate_schema", "clean_reviews", "normalize_header",
           "clean_text", "parse_rating", "parse_number", "parse_dates", "sentiment_from_rating",
           "add_derived_columns", "canonicalize_labels"]


class DataValidationError(ValueError):
    """Raised when a file cannot be analysed. The message is user-facing."""


@dataclass
class CleaningReport:
    """A record of everything the pipeline did to the uploaded data."""

    rows_received: int = 0
    rows_retained: int = 0
    profile_name: str = ""
    profile_reason: str = ""
    roles: Roles = field(default_factory=Roles)
    column_mapping: Dict[str, str] = field(default_factory=dict)
    extra_columns: List[str] = field(default_factory=list)
    missing_optional_columns: List[str] = field(default_factory=list)
    row_number_columns: List[str] = field(default_factory=list)
    missing_values_before: Dict[str, int] = field(default_factory=dict)
    removed: Dict[str, int] = field(default_factory=dict)  # rows dropped, by reason
    flagged: Dict[str, int] = field(default_factory=dict)  # rows kept, value fixed/cleared
    warnings: List[str] = field(default_factory=list)      # problems the user should see
    notes: List[str] = field(default_factory=list)         # facts about the dataset

    def add_removed(self, reason: str, count: int) -> None:
        if count:
            self.removed[reason] = self.removed.get(reason, 0) + int(count)

    def add_flagged(self, issue: str, count: int) -> None:
        if count:
            self.flagged[issue] = self.flagged.get(issue, 0) + int(count)

    @property
    def rows_removed(self) -> int:
        return sum(self.removed.values())

    @property
    def has_rating(self) -> bool:
        return self.roles.has_rating

    def summary_table(self) -> pd.DataFrame:
        """One row per action, ready to show in a table."""
        rows = [("Removed", reason, n) for reason, n in self.removed.items()]
        rows += [("Kept, value fixed or cleared", issue, n) for issue, n in self.flagged.items()]
        return pd.DataFrame(rows, columns=["action", "reason", "rows"])


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_csv(source: CsvSource) -> pd.DataFrame:
    """Read a CSV into a DataFrame of strings, with user-friendly errors.

    Everything is read as text so that parsing rules (ratings like "4/5",
    prices like "$12.50") are applied consistently by this module rather than
    guessed by pandas.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raw = _read_bytes(source)
        if not raw.strip():
            raise DataValidationError("The uploaded file is empty.")
        df = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=encoding,
                                 skipinitialspace=True)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise DataValidationError("The uploaded file is empty.") from None
            except pd.errors.ParserError as exc:
                raise DataValidationError(
                    "The file could not be read as a CSV. Check that it is "
                    f"comma-separated and that quoted text is closed. ({exc})"
                ) from None

    if df is None or df.shape[1] == 0:
        raise DataValidationError("No columns were found in the file.")
    if df.empty:
        raise DataValidationError("The file has column headers but no data rows.")
    if len(df) > config.MAX_ROWS:
        raise DataValidationError(
            f"The file has {len(df):,} rows; the limit is {config.MAX_ROWS:,}. "
            "Please upload a sample (for example, the most recent reviews)."
        )
    return df


def _read_bytes(source: CsvSource) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"File not found: {path}")
        return path.read_bytes()
    if hasattr(source, "read"):  # file-like, e.g. a Streamlit UploadedFile
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        return data.encode("utf-8") if isinstance(data, str) else data
    raise DataValidationError("Unsupported input. Please upload a CSV file.")


# ---------------------------------------------------------------------------
# 2. Column standardisation and schema validation
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame, report: CleaningReport,
                        profile: Optional[Profile] = None) -> pd.DataFrame:
    """Rename headers using, in order: the profile's explicit column map, the
    canonical name itself, the profile's synonyms, then the core synonyms."""
    profile = profile or load_profile()
    core_lookup = {syn: canonical for canonical, syns in config.CORE_COLUMN_SYNONYMS.items()
                   for syn in syns}
    normalized = [normalize_header(c) for c in df.columns]
    present = set(normalized)
    renamed, used = [], set()
    for original, norm in zip(df.columns, normalized):
        if norm in profile.columns:
            target = profile.columns[norm]
        elif norm in config.CORE_COLUMNS:
            target = norm
        else:
            target = norm
            candidate = profile.column_synonyms.get(norm) or core_lookup.get(norm)
            # Only use a synonym if the target column is not already present.
            if candidate and candidate not in present and candidate not in used:
                target = candidate
        if target in used:  # two headers mapping to the same name
            report.warnings.append(f"Duplicate column '{original}' was ignored.")
            target = f"{target}__duplicate_{len(renamed)}"
        if str(original) != target:
            report.column_mapping[str(original)] = target
        if _ROW_NUMBER_HEADER.match(norm):
            report.row_number_columns.append(target)
        used.add(target)
        renamed.append(target)

    out = df.copy()
    out.columns = renamed
    out = out.loc[:, [c for c in out.columns if "__duplicate_" not in c]]

    role_columns = _profile_role_columns(profile)
    report.extra_columns = [c for c in out.columns
                            if c not in config.CORE_COLUMNS and c not in role_columns]
    declared = [c for c in config.CORE_OPTIONAL_COLUMNS if c != "review_id"]
    declared += [c for c in sorted(role_columns) if not _is_band_column(profile, c)]
    report.missing_optional_columns = [c for c in declared if c not in out.columns]
    return out


def _profile_role_columns(profile: Profile) -> set:
    cols = set()
    if "column" in profile.entity:
        cols.add(profile.entity["column"])
    cols.update(profile.entity.get("candidates", []))
    for spec in profile.segments:
        cols.update([spec["column"]] if "column" in spec else spec.get("candidates", []))
    for spec in profile.numeric:
        cols.add(spec.column)
        if spec.band_column:
            cols.add(spec.band_column)
    cols.update(o.column for o in profile.outcomes)
    return cols


def _is_band_column(profile: Profile, column: str) -> bool:
    return any(spec.band_column == column for spec in profile.numeric)


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the review text column is absent."""
    if "review_text" not in df.columns:
        found = ", ".join(map(str, df.columns)) or "none"
        raise DataValidationError(
            "Missing required column 'review_text' — the customer's written feedback "
            "(e.g. 'review', 'comment', 'feedback'). Optional columns such as 'rating', "
            f"'date' and 'title' are used when present. Columns found: {found}."
        )


# ---------------------------------------------------------------------------
# 3. Value parsers (small, pure functions — easy to unit test)
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_RATING = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(?:/\s*\d+|stars?|★)?$")
_WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_CLEAN_ENDING = re.compile(r"[.!?)\"'”’]\s*$")


def is_missing_token(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in config.MISSING_TOKENS


def clean_text(value: object) -> Optional[str]:
    """Remove HTML, decode entities and collapse whitespace. Empty -> None."""
    if is_missing_token(value):
        return None
    text = html.unescape(str(value))
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return None if is_missing_token(text) else text


def normalize_for_matching(text: Optional[str]) -> Optional[str]:
    """Lower-case, punctuation-free version used to detect duplicate reviews."""
    if text is None:
        return None
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def parse_rating(value: object, scale_max: float = config.RATING_MAX) -> float:
    """'4', '4.0', '4/5', '4 stars', 'four' -> 4.0; anything else -> NaN.

    A fraction such as '4/10' is accepted only if its denominator equals the
    scale maximum; the range itself is checked separately in clean_reviews."""
    if is_missing_token(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    words = text.replace("stars", "").replace("star", "").strip()
    if words in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[words])
    match = _RATING.match(text)
    if not match:
        return np.nan
    if "/" in text and not text.replace(" ", "").endswith(f"/{_fmt(scale_max)}"):
        return np.nan  # e.g. "4/10" on a 1–5 scale is ambiguous
    return float(match.group(1))


def parse_number(value: object) -> float:
    """Extract a number from strings such as '35 min', '$1,234.50', 'SAR 40'."""
    if is_missing_token(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = _NUMBER.search(str(value))
    if not match:
        return np.nan
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return np.nan


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates written in mixed formats; unparseable values become NaT."""
    values = series.where(~series.map(is_missing_token))
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce",
                                dayfirst=config.DATE_DAYFIRST, utc=True)
    except (ValueError, TypeError):  # very unusual input: fall back to one-by-one
        parsed = pd.to_datetime(values.map(_parse_one_date), utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _parse_one_date(value: object):
    try:
        return pd.to_datetime(value, errors="coerce", dayfirst=config.DATE_DAYFIRST, utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def canonicalize_labels(series: pd.Series) -> pd.Series:
    """Merge spelling variants such as ' acme store' and 'Acme Store'.

    Each case-insensitive group is replaced by its most frequent spelling.
    """
    cleaned = series.map(clean_text)
    keys = cleaned.str.lower()
    preferred = (
        pd.DataFrame({"key": keys, "label": cleaned})
        .dropna()
        .groupby("key")["label"]
        .agg(lambda s: s.value_counts().index[0])
    )
    return keys.map(preferred)


def _parse_outcome(value: object, positive: Tuple[str, ...], negative: Tuple[str, ...]) -> float:
    if is_missing_token(value):
        return np.nan
    text = str(value).strip().lower()
    try:  # "1.0" -> "1"
        number = float(text)
        if number.is_integer():
            text = str(int(number))
    except ValueError:
        pass
    if text in positive:
        return 1.0
    if text in negative:
        return 0.0
    return np.nan


def _join_words(words: List[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _fmt(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------

def clean_reviews(
    df: pd.DataFrame,
    report: CleaningReport,
    reference_date: Optional[pd.Timestamp] = None,
    profile: Optional[Profile] = None,
) -> pd.DataFrame:
    """Clean a DataFrame that already has standardised column names."""
    profile = profile or load_profile()
    scale = profile.rating_scale
    reference_date = pd.Timestamp(reference_date or pd.Timestamp.today()).normalize()
    data = df.copy()
    roles_before = profile.resolve_roles(data.columns)
    report.missing_values_before = {
        col: int(data[col].map(is_missing_token).sum())
        for col in data.columns if col in config.CORE_COLUMNS
    }

    # Exact duplicate rows (e.g. the same export appended twice). Row-number
    # columns written by spreadsheet/pandas exports ("Unnamed: 0") differ on
    # every row, so they are ignored when comparing.
    compare = [c for c in data.columns if c not in report.row_number_columns]
    exact_dupes = data.duplicated(subset=compare, keep="first")
    reason = ("Exact duplicate row" if len(compare) == len(data.columns)
              else "Exact duplicate row (ignoring the row-number column)")
    report.add_removed(reason, exact_dupes.sum())
    data = data.loc[~exact_dupes]

    # --- text: review text, optional title, and the text used for analysis
    data["review_text"] = data["review_text"].map(clean_text)
    report.add_flagged("Missing or empty review text", data["review_text"].isna().sum())
    if roles_before.has_title:
        data["review_title"] = data["review_title"].map(clean_text)
        title_only = data["review_text"].isna() & data["review_title"].notna()
        report.add_flagged("Title used as text (review text missing)", title_only.sum())
    data["analysis_text"] = _combine_title_and_text(data)
    data["has_text"] = data["analysis_text"].notna()
    _flag_source_truncation(data, report)

    # --- rating (optional) ------------------------------------------------
    if roles_before.has_rating:
        parsed = data["rating"].map(lambda v: parse_rating(v, scale.max))
        in_range = parsed.between(scale.min, scale.max)
        had_rating = ~data["rating"].map(is_missing_token)
        report.add_flagged("Invalid rating (cleared)", (had_rating & ~in_range).sum())
        report.add_flagged("Missing rating", (~had_rating).sum())
        rounded = in_range & (parsed != parsed.round())
        report.add_flagged("Fractional rating rounded to whole star", rounded.sum())
        # Round half up (4.5 -> 5) rather than numpy's round-half-to-even.
        data["rating"] = np.floor(parsed.where(in_range) + 0.5)
    else:
        data["rating"] = np.nan
        report.notes.append("No rating column: rating analytics are skipped; sentiment "
                            "comes from the text labelling instead.")

    # --- date (optional) --------------------------------------------------
    if roles_before.has_date:
        had_value = ~data["date"].map(is_missing_token)
        dates = parse_dates(data["date"])
        report.add_flagged("Unreadable date (cleared)", (had_value & dates.isna()).sum())
        too_late = dates > reference_date
        too_early = dates < pd.Timestamp(config.EARLIEST_VALID_DATE)
        report.add_flagged("Future or implausible date (cleared)", (too_late | too_early).sum())
        data["date"] = dates.mask(too_late | too_early)
    else:
        report.notes.append("No date column: trend analysis is not available.")

    # --- profile value fixes (e.g. a misspelt category in the source) -----
    for col, aliases in profile.value_aliases.items():
        if col in data.columns:
            fixed = data[col].isin(list(aliases))
            report.add_flagged(f"{roles_before.label_for(col)} value corrected", fixed.sum())
            data[col] = data[col].replace(aliases)

    # --- categorical labels: entity and segments --------------------------
    for col, label in roles_before.dimensions:
        before = data[col].map(clean_text)
        data[col] = canonicalize_labels(data[col])
        merged = (before.notna() & (before != data[col])).sum()
        report.add_flagged(f"{label} name variant merged", merged)

    # --- numeric drivers, with optional bands -----------------------------
    for spec in roles_before.numeric:
        values = data[spec.column].map(parse_number)
        had_value = ~data[spec.column].map(is_missing_token)
        if spec.valid_range:
            bad = had_value & ~values.between(*spec.valid_range)
        else:
            bad = had_value & values.isna()
        report.add_flagged(f"Implausible {spec.label.lower()} (cleared)", bad.sum())
        data[spec.column] = values.mask(bad)
        if spec.bins:
            data[spec.band_column] = pd.cut(
                data[spec.column], bins=list(spec.bins), labels=list(spec.band_labels),
                right=spec.closed == "right", include_lowest=True).astype(object)

    # --- outcomes (yes/no columns such as "Recommended") ------------------
    for spec in roles_before.outcomes:
        values = data[spec.column].map(
            lambda v: _parse_outcome(v, spec.positive_values, spec.negative_values))
        had_value = ~data[spec.column].map(is_missing_token)
        report.add_flagged(f"Unreadable {spec.label.lower()} value (cleared)",
                           (had_value & values.isna()).sum())
        data[spec.column] = values

    # --- unusable rows ----------------------------------------------------
    unusable = ~data["has_text"] & data["rating"].isna()
    report.add_removed("No review text and no valid rating" if roles_before.has_rating
                       else "No review text", unusable.sum())
    data = data.loc[~unusable]

    # --- content duplicates (same customer submitting twice) -------------
    # A re-submission has the same text, rating, entity and day. Matching on
    # text alone would wrongly merge different customers writing similar reviews.
    data["_match_text"] = data["analysis_text"].map(normalize_for_matching)
    long_enough = data["_match_text"].str.len() >= config.MIN_DUPLICATE_TEXT_LENGTH
    key_cols, words = ["_match_text"], ["text"]
    if roles_before.has_rating:
        key_cols.append("rating"); words.append("rating")
    if roles_before.entity:
        key_cols.append(roles_before.entity); words.append(roles_before.entity_label.lower())
    if roles_before.has_date:
        key_cols.append("date"); words.append("date")
    content_dupes = long_enough & data.duplicated(subset=key_cols, keep="first")
    label = ("Re-submitted review (same text)" if len(words) == 1
             else f"Re-submitted review (same {_join_words(words)})")
    report.add_removed(label, content_dupes.sum())
    data = data.loc[~content_dupes].drop(columns="_match_text")

    # --- review ids -------------------------------------------------------
    data = _ensure_review_ids(data, report)

    data = add_derived_columns(data, scale)
    report.roles = profile.resolve_roles(data.columns)
    report.roles.has_rating = roles_before.has_rating
    report.rows_retained = len(data)
    return data.reset_index(drop=True)


def _combine_title_and_text(data: pd.DataFrame) -> pd.Series:
    """Title + text, so the AI sees the customer's headline too."""
    text = data["review_text"]
    if "review_title" not in data.columns:
        return text
    title = data["review_title"]
    joiner = title.fillna("").str.contains(r"[.!?]$").map({True: " ", False: ". "})
    combined = title + joiner + text
    return combined.where(title.notna() & text.notna(), text.fillna(title))


def _flag_source_truncation(data: pd.DataFrame, report: CleaningReport) -> None:
    """Detect reviews cut off at a fixed length by the data source."""
    lengths = data["review_text"].str.len()
    data["text_truncated"] = False
    if lengths.notna().sum() == 0:
        return
    longest = int(lengths.max())
    near_max = lengths >= longest - config.TRUNCATION_WINDOW
    if near_max.sum() < max(20, config.TRUNCATION_MIN_SHARE * lengths.notna().sum()):
        return
    cut = near_max & ~data["review_text"].fillna("").str.contains(_CLEAN_ENDING)
    data["text_truncated"] = cut
    report.add_flagged(f"Text appears cut off by the source (~{longest} characters)", cut.sum())
    report.notes.append(f"The source seems to truncate reviews at about {longest} characters; "
                        f"{int(cut.sum()):,} reviews end mid-sentence.")


def _ensure_review_ids(data: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    if "review_id" not in data.columns:
        data["review_id"] = None
    ids = data["review_id"].map(clean_text)
    missing = ids.isna()
    report.add_flagged("Missing review_id (generated)", missing.sum())
    ids[missing] = [f"gen_{i:06d}" for i in range(1, int(missing.sum()) + 1)]
    repeated = ids.duplicated(keep="first")
    if repeated.any():
        report.add_flagged("Repeated review_id with different content (suffixed)", repeated.sum())
        ids[repeated] = ids[repeated] + "_dup" + ids[repeated].groupby(ids[repeated]).cumcount().add(1).astype(str)
    data["review_id"] = ids
    return data


def sentiment_from_rating(rating: float, scale: Optional[RatingScale] = None) -> Optional[str]:
    """Rating-based sentiment: the non-AI baseline when a rating exists."""
    return (scale or RatingScale()).sentiment(rating)


def add_derived_columns(data: pd.DataFrame, scale: Optional[RatingScale] = None) -> pd.DataFrame:
    scale = scale or RatingScale()
    data = data.copy()
    if "analysis_text" not in data.columns:
        data["analysis_text"] = data["review_text"]
    data["word_count"] = data["review_text"].fillna("").str.split().str.len()
    data["sentiment_from_rating"] = data["rating"].map(scale.sentiment)
    data["is_low_rating"] = data["rating"] <= scale.negative_max
    if "date" in data.columns:
        data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    ordered = [c for c in config.CORE_COLUMNS if c in data.columns]
    rest = [c for c in data.columns if c not in ordered]
    return data[ordered + rest]


# ---------------------------------------------------------------------------
# 5. One-call entry point
# ---------------------------------------------------------------------------

def process_reviews(
    source: CsvSource,
    reference_date: Optional[pd.Timestamp] = None,
    profile: Union[str, Path, Profile, None] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Load, validate and clean a feedback file.

    ``profile``: a profile name/path, or None to detect it from the headers
    (falls back to the generic profile). Returns the cleaned DataFrame and a
    CleaningReport (which includes the chosen profile and resolved roles).
    Raises DataValidationError with a user-facing message if the file is unusable.
    """
    raw = load_csv(source)
    if profile is None:
        chosen, reason = detect_profile(raw.columns)
    else:
        chosen, reason = load_profile(profile), "selected explicitly"
    report = CleaningReport(rows_received=len(raw), profile_name=chosen.name,
                            profile_reason=reason)
    data = standardize_columns(raw, report, chosen)
    validate_schema(data)

    text_empty = data["review_text"].map(is_missing_token).all()
    title_empty = "review_title" not in data.columns or data["review_title"].map(is_missing_token).all()
    if text_empty and title_empty:
        raise DataValidationError("The review text column is empty in every row.")

    cleaned = clean_reviews(data, report, reference_date=reference_date, profile=chosen)
    if cleaned.empty:
        raise DataValidationError(
            "No usable reviews remain after cleaning: every row was missing review text"
            + (" and a valid rating." if report.has_rating else ".")
        )
    scale = chosen.rating_scale
    if report.has_rating and cleaned["rating"].isna().all():
        report.warnings.append(f"No valid {_fmt(scale.min)}–{_fmt(scale.max)} ratings were found; "
                               "rating charts will be empty.")
    if report.roles.has_date and cleaned["date"].isna().all():
        report.warnings.append("No readable dates were found; trend charts will be hidden.")
    return cleaned, report
