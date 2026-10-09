"""Domain profiles: everything domain-specific lives in profiles/*.json.

The Feedlytics core only knows about feedback records and generic column
*roles* (text, title, rating, date, entity, segments, numeric drivers,
outcomes). A profile supplies the domain: how the dataset's columns map to
those roles, the topic/issue taxonomy for AI labelling, outcome flags,
baseline keywords, numeric bands and notes about the data.

Bundled profiles:
  generic             default for unknown datasets
  apparel_ecommerce   demo: Kaggle women's clothing reviews
  food_delivery_demo  legacy demo: the synthetic food-delivery dataset
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from app import config

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
DEFAULT_PROFILE = "generic"
CORE_ROLES = ("review_id", "review_text", "review_title", "rating", "date")


class ProfileError(ValueError):
    """A profile file is missing or invalid. The message says what to fix."""


def normalize_header(name: object) -> str:
    """'Review Text ' -> 'review_text'."""
    text = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower())
    return text.strip("_")


# ---------------------------------------------------------------------------
# Profile parts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatingScale:
    min: float = config.RATING_MIN
    max: float = config.RATING_MAX
    negative_max: float = config.NEGATIVE_MAX_RATING
    positive_min: float = config.POSITIVE_MIN_RATING

    def sentiment(self, rating: float) -> Optional[str]:
        if rating is None or rating != rating:  # None or NaN
            return None
        if rating <= self.negative_max:
            return "Negative"
        if rating >= self.positive_min:
            return "Positive"
        return "Neutral"


@dataclass(frozen=True)
class BaselineSentiment:
    """Optional calibration of the non-AI baseline's review-level sentiment.

    score = VADER compound − complaint_penalty × (complaints detected);
    score ≤ negative_max → Negative, ≥ positive_min → Positive, else Neutral.
    Profiles without this block keep the default baseline rule.
    """
    negative_max: float
    positive_min: float
    complaint_penalty: float = 0.0


@dataclass(frozen=True)
class NumericSpec:
    column: str
    label: str
    valid_range: Optional[Tuple[float, float]] = None
    bins: Optional[Tuple[float, ...]] = None
    band_labels: Optional[Tuple[str, ...]] = None
    closed: str = "right"  # "right": (a, b]   "left": [a, b)
    band_column: Optional[str] = None


@dataclass(frozen=True)
class OutcomeSpec:
    """A yes/no column such as 'Recommended'. Stored as 1.0 / 0.0 / NaN."""
    column: str
    label: str
    positive_values: Tuple[str, ...] = ("1", "yes", "true", "y")
    negative_values: Tuple[str, ...] = ("0", "no", "false", "n")


@dataclass(frozen=True)
class Taxonomy:
    topics: Dict[str, Dict[str, str]]       # topic -> {issue: definition}
    flags: Dict[str, str] = field(default_factory=dict)  # flag name -> definition

    @property
    def topic_names(self) -> List[str]:
        return list(self.topics)

    @property
    def issue_names(self) -> List[str]:
        return [issue for issues in self.topics.values() for issue in issues]

    @property
    def issue_to_topic(self) -> Dict[str, str]:
        return {issue: topic for topic, issues in self.topics.items() for issue in issues}

    @property
    def definitions(self) -> Dict[str, str]:
        return {issue: d for issues in self.topics.values() for issue, d in issues.items()}


@dataclass
class Roles:
    """Which columns of a cleaned dataset play which role."""
    has_title: bool = False
    has_rating: bool = False
    has_date: bool = False
    entity: Optional[str] = None
    entity_label: Optional[str] = None
    segments: List[Tuple[str, str]] = field(default_factory=list)  # (column, label)
    numeric: List[NumericSpec] = field(default_factory=list)
    outcomes: List[OutcomeSpec] = field(default_factory=list)

    @property
    def dimensions(self) -> List[Tuple[str, str]]:
        """Entity first, then segments — the breakdowns the analytics can use."""
        dims = [(self.entity, self.entity_label or "Entity")] if self.entity else []
        return dims + list(self.segments)

    def label_for(self, column: str) -> str:
        for col, label in self.dimensions:
            if col == column:
                return label
        return column.replace("_", " ").capitalize()


@dataclass
class Profile:
    name: str
    display_name: str
    domain_description: str
    rating_scale: RatingScale
    columns: Dict[str, str]           # normalised source header -> column name
    column_synonyms: Dict[str, str]   # normalised alternative header -> column name
    detect_columns: Tuple[str, ...]
    value_aliases: Dict[str, Dict[str, str]]
    entity: Dict
    segments: List[Dict]
    numeric: List[NumericSpec]
    outcomes: List[OutcomeSpec]
    taxonomy: Taxonomy
    baseline_keywords: Dict[str, Dict[str, List[str]]]
    notes: List[str]
    source: Optional[Path] = None
    baseline_sentiment: Optional[BaselineSentiment] = None

    @property
    def fingerprint(self) -> str:
        """Short hash of the taxonomy + description; part of the AI cache key."""
        payload = json.dumps({"d": self.domain_description, "t": self.taxonomy.topics,
                              "f": self.taxonomy.flags}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def resolve_roles(self, columns: Iterable[str]) -> Roles:
        """Match the profile's role definitions against a cleaned dataset's columns."""
        present = list(columns)
        roles = Roles(has_title="review_title" in present, has_rating="rating" in present,
                      has_date="date" in present)
        entity_col = _pick(self.entity, present)
        if entity_col:
            roles.entity = entity_col
            roles.entity_label = self.entity.get("label") or _pretty(entity_col)
        for spec in self.segments:
            if "candidates" in spec:
                cols = [c for c in spec["candidates"] if c in present and c != roles.entity]
                roles.segments += [(c, _pretty(c)) for c in cols]
            elif spec["column"] in present and spec["column"] != roles.entity:
                roles.segments.append((spec["column"], spec.get("label") or _pretty(spec["column"])))
        roles.numeric = [n for n in self.numeric if n.column in present]
        roles.outcomes = [o for o in self.outcomes if o.column in present]
        return roles


def _merge_tokens(extra: Optional[Sequence], standard: Tuple[str, ...]) -> Tuple[str, ...]:
    values = list(standard) + [str(v).strip().lower() for v in (extra or [])]
    return tuple(dict.fromkeys(values))


def _pick(spec: Dict, present: Sequence[str]) -> Optional[str]:
    if not spec:
        return None
    if "column" in spec:
        return spec["column"] if spec["column"] in present else None
    return next((c for c in spec.get("candidates", []) if c in present), None)


def _pretty(column: str) -> str:
    return column.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def available_profiles() -> List[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


_CACHE: Dict[str, Profile] = {}


def load_profile(name_or_path: Union[str, Path, Profile, None] = None) -> Profile:
    """Load a bundled profile by name, or any profile JSON file by path."""
    if isinstance(name_or_path, Profile):
        return name_or_path
    name_or_path = name_or_path or DEFAULT_PROFILE
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = PROFILE_DIR / f"{name_or_path}.json"
    key = str(path.resolve())
    if key in _CACHE:
        return _CACHE[key]
    if not path.exists():
        raise ProfileError(
            f"Profile '{name_or_path}' not found. Available: {', '.join(available_profiles())}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path.name} is not valid JSON: {exc}") from None
    profile = parse_profile(raw, source=path)
    _CACHE[key] = profile
    return profile


def parse_profile(raw: Dict, source: Optional[Path] = None) -> Profile:
    """Build and validate a Profile from parsed JSON. Raises ProfileError."""
    where = source.name if source else "profile"
    errors: List[str] = []

    for key in ("name", "taxonomy"):
        if key not in raw:
            errors.append(f"missing required key '{key}'")
    if errors:
        raise ProfileError(f"{where}: " + "; ".join(errors))

    scale_raw = raw.get("rating_scale") or {}
    scale = RatingScale(**{k: float(v) for k, v in scale_raw.items()}) if scale_raw else RatingScale()
    if not (scale.min <= scale.negative_max < scale.positive_min <= scale.max):
        errors.append("rating_scale must satisfy min ≤ negative_max < positive_min ≤ max")

    columns = {normalize_header(k): v for k, v in (raw.get("columns") or {}).items()}
    synonyms = {normalize_header(k): v for k, v in (raw.get("column_synonyms") or {}).items()}

    # Taxonomy: every topic needs at least one issue; issue names must be unique
    # (each issue belongs to exactly one topic); "Other/Other" must exist.
    topics = raw["taxonomy"]
    seen: Dict[str, str] = {}
    if not isinstance(topics, dict) or not topics:
        errors.append("taxonomy must be a non-empty object of topic -> {issue: definition}")
        topics = {}
    for topic, issues in topics.items():
        if not isinstance(issues, dict) or not issues:
            errors.append(f"topic '{topic}' has no issues")
            continue
        for issue in issues:
            if issue in seen:
                errors.append(f"issue '{issue}' appears in both '{seen[issue]}' and '{topic}'")
            seen[issue] = topic
    if topics and topics.get("Other", {}).get("Other") is None:
        errors.append("taxonomy must include topic 'Other' with issue 'Other'")

    numeric = []
    for spec in raw.get("numeric") or []:
        bins = tuple(spec["bins"]) if spec.get("bins") else None
        labels = tuple(spec["band_labels"]) if spec.get("band_labels") else None
        if bins and (not labels or len(labels) != len(bins) - 1):
            errors.append(f"numeric '{spec.get('column')}': need exactly {len(bins) - 1} band_labels")
        if bins and list(bins) != sorted(set(bins)):
            errors.append(f"numeric '{spec.get('column')}': bins must be increasing")
        if spec.get("closed", "right") not in ("left", "right"):
            errors.append(f"numeric '{spec.get('column')}': closed must be 'left' or 'right'")
        vr = spec.get("valid_range")
        numeric.append(NumericSpec(
            column=spec["column"], label=spec.get("label") or _pretty(spec["column"]),
            valid_range=(float(vr[0]), float(vr[1])) if vr else None, bins=bins, band_labels=labels,
            closed=spec.get("closed", "right"),
            band_column=spec.get("band_column") or (f"{spec['column']}_band" if bins else None)))

    # Profile values extend (never replace) the standard yes/no/true/false/1/0 tokens.
    outcomes = [OutcomeSpec(column=o["column"], label=o.get("label") or _pretty(o["column"]),
                            positive_values=_merge_tokens(o.get("positive_values"), OutcomeSpec.positive_values),
                            negative_values=_merge_tokens(o.get("negative_values"), OutcomeSpec.negative_values))
                for o in raw.get("outcomes") or []]

    keywords = raw.get("baseline_keywords") or {}
    for issue in (keywords.get("issues") or {}):
        if issue not in seen:
            errors.append(f"baseline keyword issue '{issue}' is not in the taxonomy")
    for topic in (keywords.get("praise") or {}):
        if topic not in topics:
            errors.append(f"baseline praise topic '{topic}' is not in the taxonomy")
    flags = raw.get("outcome_flags") or {}
    for flag in flags:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", flag):
            errors.append(f"outcome flag '{flag}' must be snake_case")
    for flag in (keywords.get("flags") or {}):
        if flag not in flags:
            errors.append(f"baseline keyword flag '{flag}' is not an outcome flag")

    for spec in raw.get("segments") or []:
        if "column" not in spec and "candidates" not in spec:
            errors.append("each segment needs 'column' or 'candidates'")

    calibration = None
    if raw.get("baseline_sentiment") is not None:
        bs = raw["baseline_sentiment"]
        try:
            calibration = BaselineSentiment(negative_max=float(bs["negative_max"]),
                                            positive_min=float(bs["positive_min"]),
                                            complaint_penalty=float(bs.get("complaint_penalty", 0.0)))
        except (KeyError, TypeError, ValueError):
            errors.append("baseline_sentiment needs numeric negative_max and positive_min")
        else:
            if not calibration.negative_max < calibration.positive_min:
                errors.append("baseline_sentiment: negative_max must be below positive_min")
            if calibration.complaint_penalty < 0:
                errors.append("baseline_sentiment: complaint_penalty must be ≥ 0")

    if errors:
        raise ProfileError(f"{where}: " + "; ".join(errors))

    return Profile(
        name=raw["name"],
        display_name=raw.get("display_name") or raw["name"],
        domain_description=raw.get("domain_description") or "customer feedback",
        rating_scale=scale,
        columns=columns,
        column_synonyms=synonyms,
        detect_columns=tuple(normalize_header(c) for c in raw.get("detect_columns") or []),
        value_aliases=raw.get("value_aliases") or {},
        entity=raw.get("entity") or {},
        segments=list(raw.get("segments") or []),
        numeric=numeric,
        outcomes=outcomes,
        taxonomy=Taxonomy(topics=topics, flags=flags),
        baseline_keywords={k: dict(keywords.get(k) or {}) for k in ("issues", "praise", "flags")},
        notes=list(raw.get("notes") or []),
        source=source,
        baseline_sentiment=calibration,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_profile(headers: Iterable[object]) -> Tuple[Profile, str]:
    """Pick the profile whose distinctive columns best match the file's headers.

    Each profile lists ``detect_columns`` — headers that only make sense in its
    domain. The profile with the most matches wins; no match (or a tie) falls
    back to the generic profile. Returns the profile and a human-readable reason.
    """
    normalized = {normalize_header(h) for h in headers}
    scores = []
    for name in available_profiles():
        profile = load_profile(name)
        hits = sorted(normalized & set(profile.detect_columns))
        if hits:
            scores.append((len(hits), name, hits))
    scores.sort(reverse=True)
    if not scores or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        reason = ("no domain-specific columns recognised" if not scores
                  else "columns match several profiles equally")
        return load_profile(DEFAULT_PROFILE), f"generic profile used: {reason}"
    _, name, hits = scores[0]
    return load_profile(name), f"matched columns: {', '.join(hits)}"
