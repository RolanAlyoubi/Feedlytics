"""Generate a SYNTHETIC food-delivery review dataset for development and demos.

!! Everything produced by this script is invented. !!
The patterns in the data come from the rules declared in this file
(RESTAURANTS, ISSUES, time_multipliers). Any "insight" found in this dataset
is those rules being rediscovered — it says nothing about real customers.
That is useful for testing the pipeline: we know what the analysis *should* find.

Outputs (in data/):
  sample_reviews_SYNTHETIC.csv               the messy CSV a user would upload
  sample_reviews_SYNTHETIC_ground_truth.csv  the true sentiment/topic/issue of
                                             every review, for evaluating the
                                             AI labelling step in phase 2

Usage:
  python -m scripts.generate_synthetic_data            # default seed 42
  python -m scripts.generate_synthetic_data --seed 7 --out-dir data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.profiles import load_profile

# The issue -> topic mapping comes from the food-delivery demo profile.
ISSUE_TO_TOPIC = load_profile("food_delivery_demo").taxonomy.issue_to_topic

START_MONTH = "2025-01-01"
N_MONTHS = 18                 # Jan 2025 – Jun 2026
BASE_REVIEWS_PER_MONTH = 130
MONTHLY_GROWTH = 0.03         # platform volume grows ~3% per month
DECEMBER_VOLUME_BOOST = 1.4   # holiday season

# Shares of review types before restaurant / time effects.
BASE_NEGATIVE = 0.27
BASE_MIXED = 0.10
BASE_NEUTRAL = 0.12           # remainder is positive
RATING_NOISE_RATE = 0.04      # star rating that contradicts the text (mis-click)


# ---------------------------------------------------------------------------
# Issues: base frequency weight, typical star rating, complaint phrasings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    weight: float
    mean_rating: float
    phrases: Tuple[str, ...]


ISSUES: Dict[str, Issue] = {
    "Late delivery": Issue(3.0, 1.9, (
        "My order arrived {late_by} minutes later than the estimate.",
        "Delivery took {minutes} minutes, way longer than promised.",
        "The driver showed up almost an hour late.",
        "Waited over {minutes} minutes for my food, the ETA kept changing.",
        "Delivery was really slow again tonight.",
        "It said 25 minutes and it took {minutes}.",
    )),
    "Courier behaviour": Issue(0.8, 2.0, (
        "The courier was rude when he handed over the bag.",
        "Driver refused to come up to the door and left the food downstairs.",
        "The delivery guy didn't follow the drop-off instructions at all.",
        "Courier called me five times and then got angry.",
    )),
    "Order not delivered": Issue(0.6, 1.1, (
        "The app says delivered but I never received anything.",
        "My order never arrived and nobody could tell me where it was.",
        "Food was marked as delivered to the wrong building, never got it.",
    )),
    "Cold food": Issue(1.6, 2.2, (
        "The {item} was cold when it arrived.",
        "Everything was lukewarm at best.",
        "Food came cold, had to reheat the {item} myself.",
        "Fries were cold and soggy.",
    )),
    "Poor taste": Issue(1.2, 2.3, (
        "The {item} was bland and tasted nothing like before.",
        "Honestly the {item} was not good, way too salty.",
        "Quality has gone down, the {item} tasted cheap.",
        "Didn't enjoy the food at all this time.",
    )),
    "Undercooked or stale food": Issue(0.6, 1.6, (
        "The {item} was undercooked in the middle.",
        "Bread was stale and the {item} tasted old.",
        "Chicken in the {item} was still pink inside, not safe.",
    )),
    "Missing items": Issue(1.8, 1.8, (
        "My {item} was missing from the bag.",
        "Ordered three items and only got two.",
        "The drinks were missing again.",
        "Half of my order was not in the bag, missing the {item}.",
        "No sauces or cutlery even though I asked for them.",
    )),
    "Wrong items": Issue(1.0, 2.0, (
        "I got someone else's order instead of mine.",
        "Received the wrong {item}, not what I ordered.",
        "They sent a completely different {item}.",
    )),
    "Spilled or leaking": Issue(0.9, 2.2, (
        "The soup leaked all over the bag.",
        "Sauce spilled everywhere, the box was a mess.",
        "Drink was spilled and soaked the {item}.",
    )),
    "Poor packaging": Issue(0.7, 2.6, (
        "Packaging was flimsy and the {item} got squashed.",
        "The container was crushed when it arrived.",
        "Everything was thrown in one bag with no separation.",
    )),
    "Unresponsive support": Issue(0.9, 1.7, (
        "Customer support never replied to my complaint.",
        "Tried to contact support, waited 40 minutes in chat with no answer.",
        "Support just sent a copy-paste message and closed the ticket.",
    )),
    "Refund problems": Issue(0.7, 1.6, (
        "Still waiting for my refund after two weeks.",
        "They refused to refund the missing items.",
        "Refund was only a voucher, not my money back.",
    )),
    "App crashes or bugs": Issue(0.7, 2.3, (
        "The app keeps crashing at checkout.",
        "App froze twice while I was placing the order.",
        "After the update the app is so buggy, couldn't even open the menu.",
        "The app logged me out in the middle of ordering.",
    )),
    "Payment failure": Issue(0.5, 2.1, (
        "Payment failed but I was still charged.",
        "My card was declined in the app but works everywhere else.",
        "Got charged twice for the same order.",
    )),
    "Inaccurate tracking": Issue(0.6, 2.5, (
        "Tracking map showed the driver going in circles.",
        "The live tracking was useless, it said arriving for 20 minutes.",
        "Order status never updated after it was picked up.",
    )),
    "Too expensive": Issue(0.9, 2.6, (
        "Way too expensive for the portion size.",
        "Prices went up again, {order_value} for a small meal is too much.",
        "Not worth the price anymore.",
    )),
    "Hidden or high fees": Issue(0.6, 2.4, (
        "The service fee is ridiculous now.",
        "Delivery fee plus service fee doubled the price of my order.",
        "New fees appeared at checkout that weren't shown before.",
        "Paid more in fees than the food itself.",
    )),
    "Promo not applied": Issue(0.5, 2.7, (
        "My promo code didn't work even though it was valid.",
        "The discount wasn't applied and support couldn't fix it.",
    )),
}

PRAISE = {
    "Food Quality": (
        "The {item} was delicious.",
        "Food was fresh and really tasty.",
        "Best {item} I've had in a while.",
        "Great flavour, generous portions.",
        "The {item} was hot and perfectly cooked.",
    ),
    "Delivery": (
        "Delivery was fast, arrived in {minutes} minutes.",
        "Arrived earlier than expected.",
        "The driver was friendly and quick.",
    ),
    "Packaging": (
        "Everything was packed neatly and nothing spilled.",
        "Good packaging, food stayed warm.",
    ),
    "App": (
        "Ordering through the app was easy.",
        "Tracking was accurate the whole way.",
    ),
    "Pricing": (
        "Good value for money.",
        "Great deal with the promo.",
    ),
    "Customer Service": (
        "Support sorted out my issue quickly.",
    ),
}
PRAISE_WEIGHTS = {"Food Quality": 5, "Delivery": 2.5, "Packaging": 1, "App": 1,
                  "Pricing": 1, "Customer Service": 0.5}

SHORT_POSITIVE = ("Great food!", "Love it", "Perfect as always", "Amazing!", "10/10 would order again")
NEUTRAL_TEXTS = (
    "Order was okay, nothing special.",
    "Average experience. Food was fine and arrived on time.",
    "It was alright. Not bad, not great.",
    "Decent {item} but I've had better.",
    "Food was fine. Delivery was a bit slow but acceptable.",
    "Standard order, no complaints but nothing memorable.",
    "The {item} was okay. Arrived in about {minutes} minutes.",
    "Fine for a quick meal, the {item} was average.",
    "Nothing wrong with it, nothing exciting either.",
    "{item} was as expected. Delivery took {minutes} minutes.",
    "Mixed feelings, the {item} was good but the sides were meh.",
)
CONTEXT_PREFIXES = (
    "Ordered the {item} for dinner.", "Lunch order from {restaurant}.",
    "Ordered for the family on the weekend.", "Late night order.",
    "Second order this week.", "First time trying {restaurant}.",
    "Got the {item} and a drink.", "Ordered to the office today.",
    "Rainy evening order.", "Ordered {item} for two.",
)
CONTEXT_RATE = 0.45
LOWERCASE_RATE = 0.08
NEGATIVE_OPENERS = ("Very disappointed.", "Not happy with this order.", "Terrible experience.",
                    "Ugh.", "Second time this happened.", "")
NEGATIVE_CLOSERS = ("Won't order again.", "Please fix this.", "Really frustrating.",
                    "Expected better.", "")
POSITIVE_CLOSERS = ("Will order again!", "Highly recommend.", "Thanks!", "")
MIXED_CONNECTORS = (" But ", " However, ", " Unfortunately ", " The only problem: ")


# ---------------------------------------------------------------------------
# Restaurants (fictional) and their built-in strengths / weaknesses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Restaurant:
    name: str
    cuisine: str
    popularity: float
    avg_order: float
    base_delivery: float
    negative_factor: float
    issue_multipliers: Dict[str, float]
    items: Tuple[str, ...]


RESTAURANTS: Tuple[Restaurant, ...] = (
    Restaurant("Burger Barn", "Burgers", 1.6, 55, 34, 1.15, {"Late delivery": 2.0, "Cold food": 1.5},
               ("burger", "fries", "chicken sandwich", "milkshake")),
    Restaurant("Slice Republic", "Pizza", 1.5, 70, 36, 1.15, {"Late delivery": 2.2},
               ("pizza", "garlic bread", "wings", "pasta")),
    Restaurant("Sakura Sushi House", "Sushi", 0.9, 130, 38, 0.9, {"Too expensive": 2.5, "Hidden or high fees": 1.5},
               ("sushi platter", "ramen", "salmon roll", "gyoza")),
    Restaurant("Spice Route", "Indian", 1.0, 80, 40, 1.1, {"Missing items": 2.5},
               ("biryani", "butter chicken", "naan", "samosa")),
    Restaurant("Green Bowl", "Healthy", 0.8, 60, 30, 0.55, {},
               ("salad bowl", "smoothie", "wrap", "acai bowl")),
    Restaurant("Cedar Grill", "Middle Eastern", 1.1, 75, 35, 0.9, {},
               ("mixed grill", "hummus", "kebab plate", "falafel")),
    Restaurant("Golden Wok", "Chinese", 1.0, 65, 37, 1.1, {"Spilled or leaking": 3.0, "Poor packaging": 1.5},
               ("noodles", "fried rice", "dumplings", "hot and sour soup")),
    Restaurant("Sweet Spot Desserts", "Desserts", 0.6, 35, 30, 0.8, {"Poor packaging": 2.0},
               ("cheesecake", "cupcakes", "brownie", "ice cream")),
    Restaurant("Taco Fiesta", "Mexican", 0.8, 50, 33, 1.0, {"Wrong items": 1.8},
               ("tacos", "burrito", "nachos", "quesadilla")),
    Restaurant("Pasta Piazza", "Italian", 0.8, 75, 36, 0.95, {"Poor taste": 1.6},
               ("lasagna", "carbonara", "risotto", "tiramisu")),
    Restaurant("Shawarma Corner", "Middle Eastern", 1.2, 40, 28, 1.05, {"Cold food": 2.2},
               ("shawarma wrap", "fries", "chicken plate", "garlic sauce")),
    Restaurant("Morning Brew Café", "Breakfast & Coffee", 0.7, 38, 26, 0.85, {},
               ("latte", "croissant", "breakfast sandwich", "pancakes")),
)


def time_multipliers(month_index: int) -> Dict[str, float]:
    """Platform-wide events, by month (0 = Jan 2025).

    - Sep–Oct 2025 (index 8–9): simulated buggy app release -> app crashes ×4.
    - From Mar 2026 (index 14): simulated new service fee -> fee complaints ×2.5.
    - During 2026 (index 12+): late deliveries rise gradually, up to ×1.6.
    """
    m: Dict[str, float] = {}
    if month_index in (8, 9):
        m["App crashes or bugs"] = 4.0
    if month_index >= 14:
        m["Hidden or high fees"] = 2.5
    if month_index >= 12:
        m["Late delivery"] = 1.0 + 0.6 * (month_index - 11) / (N_MONTHS - 12)
    return m


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

BASE_ISSUE_WEIGHT = sum(i.weight for i in ISSUES.values())


class Generator:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def choice(self, options, p=None):
        return options[self.rng.choice(len(options), p=p)]

    def issue_weights(self, restaurant: Restaurant, month_index: int) -> Dict[str, float]:
        timed = time_multipliers(month_index)
        return {
            name: issue.weight * restaurant.issue_multipliers.get(name, 1.0) * timed.get(name, 1.0)
            for name, issue in ISSUES.items()
        }

    def pick_issue(self, weights: Dict[str, float], exclude: Optional[str] = None) -> str:
        names = [n for n in weights if n != exclude]
        w = np.array([weights[n] for n in names])
        return names[self.rng.choice(len(names), p=w / w.sum())]

    def pick_praise_topic(self) -> str:
        topics = list(PRAISE_WEIGHTS)
        w = np.array(list(PRAISE_WEIGHTS.values()))
        return topics[self.rng.choice(len(topics), p=w / w.sum())]

    def fill(self, template: str, ctx: Dict) -> str:
        return template.format(**ctx)

    def review(self, restaurant: Restaurant, month_index: int, date: pd.Timestamp) -> Dict:
        weights = self.issue_weights(restaurant, month_index)
        # More issue "pressure" (restaurant weaknesses, platform events) -> more negatives.
        pressure = sum(weights.values()) / BASE_ISSUE_WEIGHT
        p_neg = min(0.6, BASE_NEGATIVE * restaurant.negative_factor * pressure)
        p_mix = BASE_MIXED * restaurant.negative_factor
        p_neu = BASE_NEUTRAL
        kind = self.choice(["negative", "mixed", "neutral", "positive"],
                           p=[p_neg, p_mix, p_neu, 1 - p_neg - p_mix - p_neu])

        delivery = float(self.rng.lognormal(np.log(restaurant.base_delivery), 0.22))
        order_value = float(self.rng.lognormal(np.log(restaurant.avg_order), 0.35))
        issues: List[str] = []
        praise: List[str] = []
        if kind in ("negative", "mixed"):
            issues.append(self.pick_issue(weights))
            if kind == "negative" and self.rng.random() < 0.25:
                issues.append(self.pick_issue(weights, exclude=issues[0]))
        if kind in ("positive", "mixed"):
            praise.append(self.pick_praise_topic())
            if kind == "positive" and self.rng.random() < 0.35:
                second = self.pick_praise_topic()
                if second != praise[0]:
                    praise.append(second)

        # Make numeric fields consistent with what the text says.
        if "Late delivery" in issues:
            delivery += float(self.rng.uniform(20, 60))
        elif "Delivery" in praise:
            delivery *= 0.8
        if {"Too expensive", "Hidden or high fees"} & set(issues):
            order_value *= 1.25
        delivery = round(delivery)
        order_value = round(order_value, 2)

        ctx = {
            "item": self.choice(restaurant.items),
            "restaurant": restaurant.name,
            "minutes": delivery,
            "late_by": max(10, delivery - restaurant.base_delivery),
            "order_value": f"{order_value:.0f}",
        }
        text, rating = self.compose(kind, issues, praise, ctx)
        # Writing-style variety so that reviews are not copies of a few templates.
        if len(text) > 25 and self.rng.random() < CONTEXT_RATE:
            text = self.fill(self.choice(CONTEXT_PREFIXES), ctx) + " " + text
        if self.rng.random() < LOWERCASE_RATE:
            text = text.lower()

        # Ground-truth sentiment describes the TEXT. Mixed reviews take the
        # overall tone implied by the rating they were written with.
        if kind == "mixed":
            true_sentiment = "Negative" if rating <= 2 else "Positive" if rating >= 4 else "Neutral"
        else:
            true_sentiment = kind.title()

        if self.rng.random() < RATING_NOISE_RATE:  # star rating that contradicts the text
            rating = int(self.rng.integers(1, 6))

        primary_topic = (ISSUE_TO_TOPIC[issues[0]] if issues
                         else praise[0] if praise else "Other")
        aspects = [f"{ISSUE_TO_TOPIC[i]}:negative:{i}" for i in issues] + \
                  [f"{p}:positive:" for p in praise]
        return {
            "date": date,
            "restaurant": restaurant.name,
            "category": restaurant.cuisine,
            "review_text": text,
            "rating": rating,
            "delivery_time": delivery,
            "order_value": order_value,
            "_kind": kind,
            "_true_sentiment": true_sentiment,
            "_true_topic": primary_topic,
            "_true_issue": issues[0] if issues else "",
            "_aspects": "|".join(aspects),
        }

    def compose(self, kind: str, issues: List[str], praise: List[str], ctx: Dict) -> Tuple[str, int]:
        if kind == "neutral":
            rating = self.choice([3, 3, 3, 3, 2, 4])
            return self.fill(self.choice(NEUTRAL_TEXTS), ctx), rating

        if kind == "positive":
            if self.rng.random() < 0.12:
                return self.choice(SHORT_POSITIVE), self.choice([5, 5, 4])
            parts = [self.fill(self.choice(PRAISE[t]), ctx) for t in praise]
            parts.append(self.choice(POSITIVE_CLOSERS))
            return " ".join(p for p in parts if p), self.choice([5, 5, 5, 4, 4])

        issue_phrases = [self.fill(self.choice(ISSUES[i].phrases), ctx) for i in issues]
        mean = min(ISSUES[i].mean_rating for i in issues)
        if kind == "negative":
            rating = int(np.clip(round(self.rng.normal(mean, 0.6)), 1, 3))
            parts = [self.choice(NEGATIVE_OPENERS)] + issue_phrases + [self.choice(NEGATIVE_CLOSERS)]
            return " ".join(p for p in parts if p), rating

        # mixed: praise first, then the problem
        rating = int(np.clip(round(self.rng.normal(mean + 1.1, 0.6)), 2, 4))
        good = self.fill(self.choice(PRAISE[praise[0]]), ctx)
        bad = issue_phrases[0]
        return good + self.choice(MIXED_CONNECTORS) + bad[0].lower() + bad[1:], rating


def generate_clean(seed: int = 42) -> pd.DataFrame:
    gen = Generator(seed)
    popularity = np.array([r.popularity for r in RESTAURANTS])
    popularity = popularity / popularity.sum()
    rows = []
    months = pd.date_range(START_MONTH, periods=N_MONTHS, freq="MS")
    for idx, month_start in enumerate(months):
        volume = BASE_REVIEWS_PER_MONTH * (1 + MONTHLY_GROWTH) ** idx
        if month_start.month == 12:
            volume *= DECEMBER_VOLUME_BOOST
        n = int(gen.rng.poisson(volume))
        days = month_start.days_in_month
        for _ in range(n):
            restaurant = RESTAURANTS[gen.rng.choice(len(RESTAURANTS), p=popularity)]
            ts = month_start + pd.Timedelta(days=int(gen.rng.integers(0, days)),
                                            minutes=int(gen.rng.integers(10 * 60, 23 * 60 + 59)))
            rows.append(gen.review(restaurant, idx, ts))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.insert(0, "review_id", [f"R{i:05d}" for i in range(1, len(df) + 1)])
    return df


# ---------------------------------------------------------------------------
# Deliberate data-quality problems, so the cleaning step has real work to do
# ---------------------------------------------------------------------------

def add_data_quality_problems(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    n = len(out)

    def pick(rate: float) -> np.ndarray:
        return rng.random(n) < rate

    # Dates written in several formats.
    fmt = rng.choice(4, size=n, p=[0.80, 0.10, 0.05, 0.05])
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y", "%d %b %Y"]
    out["date"] = [d.strftime(formats[f]) for d, f in zip(out["date"], fmt)]
    bad_date = pick(0.005)
    out.loc[bad_date, "date"] = rng.choice(["not available", "2031-01-15", "13/45/2025", ""],
                                           size=bad_date.sum())

    # Ratings: some messy-but-valid, some invalid, some missing.
    out["rating"] = out["rating"].astype(object)
    messy = pick(0.01)
    out.loc[messy, "rating"] = [rng.choice([f"{r}/5", f"{r} stars", f"{r}.0"])
                                for r in out.loc[messy, "rating"]]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    worded = pick(0.003) & ~messy
    out.loc[worded, "rating"] = [words[int(r)] for r in out.loc[worded, "rating"]]
    invalid = pick(0.01) & ~messy & ~worded
    out.loc[invalid, "rating"] = rng.choice(["6", "0", "10", "-1", "abc", ""], size=invalid.sum())

    # Review text: missing, placeholder, HTML / whitespace noise.
    empty_text = pick(0.015)
    out.loc[empty_text, "review_text"] = rng.choice(["", "N/A", "   "], size=empty_text.sum())
    noisy = pick(0.01) & ~empty_text
    out.loc[noisy, "review_text"] = ["  " + t.replace(". ", ".<br>  ") + "  "
                                     for t in out.loc[noisy, "review_text"]]

    # Restaurant names typed inconsistently.
    variant = pick(0.02)
    out.loc[variant, "restaurant"] = [rng.choice([name.lower(), name.upper(), f"  {name} "])
                                      for name in out.loc[variant, "restaurant"]]

    # Delivery time: units in the text, missing, impossible values.
    out["delivery_time"] = out["delivery_time"].astype(object)
    with_unit = pick(0.02)
    out.loc[with_unit, "delivery_time"] = [f"{v} min" for v in out.loc[with_unit, "delivery_time"]]
    out.loc[pick(0.02), "delivery_time"] = ""
    impossible = pick(0.005)
    out.loc[impossible, "delivery_time"] = rng.choice(["-5", "999", "0"], size=impossible.sum())

    # Order value: currency symbols, missing, impossible values.
    out["order_value"] = out["order_value"].astype(object)
    currency = pick(0.02)
    out.loc[currency, "order_value"] = [rng.choice([f"${v:.2f}", f"SAR {v:.2f}"])
                                        for v in out.loc[currency, "order_value"]]
    out.loc[pick(0.02), "order_value"] = ""
    bad_value = pick(0.003)
    out.loc[bad_value, "order_value"] = rng.choice(["0", "-20"], size=bad_value.sum())

    # Missing review ids.
    out.loc[pick(0.005), "review_id"] = ""

    # Duplicates: exact copies, and the same review re-submitted under a new id.
    exact = out.sample(frac=0.02, random_state=seed)
    resubmitted = out[~out.index.isin(exact.index)].sample(frac=0.01, random_state=seed + 2).copy()
    resubmitted["review_id"] = [f"R9{i:04d}" for i in range(len(resubmitted))]
    out = pd.concat([out, exact, resubmitted], ignore_index=True)

    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


PUBLIC_COLUMNS = ["review_id", "review_text", "rating", "date", "restaurant",
                  "category", "delivery_time", "order_value"]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    clean = generate_clean(args.seed)
    truth = clean[["review_id", "_kind", "_true_sentiment", "_true_topic", "_true_issue", "_aspects"]]
    truth.columns = ["review_id", "review_type", "true_sentiment", "true_topic", "true_issue", "true_aspects"]
    messy = add_data_quality_problems(clean[PUBLIC_COLUMNS], args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = args.out_dir / "sample_reviews_SYNTHETIC.csv"
    truth_path = args.out_dir / "sample_reviews_SYNTHETIC_ground_truth.csv"
    messy.to_csv(reviews_path, index=False)
    truth.to_csv(truth_path, index=False)
    print(f"Wrote {len(messy):,} rows -> {reviews_path}  (SYNTHETIC)")
    print(f"Wrote {len(truth):,} rows -> {truth_path}")


if __name__ == "__main__":
    main()
