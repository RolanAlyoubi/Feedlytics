# Feedlytics — AI-Powered Customer Insights

Feedlytics is a domain-neutral customer-feedback analytics platform. It turns unstructured
reviews, survey comments or support feedback into ranked problems and evidence-backed
recommendations, and every recommendation can be traced back to a number in the data.
The business domain is plugged in as a **profile**: it isn't hard-coded.

> **Project status**
> Done: domain profiles, data cleaning, analytics, end-to-end pipeline, AI labelling layer (with
> non-AI fallback, caching and weighted sampling), evaluation tooling, rule-based insights and
> recommendations, a single-page Streamlit dashboard, 252 tests.
> **The demo runs on the non-AI baseline (VADER + keyword rules).**
>
> Evaluation on 200 hand-labelled apparel reviews is limited, achieving about 42% text-sentiment accuracy and 50% main-topic accuracy. The apparel evaluation and calibration reports are excluded from this repository because the dataset's redistribution terms have not been verified. These accuracy figures are indicative and should be interpreted with caution.
> Insights are rule-based templates over the computed tables. AI-written findings are not implemented.
>
> **No Anthropic API access.** This project has no Anthropic API credits, so **Claude has never
> been run on real data and its accuracy has not been measured.** The Claude integration is
> implemented and tested only against a mocked API, and is kept for future use. Everything
> else (cleaning, profiles, analytics, sampling and weights, the non-AI baseline and the
> evaluations) runs locally without an API key. See [Running without an API key](#running-without-an-api-key).

---

## Problem statement

Companies collect large volumes of customer feedback, but raw text is unstructured, hard to
categorise and slow to read. A rating shows *that* customers are unhappy, not *why*.
Teams struggle to answer:

- What are customers most dissatisfied with, and what do they praise?
- Which problems are most common, and which hurt ratings or recommendations most?
- Are complaints increasing or decreasing (when the data has dates)?
- What should the business fix first?

## Objective

A transparent pipeline: **Data → Analysis → AI → Insight → Decision**.

1. Code computes every statistic with pandas.
2. AI labels each review's text (sentiment, topic, specific issues, outcome flags). It
   interprets text; it does not compute statistics.
3. Outputs are labelled as *observed data*, *AI interpretation* or *recommendation*.

## Domain profiles

Everything domain-specific lives in a JSON file in [`profiles/`](profiles/). The core
modules contain no restaurant, clothing or other domain logic.

| Profile | Used for | Topics |
|---|---|---|
| [`generic`](profiles/generic.json) | Default for any unrecognised dataset | Product, Service, Price, Usability, Delivery, Other |
| [`apparel_ecommerce`](profiles/apparel_ecommerce.json) | Demo: real clothing reviews (Kaggle) | Fit & Sizing, Fabric & Comfort, Quality & Durability, Style & Design, Appearance vs Listing, Price & Value, Order & Service, Other |
| [`food_delivery_demo`](profiles/food_delivery_demo.json) | Legacy demo: synthetic food-delivery reviews with dates (trend analysis) | Delivery, Food Quality, Order Accuracy, Packaging, Customer Service, App, Pricing, Other |

A profile defines:
- **Column mapping and synonyms**: e.g. `Clothing ID` → `product_id`.
- **Roles**: the *entity* being reviewed (product, restaurant, store), *segments* to break
  results down by, *numeric drivers* with bands, and yes/no *outcomes* such as "Recommended".
- **Value fixes**: e.g. a misspelling in the source.
- **Rating scale.**
- **Topic → issue taxonomy** with definitions, and **outcome flags** for the AI
  (e.g. `mentions_return`).
- **Baseline keywords** for the non-AI labeller.
- **Data notes** shown to the user.

The profile is detected automatically from distinctive column names. If nothing matches,
or two profiles match equally, the generic profile is used. You can also choose one
explicitly (`--profile`). A new domain needs a new JSON file, not new code.

## Features

| Feature | Status |
|---|---|
| Upload any feedback CSV; only a text column is required (rating, date, title, id optional) | ✅ |
| Profile detection and role resolution (entity, segments, numeric drivers, outcomes) | ✅ |
| Validation with user-friendly errors (empty file, no text column, too many rows, encoding) | ✅ |
| Cleaning: duplicates (incl. row-number exports), invalid ratings, mixed dates, HTML, name variants, implausible numbers, source truncation | ✅ |
| Data-quality report listing every row removed or value cleared, and why | ✅ |
| KPIs, rating and sentiment distributions, monthly trends (if dated) | ✅ |
| Breakdowns by entity and segments, with outcome rates (e.g. % recommended) | ✅ |
| Numeric drivers vs rating (bands + Spearman), low-rating drivers with multiple-testing correction | ✅ |
| Trend direction with a significance test | ✅ |
| Priority scoring (frequency × severity), working with or without ratings | ✅ |
| AI labelling: sentiment, topic, issues, outcome flags; batched, cached, schema-constrained | Implemented, mock-tested; **never run live (no API credits)** |
| Non-AI baseline (VADER + keywords) and automatic fallback on API errors | ✅ |
| Stratified AI sampling with weights, so population percentages are not inflated | ✅ |
| Evaluation: accuracy, macro-F1, confusion matrices, Wilson CIs, hand-labelling workflow | ✅ |
| Rule-based findings and recommendations, each traced to its source table (no AI) | ✅ |
| Streamlit dashboard: KPIs, charts, priority matrix, one segment filter, insights | ✅ |
| AI-written findings with a grounding check against computed numbers | Not built (no API access) |

## Architecture

```
CSV ──► profiles.py: detect profile ──► data_processing.py ──► cleaned data + CleaningReport
                                        map columns → validate → clean → derive bands/outcomes
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         ▼                               ▼                               ▼
                   analytics.py                    ai_analysis.py                  evaluation.py
          KPIs · trends · breakdowns ·      sample (stratified, weighted) →     accuracy vs ground truth
          drivers · significance ·          label (Claude via llm_client.py,     or hand labels; rating
          priority  (pandas only;           or VADER baseline) → cache           vs outcome agreement
          weighted when AI-sampled)
                         │
                         ▼
          pipeline.py: run_analysis() — clean, label, weight and build every table in one pass
                         │
                         ▼
          insights.py: rule-based findings from the tables ──► dashboard.py (Streamlit)
```

### Repository structure

```
Feedlytics/
├── app/
│   ├── config.py             # domain-neutral thresholds and AI settings
│   ├── profiles.py           # load/validate/detect profiles; resolve column roles
│   ├── data_processing.py    # load, map, validate, clean, data-quality report
│   ├── analytics.py          # descriptive statistics, significance tests, priority
│   ├── llm_client.py         # the only module that calls the Claude API
│   ├── ai_analysis.py        # AI + baseline labelling, sampling weights, cache
│   ├── evaluation.py         # metrics against ground truth
│   ├── pipeline.py           # run_analysis(): one entry point from CSV to every table
│   ├── insights.py           # rule-based findings and recommendations from the tables
│   └── dashboard.py          # single-page Streamlit dashboard
├── profiles/                 # generic.json, apparel_ecommerce.json, food_delivery_demo.json
├── data/
│   ├── README.md             # dataset cards (synthetic food demo, apparel demo)
│   ├── sample_reviews_SYNTHETIC.csv
│   ├── sample_reviews_SYNTHETIC_ground_truth.csv
│   └── raw/                  # downloaded third-party data (git-ignored)
├── scripts/
│   ├── profile_dataset.py        # data-quality report + analytics for any CSV
│   ├── evaluate_labels.py        # compare labelling methods against ground truth
│   ├── make_labeling_sample.py   # build a hand-labelling sample for real data
│   └── generate_synthetic_data.py
├── evaluation/               # RESULTS_<profile>.md, results/*.json, labeling/
├── tests/                    # 252 pytest tests
├── requirements.txt  pytest.ini  .env.example  .gitignore
```

## Technology stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Data | pandas, NumPy |
| AI | Claude API (`anthropic` SDK), structured JSON output; default model `claude-opus-5-5` |
| Non-AI baseline | VADER sentiment + profile keyword rules |
| Testing | pytest; the API is mocked at the HTTP layer |
| Config / secrets | python-dotenv (`.env`, never committed) |
| Dashboard | Streamlit, Plotly |

## Why AI?

A rating alone can't say what went wrong, and keyword rules miss phrasing like
*"look carefully at the model, she's positioned to hide how boxy it is"*. They also break on
negation and mixed reviews. A language model can map free text onto a **fixed** taxonomy, so
thousands of reviews become countable categories.

AI is **not** used where ordinary code is more reliable. Counting, averaging, significance
tests and ranking are all done in pandas, which keeps the numbers reproducible and stops
the model from inventing statistics.

## How AI is used

| Step | Done by | Notes |
|---|---|---|
| Cleaning, KPIs, trends, correlations, significance tests, priority | **Python** | Deterministic and tested |
| Choosing which reviews the AI labels | **Python** | Stratified sample, weighted back to the population |
| Sentiment, topic, issues, outcome flags per review | **AI** | Profile taxonomy as a JSON schema; invalid values repaired; cached |
| Fallback when there's no API key or an API error | **Python** (VADER + keywords) | Every label records its source |
| Findings and recommendations | **Python** (rule-based templates) | No AI; every number comes from one named table. AI-written findings: not implemented |
| Accuracy check | **Python** | Against ground truth or hand labels |

**Sampling weights.** Labelling every review is unnecessary and expensive: about $66 for the
22,641 apparel reviews with text, versus about $6 for a 2,000-review sample (upper-bound
estimates). The sample over-represents low ratings so complaints are well covered (50%
negative, 25% neutral, 25% positive). Each sampled review is weighted by
*stratum size in the data ÷ stratum size in the sample*. All AI-based percentages are
computed with these weights, so they estimate the whole dataset. In the apparel demo, 50% of
the sample is low-rated reviews against about 10% of the dataset, and the weights restore the
real mix. Raw counts are always shown alongside, so the sample size behind each number stays
visible.

## Data analysis methodology

**Cleaning** ([`data_processing.py`](app/data_processing.py)):

- **Text:** HTML and entities are removed and whitespace collapsed; placeholders like `N/A`
  count as missing. If a title exists, the AI sees *title + text*.
- **Rating** (optional): accepts `4`, `4.0`, `4/5`, `4 stars`, `four`. Values outside the
  profile's scale are cleared, and half-stars are rounded half-up.
- **Dates** (optional): mixed formats are parsed; unreadable, future or pre-2000 dates are cleared.
- **Duplicates:** exact duplicate rows are removed, ignoring row-number columns from
  spreadsheet/pandas exports (`Unnamed: 0`). A *re-submission* (same normalised text, rating,
  entity and day) is also removed. Texts under 20 characters are never treated as duplicates.
- **Labels:** spelling variants of entities and segments are merged, and profile value fixes
  are applied.
- **Numbers:** units and currency symbols are stripped, implausible values cleared, and
  profile bands derived (e.g. age bands).
- **Truncation:** reviews cut off at a fixed length by the source are flagged.
- A row is dropped only if it has no text and no valid rating. Otherwise the bad value is
  cleared and the row is kept.

**Analysis** ([`analytics.py`](app/analytics.py)):

- **Primary sentiment** (`sentiment`, with `sentiment_source`): the rating, using the profile's
  scale, whenever a review has a usable rating. Otherwise the text label is used: Claude
  (`text_ai`) or the non-AI baseline (`text_baseline`). For `apparel_ecommerce`, the baseline's
  sentiment cut-offs were tuned to match star ratings (1–2★ Negative, 3★ Neutral, 4–5★ Positive)
  on half of the data, excluding the 200 hand-labelled reviews. On the other half (11,182 reviews),
  its text labels agree with the rating-derived sentiment 76.3% of the time (macro-F1 0.515), versus
  57.3% for the uncalibrated rule. This is agreement with star ratings, not accuracy: ratings are only
  a proxy, and because most reviews are positive the figure is driven mainly by the Positive class
  (recall 0.90 versus 0.29 for Negative). Against human labels, text-sentiment accuracy is much
  lower, about 42% on the 200 hand-labelled reviews.
  It still misses most complaints (recall on 1–2★ reviews is 0.29). Complaint and topic labels
  never depend on the sentiment source, and evaluations score the text label (`sentiment_ai`) alone.
- Groups with fewer than 15 reviews are marked unreliable.
- Numeric drivers vs rating use bands plus Spearman rank correlation.
- Low-rating drivers report *lift* with a two-proportion z-test against the rest of the data,
  plus a **Benjamini–Hochberg** correction, because hundreds of groups are tested at once.
  In the apparel demo this cuts "significant" groups from 46 to 21.
- Trend direction compares the last 3 months with the 3 before. A change counts only if it is
  ≥ 2 percentage points **and** p < 0.05. With weights, the test uses the effective sample size.
- Priority = frequency × severity. Severity = share negative × rating gap, or the share
  negative alone when there are no ratings. Levels are relative to the top issue.

## Demo datasets

| | Apparel e-commerce (real) | Food delivery (synthetic, legacy) |
|---|---|---|
| Source | [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), download to `data/raw/reviews.csv` (not committed) | Generated by [`scripts/generate_synthetic_data.py`](scripts/generate_synthetic_data.py) |
| Size | 23,486 rows → 23,465 after cleaning | 3,193 rows → 3,102 after cleaning |
| Dates | **None**: no trend analysis | Jan 2025 – Jun 2026 |
| Demonstrates | Real text, product/segment breakdowns, Recommended outcome, age bands | Trends, significance tests, ground-truth evaluation |
| Caveats | Source truncates text at ~508 characters (3,769 reviews end mid-sentence) | Patterns are built into the generator; they are not findings |

See the [dataset cards](data/README.md) for details.

To use your own data, upload a CSV with at least a text column. Use anonymised data only,
and don't include names, emails or phone numbers.

## Evaluation

| Method | Data | Sentiment accuracy | Issue found | False complaints |
|---|---|---|---|---|
| Rating rule (no text) | synthetic, n=296 | 85.8% (95% CI 81.4–89.3) | — | — |
| Baseline: VADER + keywords | synthetic, n=301 | 69.4% (64.0–74.4) | 81.1% | 11.2% |
| Claude | — | **not run: no Anthropic API credits** | | |

The synthetic text is template-based, so these numbers are optimistic. Real-data evaluation
uses hand labels: [`scripts/make_labeling_sample.py`](scripts/make_labeling_sample.py)
creates a 200-review sample (over-sampling low ratings) plus a labelling guide.
For the apparel data, rating and "Recommended" agree on 98.5% of clearly positive/negative
reviews, a label-free sanity check. Full report:
[`evaluation/RESULTS_food_delivery_demo.md`](evaluation/RESULTS_food_delivery_demo.md).

## Running without an API key

Feedlytics is designed to run fully without Anthropic API access, and that's how this project
is currently run. No API key, `.env` file or network connection is needed for:

- cleaning and profiling any CSV (all three profiles);
- rating-based and baseline (VADER + keywords) labelling;
- all analytics: breakdowns, drivers, significance tests and priority scoring;
- stratified sampling and weights;
- the synthetic-data evaluation and the hand-labelling workflow;
- the full test suite.

Safeguards:
- **No accidental API calls.** The API is only called with an explicit `--ai` (or `--smoke`)
  flag, and those ask for confirmation first. Without API access, `--ai` stops immediately
  with a clear message instead of silently skipping.
- **The tests can't reach the API.** [`tests/conftest.py`](tests/conftest.py) removes any
  Anthropic credentials, ignores `.env` and blocks outbound network connections for every
  test. The Claude client is tested against an in-memory mock.
- **The SDK is optional.** If `anthropic` isn't installed, all non-AI code still runs and its
  client tests are skipped.

## Installation

```bash
git clone <your-repo-url> Feedlytics
cd Feedlytics
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional, AI path only (not used in this project yet): cp .env.example .env and add a key
```

## How to run

```bash
# Dashboard (local, non-AI baseline; opens http://localhost:8501).
# It opens on the apparel dataset, which is not included: download the Kaggle
# "Women's E-Commerce Clothing Reviews" CSV yourself to data/raw/reviews.csv
# (check its licence), or upload your own CSV in the sidebar.
streamlit run app/dashboard.py --browser.gatherUsageStats false
# No Kaggle file? Upload `data/sample_reviews_SYNTHETIC.csv` in the sidebar to try the dashboard.

# Data-quality report + analytics (profile auto-detected)
python -m scripts.profile_dataset                          # synthetic food demo
python -m scripts.profile_dataset data/raw/reviews.csv     # apparel demo
python -m scripts.profile_dataset my.csv --profile generic

# Evaluate labelling locally: rating rule + baseline, no API key needed
python -m scripts.evaluate_labels

# AI path only: free setup check (key, model, labels, cost); --smoke adds ONE ~$0.01 call
python -m scripts.check_ai_setup
python -m scripts.check_ai_setup --smoke

# Real data: build a hand-labelling sample, label it, then evaluate
python -m scripts.make_labeling_sample --data data/raw/reviews.csv --n 200
python -m scripts.evaluate_labels --data data/raw/reviews.csv \
    --truth evaluation/labeling/apparel_ecommerce_labeled_audited.csv

# Tests
python -m pytest
```

## Screenshots

1. Dashboard Overview



2. Ratings, Sentiment & Issue Priority



3. Priority Matrix, Praise & Flags

## Limitations

- **Claude has not been run.** There are no Anthropic API credits, so no Claude labels or
  accuracy figures exist. The AI layer is tested only against a mocked API. All reported
  numbers come from Python analytics, the rating rule or the VADER + keyword baseline.
- The synthetic demo's "insights" reflect the generator's rules, and accuracy measured on its
  template text is optimistic.
- The apparel demo has no dates (no trends), and the source truncated about one in six reviews.
- AI results on large datasets are estimates from a weighted sample, not a full census.
- "Business impact" in the priority score is a proxy (frequency × severity). No revenue,
  return-rate or churn data is used.
- Correlations and lifts show association, not cause.
- English-only. Ambiguous dates such as `03/04/2025` are read month-first (configurable).
- Files are limited to 200,000 rows.

## Future improvements

- Planned, not implemented: AI-based labelling and AI-written findings with a grounding check. The current demo uses the non-AI baseline and rule-based insights.
- An AI-suggested taxonomy for new domains, reviewed by a person before use.
- Regression modelling of rating and recommendation drivers.
- Alerts when an issue's complaint rate rises significantly.
- Arabic and multilingual feedback.

## Author

**Rolan Alyoubi** · [LinkedIn](https://www.linkedin.com/in/rolan-alyoubi-a67288386/)

## License

Code is licensed under MIT (see `LICENSE`). The Kaggle dataset is not included and is subject to its own terms.
