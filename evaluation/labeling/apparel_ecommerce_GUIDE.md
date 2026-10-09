# Hand-labelling guide — profile `apparel_ecommerce`

You are labelling customer reviews of clothing sold by an online fashion retailer. Your labels are the **ground truth** used to
measure how accurate the labelling methods are, so consistency matters more than speed.

## Workflow

1. Copy `apparel_ecommerce_to_label.csv` to **`apparel_ecommerce_labeled.csv`** in this folder and
   label the copy (the original stays as a clean template).
2. Open it in a spreadsheet. Fill in the `true_*` columns for each row; leave every other column
   unchanged. Save as **CSV** (comma-separated; "CSV UTF-8" is fine).
3. Check your progress and catch typos at any time (no API, no cost):
   `python -m scripts.evaluate_labels --data data/raw/reviews.csv --truth evaluation/labeling/apparel_ecommerce_labeled.csv --check-only`
4. You can stop part-way: rows with an empty `true_sentiment` are skipped.

**The rating is hidden on purpose.** Judge the text (title + review) only.

## Columns

| Column | What to enter | Allowed values |
|---|---|---|
| `true_sentiment` | The overall tone of the text | `Positive`, `Neutral`, `Negative` |
| `true_topic` | The single topic the review is mainly about: its main complaint if it has one, otherwise its main praise | one topic name (list below) |
| `true_issues` | **Every** complaint the text mentions, most important first, separated by `;` — leave **empty** if there is no complaint | issue names (list below) |
| `true_praise_topics` | Every topic the customer praises, separated by `;` — leave empty if none | topic names (list below) |
| `true_mentions_return` | the customer says they returned, will return, or sent back the item | `true` or `false` |
| `true_mentions_repurchase` | the customer says they bought, or would buy, it again or in another colour/size | `true` or `false` |

Values must match the lists below exactly (capitals and spelling). The check command
suggests a fix for near-misses such as `negative` → `Negative`.

## Sentiment rules

- **Negative** — the main point is a complaint or disappointment, even if something is praised
  ("Love the colour, but it fell apart after one wash and I'm returning it").
- **Positive** — mostly happy; a small reservation does not change that
  ("Beautiful dress, runs a little long but I'll keep it").
- **Neutral** — mixed with no clear winner, indifferent, or purely factual
  ("It's okay. Fits as expected.").

## Complaint (issue) rules

- List **every** distinct complaint, not just the main one. Put the main complaint first.
- A complaint counts even inside a positive review ("Love it, but it runs small" → `Runs small`).
- Neutral sizing *advice* is still a complaint about fit if it says the size is off
  ("order a size down" → `Runs large`). Pure facts ("I'm 5'4 and ordered a S") are not.
- Use `Other` only for a real complaint that fits no issue below.
- Each issue belongs to one topic; the main issue's topic should normally equal `true_topic`.

## Praise rules

- List the topics the customer explicitly praises ("so soft" → `Fabric & Comfort`,
  "flattering" → `Style & Design`). General enthusiasm with no topic ("Love it!") → leave empty.

## Outcome flags

- `true_mentions_return`: `true` if the customer says they returned, will return, or sent back the item; otherwise `false`.
- `true_mentions_repurchase`: `true` if the customer says they bought, or would buy, it again or in another colour/size; otherwise `false`.

## Truncated reviews

Some reviews were cut off by the data source mid-sentence. Label only what is there; do not
guess how the sentence ended.

## Topics and issues

**Fit & Sizing**
- `Runs small` — smaller or tighter than the customer's usual size
- `Runs large` — bigger, looser or more oversized than the customer's usual size
- `Wrong length` — too long or too short (hem, sleeves, inseam or torso)
- `Poor fit in one area` — fits badly in a specific area such as bust, waist, hips, shoulders or arms
- `Doesn't suit body type or height` — cut only works for certain heights or body shapes (e.g. petite, tall, curvy)
- `Inconsistent sizing` — sizing differs from the brand's usual sizing, between colours, or from the size chart

**Fabric & Comfort**
- `Thin or see-through` — fabric is thin, sheer or see-through, or needs a layer underneath
- `Itchy or uncomfortable` — fabric is itchy, scratchy or uncomfortable to wear
- `Stiff, heavy or unexpected texture` — fabric is stiff, heavy, rough or a different texture than expected

**Quality & Durability**
- `Poor construction` — problems with seams, stitching, zippers, buttons or lining
- `Pilling, wear or falling apart` — pills, frays, gets holes or falls apart after little wear
- `Damaged or shrunk in the wash` — shrinks, fades or is damaged by washing
- `Looks or feels cheap` — looks or feels cheaply made for what it is

**Style & Design**
- `Unflattering cut or shape` — shape is boxy, shapeless, frumpy or unflattering
- `Design detail problem` — a specific design element (neckline, pockets, straps, slit, lining placement) does not work
- `Pattern or print disappointing` — pattern, print or embellishment is disappointing

**Appearance vs Listing**
- `Looks different from photo or model` — looks different in person than in the product photos or on the model
- `Colour different from shown` — colour differs from what the listing showed

**Price & Value**
- `Overpriced for the quality` — price is too high for the quality, or only worth buying on sale

**Order & Service**
- `Arrived damaged or defective` — item arrived damaged, stained or defective
- `Wrong item or size sent` — received a different item, colour or size than ordered
- `Shipping or delivery problem` — late, lost or otherwise problematic shipping
- `Return or exchange problem` — difficulty returning or exchanging the item
- `Size or colour unavailable` — wanted size or colour was sold out or not offered

**Other**
- `Other` — a complaint that fits none of the issues above

## Tips

- Label in a few sittings and keep a short note of hard cases and how you decided them.
- Expect roughly 30–60 seconds per review (about 2–3 hours for 200).
- Don't look up the rating, and don't change `review_id`, the title or the text.
