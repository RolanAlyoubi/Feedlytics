# Hand-labelling guide — profile `apparel_ecommerce`

Label each review from its **text only** (ignore the rating column when judging sentiment).

| Column | Allowed values |
|---|---|
| `true_sentiment` | Positive, Neutral, Negative — the overall tone of the text |
| `true_topic` | the main topic (list below) |
| `true_issue` | the main complaint's issue (list below); **leave empty** if there is no complaint |
| `true_mentions_return` | true / false — the customer says they returned, will return, or sent back the item |
| `true_mentions_repurchase` | true / false — the customer says they bought, or would buy, it again or in another colour/size |

## Topics and issues

**Fit & Sizing**
- Runs small — smaller or tighter than the customer's usual size
- Runs large — bigger, looser or more oversized than the customer's usual size
- Wrong length — too long or too short (hem, sleeves, inseam or torso)
- Poor fit in one area — fits badly in a specific area such as bust, waist, hips, shoulders or arms
- Doesn't suit body type or height — cut only works for certain heights or body shapes (e.g. petite, tall, curvy)
- Inconsistent sizing — sizing differs from the brand's usual sizing, between colours, or from the size chart

**Fabric & Comfort**
- Thin or see-through — fabric is thin, sheer or see-through, or needs a layer underneath
- Itchy or uncomfortable — fabric is itchy, scratchy or uncomfortable to wear
- Stiff, heavy or unexpected texture — fabric is stiff, heavy, rough or a different texture than expected

**Quality & Durability**
- Poor construction — problems with seams, stitching, zippers, buttons or lining
- Pilling, wear or falling apart — pills, frays, gets holes or falls apart after little wear
- Damaged or shrunk in the wash — shrinks, fades or is damaged by washing
- Looks or feels cheap — looks or feels cheaply made for what it is

**Style & Design**
- Unflattering cut or shape — shape is boxy, shapeless, frumpy or unflattering
- Design detail problem — a specific design element (neckline, pockets, straps, slit, lining placement) does not work
- Pattern or print disappointing — pattern, print or embellishment is disappointing

**Appearance vs Listing**
- Looks different from photo or model — looks different in person than in the product photos or on the model
- Colour different from shown — colour differs from what the listing showed

**Price & Value**
- Overpriced for the quality — price is too high for the quality, or only worth buying on sale

**Order & Service**
- Arrived damaged or defective — item arrived damaged, stained or defective
- Wrong item or size sent — received a different item, colour or size than ordered
- Shipping or delivery problem — late, lost or otherwise problematic shipping
- Return or exchange problem — difficulty returning or exchanging the item
- Size or colour unavailable — wanted size or colour was sold out or not offered

**Other**
- Other — a complaint that fits none of the issues above

## Tips
- A review that praises something but whose main point is a complaint is Negative.
- If a review was cut off mid-sentence, label only what is there.
- Label in one sitting if possible and keep notes on hard cases; consistency matters more than speed.