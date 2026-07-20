# LLM-judge rubric for feature abstractness

**Purpose.** The triviality metrics (modal-word %, distinct-word count) only measure *lexical* diversity — they can't tell a noise feature that fires on many varied words apart from a genuine *abstract* feature like "negation." This LLM judge measures **abstractness/complexity** directly, so we can compare the abstractness *distribution* across the three SAEs (full / hybrid / outbias).

**How it's run.**
- The judge is an LLM (currently `gpt-5.6-terra`, via the Batch API) rating one feature at a time.
- It is **blind to which SAE a feature came from** — so the rubric cannot bias the full-vs-hybrid-vs-outbias comparison.
- For each feature it sees two blocks of examples: **PEAK** (strongest activations) and **TYPICAL** (a uniform random sample of firings), and rates them independently. Each example shows the activating token wrapped in `《》` with a few tokens of context.
- Output per block: three integer axes (below) + a short `label` and `rationale`.

---

## How to read the examples
- **Judge from the `《》` token itself** — ask "what property of *this* token makes the feature fire?" The surrounding words are only context. Rate what the highlighted tokens share, not what the paragraphs happen to be about.
- **Subwords.** The `《》` token is often just a *piece* of a word (the tokenizer splits words, e.g. `Cr|umble`, `Vie|ja`). Read the whole word and the small construction it completes, not the bare fragment — the shared property may live there (e.g. `umble / Dip / Soup / Cr` all being "the final piece of a dish name in a recipe title" → coherent). But do **not** infer a property the token or its word doesn't itself carry.

## Rate COHERENCE first — it gates abstractness.

### COHERENCE (1–5)
Do the `《》` tokens in this block share one consistent property (a meaning, a role, or a surface form) that explains the firing?

| | |
|---|---|
| **1** | none: highlighted tokens look unrelated; no property explains them (noise) |
| **2** | weak: a minority share a property; most do not |
| **3** | mixed: a clear property holds for most, with several outliers |
| **4** | strong: nearly all firings share one clear property |
| **5** | exact: every firing has the same clear property, no exceptions |

### ABSTRACTNESS (0–5)
Given a coherent property, how abstract is it? Rate the property named for coherence.

| | |
|---|---|
| **0** | No real pattern. Use whenever coherence ≤ 2 — don't invent a concept out of noise |
| **1** | The exact same word or spelling every time — fires on one specific token or letter-pattern, regardless of meaning (e.g. always `the`; or always words ending in `-ing`) |
| **2** | One word or its close variants — a single word across its senses, or a few synonyms for one thing (e.g. big / large / huge) |
| **3** | One topic — many different words, but all from the same subject area (e.g. healthcare, elections, basketball) |
| **4** | A role or relationship, not a topic — many unrelated words linked by what they *do* in the sentence, not what they're *about* (e.g. negation words, comparisons, or "an organization being founded," which shows up in sports, business, and charities alike) |
| **5** | An abstract idea no word list could capture — spans many topics and roles at once (e.g. uncertainty, formality, politeness) |

**3-vs-4 test.** Look at what the passages are *about*. If firings stay locked to one subject (sports words never appear outside sports) → **3**. If the same relation/action/role recurs across many *different* subjects → **4**. A noun you could name a subject ("about sports/law") is a topic (3); a verb-y action or relation that happens in any subject ("something is being founded / negated / compared") is a frame (4).

### BREADTH (1–5)
How many distinct tokens / words / contexts does the feature respond to?

| | |
|---|---|
| **1** | one surface form only (a single token or word) |
| **2** | a few surface variants of one word |
| **3** | a handful of related words |
| **4** | many words within one area |
| **5** | many varied words across many contexts |

### Then give
- **`label`** — 2–6 words naming what the feature detects (write "incoherent" if coherence ≤ 2).
- **`rationale`** — one sentence naming the concrete shared property of the `《》` tokens (or stating there is none), citing the actual property, not a vibe.

---

## Lineage (Scaling Monosemanticity)
This reuses the paper's automated-interpretability recipe — an LLM judge on a small anchored scale, grounded in the activating token, with a "no interpretation here" floor (`0`), validated against human labels — but points it at **abstractness** rather than the paper's **specificity**. The abstractness ladder and the coherence-gates-abstractness structure are specific to this project (they're the axis word-counting can't reach).

## Validation status
Validated against hand labels on a seeded, SAE-balanced sample of features (blind, same rubric). Agreement was strong on coherence, breadth, and the "no-pattern" gate; the weakest axis was abstractness, traced to two causes: (a) the judge under-crediting features that are semantically coherent but lexically varied, and (b) subword tokenization. The **subword rule above** was added to address these. A fresh-sample re-validation (a held-out seed, so the fix isn't measured on the features it was tuned on) is the next step before scaling to the full feature set.
