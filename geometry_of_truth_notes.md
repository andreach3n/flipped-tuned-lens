# The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations
**arxiv: 2310.06824**

## Core Research Question
Do LLMs linearly represent the truthfulness of factual statements? The paper investigates whether previous "truthfulness probes" actually detect truth or just spurious correlations.

---

## Datasets

**Curated** (simple, unambiguous statements):
- Cities: "The city of [X] is in [Y]"
- Spanish-English translations: "The Spanish word '[X]' means '[Y]'"
- Numerical comparisons: "X is larger/smaller than Y"
- Negations of the above
- Logical combinations (conjunctions/disjunctions)

**Uncurated**: More complex, sometimes ambiguous claims about companies and general knowledge (from prior work).

**Control**: "Likely" text — statements where the final token is either the most probable or 100th-most probable next token. Used to test whether models represent truth vs. textual plausibility.

---

## Key Findings on Geometric Structure

**Linear separability**: PCA visualizations show true and false statements separate in the top few principal components. Nearly all linearly-accessible truth information is concentrated there.

**Misalignment problem**: The naive "truth direction" (vector from mean false → mean true) doesn't consistently align across datasets. E.g., the direction for cities vs. negated cities statements is approximately *orthogonal* — even though both encode truth.

---

## Three Hypotheses for Misalignment

1. **H1**: Models lack a unified truth representation; they represent diverse features (size, translation equivalence, etc.) that correlate with truth contextually.
2. **H2**: Truth is represented distinctly for different statement types (negations, conjunctions) without unified encoding.
3. **H3 — MCI (Misalignment from Correlational Inconsistency)**: Models encode *both* truth and non-truth features; spurious features correlate with truth *inconsistently* across datasets.

**Evidence supports H3.** When probes are trained on statements *and* their negations, generalization improves substantially — because the model is forced to find a direction that works for both, canceling out the spurious surface correlates.

---

## Mass-Mean Probing

**Problem with logistic regression**: When non-truth features non-orthogonally interfere with the true truth direction, logistic regression's maximum-margin solution can diverge from it.

**Mass-mean probe**: Use the vector from mean(false representations) → mean(true representations). The IID variant:

$$p_\text{mm}^\text{iid}(x) = \sigma(\theta_\text{mm}^T \Sigma^{-1} x)$$

where Σ is the data covariance. The Σ⁻¹ transformation whitens data to handle interference from spurious features.

**Why it's better**: Mass-mean more directly identifies the candidate feature direction. Under Gaussian assumptions, it coincides on average with logistic regression — but is more robust when spurious features are present.

---

## Generalization Results

- Probes trained only on numerical comparisons → 92%+ accuracy on Spanish-English translation (good cross-dataset transfer)
- Training on paired statements (true + negations) → better generalization (supports MCI hypothesis)
- Probes trained on "likely" text perform poorly → models encode truth beyond textual plausibility

---

## Causal Intervention Evidence

Using activation patching on LLaMA-13B, three groups of causally relevant hidden states were identified. In the middle group (layers 7–13), truth computation occurs above end-of-clause tokens.

**Adding truth vectors** (normalized to shift average false → average true) causally influences outputs:
- Best mass-mean intervention: NIE = 0.95 for false→true direction
- Swings model from 77% confidence TRUE (for false statements) to 92% confidence FALSE
- Logistic regression interventions: weaker effect (NIE ~0.52–0.66)
- "Likely" dataset interventions: minimal causal impact

---

## What Information Do Probes Use?

Three distinct things to measure:
| Property | Meaning |
|---|---|
| Linear separability | Can some direction separate true/false? (easy, many directions work) |
| Causal involvement | Does the direction actually mediate truth judgments in the model? |
| Genuine feature detection | Is the direction the actual truth feature, not a spurious correlate? |

Mass-mean probes score highest on causal mediation, especially when trained on diverse/negated statements.

---

## Limitations

- Only tests simple, uncontroversial statements
- Cannot distinguish "true" from "commonly believed" or "verifiable"
- Optimal bias term for probes is underdetermined
- Only two models tested (from the same family)
