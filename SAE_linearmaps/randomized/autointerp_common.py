"""Pure-python core of the standard-autointerp experiment (no torch / sae_lens / openai).

Shared by autointerp_collect.py (pod), autointerp_explain.py + autointerp_score.py (Mac).
Everything here is unit-testable locally — the GPU and API scripts stay thin wrappers.

The experiment: fuzzing/detection auto-interpretability (Paulo et al. 2024, the protocol
Heap et al. arXiv:2501.17727 use) on the 2x2 of {trained, rand_all seed 0} x {full, resid}
20M-token SAEs, scored alongside the validated abstractness rubric on the SAME latents.

Marking conventions (deliberate, documented):
  <<tok>>  autointerp blocks (explain / fuzz) — delphi's delimiter, ALL activating tokens marked.
  《tok》    rubric blocks — anchor token only, matching the format the rubric was
            human-validated on (week 8). The two styles coexist on purpose.
"""
import math
import random

CELLS = [("trained", "full"), ("trained", "resid"), ("rand_all", "full"), ("rand_all", "resid")]


def cell_name(arm, mode):
    return f"{arm}/{mode}"


# ---------------------------------------------------------------- freq bins
def rate_bin(rate):
    """Half-order-of-magnitude bin of a firing RATE (firings per token scanned).
    Rate, not absolute count, so bins are comparable across scans of different length."""
    if rate <= 0:
        return None
    return int(math.floor(2.0 * math.log10(rate)))


def bin_label(b):
    return f"1e{b/2:.1f}-1e{(b+1)/2:.1f}"


def stratified_sample(cell_rates, target_per_bin, seed, min_rate=0.0):
    """Equal-N-per-bin sampling matched across ALL cells (the 4-cell version of
    build_judge_features.sample_features).

    cell_rates: {cell: {feat_id: rate}} using only features alive enough to have examples.
    Returns ({cell: [feat_id,...]}, plan) where plan rows are
    (bin, {cell: n_alive}, n_chosen). For each bin present in EVERY cell, draws
    min(target, smallest alive count) features per cell -> identical freq marginals
    by construction."""
    rng = random.Random(seed)
    members = {c: {} for c in cell_rates}
    for c, rates in cell_rates.items():
        for fid, r in rates.items():
            if r <= min_rate:
                continue
            b = rate_bin(r)
            if b is not None:
                members[c].setdefault(b, []).append(fid)

    all_bins = sorted(set().union(*[set(m) for m in members.values()]))
    selected = {c: [] for c in cell_rates}
    plan = []
    for b in all_bins:
        counts = {c: len(members[c].get(b, [])) for c in cell_rates}
        n_b = min(target_per_bin, min(counts.values()))
        if n_b > 0:
            for c in cell_rates:
                pick = rng.sample(sorted(members[c][b]), n_b)   # sorted pool -> deterministic
                selected[c].extend(pick)
        plan.append((b, counts, n_b))
    return selected, plan


def print_plan(plan, cells):
    head = f"{'bin (rate)':>16} | " + " | ".join(f"{cell_name(*c):>16}" for c in cells) + " | chosen"
    print(head)
    print("-" * len(head))
    for b, counts, n_b in plan:
        row = " | ".join(f"{counts[cell_name(*c)]:>16,}" for c in cells)
        note = "" if n_b > 0 else "   (skipped: empty in a cell)"
        print(f"{bin_label(b):>16} | {row} | {n_b:>6}{note}")
    total = sum(n for _, _, n in plan)
    print(f"{'total / cell':>16} : {total}")


# ---------------------------------------------------------------- rendering
def render_marked(tokens, acts, open_mark="<<", close_mark=">>", threshold=0.0):
    """Join decoded window `tokens` (list of str), wrapping every token whose activation
    exceeds `threshold` in the marks. acts aligned with tokens (list of float)."""
    out = []
    for tok, a in zip(tokens, acts):
        out.append(f"{open_mark}{tok}{close_mark}" if a > threshold else tok)
    return "".join(out)


def render_anchor(tokens, anchor_idx, open_mark="《", close_mark="》"):
    """Rubric-format rendering: mark ONLY the anchor token (the validated week-8 format)."""
    out = []
    for i, tok in enumerate(tokens):
        out.append(f"{open_mark}{tok}{close_mark}" if i == anchor_idx else tok)
    return "".join(out)


def plant_wrong_marks(tokens, acts, n_marks, seed):
    """Fuzz negative: mark n_marks tokens chosen among NON-activating positions.
    Returns marked text, or None if the window has too few non-activating tokens."""
    idle = [i for i, a in enumerate(acts) if a <= 0]
    if len(idle) < n_marks:
        return None
    rng = random.Random(seed)
    chosen = set(rng.sample(idle, n_marks))
    return "".join(f"<<{t}>>" if i in chosen else t for i, t in enumerate(tokens))


# ---------------------------------------------------------------- splits
def split_examples(score_items, n_detect, n_fuzz, seed):
    """Partition a feature's held-out examples into detection positives and fuzz
    positives (disjoint). Returns (detect, fuzz) — short lists are split proportionally."""
    rng = random.Random(seed)
    items = list(score_items)
    rng.shuffle(items)
    want = n_detect + n_fuzz
    if len(items) < want:
        n_detect = max(1, len(items) * n_detect // want) if items else 0
    return items[:n_detect], items[n_detect:n_detect + n_fuzz]


# ---------------------------------------------------------------- scoring
def auroc(labels, scores):
    """Rank AUROC with tie handling (average ranks). labels: 1/0, scores: floats.
    Returns None if a class is missing."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank across the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    """Standard error of a cell's mean AUROC. None if fewer than 2 values.

    AUROC is computed PER LATENT (one scoring call each) and the cell figure is the mean over
    latents, so latent-to-latent spread IS the sampling distribution of that mean -- sd/sqrt(n)
    is the correct error bar. Do NOT use the per-example count: the examples within one latent
    are not independent draws for this purpose.
    """
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


# ---------------------------------------------------------------- prompts
# Adapted from the delphi (EleutherAI) explainer/scorer prompts; simplified wording,
# same task structure. Cite as "following Paulo et al. (2024)".
EXPLAIN_SYSTEM = """\
You are explaining the behavior of a neuron in a language model. You will see text \
examples where the neuron activates; the specific tokens it activates on are wrapped in \
<< >>. The activation strength is shown in brackets before each example.

Look for the pattern shared by the << >> tokens (and their immediate context). Tokens are \
often word-pieces — read the whole word the piece belongs to. Describe the single most \
specific pattern that covers most marked tokens. If the marked tokens share no discernible \
pattern, say exactly: "no clear pattern".

Reply with a short explanation of at most 12 words. Examples of good explanations: \
"the token 'district', usually in legal contexts", "sports terminology in news articles", \
"final piece of a dish name in recipe titles"."""

DETECT_SYSTEM = """\
You are scoring how well an explanation of a language-model neuron predicts its behavior.

NEURON EXPLANATION: {explanation}

You will see numbered text examples. For EACH example, rate 0-10 how likely it is that \
this neuron (as described by the explanation) activates somewhere in the example. \
0 = certainly does not activate, 10 = certainly activates. Judge only from the \
explanation — do not invent additional behaviors for the neuron."""

FUZZ_SYSTEM = """\
You are scoring how well an explanation of a language-model neuron predicts WHICH tokens \
it activates on.

NEURON EXPLANATION: {explanation}

You will see numbered text examples, each with some tokens wrapped in << >>. In some \
examples the marked tokens are truly the ones the neuron activates on; in others the marks \
were placed on the wrong tokens. For EACH example, rate 0-10 how likely it is that the \
<< >> marks are correctly placed on tokens this neuron (per the explanation) activates on. \
0 = certainly wrong tokens, 10 = certainly the right tokens."""


def numbered_block(texts):
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))


def ratings_schema(n):
    """Strict json_schema for a fixed-length list of 0-10 integer ratings."""
    return {"type": "json_schema",
            "json_schema": {"name": "example_ratings", "strict": True, "schema": {
                "type": "object",
                "properties": {"ratings": {
                    "type": "array", "minItems": n, "maxItems": n,
                    "items": {"type": "integer", "minimum": 0, "maximum": 10}}},
                "required": ["ratings"], "additionalProperties": False}}}


EXPLANATION_SCHEMA = {"type": "json_schema",
                      "json_schema": {"name": "neuron_explanation", "strict": True, "schema": {
                          "type": "object",
                          "properties": {"explanation": {"type": "string"}},
                          "required": ["explanation"], "additionalProperties": False}}}
