"""What do these latents actually LOOK like? Joins delphi's explanations to AUROC and firing count.

WHY THIS EXISTS. Every number in this project so far is an aggregate over latents, and the
frequency analysis showed the aggregate hides a sign change. The obvious next question -- and the
one a mentor asks -- is what the latents at each end of that curve actually are. In particular,
for the RANDOM arm's skip-embed SAE, per-latent fuzz AUROC is ~0.87 among rare latents and ~0.47
(chance) among the most frequent ones. Those two populations should look qualitatively different,
and if they do not, the AUROC split is suspect.

Three views per cell, because "interpretable" is not one thing:
  TOP BY AUROC     -- what a describable latent looks like here
  BOTTOM BY AUROC  -- what the judge could not characterise
  MOST FREQUENT    -- the high-firing tail; on the random arm this is the population that
                      collapsed to chance after P[tok] was subtracted, so its explanations are
                      the direct evidence for what skip-embed removed

Firing counts come from delphi's own log/hookpoint_firing_counts.pt (full dictionary, survives
`tar --exclude latents`), same source as the frequency analysis, so the bins line up exactly.

EXAMPLES. delphi's score files carry `str_tokens` and per-token `activations` for every example
it showed the judge, so we can print what the latent ACTUALLY fires on rather than only the
judge's paraphrase of it. That matters here: the judge's text is itself the thing under
suspicion -- on the random arm it writes "various words and phrases that appear to be randomly
selected", and you cannot tell from the paraphrase alone whether the latent is genuinely
patternless or the judge simply failed. The activating contexts settle it. The peak token is
wrapped in »«.

    RESULTS_DIR=/dev/shm/delphi_run/results CELL=pythia1b_rand_resid_L8_lr1e-3 \
      python -u inspect_explanations.py
    N=15 EX=4 MIN_AUROC=0.8 python -u inspect_explanations.py      # only the describable ones
"""
import glob
import json
import os
import re

ROOT   = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
CELL   = os.environ["CELL"]
SCORER = os.environ.get("SCORER", "fuzz")
N      = int(os.environ.get("N", 8))
WIDTH  = int(os.environ.get("WIDTH", 150))
EX     = int(os.environ.get("EX", 3))            # activating examples printed per latent
CTX    = int(os.environ.get("EX_CTX", 7))        # tokens of context either side of the peak
MIN_AUROC = float(os.environ.get("MIN_AUROC", 0))
VIEWS  = os.environ.get("VIEWS", "top,bottom,frequent").split(",")
# Restrict to a latent-id range. For the T-SAE cells, delphi_tsae.py's column permutation puts
# 500 Matryoshka high-level latents at ids 0..499 and 500 low-level ones at 500..999, so
# LID_MAX=500 selects the contrastively-regularised group and LID_MIN=500 the unregularised one.
# Without this, VIEWS=top just returns the highest-AUROC latents overall -- which skews to the
# low-level group, since its mean is higher (0.824 vs 0.779) -- and silently mixes the two
# populations the whole control rests on separating.
LID_MIN = int(os.environ.get("LID_MIN", 0))
LID_MAX = int(os.environ.get("LID_MAX", 10 ** 9))


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def latent_id(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def render(rec):
    """One activating example, trimmed to a window around the peak token, peak marked »«."""
    toks, acts = rec.get("str_tokens") or [], rec.get("activations") or []
    if not toks or not acts or max(acts) <= 0:
        return None
    i = acts.index(max(acts))
    lo, hi = max(0, i - CTX), min(len(toks), i + CTX + 1)
    parts = list(toks[lo:hi])
    parts[i - lo] = f"»{toks[i]}«"
    s = "".join(parts).replace("\n", "⏎")
    return ("…" if lo else "") + " ".join(s.split()) + ("…" if hi < len(toks) else "")


scores, examples = {}, {}
for f in sorted(glob.glob(f"{ROOT}/{CELL}/scores/{SCORER}/*.txt")):
    lid = latent_id(f)
    if lid is None:
        continue
    all_recs = json.load(open(f))
    recs = [r for r in all_recs if r.get("prediction") is not None]
    pos = [r["probability"] for r in recs if r["activating"] and r.get("probability") is not None]
    neg = [r["probability"] for r in recs if not r["activating"] and r.get("probability") is not None]
    a = auroc(pos, neg)
    if a is None:
        continue
    scores[lid] = a
    # Strongest activations first -- the peak examples are the ones the explainer was shown.
    act = [r for r in all_recs if r.get("activating") and (r.get("activations") or [0])]
    act.sort(key=lambda r: -max(r["activations"] or [0]))
    examples[lid] = [e for e in (render(r) for r in act[: EX * 3]) if e][:EX]

expl = {}
for f in glob.glob(f"{ROOT}/{CELL}/explanations/*.txt"):
    lid = latent_id(f)
    if lid is not None:
        # delphi writes the explanation as a JSON string, so it arrives quoted and escaped.
        raw = open(f).read().strip()
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pass
        expl[lid] = " ".join(str(raw).split())

p = f"{ROOT}/{CELL}/log/hookpoint_firing_counts.pt"
counts = {}
if os.path.exists(p):
    import torch as t
    obj = t.load(p, weights_only=True, map_location="cpu")
    v = list(obj.values())[0] if isinstance(obj, dict) else obj
    counts = {i: int(c) for i, c in enumerate(v.tolist())}

rows = [(lid, counts.get(lid, -1), a, expl.get(lid, "<no explanation>"))
        for lid, a in scores.items()
        if a >= MIN_AUROC and LID_MIN <= lid < LID_MAX]
rng = "" if LID_MAX > 10 ** 8 and LID_MIN == 0 else f", latent ids [{LID_MIN}..{LID_MAX - 1}]"
mean = sum(r[2] for r in rows) / len(rows) if rows else float("nan")
print(f"\n=== {CELL} — {len(rows)} of {len(scores)} scored latents shown "
      f"(MIN_AUROC={MIN_AUROC}{rng}), scorer={SCORER}, mean AUROC {mean:.3f} ===")


def show(title, items):
    print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))
    for lid, fc, a, e in items:
        print(f"\n  latent {lid:>5}  fires {fc:>9,}  AUROC {a:.3f}")
        print(f"      {e[:WIDTH]}{'…' if len(e) > WIDTH else ''}")
        for ex in examples.get(lid, []):
            print(f"        · {ex[:WIDTH + 20]}")


if "top" in VIEWS:
    show(f"TOP {N} BY AUROC", sorted(rows, key=lambda r: -r[2])[:N])
if "bottom" in VIEWS:
    show(f"BOTTOM {N} BY AUROC", sorted(rows, key=lambda r: r[2])[:N])
if "frequent" in VIEWS:
    show(f"MOST FREQUENT {N}", sorted(rows, key=lambda r: -r[1])[:N])
