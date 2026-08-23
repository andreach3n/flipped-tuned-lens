"""AUROC per latent, clustered by latent, with standard errors -- trained vs random Pythia-1b.

THE CLUSTERING IS THE POINT. delphi samples LATENTS and then asks the judge ~100 questions
about each one. Pooling those ~40k individual scoring decisions would treat a few hundred
latents as tens of thousands of independent trials and shrink the standard errors by roughly
4x, which is how you manufacture a significant gap out of nothing. So every latent yields one
AUROC, and the mean and SE are taken over latents.

AUROC per latent is the exact Mann-Whitney statistic,
    P(score(activating) > score(non-activating)) + 0.5 * P(equal),
written as the pairwise sum rather than via ranks: it is the definition, it handles ties
without a correction term, and at ~50x50 pairs per latent the cost is irrelevant. It needs
delphi's per-example `probability` field, which is only populated with `--log_probs` AND a
judge that actually returns token logprobs -- over OpenRouter, Llama-3.1-70B returns none
(DeepInfra) or one (CoreWeave), so the judge has to be served locally. See ../DELPHI_SETUP.md.

Why AUROC and not delphi's default class-balanced accuracy: balanced accuracy folds
discrimination together with response bias, and this judge has a lot of response bias on
random-model features (Gemma random arm: fuzz TPR 0.797 / TNR 0.478, i.e. below chance --
it agrees rather than discriminates). AUROC is rank-based and bias-free. Both are reported
here, along with TPR/TNR, precisely so bias stays visible instead of being averaged away.

TWO LEARNING RATES ARE SCORED, and that is a deliberate robustness axis rather than a sweep
left half-finished. The LR sweep (see README) put 1e-3 and 2e-3 in direct conflict: 2e-3 wins
on reconstruction by ~4% while losing 24 points of alive% on the trained arm and 12 points of
>=10-firings on the random one. Rather than resolve that by choosing, both are scored, and the
question becomes whether the trained-vs-random conclusion is the SAME at both -- which is a
far stronger answer to "your SAEs were undertrained" than any single defensible pick.

    RESULTS_DIR=/dev/shm/delphi_run/results python -u report_auroc.py

Expects RESULTS_DIR/<cell>/scores/{detection,fuzz}/*.txt, cells named as CELL_FMT below.
Set LRS="1e-3" to report a single configuration.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
LAYER = int(os.environ.get("LAYER", 8))
LRS = os.environ.get("LRS", "1e-3,2e-3").split(",")
CELL_FMT = os.environ.get("CELL_FMT", "pythia1b_{arm}_L{layer}_lr{lr}")
ARMS = [("trained", "trained"), ("rand", "re-randomized\n(incl. embeddings)")]
SCORERS = ["detection", "fuzz"]

# Okabe-Ito. Validated for this use: adjacent-pair CVD separation dE 29.2 (protan) / 30.9
# (tritan), normal-vision 36.2, all well above the 8 floor. The orange sits at 2.19:1 against
# the surface, which obliges visible relief -- discharged by the direct value labels on every
# bar and the full table printed to stdout, both of which are load-bearing here, not decoration.
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
CHANCE = 0.5


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def per_latent(cell, scorer):
    """One row per latent: (AUROC, balanced accuracy, TPR, TNR). The latent is the unit."""
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
        # A parse failure drops that EXAMPLE from the AUROC, not the whole latent -- the
        # accuracy columns still use every example the judge answered.
        a = auroc([r["probability"] for r in pos if r.get("probability") is not None],
                  [r["probability"] for r in neg if r.get("probability") is not None])
        rows.append((a, (tpr + tnr) / 2, tpr, tnr))
    return rows


def stats(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (float("nan"), float("nan"), len(xs))
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


D, missing = {}, []
for lr in LRS:
    for arm, _ in ARMS:
        cell = CELL_FMT.format(arm=arm, layer=LAYER, lr=lr)
        for s in SCORERS:
            rows = per_latent(cell, s)
            if not rows:
                missing.append(f"{cell}/{s}")
            D[(lr, arm, s)] = rows
if missing:
    raise SystemExit(f"no scores found for: {missing}\nunder {ROOT} -- check RESULTS_DIR / "
                     f"--name / LRS ({LRS})")

# ---------------------------------------------------------------- table (the deliverable)
print(f"\ndelphi on pythia-1b layer {LAYER}, per-latent, clustered by latent (+/- 1 SE)\n")
print(f"{'LR':>7} {'arm':>8} {'scorer':>10} {'AUROC':>16} {'bal acc':>16} "
      f"{'TPR':>16} {'TNR':>16} {'n':>6}")
for lr in LRS:
    for arm, _ in ARMS:
        for s in SCORERS:
            cols = [stats([r[i] for r in D[(lr, arm, s)]]) for i in range(4)]
            print(f"{lr:>7} {arm:>8} {s:>10} "
                  + " ".join(f"{m:9.3f}+/-{e:.3f}" for m, e, _ in cols)
                  + f" {cols[0][2]:6d}")
print()

gaps = {}
for idx, lab in ((0, "AUROC"), (1, "bal acc")):
    for lr in LRS:
        for s in SCORERS:
            (mt, et, nt) = stats([r[idx] for r in D[(lr, "trained", s)]])
            (mr, er, nr) = stats([r[idx] for r in D[(lr, "rand", s)]])
            d, se = mt - mr, math.sqrt(et ** 2 + er ** 2)
            gaps[(idx, lr, s)] = (d, se)
            print(f"  {lab:>8} gap {s:>9} @ LR {lr:>5}: {d:+.3f} +/- {se:.3f}  z = {d / se:6.2f}"
                  f"   (n = {nt} trained vs {nr} random)")
    print()

if len(LRS) > 1:
    print("  LR-ROBUSTNESS -- the reason both were scored. The conclusion is only safe if the")
    print("  sign and rough size of the gap agree across LRs; if they disagree, the finding is")
    print("  about SAE training, not about the models.")
    for idx, lab in ((0, "AUROC"),):
        for s in SCORERS:
            vals = [gaps[(idx, lr, s)] for lr in LRS]
            spread = max(v[0] for v in vals) - min(v[0] for v in vals)
            joint_se = math.sqrt(sum(v[1] ** 2 for v in vals))
            agree = all(v[0] > 0 for v in vals) or all(v[0] < 0 for v in vals)
            print(f"    {lab} {s:>9}: gaps " + ", ".join(f"{v[0]:+.3f}" for v in vals)
                  + f"  spread {spread:.3f} (+/- {joint_se:.3f})"
                  + ("  SAME SIGN" if agree else "  *** SIGN FLIPS -- do not pool ***"))
    print()

print("  Heap et al. replicates iff these gaps are ~0. A clearly positive gap is a "
      "NON-replication\n  on pythia-1b -- the paper's own model family, so the "
      "'gemma is not pythia' caveat is gone.\n")

# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.6 + 2.6 * len(LRS), 4.9), dpi=200)
W, EB = 0.34, dict(ecolor=INK, capsize=4, elinewidth=1.4)
# x positions: arms adjacent within an LR, a visible gap BETWEEN LR groups, so the eye compares
# trained-vs-rand first (the question) and LR second (the robustness check).
XS, XT, GROUPS = [], [], []
for gi, lr in enumerate(LRS):
    base = gi * (len(ARMS) + 0.8)
    for ai, (_, lab) in enumerate(ARMS):
        XS.append(base + ai)
        XT.append(lab)
    GROUPS.append((base + (len(ARMS) - 1) / 2, lr))


def draw(ax, idx, ylabel, title):
    """Bars measure distance ABOVE CHANCE, because 0.5 is the meaningful zero for both metrics.
    Starting them at chance makes bar length encode discriminative signal honestly, instead of
    cropping the axis to exaggerate a difference."""
    handles, top = [], 0.0
    for i, (s, colour) in enumerate(zip(SCORERS, (BLUE, ORANGE))):
        xs = [x + (i - 0.5) * W for x in XS]
        ms = [stats([r[idx] for r in D[(lr, arm, s)]]) for lr in LRS for arm, _ in ARMS]
        handles.append(ax.bar(xs, [m - CHANCE for m, _, _ in ms], W,
                              yerr=[e for _, e, _ in ms], error_kw=EB, zorder=3,
                              color=colour, label=s))
        for x, (m, e, _) in zip(xs, ms):               # direct labels: the contrast relief
            ax.text(x, m - CHANCE + e + 0.008, f"{m:.3f}", ha="center", fontsize=8, color=MUTED)
            top = max(top, m - CHANCE + e)
    ax.axhline(0, lw=1, color=MUTED, zorder=2)
    ax.set_xticks(XS, XT, fontsize=8.5)
    ax.set_xlim(min(XS) - 0.7, max(XS) + 0.7)
    # Headroom for value labels; the legend lives OUTSIDE the axes (fig.legend) because the
    # outcome this figure tests -- random scoring AS HIGH AS trained -- is exactly the case
    # where an in-axes legend would sit on the right-hand bar's label.
    ax.set_ylim(0, top * 1.20 + 0.02)
    for x, lr in GROUPS:
        ax.text(x, top * 1.13 + 0.02, f"LR {lr}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    return handles


ns = "/".join(str(stats([r[0] for r in D[(LRS[0], a, "detection")]])[2]) for a, _ in ARMS)
handles = draw(axA, 0, "AUROC above chance", f"A  AUROC — bias-free (n = {ns} latents at LR {LRS[0]})")
draw(axB, 1, "balanced accuracy above chance", "B  the same latents, class-balanced accuracy")

fig.suptitle(f"delphi + Llama-3.1-70B on pythia-1b layer {LAYER} — trained vs re-randomized",
             fontsize=11, color=INK, y=0.995)
fig.legend(handles, SCORERS, loc="upper center", bbox_to_anchor=(0.5, 0.945),
           ncol=len(SCORERS), frameon=False, fontsize=9)
fig.text(0.5, 0.005,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars are "
         "±1 SE over LATENTS, the unit delphi samples — not over individual judge decisions. "
         "Two learning rates are shown as a robustness check, not as a sweep.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))

out = os.path.join(HERE, "plots", f"pythia1b_L{LAYER}_auroc.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
