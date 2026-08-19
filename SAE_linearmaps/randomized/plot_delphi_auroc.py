"""Figure: delphi's AUROC on trained vs randomized gemma — the metric the accuracy runs could not give.

This is the run the whole vLLM detour was for. delphi only populates the per-example `probability`
field when the judge returns token logprobs, and logprobs require a LOCAL backend: OpenRouter's
client never requests them, and of the two providers serving llama-3.1-70b there, DeepInfra returns
none and CoreWeave returns only the final token's. So every earlier delphi figure here reports
class-balanced accuracy, computed from hard yes/no predictions, and none of them could report AUROC.

Why that mattered: balanced accuracy throws away the judge's confidence and collapses two
independent axes (discrimination and response bias) into one number. plot_delphi_n500.py showed
exactly that pathology -- the arms matched on balanced accuracy while differing sharply in TPR/TNR
bias, which is what a vacuous explanation ("unrelated texts from various sources") produces: it
matches every example shown, so accepting everything scores at chance rather than below it. AUROC is
rank-based and bias-free, so it measures discrimination alone. If the arms separate here but not on
accuracy, the disagreement with our own pipeline is about the METRIC, not about the models.

Panel A is therefore the result; panel B is the same latents scored the old way, kept alongside so
the comparison is visible in one figure rather than across two.

Per-latent AUROC is computed as the exact Mann-Whitney statistic with ties counted at half weight,
P(score(activating) > score(non-activating)) + 0.5*P(equal), over that latent's own examples.
Everything is then clustered BY LATENT -- the unit delphi samples. Pooling the ~36k individual
scoring decisions would treat 198 latents as if they were 19800 independent trials and understate
the standard errors roughly fourfold.

Provenance: delphi @ EleutherAI (vllm==0.10.2 / torch==2.8.0 / transformers==4.56.1 -- the version
triple its unpinned pyproject.toml actually needs), explainer AND scorer
hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 served locally, --log_probs, our TopK SAEs (the
paper's architecture), gemma-2-2b layer 13, 100M-token SAEs, --max_latents 500 --n_tokens 30000000,
openwebtext. 198 trained / 161 random latents survive delphi's min_examples=200 filter.

    python randomized/plot_delphi_auroc.py
"""
import glob
import json
import math
import os
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [("trained", "andreayhchen/gemma2-2b-linearmap-saes-trained-20m",
            "delphi_trained_L13_local_llama70b.tar.gz", "trained gemma"),
           ("rand", "andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0",
            "delphi_rand_L13_local_llama70b.tar.gz", "randomized gemma")]
SCORERS = ["detection", "fuzz"]
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
CHANCE = 0.5


def results_root():
    d = os.environ.get("RESULTS_DIR")
    if d:
        return d
    cache = os.path.join(HERE, ".delphi_auroc")
    if not os.path.isdir(os.path.join(cache, "results")):
        from huggingface_hub import hf_hub_download
        os.makedirs(cache, exist_ok=True)
        for _, repo, arch, _ in SOURCES:
            tarfile.open(hf_hub_download(repo, arch, repo_type="model")).extractall(cache)
    return os.path.join(cache, "results")


ROOT = results_root()


def auroc(pos, neg):
    """P(pos > neg) + 0.5*P(pos == neg) -- the exact Mann-Whitney U statistic.

    Written out as the pairwise sum rather than via ranks because it is the definition, it handles
    ties without a rank-correction term, and at ~50x50 pairs per latent the cost is irrelevant.
    """
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def per_latent(arm, scorer):
    """(AUROC, balanced accuracy, TPR, TNR) per latent. The LATENT is the sampling unit."""
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{arm}-L13-local/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
        # A latent contributes to AUROC only if the judge returned probabilities for BOTH classes;
        # a parse failure drops that example rather than the latent's accuracy numbers.
        a = auroc([r["probability"] for r in pos if r.get("probability") is not None],
                  [r["probability"] for r in neg if r.get("probability") is not None])
        rows.append((a, (tpr + tnr) / 2, tpr, tnr))
    return rows


def stats(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


D = {(a, s): per_latent(a, s) for a, _, _, _ in SOURCES for s in SCORERS}
NS = {a: len(D[(a, "detection")]) for a, _, _, _ in SOURCES}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.9), dpi=200)
W, EB = 0.34, dict(ecolor=INK, capsize=4, elinewidth=1.4)
XT = [lab for _, _, _, lab in SOURCES]


def draw(ax, idx, ylim, ylabel, title, note_y):
    """Grouped bars of (metric - chance), one pair per arm. Chance is the meaningful zero for both
    AUROC and balanced accuracy, so bars start there: length encodes discriminative signal and the
    baseline is honest rather than a cropped axis."""
    for i, (sc, colour) in enumerate(zip(SCORERS, (BLUE, ORANGE))):
        xs = [j + (i - 0.5) * W for j in range(len(SOURCES))]
        ms = [stats([r[idx] for r in D[(a, sc)]]) for a, _, _, _ in SOURCES]
        ax.bar(xs, [m - CHANCE for m, _, _ in ms], W, yerr=[e for _, e, _ in ms], error_kw=EB,
               zorder=3, color=colour, label=sc)
        for x, (m, e, _) in zip(xs, ms):
            off = (e + 0.010) if m >= CHANCE else -(e + 0.030)
            ax.text(x, m - CHANCE + off, f"{m:.3f}", ha="center", fontsize=8.5, color=MUTED)
    ax.axhline(0, lw=1, color=MUTED, zorder=2)
    ax.set_xticks(range(len(SOURCES)), XT)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    tw = ax.twinx(); tw.set_ylim(CHANCE + ylim[0], CHANCE + ylim[1])
    tw.set_ylabel("(raw value)", color=MUTED, fontsize=9)
    tw.tick_params(colors=MUTED, labelsize=8); tw.spines["top"].set_visible(False)
    for sc, y in zip(SCORERS, note_y):
        (mt, et, _), (mr, er, _) = (stats([r[idx] for r in D[(a, sc)]]) for a, _, _, _ in SOURCES)
        d, s = mt - mr, math.sqrt(et ** 2 + er ** 2)
        ax.text(0.5, y, f"{sc}: gap {d:+.3f} ± {s:.3f}   z = {d / s:.2f}",
                ha="center", fontsize=8, color=MUTED)


draw(axA, 0, (0, 0.42), "AUROC above chance",
     f"A  AUROC — the bias-free measure (n = {NS['trained']} / {NS['rand']} latents)",
     (0.405, 0.383))
axA.legend(frameon=False, fontsize=9, loc="lower right")

draw(axB, 1, (0, 0.42), "class-balanced accuracy above chance",
     "B  the same latents, scored by class-balanced accuracy", (0.405, 0.383))
axB.legend(frameon=False, fontsize=9, loc="lower right")

for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

fig.suptitle("delphi + local Llama-3.1-70B on gemma-2-2b layer 13 — AUROC vs balanced accuracy",
             fontsize=11, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars: ±1 SE over "
         "LATENTS, the unit delphi samples. AUROC is the exact Mann-Whitney statistic per latent, "
         "from the per-example logprobs a locally served judge makes available.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))

out = os.path.join(HERE, "plots", "delphi_L13_auroc.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}\n")

print(f"{'arm':>8} {'scorer':>10} {'AUROC':>15} {'bal acc':>15} {'TPR':>15} {'TNR':>15} {'n':>5}")
for a, _, _, _ in SOURCES:
    for sc in SCORERS:
        cols = [stats([r[i] for r in D[(a, sc)]]) for i in (0, 1, 2, 3)]
        (au, aue, n) = cols[0]
        print(f"{a:>8} {sc:>10} {au:8.3f}±{aue:.3f} " +
              " ".join(f"{m:8.3f}±{e:.3f}" for m, e, _ in cols[1:]) + f" {n:5d}")
print()
for idx, lab in ((0, "AUROC"), (1, "bal acc")):
    for sc in SCORERS:
        (mt, et, nt), (mr, er, nr) = (stats([r[idx] for r in D[(a, sc)]]) for a, _, _, _ in SOURCES)
        d, s = mt - mr, math.sqrt(et ** 2 + er ** 2)
        print(f"  {lab:>8} gap {sc:>9}: {d:+.3f} ± {s:.3f}  z = {d / s:.2f}"
              f"  (clustered by latent, n={nt} vs {nr})")
