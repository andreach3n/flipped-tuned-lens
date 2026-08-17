"""Figure: delphi at proper sample size — the arms match on accuracy but differ in BIAS.

Supersedes the n=35/23 read. That run showed a +0.075 balanced-accuracy gap; at 195/161 latents the
gap is zero. The trained arm barely moved (0.684 -> 0.699) while the random arm went 0.609 -> 0.704,
so the earlier "delphi confirms our result" was small-sample noise. Do not cite those numbers.

What IS there at proper n is a response-bias difference, and it is significant where the accuracy
difference is not: on the random arm the judge over-accepts (high TPR, TNR below chance), and the
two errors cancel inside balanced accuracy. That is the signature of vacuous explanations --
"unrelated texts from various sources" matches every example a scorer is shown.

Everything is computed from the PER-LATENT score files and clustered by latent, which is the unit
delphi actually samples. Pooling the ~15k individual scoring decisions would understate the errors
about fourfold.

Provenance: delphi @ EleutherAI, explainer meta-llama/llama-3.1-70b-instruct via OpenRouter (the
paper's default), our TopK SAEs (the paper's architecture), gemma-2-2b layer 13, 100M-token SAEs,
--max_latents 500 --n_tokens 30000000, openwebtext.

    python randomized/plot_delphi_n500.py
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
            "delphi_trained_L13_n500.tar.gz", "trained gemma"),
           ("rand", "andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0",
            "delphi_rand_L13_n500.tar.gz", "randomized gemma")]
SCORERS = ["detection", "fuzz"]
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#666666", "#d9d9d9"
CHANCE = 0.5


def results_root():
    d = os.environ.get("RESULTS_DIR")
    if d:
        return d
    cache = os.path.join(HERE, ".delphi_n500")
    if not os.path.isdir(os.path.join(cache, "results")):
        from huggingface_hub import hf_hub_download
        os.makedirs(cache, exist_ok=True)
        for _, repo, arch, _ in SOURCES:
            tarfile.open(hf_hub_download(repo, arch, repo_type="model")).extractall(cache)
    return os.path.join(cache, "results")


ROOT = results_root()


def per_latent(arm, scorer):
    """(balanced accuracy, TPR, TNR) per latent -- the latent is the sampling unit."""
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{arm}-L13-n500/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
        rows.append(((tpr + tnr) / 2, tpr, tnr))
    return rows


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


D = {(a, s): per_latent(a, s) for a, _, _, _ in SOURCES for s in SCORERS}
NS = {a: len(D[(a, "detection")]) for a, _, _, _ in SOURCES}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.9), dpi=200)
W, EB = 0.34, dict(ecolor=INK, capsize=4, elinewidth=1.4)
XT = [lab for _, _, _, lab in SOURCES]


def draw(ax, series, idx_of, ylim, ylabel, title):
    """Grouped bars of (metric - chance). Chance is the meaningful zero for these metrics, so the
    baseline is honest and bar length encodes discriminative signal rather than a cropped axis."""
    for i, (lab, colour) in enumerate(series):
        xs = [j + (i - 0.5) * W for j in range(len(SOURCES))]
        ms = [stats([r[idx_of(lab)] for r in D[(a, sc_of(lab))]]) for a, _, _, _ in SOURCES]
        ax.bar(xs, [m - CHANCE for m, _, _ in ms], W, yerr=[e for _, e, _ in ms], error_kw=EB,
               zorder=3, color=colour, label=lab)
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


sc_of = lambda lab: lab                      # panel A: the label IS the scorer
idx_A = lambda lab: 0
draw(axA, [("detection", BLUE), ("fuzz", ORANGE)], idx_A, (0, 0.30),
     "class-balanced accuracy above chance",
     f"A  no difference in accuracy (n = {NS['trained']} / {NS['rand']} latents)")
axA.legend(frameon=False, fontsize=9, loc="lower right")
for sc, y in zip(SCORERS, (0.285, 0.267)):
    (mt, et, _), (mr, er, _) = (stats([r[0] for r in D[(a, sc)]]) for a, _, _, _ in SOURCES)
    d, s = mt - mr, math.sqrt(et ** 2 + er ** 2)
    axA.text(0.5, y, f"{sc}: gap {d:+.3f} ± {s:.3f}   z = {d / s:.2f}",
             ha="center", fontsize=8, color=MUTED)

sc_of = lambda lab: "fuzz"                   # panel B: both series come from the fuzz scorer
idx_B = lambda lab: 1 if lab.startswith("true positive") else 2
draw(axB, [("true positive rate", BLUE), ("true negative rate", ORANGE)], idx_B, (-0.175, 0.40),
     "rate above chance (fuzz scorer)",
     "B  but a large, significant difference in BIAS")
axB.legend(frameon=False, fontsize=9, loc="upper left")
for lab, i, y in (("TPR", 1, -0.115), ("TNR", 2, -0.142)):
    (mt, et, _), (mr, er, _) = (stats([r[i] for r in D[(a, "fuzz")]]) for a, _, _, _ in SOURCES)
    d, s = mr - mt, math.sqrt(et ** 2 + er ** 2)
    axB.text(1.58, y, f"{lab}: random − trained {d:+.3f} ± {s:.3f}   z = {d / s:.1f}",
             ha="right", fontsize=8, color=MUTED)


for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

fig.suptitle("delphi + Llama-3.1-70B on gemma-2-2b layer 13 — trained vs randomized",
             fontsize=11, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Bars show distance above chance (0.5); labels give the raw value. Error bars: ±1 SE over "
         "LATENTS, the unit delphi samples. Same models and SAEs on which our own pipeline separates "
         "the arms by 0.122 (z > 10).",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))

out = os.path.join(HERE, "plots", "delphi_L13_n500.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}\n")

print(f"{'arm':>8} {'scorer':>10} {'bal acc':>15} {'TPR':>15} {'TNR':>15} {'n':>5}")
for a, _, _, _ in SOURCES:
    for sc in SCORERS:
        (m, e, n), (tp, te, _), (tn, tne, _) = (stats([r[i] for r in D[(a, sc)]]) for i in (0, 1, 2))
        print(f"{a:>8} {sc:>10} {m:8.3f}±{e:.3f} {tp:8.3f}±{te:.3f} {tn:8.3f}±{tne:.3f} {n:5d}")
