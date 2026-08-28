"""Is the trained arm's AUROC driven by SELECTION on firing frequency? Pure re-analysis, no GPU.

THE QUESTION. At LR 1e-3 the trained arm scores fuzz AUROC 0.626 over 397 latents; at 2e-3 it
scores 0.841 over 192. The dictionary got WORSE (alive 96.1% -> 71.9%) and the score went UP,
which degradation cannot explain but selection can: delphi only scores latents clearing
`min_examples=200`, so killing a quarter of the dictionary removes the rare, weakly-firing
latents -- exactly the ones hardest to describe -- and leaves a self-selected, easier sample.

If that is what happened, then "Heap et al. replicates at 2e-3" is an artifact of the sample,
not a fact about the models, and the 1e-3 cell is the result.

THE TEST, entirely within ONE cell so nothing is confounded with LR:
  1. Does per-latent AUROC rise with per-latent firing count? (Spearman, plus deciles.)
  2. Restricted to the TOP_N most-active latents -- matching the other cell's n -- does the
     mean AUROC climb toward that cell's value?
  Climbs  => selection confirmed; 2e-3's "replication" is a measurement artifact.
  Flat    => selection is NOT the explanation; high LR genuinely yields more describable
             trained features, which is a real finding about SAE training rather than a bug.

The deciles matter as much as the headline: a monotone rise across all ten is strong evidence,
whereas a jump confined to the top decile would say something narrower (a few very active
latents, not a general frequency effect).

Firing counts come from the CACHE, not the scores -- `locations[:, 2] + start` over every
shard, bincounted. That is the same quantity `min_examples` filters on, which is what makes it
the right covariate rather than a proxy.

    RESULTS_DIR=/dev/shm/delphi_run/results CELL=pythia1b_trained_L8_lr1e-3 \
      TOP_N=192 TARGET=0.841 python -u firing_vs_auroc.py

Run it on the random arm too (TOP_N=410): if the effect is specific to the trained arm, that is
itself informative -- it would mean frequency predicts describability only where the dictionary
is heavy-tailed.
"""
import glob
import json
import math
import os
import re

import numpy as np
from safetensors.numpy import load_file

ROOT    = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
CELL    = os.environ.get("CELL", "pythia1b_trained_L8_lr1e-3")
LAYER   = int(os.environ.get("LAYER", 8))
HOOKPOINT = os.environ.get("HOOKPOINT", f"layers.{LAYER}")   # skip-embed: layers.8.skipembed
TOP_N   = int(os.environ.get("TOP_N", 192))
TARGET  = float(os.environ.get("TARGET", 0)) or None   # the other cell's AUROC, for context
SCORERS = os.environ.get("SCORERS", "detection,fuzz").split(",")


def auroc(pos, neg):
    """Exact Mann-Whitney: P(pos > neg) + 0.5*P(pos == neg). Same statistic as report_auroc.py."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def latent_id(path):
    """delphi names score files per latent; take the LAST integer in the stem.

    Written defensively because the naming has varied across delphi versions -- the script
    prints a few parsed examples so a silent mis-parse cannot pass unnoticed.
    """
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def per_latent_auroc(scorer):
    out = {}
    files = sorted(glob.glob(f"{ROOT}/{CELL}/scores/{scorer}/*.txt"))
    for f in files:
        lid = latent_id(f)
        if lid is None:
            continue
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r["probability"] for r in recs if r["activating"] and r.get("probability") is not None]
        neg = [r["probability"] for r in recs if not r["activating"] and r.get("probability") is not None]
        a = auroc(pos, neg)
        if a is not None:
            out[lid] = a
    return out, files


def firing_counts():
    """Per-latent firing count.

    Prefers delphi's log/hookpoint_firing_counts.pt: full dictionary, sub-MB, and present even
    when the archive excluded the 14 GB latents/ tree. Falls back to bincounting the shards,
    where locations[:, 2] is the latent index MINUS the shard's start.
    """
    p = f"{ROOT}/{CELL}/log/hookpoint_firing_counts.pt"
    if os.path.exists(p):
        import torch as t
        obj = t.load(p, weights_only=True, map_location="cpu")
        v = list(obj.values())[0] if isinstance(obj, dict) else obj
        return {i: int(c) for i, c in enumerate(v.tolist()) if c > 0}

    counts = {}
    shards = sorted(glob.glob(f"{ROOT}/{CELL}/latents/{HOOKPOINT}/*.safetensors"))
    if not shards:
        raise SystemExit(f"no firing counts for {CELL}: neither {p} nor a latents/{HOOKPOINT} "
                         f"cache. Set HOOKPOINT if it is not layers.{LAYER}.")
    for s in shards:
        start = int(os.path.basename(s).split("_")[0])
        loc = load_file(s)["locations"]
        idx = loc[:, 2].astype(np.int64) + start
        for lid, c in zip(*np.unique(idx, return_counts=True)):
            counts[int(lid)] = counts.get(int(lid), 0) + int(c)
    return counts


def stats(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan"), float("nan"), len(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs)), len(xs)


def spearman(xs, ys):
    """Rank correlation, average ranks for ties. Hand-rolled to avoid a scipy dependency."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# =======================================================================================
print(f"\n=== firing_vs_auroc: {CELL} (layer {LAYER}) ===\n")
counts = firing_counts()
print(f"cache: {len(counts)} latents with >=1 firing")

for scorer in SCORERS:
    a_by_latent, files = per_latent_auroc(scorer)
    if not a_by_latent:
        print(f"\n[{scorer}] no scored latents found -- skipping")
        continue
    if scorer == SCORERS[0]:
        ex = [(os.path.basename(f), latent_id(f)) for f in files[:3]]
        print(f"filename parse check (name -> latent id): {ex}")

    common = sorted(set(a_by_latent) & set(counts))
    missing = len(a_by_latent) - len(common)
    fs = [counts[i] for i in common]
    au = [a_by_latent[i] for i in common]

    m, se, n = stats(au)
    r = spearman(fs, au)
    z = r * math.sqrt(max(n - 1, 1))

    print(f"\n[{scorer}]  n = {n} scored latents"
          + (f"  ({missing} scored latents absent from cache -- check the shard)" if missing else ""))
    print(f"  mean AUROC (all)            {m:.3f} +/- {se:.3f}")
    print(f"  Spearman(firing, AUROC)     {r:+.3f}   z = {z:+.2f}")
    print(f"  firing count range          {min(fs)} .. {max(fs)}  (median {sorted(fs)[len(fs)//2]})")

    # Deciles: a monotone rise is much stronger evidence than a top-decile-only jump.
    order = sorted(range(n), key=lambda i: fs[i])
    print(f"\n  {'decile':>7} {'n':>5} {'median firings':>15} {'mean AUROC':>18}")
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        if hi <= lo:
            continue
        seg = order[lo:hi]
        sm, sse, sn = stats([au[i] for i in seg])
        med = sorted(fs[i] for i in seg)[sn // 2]
        print(f"  {d+1:>7} {sn:>5} {med:>15} {sm:>11.3f} +/- {sse:.3f}")

    # The headline: restrict to the TOP_N most-active latents, matching the other cell's n.
    if TOP_N < n:
        top = order[-TOP_N:]
        tm, tse, tn = stats([au[i] for i in top])
        print(f"\n  TOP {TOP_N} by firing count   {tm:.3f} +/- {tse:.3f}   (all {n}: {m:.3f} +/- {se:.3f})")
        print(f"  shift from selection alone  {tm - m:+.3f} +/- {math.sqrt(tse**2 + se**2):.3f}")
        if TARGET:
            print(f"  other cell's value          {TARGET:.3f}")
            # Only meaningful when the target sits ABOVE this cell's mean -- otherwise there is
            # no gap for selection to close and the ratio divides by a negative.
            if TARGET - m > 1e-9:
                closed = (tm - m) / (TARGET - m)
                print(f"  fraction of the gap closed by selection alone: {closed:.0%}")
                print("    ~100% => selection fully explains the other cell; "
                      "~0% => it explains none of it.")
            else:
                print("  (target is not above this cell's mean -- no gap for selection to "
                      "close, ratio not reported)")
    else:
        print(f"\n  TOP_N={TOP_N} >= n={n}; nothing to restrict.")

print("\nNOTE: firing count is measured on the SCORING corpus via the cache, i.e. the same\n"
      "quantity min_examples filters on -- not a proxy for it.\n")
