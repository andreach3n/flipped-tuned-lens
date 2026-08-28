"""Trained vs random at MATCHED firing frequency -- the unconfounded comparison. No GPU.

WHY THIS EXISTS. `firing_vs_auroc.py` showed the two arms' dictionaries are not comparable
populations. On the trained arm, per-latent AUROC rises steeply with firing count (Spearman
+0.617): its bottom seven deciles sit at 0.50-0.59, i.e. CHANCE, and its top decile reaches
0.884. On the random arm there is no relationship at all (Spearman -0.012) -- every decile is
~0.81. And the two frequency distributions differ by more than 5x at the median (trained 459,
random 2482 firings).

So the naive headline -- mean AUROC 0.626 trained vs 0.808 random -- is a comparison of two
different frequency MIXTURES, not of two models. The trained arm carries a large mass of rare
latents scoring at chance, which drags its mean down; the random arm has almost none.

This script removes that confound by comparing arms only WITHIN matched firing-count bins,
then combining bins by direct standardization to a common (pooled) frequency distribution.

    naive gap      = mean_trained - mean_random                      (confounded)
    adjusted gap   = sum_b w_b*(mean_trained,b - mean_random,b)      (frequency-matched)
    with w_b the POOLED share of latents in bin b, so both arms are evaluated against the
    same frequency profile. SE = sqrt(sum_b w_b^2 * (se_t,b^2 + se_r,b^2)).

READ THE PER-BIN TABLE, NOT JUST THE ADJUSTED NUMBER. Standardization assumes the gap is
reasonably stable across bins; if it flips sign between bins, one summary number is the wrong
object and the bins themselves are the result.

Bins with fewer than MIN_PER_BIN latents in EITHER arm are dropped -- a bin one arm barely
populates cannot support a within-bin contrast. The script reports how much mass that discards,
because a large discard would mean the arms' frequency ranges barely overlap, and then no
matched comparison is possible at all.

    RESULTS_DIR=/dev/shm/delphi_run/results LR=1e-3 python -u matched_comparison.py

Helpers duplicated from firing_vs_auroc.py deliberately -- every analysis script in this
project stands alone (see ../plot_delphi_auroc.py, ../plot_delphi_2x2.py).
"""
import glob
import json
import math
import os
import re

import numpy as np
from safetensors.numpy import load_file

ROOT     = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
LAYER    = int(os.environ.get("LAYER", 8))
LR       = os.environ.get("LR", "1e-3")
CELL_FMT = os.environ.get("CELL_FMT", "pythia1b_{arm}_L{layer}_lr{lr}")
# Skip-embed cells hook `layers.8.skipembed`; only the shard fallback in firing_counts() uses it.
HOOKPOINT = os.environ.get("HOOKPOINT", f"layers.{LAYER}")
SCORERS  = os.environ.get("SCORERS", "detection,fuzz").split(",")
MIN_PER_BIN = int(os.environ.get("MIN_PER_BIN", 5))
# Log-spaced (doubling) bins: firing counts span 200 to ~650,000, so linear bins would put
# almost everything in one bucket.
EDGES = [float(x) for x in os.environ.get(
    "BIN_EDGES", "0,400,800,1600,3200,6400,12800,25600,51200,1e18").split(",")]
ARMS = ["trained", "rand"]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def latent_id(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else None


def per_latent_auroc(cell, scorer):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{scorer}/*.txt")):
        lid = latent_id(f)
        if lid is None:
            continue
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r["probability"] for r in recs if r["activating"] and r.get("probability") is not None]
        neg = [r["probability"] for r in recs if not r["activating"] and r.get("probability") is not None]
        a = auroc(pos, neg)
        if a is not None:
            out[lid] = a
    return out


def firing_counts(cell):
    """Prefers delphi's log/hookpoint_firing_counts.pt -- full dictionary, sub-MB, and it
    survives the `--exclude <cell>/latents` used when archiving. Shard bincount is the fallback.
    """
    p = f"{ROOT}/{cell}/log/hookpoint_firing_counts.pt"
    if os.path.exists(p):
        import torch as t
        obj = t.load(p, weights_only=True, map_location="cpu")
        v = list(obj.values())[0] if isinstance(obj, dict) else obj
        return {i: int(c) for i, c in enumerate(v.tolist()) if c > 0}

    counts = {}
    shards = sorted(glob.glob(f"{ROOT}/{cell}/latents/{HOOKPOINT}/*.safetensors"))
    if not shards:
        raise SystemExit(f"no firing counts for {cell}: neither {p} nor a latents/{HOOKPOINT} "
                         f"cache. Set HOOKPOINT if it is not layers.{LAYER}.")
    for s in shards:
        start = int(os.path.basename(s).split("_")[0])
        idx = load_file(s)["locations"][:, 2].astype(np.int64) + start
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


def fmt(x):
    return f"{x:>7.0f}" if x < 1e17 else "    inf"


# =======================================================================================
print(f"\n=== matched_comparison: pythia-1b layer {LAYER}, LR {LR} ===")

data = {}
for arm in ARMS:
    cell = CELL_FMT.format(arm=arm, layer=LAYER, lr=LR)
    fc = firing_counts(cell)
    data[arm] = {s: {"auroc": per_latent_auroc(cell, s), "fc": fc} for s in SCORERS}
    n0 = len(data[arm][SCORERS[0]]["auroc"])
    med = sorted(fc[i] for i in data[arm][SCORERS[0]]["auroc"] if i in fc)
    print(f"  {arm:>8}: {n0} scored latents, median firing "
          f"{med[len(med)//2] if med else float('nan')}")

for scorer in SCORERS:
    print(f"\n--- {scorer} " + "-" * 62)
    # Collect (firing, auroc) per arm
    pts = {}
    for arm in ARMS:
        a, fc = data[arm][scorer]["auroc"], data[arm][scorer]["fc"]
        pts[arm] = [(fc[i], a[i]) for i in sorted(a) if i in fc]

    naive = {arm: stats([v for _, v in pts[arm]]) for arm in ARMS}
    nd = naive["trained"][0] - naive["rand"][0]
    nse = math.sqrt(naive["trained"][1] ** 2 + naive["rand"][1] ** 2)

    print(f"  {'bin (firings)':>22} {'n_tr':>5} {'trained':>16} {'n_rd':>5} {'random':>16} "
          f"{'gap':>16} {'z':>6}")
    rows, kept_t, kept_r, dropped = [], 0, 0, 0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        seg = {arm: [v for f, v in pts[arm] if lo <= f < hi] for arm in ARMS}
        nt, nr = len(seg["trained"]), len(seg["rand"])
        if nt < MIN_PER_BIN or nr < MIN_PER_BIN:
            dropped += nt + nr
            continue
        mt, set_, _ = stats(seg["trained"])
        mr, ser, _ = stats(seg["rand"])
        g = mt - mr
        gse = math.sqrt(set_ ** 2 + ser ** 2)
        rows.append((nt + nr, g, gse))
        kept_t += nt
        kept_r += nr
        print(f"  {fmt(lo)}-{fmt(hi)} {nt:>5} {mt:>9.3f}+/-{set_:.3f} {nr:>5} "
              f"{mr:>9.3f}+/-{ser:.3f} {g:>+9.3f}+/-{gse:.3f} {g/gse:>6.2f}")

    if not rows:
        print("  no bin has enough latents in BOTH arms -- the frequency ranges do not "
              "overlap, so no matched comparison is possible.")
        continue

    tot = sum(n for n, _, _ in rows)
    adj = sum((n / tot) * g for n, g, _ in rows)
    adj_se = math.sqrt(sum((n / tot) ** 2 * gse ** 2 for n, _, gse in rows))
    total_latents = len(pts["trained"]) + len(pts["rand"])

    print(f"\n  naive gap (trained - random)      {nd:>+7.3f} +/- {nse:.3f}   z = {nd/nse:>6.2f}")
    print(f"  frequency-matched (adjusted) gap  {adj:>+7.3f} +/- {adj_se:.3f}   z = {adj/adj_se:>6.2f}")
    print(f"  confounding removed               {nd - adj:>+7.3f}"
          f"   ({abs(nd - adj)/abs(nd)*100 if nd else 0:.0f}% of the naive gap)")
    print(f"  latents used: {kept_t} trained + {kept_r} random of {total_latents} "
          f"({dropped} dropped in thin bins = {dropped/total_latents*100:.0f}%)")
    signs = {g > 0 for _, g, _ in rows}
    if len(signs) > 1:
        print("  *** the gap CHANGES SIGN across bins -- the single adjusted number is not a "
              "faithful summary; quote the per-bin table instead. ***")

print("\nInterpretation: adjusted ~ 0 => the naive difference was entirely the frequency\n"
      "mixture. Adjusted still large => a real per-latent difference the mixture was hiding.\n")
