"""Summarise delphi scores for ARBITRARY result dirs -- the 2x2 attribution of the false null.

plot_delphi_2x2.py hardcodes the four science cells. This does the same per-latent maths over
whatever directories you name, which is what the defect ablation needs:

    the old delphi result was produced under THREE simultaneous defects --
      (1) b_dec applied twice by convert_sae_to_sparsify.py   } the CONVERSION
      (2) the decoder-norm scaling never folded in            }
      (3) activations cached BOS-free, while the SAEs were trained on BOS-prefixed contexts

`write_delphi_cache.py` now toggles them independently: EMULATE_SPARSIFY reproduces (1)+(2),
PREPEND_BOS controls (3). On the RANDOM arm with the plain SAE -- the cell where the old numbers
were inflated ~0.14 -- the four combinations separate the causes:

    | condition                       | EMULATE_SPARSIFY | PREPEND_BOS | expectation             |
    |---------------------------------|------------------|-------------|-------------------------|
    | rand_full_abl_conv_nobos        | rand sparsify    | 0           | ~the old 0.781 fuzz     |
    | rand_full_abl_conv_bos          | rand sparsify    | 1           | conversion alone        |
    | rand_full_abl_true_nobos        | (unset)          | 0           | BOS regime alone        |
    | rand_full  (already computed)   | (unset)          | 1           | the corrected 0.632     |

Whichever cell stays near 0.781 identifies the defect that manufactured the false replication.

    RESULTS_DIR=/dev/shm/delphi_run/results python randomized/ablation_delphi.py \
        rand_full_abl_conv_nobos rand_full_abl_conv_bos rand_full_abl_true_nobos rand_full
"""
import glob
import json
import math
import os
import sys

ROOT = os.environ.get("RESULTS_DIR", "/dev/shm/delphi_run/results")
SCORERS = ["detection", "fuzz"]
CELLS = sys.argv[1:]
if not CELLS:
    raise SystemExit(__doc__)


def auroc(pos, neg):
    """Exact Mann-Whitney: P(pos > neg) + 0.5*P(pos == neg). Same definition as the main figure."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def per_latent(cell, scorer):
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{cell}/scores/{scorer}/*.txt")):
        recs = [r for r in json.load(open(f)) if r.get("prediction") is not None]
        pos = [r for r in recs if r["activating"]]
        neg = [r for r in recs if not r["activating"]]
        if not pos or not neg:
            continue
        tpr = sum(bool(r["prediction"]) for r in pos) / len(pos)
        tnr = sum(not bool(r["prediction"]) for r in neg) / len(neg)
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


print(f"{'cell':>28} {'scorer':>10} {'AUROC':>15} {'bal acc':>15} {'TPR':>15} {'TNR':>15} {'n':>5}")
vals = {}
for c in CELLS:
    for sc in SCORERS:
        rows = per_latent(c, sc)
        if not rows:
            print(f"{c:>28} {sc:>10}   (no scores under {ROOT}/{c})")
            continue
        cols = [stats([r[i] for r in rows]) for i in range(4)]
        vals[(c, sc)] = cols[0]
        print(f"{c:>28} {sc:>10} " + " ".join(f"{m:8.3f}±{e:.3f}" for m, e, _ in cols)
              + f" {cols[0][2]:5d}")

# Every condition against the LAST one named, which should be the fully corrected cell. The defect
# that matters is whichever condition stays far from it.
if len(CELLS) > 1:
    ref = CELLS[-1]
    print(f"\nAUROC difference vs {ref} (the corrected cell):")
    for c in CELLS[:-1]:
        for sc in SCORERS:
            if (c, sc) not in vals or (ref, sc) not in vals:
                continue
            m, e, _ = vals[(c, sc)]
            mr, er, _ = vals[(ref, sc)]
            d, se = m - mr, math.sqrt(e ** 2 + er ** 2)
            print(f"  {c:>28} {sc:>9}: {d:+.3f} ± {se:.3f}  z = {d / se:5.2f}")
