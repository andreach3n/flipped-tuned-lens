"""THE EXPLANATION-SWAP NULL: how much of delphi's AUROC survives a WRONG explanation?

THE QUESTION. delphi's fuzz/detection score is meant to measure whether a latent's explanation
describes it. The explanation is an INPUT to the scorer: the judge sees one sentence plus a mixed
batch of activating and non-activating windows and says which match. So the metric can be tested
directly -- hand the judge latent 412's sentence while showing it latent 7's windows. A score that
measures explanation quality must fall to 0.5. Whatever survives is the judge exploiting something
else: activating windows are selected by peak activation, so they may be more repetitive, built
around rarer tokens, or simply more "salient-looking" than the negatives, and in fuzz the judge is
only asked whether the token HIGHLIGHTING looks right -- a question answerable without reading the
explanation at all.

That residual is the number Heap et al.'s claim really rests on, and nobody in this literature
reports it.

WHY A DERANGEMENT AND NOT RANDOM TEXT. Every explanation is still used exactly once, so the pool of
sentences is identical in length, style, vagueness and judge-model idiosyncrasy. Only the PAIRING
is destroyed. If scores stay high, it cannot be blamed on the swapped text being worse -- it is the
same text. A derangement also guarantees no latent keeps its own explanation, which a plain shuffle
does not (a random permutation of n leaves ~1 fixed point in expectation).

HOW IT RUNS, AND WHY IT IS CHEAP. delphi skips caching when `latents/<hookpoint>` already exists
(`non_redundant_hookpoints`), and `--explainer none` makes it load each explanation from disk via
NoOpExplainer instead of generating one. So this script builds a sibling results directory whose
`latents/` is HARDLINKED to the original (instant, zero extra bytes on tmpfs) and whose
`explanations/` holds the deranged copies, and the delphi run that follows does scoring only.

    python swap_explanations.py --results /dev/shm/delphi_run/results --cell pythia1b_trained_R8_L8_lr2e-3

then, with the SAME model/sae/dataset arguments as the original run plus `--explainer none`:

    python delphi_cuda.py <model> <sae_dir> --hookpoints layers.8 --scorers fuzz detection \
      --log_probs --max_latents 500 --n_tokens 30000000 --num_gpus 1 \
      --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' \
      --explainer none --name <cell>_swap

WHAT TO COMPARE. The swapped cell against the original, per latent, same statistic as everywhere
else in this project (exact Mann-Whitney, clustered by latent). Report per ARM: true AUROC, swapped
AUROC, and the difference -- the explanation-dependent component. Run both arms; the interesting
quantity is whether the two arms lose the same amount.

CAVEAT WORTH STATING UP FRONT: delphi's own explainer and scorer example sets already overlap (both
are drawn by `split_quantiles` after `random.seed(22)`, and in the top quantile the 4 training
examples are exactly the first 4 of the 5 test ones). That inflates the TRUE score a little. It
does not affect the swapped score, which never saw a matching explanation at all, so if anything it
makes the measured drop an overestimate of the explanation-dependent part.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys


def latent_id(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def derange(keys, seed):
    """A permutation of `keys` with no fixed point."""
    rng = random.Random(seed)
    if len(keys) < 2:
        raise SystemExit("need at least 2 explanations to derange")
    order = list(keys)
    while True:
        shuffled = order[:]
        rng.shuffle(shuffled)
        # Repair fixed points by swapping each with its successor; with n >= 2 this always
        # terminates, and rejection-sampling whole permutations would be needlessly slow.
        for i in range(len(shuffled)):
            if shuffled[i] == order[i]:
                j = (i + 1) % len(shuffled)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        if all(a != b for a, b in zip(order, shuffled)) and sorted(shuffled) == sorted(order):
            return dict(zip(order, shuffled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="delphi results root")
    ap.add_argument("--cell", required=True, help="source cell name")
    ap.add_argument("--suffix", default="_swap")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hookpoint", default="layers.8")
    args = ap.parse_args()

    src = os.path.join(args.results, args.cell)
    dst = os.path.join(args.results, args.cell + args.suffix)
    if not os.path.isdir(os.path.join(src, "latents", args.hookpoint)):
        raise SystemExit(f"no latent cache at {src}/latents/{args.hookpoint} -- "
                         "the swap needs it, and `tar --exclude latents` drops it")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists; delete it or pass a different --suffix")

    os.makedirs(dst)
    # Hardlink the cache: 14 GB per cell, and `cp -al` on tmpfs is instant and costs nothing.
    subprocess.run(["cp", "-al", os.path.join(src, "latents"), os.path.join(dst, "latents")],
                   check=True)

    src_expl = os.path.join(src, "explanations")
    dst_expl = os.path.join(dst, "explanations")
    os.makedirs(dst_expl)
    files = sorted(f for f in os.listdir(src_expl) if f.endswith(".txt"))
    if not files:
        raise SystemExit(f"no explanations in {src_expl}")

    mapping = derange(files, args.seed)
    for target, source in mapping.items():
        # target keeps its FILENAME (so delphi finds it) but receives source's CONTENT.
        shutil.copyfile(os.path.join(src_expl, source), os.path.join(dst_expl, target))

    out = {"cell": args.cell, "swapped_cell": args.cell + args.suffix, "seed": args.seed,
           "n": len(files), "fixed_points": sum(1 for k, v in mapping.items() if k == v),
           "mapping": {str(latent_id(k)): str(latent_id(v)) for k, v in mapping.items()}}
    with open(os.path.join(dst, "swap_mapping.json"), "w") as f:
        json.dump(out, f, indent=1)

    assert out["fixed_points"] == 0, "not a derangement"
    assert sorted(mapping.values()) == files, "not a permutation -- some explanation lost or reused"
    print(f"{args.cell} -> {args.cell + args.suffix}")
    print(f"  latents/   hardlinked ({len(os.listdir(os.path.join(dst,'latents',args.hookpoint)))} shards)")
    print(f"  explanations/  {len(files)} deranged, 0 fixed points")
    for k in files[:3]:
        print(f"    {k}  now carries the text written for  {mapping[k]}")
    print(f"  mapping saved to {dst}/swap_mapping.json")
    print("\nnow rerun delphi with the SAME arguments as the original plus:  "
          f"--explainer none --name {args.cell + args.suffix}")


if __name__ == "__main__":
    sys.exit(main())
