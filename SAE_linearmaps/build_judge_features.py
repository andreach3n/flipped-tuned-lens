"""Sample frequency-matched features across the full / hybrid / outbias SAEs and
emit blinded per-feature prompt blocks for the LLM abstractness judge.

This is steps (1)+(2) of the LLM-judge feature-complexity experiment: it decides
*which* features to judge (stratified so the three SAEs have identical firing-
frequency marginals — otherwise the judge inherits the frequency confound the
whole triviality writeup is about) and builds *what the judge sees* (a PEAK block
of strongest activations + a TYPICAL block of randomly-sampled firings, so the
judge can reproduce the peak-vs-typical divergence from Tests 6-7).

Output (written to CACHE_DIR):
  judge_blind.json  — shuffled list of {id, peak, typical}; NO SAE/feature label.
                      This is the only file the judge sees.
  judge_key.json    — {id -> sae, feat, freq, freq_bin, nd, nd_peak, ...} for
                      un-blinding after the ratings come back.

Run on the box (see the sae-infra note): first
  source /workspace/startup.sh && export HF_HUB_OFFLINE=1
then
  DRY_RUN=1 python -u build_judge_features.py   # print the sampling plan and stop
  python -u build_judge_features.py             # full run (streams the cache)
Env knobs: N_TOKENS, TARGET_PER_BIN, SEED, DRY_RUN.
"""
import glob
import json
import math
import os
import numpy as np
import torch as t
from transformers import AutoTokenizer
from sae_lens import BatchTopKTrainingSAE

CACHE_DIR = "/workspace/sae_cache_layer13"
MODEL_NAME = "google/gemma-2-2b"

# SAEs to compare. outbias's encoder sees the raw h (the trained map is added only
# at the *output*), so for finding activations it encodes exactly like full —
# only the learned weights differ. hybrid's encoder sees h - P_hybrid[token].
SAE_FILES = {
    "full":    f"{CACHE_DIR}/sae_full_final.pt",
    "hybrid":  f"{CACHE_DIR}/sae_hybrid_final.pt",
    "outbias": f"{CACHE_DIR}/sae_outbias_k64_final.pt",
}
STATS_FILES = {
    "full":    f"{CACHE_DIR}/stats_full.pt",
    "hybrid":  f"{CACHE_DIR}/stats_hybrid.pt",
    "outbias": f"{CACHE_DIR}/stats_outbias.pt",
}
P_HYBRID_PATH = f"{CACHE_DIR}/P_hybrid.pt"

# --- knobs -------------------------------------------------------------------
N_TOKENS       = int(os.environ.get("N_TOKENS", 3_000_000))   # tokens to scan for examples
TARGET_PER_BIN = int(os.environ.get("TARGET_PER_BIN", 130))   # cap per freq bin per SAE (~1k total)
SEED           = int(os.environ.get("SEED", 0))
DRY_RUN        = os.environ.get("DRY_RUN", "0") == "1"
K_PEAK    = 12     # strongest activating examples shown per feature
K_TYPICAL = 12     # randomly-sampled ("typical") firings shown per feature
# reservoir size == K_TYPICAL so the typical block is an UNBIASED uniform sample of
# firings (if POOL > K, render's top-by-activation cut would bias it back toward peaks)
POOL      = K_TYPICAL
WINDOW    = 8      # context tokens on each side of the activating token

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def die(msg):
    raise SystemExit(f"[build_judge_features] {msg}")


def check_files():
    missing = [p for p in list(SAE_FILES.values()) + list(STATS_FILES.values()) if not os.path.exists(p)]
    if not os.path.exists(P_HYBRID_PATH):
        missing.append(P_HYBRID_PATH)
    if not glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt"):
        missing.append(f"{CACHE_DIR}/layer_13_chunk_*.pt")
    if missing:
        die("missing required files (did outbias's eval_trivial run to write stats_outbias.pt?):\n  "
            + "\n  ".join(missing))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]


# =========================== feature sampling ================================
# Bin by half-order-of-magnitude of firing frequency (the writeup's bins):
# bin index b = floor(2*log10(freq)) -> each bin spans [10^(b/2), 10^((b+1)/2)).
def freq_bin(freq):
    b = np.full(freq.shape, -1, dtype=np.int64)
    pos = freq > 0
    b[pos] = np.floor(2.0 * np.log10(freq[pos])).astype(np.int64)
    return b


def bin_label(b):
    return f"{10**(b/2.0):,.0f}-{10**((b+1)/2.0):,.0f}"


def load_stats():
    """Per-SAE: alive feature ids, their freq, freq-bin, nd, nd_peak."""
    out = {}
    for sae, path in STATS_FILES.items():
        st = t.load(path)
        alive = st["alive"].cpu().numpy().astype(bool)
        freq = st["freq"].cpu().numpy().astype(np.float64)
        nd = st["nd"].cpu().numpy() if "nd" in st else None
        ndp = st["nd_peak"].cpu().numpy() if "nd_peak" in st else None
        out[sae] = {"alive": alive, "freq": freq, "bin": freq_bin(freq), "nd": nd, "nd_peak": ndp}
    return out


def sample_features(stats, rng):
    """Stratified equal-N-per-bin sampling. For each freq bin present in ALL three
    SAEs, draw N_b = min(TARGET_PER_BIN, smallest alive count across SAEs) features
    from each SAE -> identical frequency marginals by construction. Prints the plan."""
    saes = list(stats.keys())
    # alive members of each (sae, bin)
    members = {s: {} for s in saes}
    for s in saes:
        alive_ids = np.where(stats[s]["alive"] & (stats[s]["freq"] > 0))[0]
        for fid in alive_ids:
            members[s].setdefault(int(stats[s]["bin"][fid]), []).append(int(fid))

    all_bins = sorted(set().union(*[set(members[s]) for s in saes]))
    print(f"\n{'bin (firings)':>18} | " + " | ".join(f"{s:>8}" for s in saes) + " | chosen")
    print("-" * (18 + 3 + 11 * len(saes) + 9))

    selected = {s: [] for s in saes}
    total = {s: 0 for s in saes}
    for b in all_bins:
        counts = [len(members[s].get(b, [])) for s in saes]
        n_b = min(TARGET_PER_BIN, min(counts)) if min(counts) > 0 else 0
        for s in saes:
            if n_b > 0:
                pick = rng.choice(members[s][b], size=n_b, replace=False)
                selected[s].extend(int(x) for x in pick)
                total[s] += n_b
        flag = "" if min(counts) > 0 else "  (skipped: absent in a SAE)"
        print(f"{bin_label(b):>18} | " + " | ".join(f"{c:>8,}" for c in counts) + f" | {n_b:>6}{flag}")

    print("-" * (18 + 3 + 11 * len(saes) + 9))
    print(f"{'total / SAE':>18} | " + " | ".join(f"{total[s]:>8,}" for s in saes))
    return selected


# =========================== example collection ==============================
def sae_input(hh, tt, mode, P_hybrid):
    # hybrid encodes the residual h - P_hybrid[token]; full AND outbias encode raw h
    # (outbias only differs from full in where the trained map is applied — at the
    # output — so its encoder input is identical to full).
    if mode == "hybrid":
        return hh - P_hybrid[tt]
    return hh


def collect(mode, sae, scale, fids, P_hybrid, gen, bs=8192):
    """One streaming pass over the cache. For each of `fids` keep, in bounded
    memory: the top-K_PEAK activations, and a reservoir (uniform random sample) of
    firings for the TYPICAL block. Returns positions/values into the scanned token
    stream, which we also return so context windows can be sliced afterwards."""
    nf = len(fids)
    fids_t = t.tensor(fids, device=device)
    pk_val = [t.empty(0) for _ in range(nf)]           # top-K peak activations
    pk_pos = [t.empty(0, dtype=t.long) for _ in range(nf)]
    rs_key = [t.empty(0) for _ in range(nf)]           # random keys (min-key reservoir)
    rs_val = [t.empty(0) for _ in range(nf)]
    rs_pos = [t.empty(0, dtype=t.long) for _ in range(nf)]
    tok_parts, gpos, seen = [], 0, 0

    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0)
            tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start + bs].float().to(device)
                ttk = tc[start:start + bs].to(device)
                x = sae_input(hh, ttk, mode, P_hybrid)
                a = sae.encode(x / scale)[:, fids_t].cpu()     # (b, nf) chosen-feature acts
                tok_parts.append(tc[start:start + bs].cpu())
                b = a.shape[0]
                nz = a > 0
                # only touch features that actually fired in this batch (features are sparse)
                for j in nz.any(0).nonzero(as_tuple=False).squeeze(1).tolist():
                    rows = nz[:, j].nonzero(as_tuple=False).squeeze(1)
                    vals = a[rows, j]
                    pos = rows + gpos                            # global token positions
                    # --- peak: keep the K_PEAK largest activations seen so far ---
                    cv = t.cat([pk_val[j], vals]); cp = t.cat([pk_pos[j], pos])
                    if cv.numel() > K_PEAK:
                        top = cv.topk(K_PEAK).indices
                        cv, cp = cv[top], cp[top]
                    pk_val[j], pk_pos[j] = cv, cp
                    # --- typical: min-random-key reservoir = uniform sample of firings ---
                    keys = t.rand(vals.numel(), generator=gen)
                    ck = t.cat([rs_key[j], keys]); cvv = t.cat([rs_val[j], vals]); cpp = t.cat([rs_pos[j], pos])
                    if ck.numel() > POOL:
                        sel = ck.topk(POOL, largest=False).indices
                        ck, cvv, cpp = ck[sel], cvv[sel], cpp[sel]
                    rs_key[j], rs_val[j], rs_pos[j] = ck, cvv, cpp
                gpos += b; seen += b
                if seen >= N_TOKENS:
                    break
            print(f"  [{mode}] scanned {seen:,}/{N_TOKENS:,} tokens", flush=True)
            if seen >= N_TOKENS:
                break
    return pk_val, pk_pos, rs_val, rs_pos, t.cat(tok_parts)


def render(pos_list, val_list, toks_all, k):
    """Render up to k examples (strongest first) as {act, text}, with the
    activating token wrapped in guillemets so the judge can see where it fired."""
    order = sorted(range(len(val_list)), key=lambda i: -val_list[i])[:k]
    out = []
    for i in order:
        p = int(pos_list[i]); v = float(val_list[i])
        lo, hi = max(0, p - WINDOW), min(len(toks_all), p + WINDOW + 1)
        left = tokenizer.decode(toks_all[lo:p].tolist())
        tok = tokenizer.decode([int(toks_all[p])])
        right = tokenizer.decode(toks_all[p + 1:hi].tolist())
        out.append({"act": round(v, 2), "text": f"{left}《{tok}》{right}"})
    return out


# =============================== main ========================================
def main():
    check_files()
    rng = np.random.default_rng(SEED)
    stats = load_stats()
    selected = sample_features(stats, rng)
    if DRY_RUN:
        print("\nDRY_RUN=1 -> plan only, not collecting examples.")
        return

    P_hybrid = t.load(P_HYBRID_PATH, map_location=device)
    gen = t.Generator().manual_seed(SEED)   # reservoir randomness (reproducible)

    records = []
    empties = {}
    for sae in SAE_FILES:
        print(f"\n=== collecting examples: {sae} ({len(selected[sae])} features) ===")
        model, scale = load_sae(SAE_FILES[sae])
        pk_val, pk_pos, rs_val, rs_pos, toks_all = collect(sae, model, scale, selected[sae], P_hybrid, gen)
        n_empty = 0
        for j, feat in enumerate(selected[sae]):
            peak = render(pk_pos[j].tolist(), pk_val[j].tolist(), toks_all, K_PEAK)
            typ = render(rs_pos[j].tolist(), rs_val[j].tolist(), toks_all, K_TYPICAL)
            if not peak:
                n_empty += 1
            st = stats[sae]
            records.append({
                "sae": sae, "feat": int(feat),
                "freq": float(st["freq"][feat]),
                "freq_bin": bin_label(int(st["bin"][feat])),
                "nd": None if st["nd"] is None else float(st["nd"][feat]),
                "nd_peak": None if st["nd_peak"] is None else float(st["nd_peak"][feat]),
                "peak": peak, "typical": typ,
            })
        empties[sae] = n_empty
        del model
        t.cuda.empty_cache()

    # blind + shuffle: the judge must not be able to tell which SAE a feature is from
    order = rng.permutation(len(records))
    blind, key = [], {}
    for new_idx, old_idx in enumerate(order):
        r = records[int(old_idx)]
        aid = f"F{new_idx:04d}"
        blind.append({"id": aid, "peak": r["peak"], "typical": r["typical"]})
        key[aid] = {k: r[k] for k in ("sae", "feat", "freq", "freq_bin", "nd", "nd_peak")}
        key[aid]["n_peak"] = len(r["peak"])
        key[aid]["n_typical"] = len(r["typical"])

    with open(f"{CACHE_DIR}/judge_blind.json", "w") as f:
        json.dump(blind, f, ensure_ascii=False)
    with open(f"{CACHE_DIR}/judge_key.json", "w") as f:
        json.dump(key, f, ensure_ascii=False)

    print(f"\nwrote {len(blind)} blinded features -> {CACHE_DIR}/judge_blind.json (+ judge_key.json)")
    print("features with no examples in scanned window (thin coverage): "
          + ", ".join(f"{s}={empties[s]}" for s in empties))


if __name__ == "__main__":
    main()
