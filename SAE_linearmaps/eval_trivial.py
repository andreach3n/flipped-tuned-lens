from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput
import torch as t
import torch.nn as nn
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset
import glob
import os

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

CACHE_DIR   = "/workspace/sae_cache_layer13"
FULL_PATH   = f"{CACHE_DIR}/sae_full_final.pt"
RESID_PATH  = f"{CACHE_DIR}/sae_resid_final.pt"
HYBRID_PATH = f"{CACHE_DIR}/sae_hybrid_final.pt"
OUTBIAS_PATH = f"{CACHE_DIR}/sae_outbias_k64_final.pt"   # ablation: encoder sees full h, map at output
P_PATH      = f"{CACHE_DIR}/P.pt"
P_HYBRID_PATH = f"{CACHE_DIR}/P_hybrid.pt"
N_TOKENS = 4_000_000
bs = 8192
# EVAL_N      = 100_000     # subset of tokens for the eval (memory-bounded)
TOPK        = 20         # top activating examples per feature
TRIVIAL_THRESH = 0.8      # modal-token fraction above this = "trivial"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# commenting out for streaming instead
# h = t.cat(t.load(f"{CACHE_DIR}/layer_13_chunk_1.pt"), dim=0)[:EVAL_N]   # (100k, 2304)
# tok = t.cat(t.load(f"{CACHE_DIR}/tokens_chunk_1.pt"),   dim=0)[:EVAL_N]   # (100k,)
P = t.load(P_PATH, map_location=device)   # (V, 2304)
# only present once hybrid has been trained; stays None otherwise so full/resid eval still runs
P_hybrid = t.load(P_HYBRID_PATH, map_location=device) if os.path.exists(P_HYBRID_PATH) else None

# token_id -> normalized-word id map (needs P for vocab size)
raw = tokenizer.convert_ids_to_tokens(list(range(P.shape[0])))   # one string per token id
norm_to_id, rows = {}, []
for s in raw:
    key = s.replace("▁", "").strip().lower()   # "▁Happy" / "happy" / " happy" -> "happy"
    rows.append(norm_to_id.setdefault(key, len(norm_to_id)))
norm_map = t.tensor(rows, device=device)       # (vocab,)  token_id -> normalized-word id

def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])   # cfg was saved in the checkpoint
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]                  # <-- you MUST return the scale

sae_full,  scale_full  = load_sae(FULL_PATH)
sae_resid, scale_resid = load_sae(RESID_PATH)

# hybrid is OPTIONAL: only wire it in if both its SAE checkpoint and its trained P table exist,
# so this whole script still runs (full + resid) before hybrid has been trained.
HAVE_HYBRID = os.path.exists(HYBRID_PATH) and P_hybrid is not None
if HAVE_HYBRID:
    sae_hybrid, scale_hybrid = load_sae(HYBRID_PATH)
    print("hybrid artifacts found -> including HYBRID in the eval")

# outbias ablation: encoder saw the FULL h (like full mode), map added only at OUTPUT. For
# triviality the map is irrelevant (features are detected from h), so its stream uses x = hh.
HAVE_OUTBIAS = os.path.exists(OUTBIAS_PATH)
if HAVE_OUTBIAS:
    sae_outbias, scale_outbias = load_sae(OUTBIAS_PATH)
    print("outbias artifacts found -> including OUTBIAS in the eval")

# --- OLD: materialized the whole (N, 16384) matrix, OOMs past ~100k tokens ---
# def feature_acts(sae, scale, mode, bs=8192):
#     outs = []
#     with t.no_grad():
#         for start in range(0, h.shape[0], bs):         # encode in batches — BatchTopK makes
#             hh = h[start:start+bs].float().to(device)  # several full-size copies internally
#             tt = tok[start:start+bs].to(device)
#             x = (hh - P[tt]) if mode == "resid" else hh
#             a = sae.encode(x / scale)                  # (bs, 16384)
#             outs.append(a.cpu())                       # accumulate on CPU to free GPU
#     return t.cat(outs, dim=0)                          # (N, 16384) on CPU
#
# a_full  = feature_acts(sae_full,  scale_full,  "full")
# a_resid = feature_acts(sae_resid, scale_resid, "resid")

# --- stream, keeping per feature TWO samples of its firings (single pass) ---
#   PEAK: top-K by activation value -> biased to the strongest firings (what we had before)
#   RANGE: top-K by a RANDOM key among firings -> a uniform sample across the whole activation
#          range (reservoir sampling). This is the peak-bias fix: triviality is judged on the
#          feature's typical firings, not just its peak.
def stream_topk(sae, scale, mode, n_tokens, K=TOPK, bs=bs):
    F = sae.cfg.d_sae
    peak_vals = t.full((K, F), -1e9, device=device)          # PEAK: top-K by activation value
    peak_toks = t.zeros((K, F), dtype=t.long, device=device)
    uni_keys  = t.full((K, F), -1e9, device=device)          # UNIFORM reservoir: uniform random keys
    uni_toks  = t.zeros((K, F), dtype=t.long, device=device)
    wt_keys   = t.full((K, F), -1e9, device=device)          # WEIGHTED reservoir: keys ~ activation
    wt_toks   = t.zeros((K, F), dtype=t.long, device=device)
    freq = t.zeros(F, dtype=t.long, device=device)
    seen = 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)
                tt = tc[start:start+bs].to(device)
                if   mode == "resid":  x = hh - P[tt]          # encoder sees the residual (frozen map)
                elif mode == "hybrid": x = hh - P_hybrid[tt]   # encoder sees the residual (trained map)
                else:                  x = hh                  # full AND outbias: encoder sees the full h

                a = sae.encode(x / scale)                     # (b, F)
                fired = a > 0
                freq += fired.sum(dim=0)
                btok = tt.unsqueeze(1).expand(-1, F)          # (b, F): token id per position
                # PEAK: top-K by activation value
                peak_vals, s0 = t.cat([peak_vals, a], dim=0).topk(K, dim=0)
                peak_toks = t.cat([peak_toks, btok], dim=0).gather(0, s0)
                # UNIFORM reservoir: top-K by a uniform random key (each firing equally likely)
                uk = t.rand_like(a); uk[~fired] = -1e9
                uni_keys, s1 = t.cat([uni_keys, uk], dim=0).topk(K, dim=0)
                uni_toks = t.cat([uni_toks, btok], dim=0).gather(0, s1)
                # WEIGHTED reservoir: key = u^(1/a) (A-Res) -> firings sampled in proportion to activation
                wk = (t.rand_like(a).clamp_min(1e-12).log() / a.clamp_min(1e-6)).exp()
                wk[~fired] = -1e9
                wt_keys, s2 = t.cat([wt_keys, wk], dim=0).topk(K, dim=0)
                wt_toks = t.cat([wt_toks, btok], dim=0).gather(0, s2)
                seen += hh.shape[0]
                if seen >= n_tokens:
                    return peak_toks, uni_toks, wt_toks, freq
    return peak_toks, uni_toks, wt_toks, freq

peak_full,  uni_full,  wt_full,  freq_full  = stream_topk(sae_full,  scale_full,  "full",  N_TOKENS)
peak_resid, uni_resid, wt_resid, freq_resid = stream_topk(sae_resid, scale_resid, "resid", N_TOKENS)
if HAVE_HYBRID:
    peak_hyb, uni_hyb, wt_hyb, freq_hyb = stream_topk(sae_hybrid, scale_hybrid, "hybrid", N_TOKENS)
if HAVE_OUTBIAS:
    peak_ob, uni_ob, wt_ob, freq_ob = stream_topk(sae_outbias, scale_outbias, "outbias", N_TOKENS)

# --- OLD: took the full (N, F) matrix and did its own topk ---
# def triviality(a, tok):
#     vals, idx = a.topk(TOPK, dim=0)
#     top_tokens = tok.to(idx.device)[idx]
#     top_words  = norm_map.to(idx.device)[top_tokens]
#     modal_word, _ = t.mode(top_words, dim=0)
#     modal_frac = (top_words == modal_word).float().mean(dim=0)
#     sorted_words, _ = top_words.sort(dim=0)
#     n_distinct = 1 + (sorted_words[1:] != sorted_words[:-1]).sum(dim=0)
#     alive = (a > 0).sum(dim=0) >= TOPK
#     return modal_frac, n_distinct, alive

# --- NEW: run_toks IS already the top-K token ids per feature (streamed) ---
def triviality(run_toks, freq):
    top_words = norm_map[run_toks]                          # (K, F): token ids -> normalized-word ids
    # metric 1: modal-word fraction
    modal_word, _ = t.mode(top_words, dim=0)               # most common word per feature
    modal_frac = (top_words == modal_word).float().mean(dim=0)   # fraction of the K that are it
    # metric 2: number of distinct normalized words
    sorted_words, _ = top_words.sort(dim=0)
    n_distinct = 1 + (sorted_words[1:] != sorted_words[:-1]).sum(dim=0)
    alive = freq >= TOPK                                   # fired >= K times, else run_toks is junk-padded
    return modal_frac, n_distinct, alive

def eff_words(toks):
    """Effective # distinct normalized words per feature = exp(entropy of its word distribution)."""
    words = norm_map[toks]                                    # (K, F)
    sw, _ = words.sort(dim=0)
    is_new = t.ones_like(sw, dtype=t.bool); is_new[1:] = sw[1:] != sw[:-1]
    run_id = is_new.cumsum(0) - 1                             # which run (distinct value) each element belongs to
    counts = t.zeros_like(sw, dtype=t.float).scatter_add_(0, run_id, t.ones_like(sw, dtype=t.float))
    p = counts / counts.sum(0, keepdim=True)                 # p(word) over the K samples
    H = -(p * p.clamp_min(1e-12).log()).sum(0)
    return H.exp()                                           # (F,) effective # words (1 = single word)

def report(name, peak_toks, uni_toks, wt_toks, freq):
    print(f"\n=== {name} ===")
    for label, toks in [("PEAK     (strongest firings only)", peak_toks),
                        ("UNIFORM  (equal-weight sample)   ", uni_toks),
                        ("WEIGHTED (activation-weighted)   ", wt_toks)]:
        mf, nd, alive = triviality(toks, freq)
        mfa, nda, ewa = mf[alive], nd[alive].float(), eff_words(toks)[alive]
        print(f"  {label}: modal {mfa.mean():.3f}  distinct {nda.mean():.2f}  "
              f"eff_words {ewa.mean():.2f}  single-word {(nda == 1).float().mean():.3f}")
    return triviality(wt_toks, freq)   # WEIGHTED = principled middle, used downstream

mf_full,  nd_full,  al_full  = report("FULL",  peak_full,  uni_full,  wt_full,  freq_full)
mf_resid, nd_resid, al_resid = report("RESID", peak_resid, uni_resid, wt_resid, freq_resid)
if HAVE_HYBRID:
    mf_hyb, nd_hyb, al_hyb = report("HYBRID", peak_hyb, uni_hyb, wt_hyb, freq_hyb)
if HAVE_OUTBIAS:
    mf_ob, nd_ob, al_ob = report("OUTBIAS", peak_ob, uni_ob, wt_ob, freq_ob)

# side-by-side shift in the distribution (over each SAE's own alive features)
print("\n=== shift (resid - full) ===")
print(f"mean modal_frac:   full {mf_full[al_full].mean().item():.4f}  "
      f"resid {mf_resid[al_resid].mean().item():.4f}")
print(f"median modal_frac: full {mf_full[al_full].median().item():.4f}  "
      f"resid {mf_resid[al_resid].median().item():.4f}")

# --- frequency-binned comparison: controls for the firing-rate confound ---
# For each SAE, group its ALIVE features by how often they fired (freq), then compare
# triviality WITHIN each band. If resid is still more trivial at matched frequency,
# the effect is real; if it only shows across bins, it was a frequency artifact.
def freq_binned(mf, nd, freq, alive, edges):
    nd = nd.float()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = alive & (freq >= lo) & (freq < hi)         # alive features whose firing count lands in [lo, hi)
        n = int(m.sum())
        if n == 0:
            out.append((0, float("nan"), float("nan"), float("nan")))
        else:
            out.append((n,
                        mf[m].mean().item(),           # mean modal-word fraction in this band
                        nd[m].mean().item(),           # mean # distinct words in this band
                        (nd[m] == 1).float().mean().item()))  # frac pure single-word in this band
    return out

edges = [TOPK, 100, 1000, 10000, 100000, float("inf")]
full_bins  = freq_binned(mf_full,  nd_full,  freq_full,  al_full,  edges)
resid_bins = freq_binned(mf_resid, nd_resid, freq_resid, al_resid, edges)

print("\n=== triviality by firing-frequency bin ===")
print(f"{'freq range':>14} | {'n':>6} {'mf':>5} {'ndist':>5} {'1word':>5} (full) | "
      f"{'n':>6} {'mf':>5} {'ndist':>5} {'1word':>5} (resid)")
for (lo, hi), fb, rb in zip(zip(edges[:-1], edges[1:]), full_bins, resid_bins):
    hi_s = "inf" if hi == float("inf") else str(int(hi))
    rng = f"{int(lo)}-{hi_s}"
    print(f"{rng:>14} | {fb[0]:>6} {fb[1]:>5.3f} {fb[2]:>5.2f} {fb[3]:>5.3f}        | "
          f"{rb[0]:>6} {rb[1]:>5.3f} {rb[2]:>5.2f} {rb[3]:>5.3f}")

if HAVE_HYBRID:
    hyb_bins = freq_binned(mf_hyb, nd_hyb, freq_hyb, al_hyb, edges)
    print("\n=== HYBRID triviality by firing-frequency bin ===")
    print(f"{'freq range':>14} | {'n':>6} {'mf':>5} {'ndist':>5} {'1word':>5}")
    for (lo, hi), hb in zip(zip(edges[:-1], edges[1:]), hyb_bins):
        hi_s = "inf" if hi == float("inf") else str(int(hi))
        rng = f"{int(lo)}-{hi_s}"
        print(f"{rng:>14} | {hb[0]:>6} {hb[1]:>5.3f} {hb[2]:>5.2f} {hb[3]:>5.3f}")

if HAVE_OUTBIAS:
    ob_bins = freq_binned(mf_ob, nd_ob, freq_ob, al_ob, edges)
    print("\n=== OUTBIAS triviality by firing-frequency bin ===")
    print(f"{'freq range':>14} | {'n':>6} {'mf':>5} {'ndist':>5} {'1word':>5}")
    for (lo, hi), ob in zip(zip(edges[:-1], edges[1:]), ob_bins):
        hi_s = "inf" if hi == float("inf") else str(int(hi))
        rng = f"{int(lo)}-{hi_s}"
        print(f"{rng:>14} | {ob[0]:>6} {ob[1]:>5.3f} {ob[2]:>5.2f} {ob[3]:>5.3f}")

# --- save per-feature stats so dashboard.py can pick features without re-running the 4M eval ---
# "nd" = WEIGHTED (activation-weighted) distinct-word count [the principled metric]; "nd_peak" = old peak metric
# "eff"/"eff_peak" = effective #words = exp(entropy) on the WEIGHTED / PEAK sample [breadth metric for the LLM-judge analysis]
t.save({"freq": freq_full.cpu(),  "nd": nd_full.cpu(),  "nd_peak": triviality(peak_full, freq_full)[1].cpu(),
        "eff": eff_words(wt_full).cpu(),  "eff_peak": eff_words(peak_full).cpu(),  "alive": al_full.cpu()},  f"{CACHE_DIR}/stats_full.pt")
t.save({"freq": freq_resid.cpu(), "nd": nd_resid.cpu(), "nd_peak": triviality(peak_resid, freq_resid)[1].cpu(),
        "eff": eff_words(wt_resid).cpu(), "eff_peak": eff_words(peak_resid).cpu(), "alive": al_resid.cpu()}, f"{CACHE_DIR}/stats_resid.pt")
if HAVE_HYBRID:
    t.save({"freq": freq_hyb.cpu(), "nd": nd_hyb.cpu(), "nd_peak": triviality(peak_hyb, freq_hyb)[1].cpu(),
            "eff": eff_words(wt_hyb).cpu(), "eff_peak": eff_words(peak_hyb).cpu(), "alive": al_hyb.cpu()}, f"{CACHE_DIR}/stats_hybrid.pt")
if HAVE_OUTBIAS:
    t.save({"freq": freq_ob.cpu(), "nd": nd_ob.cpu(), "nd_peak": triviality(peak_ob, freq_ob)[1].cpu(),
            "eff": eff_words(wt_ob).cpu(), "eff_peak": eff_words(peak_ob).cpu(), "alive": al_ob.cpu()}, f"{CACHE_DIR}/stats_outbias.pt")
print(f"\nsaved per-feature stats to {CACHE_DIR}/stats_*.pt")

# --- distribution plot: is "single-word fraction" just a thresholding artifact? -------------
# The headline metric collapses each feature's top-K to a BINARY (n_distinct == 1 -> "single word").
# That hides the shape. Here we plot the FULL distribution of unique-token counts so we can see
# whether "single-token" is a real spike at 1 or just the left tail of a smooth ramp (in which
# case the single-word fraction moves a lot as you nudge the cutoff). Three panels per freq bin:
#   col 0: histogram of  # unique normalized words in the top-K        (integer, 1..K)
#   col 1: fraction of features with (# unique <= t) as t sweeps 1..K  -- the threshold sweep
#   col 2: histogram of  effective # words = exp(entropy)              (continuous, 1..K)
# Rows = firing-frequency bins, so full vs resid are compared at MATCHED frequency (the confound
# your mentor liked controlling for). Overlaid: full (blue) vs resid (orange).
import matplotlib
matplotlib.use("Agg")                       # headless box -> render to file, no display
import matplotlib.pyplot as plt
import numpy as np

C_FULL, C_RESID, C_HYBRID, C_OUTBIAS = "#4553c9", "#b5762e", "#2c885f", "#c0392b"
PLOT_BINS = [((TOPK, 1000),        "rare  (20-1k)"),
             ((1000, 100000),      "mid   (1k-100k)"),
             ((100000, float("inf")), "common (>100k)")]

def plot_unique_token_dists(sample_name, series, tag):
    # series: list of (label, color, toks, alive, freq) -- 2 (full, resid) or 3 (+ hybrid)
    ubins = np.arange(0.5, TOPK + 1.5, 1)    # integer-centered bins 1..K for the unique-count hist
    ebins = np.linspace(1, TOPK, 30)         # continuous bins for effective-#words
    ts    = np.arange(1, TOPK + 1)           # thresholds for the sweep (# unique <= t)
    # precompute per-series distinct-word count + effective-#words once
    prepared = [(label, color, triviality(toks, freq)[1].float(), eff_words(toks), alive, freq)
                for (label, color, toks, alive, freq) in series]
    nrows = len(PLOT_BINS)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.4 * nrows), squeeze=False)
    for r, ((lo, hi), lab) in enumerate(PLOT_BINS):
        # col 0: distribution of # unique tokens
        ax = axes[r][0]
        for label, color, nd, ew, alive, freq in prepared:
            m = alive & (freq >= lo) & (freq < hi)
            ax.hist(nd[m].cpu().numpy(), bins=ubins, density=True, alpha=.5, color=color,
                    label=f"{label} (n={int(m.sum())})")
        ax.set_xlabel(f"# unique words in top-{TOPK}"); ax.set_ylabel("density")
        ax.set_title(f"{lab}  ·  unique-token count"); ax.legend(fontsize=8)

        # col 1: threshold sweep -- frac of features called "single-ish" if cutoff is (# unique <= t)
        ax = axes[r][1]
        for label, color, nd, ew, alive, freq in prepared:
            m = alive & (freq >= lo) & (freq < hi)
            ndv = nd[m].cpu().numpy()
            ax.plot(ts, [ (ndv <= tt).mean() for tt in ts ], "-o", ms=3, color=color, label=label)
        ax.axvline(1, color="#999", lw=.8, ls="--")     # t=1 == the pure single-word fraction
        ax.set_xlabel("threshold t  (# unique words <= t)"); ax.set_ylabel("frac of features")
        ax.set_title(f"{lab}  ·  threshold sensitivity"); ax.legend(fontsize=8)

        # col 2: distribution of effective # words (entropy-based, continuous)
        ax = axes[r][2]
        for label, color, nd, ew, alive, freq in prepared:
            m = alive & (freq >= lo) & (freq < hi)
            ax.hist(ew[m].cpu().numpy(), bins=ebins, density=True, alpha=.5, color=color, label=label)
        ax.set_xlabel("effective # words (exp entropy)"); ax.set_ylabel("density")
        ax.set_title(f"{lab}  ·  effective #words"); ax.legend(fontsize=8)

    labels = " vs ".join(s[0] for s in series)
    fig.suptitle(f"Unique-token distribution in top-{TOPK}  —  {sample_name} sample "
                 f"({labels}, matched frequency)", y=1.002, fontsize=14)
    fig.tight_layout()
    out = f"{CACHE_DIR}/unique_tokens_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved plot -> {out}")

# PEAK = the "top-20 activating examples" reading (where the ~16% single-word number lived);
# WEIGHTED = the principled activation-weighted sample. Emit both so the contrast is visible.
peak_series = [("full", C_FULL, peak_full, al_full, freq_full), ("resid", C_RESID, peak_resid, al_resid, freq_resid)]
wt_series   = [("full", C_FULL, wt_full,   al_full, freq_full), ("resid", C_RESID, wt_resid,   al_resid, freq_resid)]
if HAVE_HYBRID:
    peak_series.append(("hybrid", C_HYBRID, peak_hyb, al_hyb, freq_hyb))
    wt_series.append(  ("hybrid", C_HYBRID, wt_hyb,   al_hyb, freq_hyb))
if HAVE_OUTBIAS:
    peak_series.append(("outbias", C_OUTBIAS, peak_ob, al_ob, freq_ob))
    wt_series.append(  ("outbias", C_OUTBIAS, wt_ob,   al_ob, freq_ob))
plot_unique_token_dists("PEAK",     peak_series, "peak")
plot_unique_token_dists("WEIGHTED", wt_series,   "weighted")

# --- 2D heatmap: feature frequency (x, log) vs # distinct words in max-20 (y) ----------------
# The mentor's preferred view: the MAX (peak) metric -- "# different words in the max 20" -- which
# is what interp dashboards show in practice, plotted against firing frequency with NO arbitrary
# buckets. Columns are normalized so each frequency slice sums to 1, i.e. we show P(#words | freq).
# That does two things: (1) exposes the frequency->complexity ridge directly (word-count should
# climb with frequency -- the confound, made visible), and (2) stops the ~80% mid-frequency bucket
# from washing everything out. Third panel = resid - full, to see if residualization moves any mass.
# NOTE: high-frequency columns hold very few features (~150 total in the common bucket), so the
# right edge of each heatmap is sparse/noisy -- read the left and middle, not the far right.
def heatmap_freq_vs_words():
    # one P(#words|freq) panel per available SAE, plus a difference panel vs FULL
    saes = [("FULL", peak_full, freq_full.float(), al_full),
            ("RESID", peak_resid, freq_resid.float(), al_resid)]
    if HAVE_HYBRID:
        saes.append(("HYBRID", peak_hyb, freq_hyb.float(), al_hyb))
    fmax = float(max(fq[al].max() for _, _, fq, al in saes))
    xedges = 10 ** np.arange(np.log10(TOPK), np.log10(fmax) + 0.5, 0.5)   # half-order-of-magnitude bins
    yedges = np.arange(0.5, TOPK + 1.5, 1)                                # integer word-count bins 1..K

    def col_norm_hist(nd, fq, alive):
        H, _, _ = np.histogram2d(fq[alive].cpu().numpy(), nd[alive].cpu().numpy(),
                                 bins=[xedges, yedges])            # (n_freq_bins, K) raw counts
        col = H.sum(axis=1, keepdims=True)                        # features per frequency column
        return np.divide(H, col, out=np.zeros_like(H), where=col > 0)   # P(#words | freq)

    Hs = {name: col_norm_hist(triviality(peak, fq)[1].float(), fq, alive) for name, peak, fq, alive in saes}
    X, Y = np.meshgrid(xedges, yedges)
    vtop = max(H.max() for H in Hs.values())                      # shared color scale across SAEs
    diff_name = "HYBRID" if HAVE_HYBRID else "RESID"              # difference panel compares this vs FULL

    n_panels = len(saes) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.2))
    for i, (name, _, _, _) in enumerate(saes):
        pc = axes[i].pcolormesh(X, Y, Hs[name].T, cmap="magma", vmin=0, vmax=vtop)
        axes[i].set_xscale("log"); axes[i].set_title(f"{name}   ·   P(#words | freq)")
        axes[i].set_xlabel("feature firing frequency (log)"); axes[i].set_ylabel("# distinct words in max-20")
        fig.colorbar(pc, ax=axes[i], fraction=.046)
    # difference: diff_name - full. RdBu -> positive (diff heavier) = blue, negative (full heavier) = red.
    D = (Hs[diff_name] - Hs["FULL"]).T
    vmax = float(np.abs(D).max()) or 1e-9
    pc = axes[-1].pcolormesh(X, Y, D, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[-1].set_xscale("log"); axes[-1].set_title(f"{diff_name} − FULL   (blue = {diff_name.lower()} more, red = full more)")
    axes[-1].set_xlabel("feature firing frequency (log)"); axes[-1].set_ylabel("# distinct words in max-20")
    fig.colorbar(pc, ax=axes[-1], fraction=.046)

    fig.suptitle("Feature frequency vs max-20 word count  (column-normalized: P(#words | freq))", fontsize=14)
    fig.tight_layout()
    out = f"{CACHE_DIR}/heatmap_freq_vs_words.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved plot -> {out}")

heatmap_freq_vs_words()

# feature inspection / dashboards moved to dashboard.py (run that to eyeball features)
