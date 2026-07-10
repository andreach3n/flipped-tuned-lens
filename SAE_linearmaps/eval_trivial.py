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

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

CACHE_DIR   = "/workspace/sae_cache_layer13"
FULL_PATH   = f"{CACHE_DIR}/sae_full_final.pt"
RESID_PATH  = f"{CACHE_DIR}/sae_resid_final.pt"
P_PATH      = f"{CACHE_DIR}/P.pt"
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
                x = (hh - P[tt]) if mode == "resid" else hh
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

# --- save per-feature stats so dashboard.py can pick features without re-running the 4M eval ---
# "nd" = WEIGHTED (activation-weighted) distinct-word count [the principled metric]; "nd_peak" = old peak metric
t.save({"freq": freq_full.cpu(),  "nd": nd_full.cpu(),  "nd_peak": triviality(peak_full, freq_full)[1].cpu(),  "alive": al_full.cpu()},  f"{CACHE_DIR}/stats_full.pt")
t.save({"freq": freq_resid.cpu(), "nd": nd_resid.cpu(), "nd_peak": triviality(peak_resid, freq_resid)[1].cpu(), "alive": al_resid.cpu()}, f"{CACHE_DIR}/stats_resid.pt")
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

C_FULL, C_RESID = "#4553c9", "#b5762e"
PLOT_BINS = [((TOPK, 1000),        "rare  (20-1k)"),
             ((1000, 100000),      "mid   (1k-100k)"),
             ((100000, float("inf")), "common (>100k)")]

def _unique_and_eff(toks, freq):
    """Per feature: # unique normalized words (int) and effective # words = exp(entropy)."""
    nd = triviality(toks, freq)[1].float()   # reuse the distinct-word count
    ew = eff_words(toks)                      # exp(entropy) of the word distribution
    return nd, ew

def plot_unique_token_dists(sample_name, toks_full, toks_resid, tag):
    ndf, ewf = _unique_and_eff(toks_full,  freq_full)
    ndr, ewr = _unique_and_eff(toks_resid, freq_resid)
    ubins = np.arange(0.5, TOPK + 1.5, 1)    # integer-centered bins 1..K for the unique-count hist
    ebins = np.linspace(1, TOPK, 30)         # continuous bins for effective-#words
    ts    = np.arange(1, TOPK + 1)           # thresholds for the sweep (# unique <= t)
    nrows = len(PLOT_BINS)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.4 * nrows), squeeze=False)
    for r, ((lo, hi), lab) in enumerate(PLOT_BINS):
        mF = (al_full  & (freq_full  >= lo) & (freq_full  < hi))   # alive full  features in this band
        mR = (al_resid & (freq_resid >= lo) & (freq_resid < hi))   # alive resid features in this band
        nF, nR = int(mF.sum()), int(mR.sum())
        ndF, ndR = ndf[mF].cpu().numpy(), ndr[mR].cpu().numpy()
        ewF, ewR = ewf[mF].cpu().numpy(), ewr[mR].cpu().numpy()

        # col 0: distribution of # unique tokens
        ax = axes[r][0]
        ax.hist(ndF, bins=ubins, density=True, alpha=.55, color=C_FULL,  label=f"full  (n={nF})")
        ax.hist(ndR, bins=ubins, density=True, alpha=.55, color=C_RESID, label=f"resid (n={nR})")
        ax.set_xlabel(f"# unique words in top-{TOPK}"); ax.set_ylabel("density")
        ax.set_title(f"{lab}  ·  unique-token count"); ax.legend(fontsize=8)

        # col 1: threshold sweep -- frac of features called "single-ish" if cutoff is (# unique <= t)
        ax = axes[r][1]
        cF = [ (ndF <= t).mean() for t in ts ]
        cR = [ (ndR <= t).mean() for t in ts ]
        ax.plot(ts, cF, "-o", ms=3, color=C_FULL,  label="full")
        ax.plot(ts, cR, "-o", ms=3, color=C_RESID, label="resid")
        ax.axvline(1, color="#999", lw=.8, ls="--")     # t=1 == the pure single-word fraction
        ax.set_xlabel("threshold t  (# unique words <= t)"); ax.set_ylabel("frac of features")
        ax.set_title(f"{lab}  ·  threshold sensitivity"); ax.legend(fontsize=8)

        # col 2: distribution of effective # words (entropy-based, continuous)
        ax = axes[r][2]
        ax.hist(ewF, bins=ebins, density=True, alpha=.55, color=C_FULL,  label="full")
        ax.hist(ewR, bins=ebins, density=True, alpha=.55, color=C_RESID, label="resid")
        ax.set_xlabel("effective # words (exp entropy)"); ax.set_ylabel("density")
        ax.set_title(f"{lab}  ·  effective #words"); ax.legend(fontsize=8)

    fig.suptitle(f"Unique-token distribution in top-{TOPK}  —  {sample_name} sample "
                 f"(full vs resid, matched frequency)", y=1.002, fontsize=14)
    fig.tight_layout()
    out = f"{CACHE_DIR}/unique_tokens_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved plot -> {out}")

# PEAK = the "top-20 activating examples" reading (where the ~16% single-word number lived);
# WEIGHTED = the principled activation-weighted sample. Emit both so the contrast is visible.
plot_unique_token_dists("PEAK",     peak_full, peak_resid, "peak")
plot_unique_token_dists("WEIGHTED", wt_full,   wt_resid,   "weighted")

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
    ndF = triviality(peak_full,  freq_full)[1].float()    # # distinct words in max-20, per feature
    ndR = triviality(peak_resid, freq_resid)[1].float()
    fqF, fqR = freq_full.float(), freq_resid.float()
    fmax = float(max(fqF[al_full].max(), fqR[al_resid].max()))
    xedges = 10 ** np.arange(np.log10(TOPK), np.log10(fmax) + 0.5, 0.5)   # half-order-of-magnitude bins
    yedges = np.arange(0.5, TOPK + 1.5, 1)                                # integer word-count bins 1..K

    def col_norm_hist(nd, fq, alive):
        H, _, _ = np.histogram2d(fq[alive].cpu().numpy(), nd[alive].cpu().numpy(),
                                 bins=[xedges, yedges])            # (n_freq_bins, K) raw counts
        col = H.sum(axis=1, keepdims=True)                        # features per frequency column
        return np.divide(H, col, out=np.zeros_like(H), where=col > 0)   # P(#words | freq)

    Hf = col_norm_hist(ndF, fqF, al_full)
    Hr = col_norm_hist(ndR, fqR, al_resid)
    X, Y = np.meshgrid(xedges, yedges)
    vtop = max(Hf.max(), Hr.max())                                # shared color scale for full & resid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, H, ttl in [(axes[0], Hf, "FULL"), (axes[1], Hr, "RESID")]:
        pc = ax.pcolormesh(X, Y, H.T, cmap="magma", vmin=0, vmax=vtop)
        ax.set_xscale("log"); ax.set_title(f"{ttl}   ·   P(#words | freq)")
        ax.set_xlabel("feature firing frequency (log)"); ax.set_ylabel("# distinct words in max-20")
        fig.colorbar(pc, ax=ax, fraction=.046)
    # difference: resid - full. RdBu -> positive (resid heavier) = blue, negative (full heavier) = red.
    D = (Hr - Hf).T
    vmax = float(np.abs(D).max()) or 1e-9
    pc = axes[2].pcolormesh(X, Y, D, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[2].set_xscale("log"); axes[2].set_title("RESID − FULL   (blue = resid more, red = full more)")
    axes[2].set_xlabel("feature firing frequency (log)"); axes[2].set_ylabel("# distinct words in max-20")
    fig.colorbar(pc, ax=axes[2], fraction=.046)

    fig.suptitle("Feature frequency vs max-20 word count  (column-normalized: P(#words | freq))", fontsize=14)
    fig.tight_layout()
    out = f"{CACHE_DIR}/heatmap_freq_vs_words.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved plot -> {out}")

heatmap_freq_vs_words()

# feature inspection / dashboards moved to dashboard.py (run that to eyeball features)
