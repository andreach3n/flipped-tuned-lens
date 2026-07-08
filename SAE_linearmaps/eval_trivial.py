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
    peak_vals = t.full((K, F), -1e9, device=device)          # PEAK: best-K activation values
    peak_toks = t.zeros((K, F), dtype=t.long, device=device)
    res_keys  = t.full((K, F), -1e9, device=device)          # RANGE: random keys for the reservoir
    res_toks  = t.zeros((K, F), dtype=t.long, device=device)
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
                btok = tt.unsqueeze(1).expand(-1, F)          # (b, F): token id per position, broadcast across features
                # PEAK: keep top-K by activation value
                peak_vals, sel = t.cat([peak_vals, a], dim=0).topk(K, dim=0)
                peak_toks = t.cat([peak_toks, btok], dim=0).gather(0, sel)
                # RANGE: keep top-K by random key, but only among positions that actually fired
                keys = t.rand_like(a); keys[~fired] = -1e9
                res_keys, sel2 = t.cat([res_keys, keys], dim=0).topk(K, dim=0)
                res_toks = t.cat([res_toks, btok], dim=0).gather(0, sel2)
                seen += hh.shape[0]
                if seen >= n_tokens:
                    return peak_toks, res_toks, freq
    return peak_toks, res_toks, freq

peak_full,  res_full,  freq_full  = stream_topk(sae_full,  scale_full,  "full",  N_TOKENS)
peak_resid, res_resid, freq_resid = stream_topk(sae_resid, scale_resid, "resid", N_TOKENS)

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

def report(name, peak_toks, res_toks, freq):
    print(f"\n=== {name} ===")
    for label, toks in [("PEAK  (top-K by activation)                 ", peak_toks),
                        ("RANGE (reservoir: uniform sample of firings)", res_toks)]:
        mf, nd, alive = triviality(toks, freq)
        mfa, nda = mf[alive], nd[alive].float()
        print(f"  {label}: alive {int(alive.sum())}  "
              f"modal {mfa.mean():.3f}  distinct {nda.mean():.2f}  single-word {(nda == 1).float().mean():.3f}")
    return triviality(res_toks, freq)   # RANGE-aware (peak-bias-corrected) metric used downstream

mf_full,  nd_full,  al_full  = report("FULL",  peak_full,  res_full,  freq_full)
mf_resid, nd_resid, al_resid = report("RESID", peak_resid, res_resid, freq_resid)

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
# "nd" = RANGE-aware (reservoir) distinct-word count [the corrected metric]; "nd_peak" = old peak metric
t.save({"freq": freq_full.cpu(),  "nd": nd_full.cpu(),  "nd_peak": triviality(peak_full, freq_full)[1].cpu(),  "alive": al_full.cpu()},  f"{CACHE_DIR}/stats_full.pt")
t.save({"freq": freq_resid.cpu(), "nd": nd_resid.cpu(), "nd_peak": triviality(peak_resid, freq_resid)[1].cpu(), "alive": al_resid.cpu()}, f"{CACHE_DIR}/stats_resid.pt")
print(f"\nsaved per-feature stats to {CACHE_DIR}/stats_*.pt")

# feature inspection / dashboards moved to dashboard.py (run that to eyeball features)
