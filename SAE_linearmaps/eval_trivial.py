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

# --- NEW: stream over millions of tokens, keeping only a running top-K per feature ---
def stream_topk(sae, scale, mode, n_tokens, K=TOPK, bs=bs):
    F = sae.cfg.d_sae                                        # number of SAE features (16384)
    # Running state — sized by FEATURES, never by tokens, so memory stays flat:
    run_vals = t.full((K, F), -1e9, device=device)          # best-K activation values per feature; init very low so any real value wins
    run_toks = t.zeros((K, F), dtype=t.long, device=device) # the token id behind each of those K values
    freq = t.zeros(F, dtype=t.long, device=device)          # how many times each feature has fired (for alive + freq bins)
    seen = 0                                                 # tokens processed so far
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")  # matching token-id chunk file
            hc = t.cat(t.load(lf), dim=0)                   # one chunk of activations, on CPU (bf16)
            tc = t.cat(t.load(tf), dim=0)                   # its token ids, on CPU
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)  # batch of activations -> GPU, float32
                tt = tc[start:start+bs].to(device)          # batch of token ids   -> GPU
                x = (hh - P[tt]) if mode == "resid" else hh # residualize (or not)
                a = sae.encode(x / scale)                   # (b, F) feature activations
                freq += (a > 0).sum(dim=0)                  # tally firings per feature this batch
                # ---- merge this batch into the running top-K ----
                cat_vals = t.cat([run_vals, a], dim=0)                 # (K+b, F): old best stacked on new values
                batch_toks = tt.unsqueeze(1).expand(-1, F)             # (b, F): same token id across every feature
                cat_toks = t.cat([run_toks, batch_toks], dim=0)        # (K+b, F): token ids aligned with cat_vals
                run_vals, sel = cat_vals.topk(K, dim=0)                # keep top-K values; sel = which rows won (K, F)
                run_toks = cat_toks.gather(0, sel)                     # carry the matching token ids along
                seen += hh.shape[0]
                if seen >= n_tokens:
                    return run_toks, freq
    return run_toks, freq

run_toks_full,  freq_full  = stream_topk(sae_full,  scale_full,  "full",  N_TOKENS)
run_toks_resid, freq_resid = stream_topk(sae_resid, scale_resid, "resid", N_TOKENS)

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

def report(name, run_toks, freq):
    modal_frac, n_distinct, alive = triviality(run_toks, freq)
    mf = modal_frac[alive]
    nd = n_distinct[alive].float()                  # distribution over ALIVE features only
    print(f"\n=== {name} ===")
    print(f"alive features: {int(alive.sum())} / {alive.numel()}")
    print(f"modal_frac  mean {mf.mean().item():.4f} | median {mf.median().item():.4f}")
    print(f"n_distinct  mean {nd.mean():.4f} | median {nd.median():.4f}  (of {TOPK}; lower=more trivial)")
    print(f"frac single-word (n_distinct==1): {(nd == 1).float().mean():.4f}")
    # threshold sweep: does a gap appear at any cutoff?
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"  frac trivial @ {thr:.1f}: {(mf > thr).float().mean().item():.4f}")
    return modal_frac, n_distinct, alive

mf_full,  nd_full,  al_full  = report("FULL",  run_toks_full,  freq_full)
mf_resid, nd_resid, al_resid = report("RESID", run_toks_resid, freq_resid)

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

# --- eyeball features: print top activating examples with context (Neuronpedia-style) ---
# Re-streams a small slice, keeping only the chosen features' activations + token positions,
# then prints each feature's top-K firings with a surrounding window of tokens.
def inspect(sae, scale, mode, feat_ids, n_tokens=200_000, window=8, top=20, bs=8192):
    feat_ids = t.tensor(feat_ids, device=device)
    acts_parts, toks_parts, seen = [], [], 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)
                tt = tc[start:start+bs].to(device)
                x = (hh - P[tt]) if mode == "resid" else hh
                a = sae.encode(x / scale)[:, feat_ids]      # only the chosen features
                acts_parts.append(a.cpu()); toks_parts.append(tt.cpu())
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    acts = t.cat(acts_parts, dim=0)                          # (N, n_feats)
    toks = t.cat(toks_parts, dim=0)                          # (N,)
    for j, f in enumerate(feat_ids.tolist()):
        vals, idx = acts[:, j].topk(top)
        print(f"\n### feature {f} ###")
        for p, v in zip(idx.tolist(), vals.tolist()):
            lo, hi = max(0, p - window), p + window + 1
            ctx   = tokenizer.decode(toks[lo:hi].tolist())
            focus = tokenizer.decode([toks[p].item()])
            print(f"  [{v:6.1f}] ...{ctx}...   <<{focus!r}>>")

# pick single-word features in the anomalous 100-1000 band of the RESID SAE
band = al_resid & (freq_resid >= 100) & (freq_resid < 1000) & (nd_resid == 1)
feat_ids = band.nonzero().squeeze(1)[:10].tolist()          # first 10 such features
print(f"\n=== inspecting {len(feat_ids)} single-word resid features (100-1000 band) ===")
inspect(sae_resid, scale_resid, "resid", feat_ids)
