"""Neuronpedia-style feature dashboards for the trained SAEs.

For each chosen feature it prints:
  - firing rate + activation histogram (from a small streamed sample)
  - logit effects (which tokens the feature's decoder direction promotes/suppresses)
  - top activating examples with surrounding context

Run this instead of re-running the full 4M eval; feature selection reads the
per-feature stats saved by eval_trivial.py (stats_*.pt).
"""
import glob
import torch as t
from transformers import AutoTokenizer
from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

CACHE_DIR   = "/workspace/sae_cache_layer13"
FULL_PATH   = f"{CACHE_DIR}/sae_full_final.pt"
RESID_PATH  = f"{CACHE_DIR}/sae_resid_final.pt"
P_PATH      = f"{CACHE_DIR}/P.pt"
MODEL_NAME  = "google/gemma-2-2b"
N_TOKENS    = 1_000_000     # tokens to scan for top activations (small -> fast)
TOPK        = 20          # top activating examples to show
WINDOW      = 8           # context tokens on each side
SHOW_LOGITS = True        # load the model's unembedding for logit effects (~5 GB, ~30 s)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
P = t.load(P_PATH, map_location=device)   # (V, 2304) per-token linear prediction

def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]

sae_full,  scale_full  = load_sae(FULL_PATH)
sae_resid, scale_resid = load_sae(RESID_PATH)

# Unembedding for logit effects (optional). Direct logit attribution: W_dec[f] @ W_U.
W_U = None
if SHOW_LOGITS:
    from transformer_lens import HookedTransformer
    _m = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
    W_U = _m.W_U.float()          # (d_model, d_vocab)
    del _m
    t.cuda.empty_cache()

def hist_text(x, bins=10, width=40):
    """Crude ASCII histogram of a 1-D tensor."""
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        return
    counts = t.histc(x.float(), bins=bins, min=lo, max=hi)
    mx = counts.max().item()
    for b in range(bins):
        edge = lo + (hi - lo) * b / bins
        bar = "#" * (int(width * counts[b].item() / mx) if mx else 0)
        print(f"    {edge:7.2f} | {bar} {int(counts[b].item())}")

def dashboard(sae, scale, mode, feat_ids, n_tokens=N_TOKENS, bs=8192):
    fids = t.tensor(feat_ids, device=device)
    # --- collect the chosen features' activations + token ids over a small sample ---
    acts_parts, toks_parts, seen = [], [], 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)
                tt = tc[start:start+bs].to(device)
                x = (hh - P[tt]) if mode == "resid" else hh
                a = sae.encode(x / scale)[:, fids]       # only the chosen features
                acts_parts.append(a.cpu()); toks_parts.append(tt.cpu())
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    acts = t.cat(acts_parts, dim=0)   # (N, n_feats)
    toks = t.cat(toks_parts, dim=0)   # (N,)

    # --- one dashboard per feature ---
    for j, f in enumerate(feat_ids):
        col = acts[:, j]
        nz = col[col > 0]
        print(f"\n{'=' * 72}\n### {mode} feature {f} ###")
        print(f"firing rate: {nz.numel()}/{col.numel()} ({100 * nz.numel() / max(1, col.numel()):.3f}%)")
        if nz.numel() > 0:
            print(f"activation: min {nz.min():.2f} | median {nz.median():.2f} | "
                  f"p90 {nz.quantile(0.9):.2f} | max {nz.max():.2f}")
            print("  histogram (nonzero activations):")
            hist_text(nz)
        if W_U is not None:
            eff = sae.W_dec[f].float() @ W_U             # (vocab,) logit push per token
            pos = eff.topk(10).indices.tolist()
            neg = (-eff).topk(10).indices.tolist()
            print("  promotes: ", [tokenizer.decode([i]) for i in pos])
            print("  suppresses:", [tokenizer.decode([i]) for i in neg])
        vals, idx = col.topk(min(TOPK, col.numel()))
        print("  top activating examples:")
        for p, v in zip(idx.tolist(), vals.tolist()):
            lo, hi = max(0, p - WINDOW), p + WINDOW + 1
            ctx = tokenizer.decode(toks[lo:hi].tolist())
            focus = tokenizer.decode([toks[p].item()])
            print(f"    [{v:6.1f}] ...{ctx}...  <<{focus!r}>>")

# ------------------------------------------------------------------
# pick which features to view
SAE, SCALE, MODE, STATS = sae_resid, scale_resid, "resid", f"{CACHE_DIR}/stats_resid.pt"

FEAT_IDS = [3]   # fallback: hardcode ids you're curious about
try:
    s = t.load(STATS)   # saved by eval_trivial.py
    # single-word features in the anomalous 100-1000 firing band:
    band = s["alive"] & (s["freq"] >= 100) & (s["freq"] < 1000) & (s["nd"] == 1)
    FEAT_IDS = band.nonzero().squeeze(1)[:8].tolist()
    print(f"selected {len(FEAT_IDS)} single-word {MODE} features (100-1000 band)")
except FileNotFoundError:
    print(f"no {STATS}; using hardcoded FEAT_IDS={FEAT_IDS}")

dashboard(SAE, SCALE, MODE, FEAT_IDS)
