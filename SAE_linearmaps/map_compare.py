"""Compare the GREEDY linear map with the JOINTLY-TRAINED (hybrid) map.

  greedy  = P.pt          -- Linear(embedding) fit to predict h on its own (minimizes ||h - P||)
  hybrid  = P_hybrid.pt   -- same map, warm-started then CO-TRAINED with the SAE

Both are (V, 2304) tables of per-token predictions of the layer-13 activation.

Reports:
  1. how much the map moved: per-token cosine(greedy, hybrid) + norm ratio, over the whole vocab
  2. R^2 of each map on h (fraction of Var(h) explained on its own) -- streamed over activations
     Hypothesis: the hybrid map explains LESS of h (it traded standalone accuracy to leave the SAE
     an easier residual). If so -> "worse predictor of h, better teammate for the SAE."
Writes map_comparison.png (cosine + norm-ratio histograms).
"""
import torch as t
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR   = "/workspace/sae_cache_layer13"
GREEDY_PATH = f"{CACHE_DIR}/P.pt"
HYBRID_PATH = f"{CACHE_DIR}/P_hybrid.pt"        # production hybrid map; or point at P_hybrid_k64.pt (sweep)
N_TOKENS    = 2_000_000
bs          = 8192
device      = t.device("cuda" if t.cuda.is_available() else "cpu")

assert os.path.exists(HYBRID_PATH), f"missing {HYBRID_PATH} — train the hybrid first (or use a P_hybrid_k*.pt)"
P_g = t.load(GREEDY_PATH, map_location=device).float()   # (V, 2304) greedy
P_h = t.load(HYBRID_PATH, map_location=device).float()   # (V, 2304) jointly-trained

# ---------- 1. how much did the map move? (over the whole vocab) ----------
cos        = t.nn.functional.cosine_similarity(P_g, P_h, dim=1)        # (V,)
norm_ratio = P_h.norm(dim=1) / P_g.norm(dim=1).clamp_min(1e-8)         # (V,)
rel_change = ((P_h - P_g).norm() / P_g.norm()).item()                 # Frobenius, whole table
print("=== how much the map moved (per token, over the vocab) ===")
print(f"cosine(greedy, hybrid): mean {cos.mean():.4f}  median {cos.median():.4f}  "
      f"p5 {cos.quantile(0.05):.4f}  p95 {cos.quantile(0.95):.4f}")
print(f"norm ratio |hybrid|/|greedy|: mean {norm_ratio.mean():.4f}  median {norm_ratio.median():.4f}")
print(f"overall relative change ||P_h - P_g|| / ||P_g|| : {rel_change:.4f}")

# ---------- 2. R^2 of each map on h (streamed; float64 accumulators) ----------
h_sum = h_sumsq = sse_g = sse_h = 0.0; n_elem = 0; seen = 0
with t.no_grad():
    for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
        tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
        hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
        for start in range(0, hc.shape[0], bs):
            hh = hc[start:start+bs].float().to(device)
            tt = tc[start:start+bs].to(device)
            h_sum   += hh.double().sum().item();          h_sumsq += (hh ** 2).double().sum().item()
            sse_g   += ((hh - P_g[tt]) ** 2).double().sum().item()
            sse_h   += ((hh - P_h[tt]) ** 2).double().sum().item()
            n_elem += hh.numel(); seen += hh.shape[0]
            if seen >= N_TOKENS: break
        if seen >= N_TOKENS: break
var_h = h_sumsq / n_elem - (h_sum / n_elem) ** 2
r2_g = 1 - sse_g / (n_elem * var_h)
r2_h = 1 - sse_h / (n_elem * var_h)
print("\n=== R^2 on h (fraction of Var(h) each map explains ON ITS OWN) ===")
print(f"  greedy map : {r2_g:.4f}   (sanity: expect ~0.55)")
print(f"  hybrid map : {r2_h:.4f}")
print(f"  -> the jointly-trained map explains {'LESS' if r2_h < r2_g else 'MORE'} of h standalone "
      f"(Δ = {r2_h - r2_g:+.4f})")

# ---------- 3. figure ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(cos.cpu().numpy(), bins=60, color="#4553c9")
ax[0].set_title("per-token cosine(greedy, hybrid)"); ax[0].set_xlabel("cosine similarity"); ax[0].set_ylabel("# tokens")
ax[1].hist(norm_ratio.clamp(max=3).cpu().numpy(), bins=60, color="#2c885f")
ax[1].axvline(1.0, color="#999", ls="--")
ax[1].set_title("norm ratio  |hybrid| / |greedy|"); ax[1].set_xlabel("ratio (clamped at 3)"); ax[1].set_ylabel("# tokens")
fig.suptitle(f"Greedy vs jointly-trained map    "
             f"(R² on h: greedy {r2_g:.3f} → hybrid {r2_h:.3f};  rel. move {rel_change:.3f})", fontsize=12)
fig.tight_layout()
out = f"{CACHE_DIR}/map_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {out}")
