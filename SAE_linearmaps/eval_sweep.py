"""Reconstruction-vs-sparsity sweep: for each (mode, k) SAE, measure FVU-on-h and the achieved L0.

Assumes the fleet was trained with train_sae_res.py at each k, producing k-tagged files:
  sae_full_k{K}_final.pt / sae_hybrid_k{K}_final.pt  (+ P_hybrid_k{K}.pt for hybrid)
Missing files are skipped, so you can run it before the whole fleet is done.

Output: sweep_results.pt = list of {mode, k, L0, FVU}. Plot with plot_sweep.py.
"""
import torch as t
import glob
import os
from sae_lens import BatchTopKTrainingSAE

CACHE_DIR = "/workspace/sae_cache_layer13"
N_TOKENS  = 2_000_000      # FVU converges fast; 2M is plenty
bs        = 8192
D_IN      = 2304
device    = t.device("cuda" if t.cuda.is_available() else "cpu")

K_LIST = [16, 32, 64, 128, 256]
MODES  = ["full", "hybrid"]

def stream():
    """Yield (h, tok) batches over the first N_TOKENS activations."""
    seen = 0
    for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
        tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
        hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
        for start in range(0, hc.shape[0], bs):
            yield hc[start:start+bs].float().to(device), tc[start:start+bs].to(device)
            seen += min(bs, hc.shape[0] - start)
            if seen >= N_TOKENS:
                return

# --- Var(h) once (float64 accumulators: summing billions of squares drifts in float32) ---
h_sum = h_sumsq = 0.0; n_elem = 0
with t.no_grad():
    for hh, _ in stream():
        h_sum += hh.double().sum().item(); h_sumsq += (hh ** 2).double().sum().item()
        n_elem += hh.numel()
var_h = h_sumsq / n_elem - (h_sum / n_elem) ** 2
print(f"Var(h) = {var_h:.4f} over {n_elem // D_IN} tokens")

def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"]); sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]

results = []
for mode in MODES:
    for k in K_LIST:
        sae_path = f"{CACHE_DIR}/sae_{mode}_k{k}_final.pt"
        if not os.path.exists(sae_path):
            print(f"  skip {mode} k={k}: {os.path.basename(sae_path)} missing")
            continue
        sae, scale = load_sae(sae_path)
        P_hyb = t.load(f"{CACHE_DIR}/P_hybrid_k{k}.pt", map_location=device) if mode == "hybrid" else None

        sse = 0.0; l0_sum = 0.0; n_tok = 0
        with t.no_grad():
            for hh, tt in stream():
                x = (hh - P_hyb[tt]) if mode == "hybrid" else hh   # SAE input (residual for hybrid)
                a = sae.encode(x / scale)                          # feature acts (post-TopK)
                recon = sae.decode(a) * scale                      # SAE reconstruction, raw units
                h_hat = (P_hyb[tt] + recon) if mode == "hybrid" else recon   # add the map back for hybrid
                sse    += ((hh - h_hat) ** 2).double().sum().item()
                l0_sum += (a > 0).float().sum().item()             # active features summed over tokens
                n_tok  += hh.shape[0]
                if n_tok * D_IN >= n_elem:                         # match the Var(h) token budget
                    break

        fvu = sse / (n_tok * D_IN * var_h)
        l0  = l0_sum / n_tok                                       # MEASURED avg active features (≈ k, but drifts)
        results.append({"mode": mode, "k": k, "L0": l0, "FVU": fvu})
        print(f"  {mode:6s} k={k:4d}  L0={l0:6.1f}  FVU={fvu:.4f}")
        del sae, P_hyb
        t.cuda.empty_cache()

t.save(results, f"{CACHE_DIR}/sweep_results.pt")
print(f"\nsaved -> {CACHE_DIR}/sweep_results.pt   ({len(results)} points)")
