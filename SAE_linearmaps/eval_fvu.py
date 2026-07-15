import torch as t
import glob
import os
from sae_lens import BatchTopKTrainingSAE

CACHE_DIR = "/workspace/sae_cache_layer13"
N_TOKENS  = 2_000_000     # FVU converges fast; 2M is plenty
bs        = 8192
device    = t.device("cuda")

P = t.load(f"{CACHE_DIR}/P.pt", map_location=device)   # (V, 2304)
# trained (jointly-optimized) map, only present once hybrid has been trained
P_HYBRID_PATH = f"{CACHE_DIR}/P_hybrid.pt"
HYBRID_PATH   = f"{CACHE_DIR}/sae_hybrid_final.pt"
P_hybrid = t.load(P_HYBRID_PATH, map_location=device) if os.path.exists(P_HYBRID_PATH) else None
# outbias ablation: encoder sees full h, jointly-trained map added at the OUTPUT (k-tagged files)
P_OUTBIAS_PATH = f"{CACHE_DIR}/P_outbias_k64.pt"
OUTBIAS_PATH   = f"{CACHE_DIR}/sae_outbias_k64_final.pt"
P_outbias = t.load(P_OUTBIAS_PATH, map_location=device) if os.path.exists(P_OUTBIAS_PATH) else None

def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]

sae_full,  scale_full  = load_sae(f"{CACHE_DIR}/sae_full_final.pt")
sae_resid, scale_resid = load_sae(f"{CACHE_DIR}/sae_resid_final.pt")

# hybrid is optional: only wire it in if both its SAE checkpoint and trained P table exist
HAVE_HYBRID = os.path.exists(HYBRID_PATH) and P_hybrid is not None
if HAVE_HYBRID:
    sae_hybrid, scale_hybrid = load_sae(HYBRID_PATH)
    print("hybrid artifacts found -> including HYBRID FVU")

HAVE_OUTBIAS = os.path.exists(OUTBIAS_PATH) and P_outbias is not None
if HAVE_OUTBIAS:
    sae_outbias, scale_outbias = load_sae(OUTBIAS_PATH)
    print("outbias artifacts found -> including OUTBIAS FVU")

def recon_full(hh):
    x = hh / scale_full
    a = sae_full.encode(x)
    return sae_full.decode(a) * scale_full          # back to raw units

def recon_resid(hh, tt):
    p = P[tt]                                       # (b, 2304) map's prediction
    r = hh - p
    x = r / scale_resid
    a = sae_resid.encode(x)
    r_hat = sae_resid.decode(a) * scale_resid       # SAE's residual recon, raw units
    return p + r_hat, r, r_hat                      # composite ĥ, plus r/r̂ for the side-metric

def recon_hybrid(hh, tt):
    p = P_hybrid[tt]                                # the JOINTLY-TRAINED prediction
    a = sae_hybrid.encode((hh - p) / scale_hybrid)
    return p + sae_hybrid.decode(a) * scale_hybrid  # composite ĥ (map + hybrid SAE)

def recon_outbias(hh, tt):
    a = sae_outbias.encode(hh / scale_outbias)      # encoder sees the FULL h (token NOT removed)
    return P_outbias[tt] + sae_outbias.decode(a) * scale_outbias   # map added at the OUTPUT

# float64 accumulators: summing ~5e9 float32 squares drifts; .double() first avoids it
sse_full = sse_comp = sse_r = 0.0
sse_hybrid = sse_outbias = 0.0
h_sum = h_sumsq = 0.0
r_sum = r_sumsq = 0.0
n_elem = 0
seen = 0

with t.no_grad():
    for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
        tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
        hc = t.cat(t.load(lf), dim=0)
        tc = t.cat(t.load(tf), dim=0)
        for start in range(0, hc.shape[0], bs):
            hh = hc[start:start+bs].float().to(device)
            tt = tc[start:start+bs].to(device)

            h_hat_full = recon_full(hh)
            h_hat_comp, r, r_hat = recon_resid(hh, tt)

            sse_full += ((hh - h_hat_full)**2).double().sum().item()
            sse_comp += ((hh - h_hat_comp)**2).double().sum().item()
            sse_r    += ((r  - r_hat     )**2).double().sum().item()
            if HAVE_HYBRID:
                sse_hybrid += ((hh - recon_hybrid(hh, tt))**2).double().sum().item()
            if HAVE_OUTBIAS:
                sse_outbias += ((hh - recon_outbias(hh, tt))**2).double().sum().item()

            h_sum += hh.double().sum().item();  h_sumsq += (hh**2).double().sum().item()
            r_sum += r.double().sum().item();   r_sumsq += (r**2).double().sum().item()
            n_elem += hh.numel()

            seen += hh.shape[0]
            if seen >= N_TOKENS: break
        if seen >= N_TOKENS: break

var_h = h_sumsq/n_elem - (h_sum/n_elem)**2
var_r = r_sumsq/n_elem - (r_sum/n_elem)**2

print(f"Var(r)/Var(h)              : {var_r/var_h:.3f}   (expect ~0.34)")
print(f"FVU on h  — full SAE       : {sse_full/(n_elem*var_h):.4f}   (expect ~0.14)")
print(f"FVU on h  — map + resid SAE: {sse_comp/(n_elem*var_h):.4f}   (expect ~0.05)")
print(f"FVU on r  — resid SAE alone: {sse_r   /(n_elem*var_r):.4f}   (expect ~0.14)")
print(f"identity check: {sse_r/(n_elem*var_r) * var_r/var_h:.4f} == composite FVU above")
if HAVE_HYBRID:
    print(f"FVU on h  — hybrid (joint)  : {sse_hybrid/(n_elem*var_h):.4f}   (beat full's {sse_full/(n_elem*var_h):.4f}?)")
if HAVE_OUTBIAS:
    print(f"FVU on h  — outbias         : {sse_outbias/(n_elem*var_h):.4f}   (encoder sees h, map at output; 20M-trained)")
