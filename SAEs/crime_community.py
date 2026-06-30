import os
import sys
import torch as t
import networkx as nx
# dictionary_learning is cloned next to this script (in SAEs/) by default;
# set the DL_PATH env var to override if your clone lives elsewhere.
DL_PATH = os.environ.get("DL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary_learning"))
sys.path.insert(0, DL_PATH)
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from transformer_lens import HookedTransformer

# pick the best available device: CUDA GPU (e.g. on RunPod) > Apple MPS > CPU.
# same script runs anywhere with no edits.
device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print("using device:", device)

model = HookedTransformer.from_pretrained("gpt2")
X = model.W_E.detach().to(device)    # [50257, 768]

d_model = 768
X = X * (d_model ** 0.5) / X.norm(dim=-1).mean()

def is_filter(i):
    s = model.tokenizer.decode([i])
    return s.startswith(" ") and s[1:].isalpha() and len(s) >= 4

keep_ids = []
for i in range(X.shape[0]):
    if is_filter(i):
        keep_ids.append(i)
keep = t.tensor(keep_ids, device=device)
robber_idx = keep_ids.index(model.tokenizer.encode(" robber")[0])

def extract_crime_community(ae):
    with t.no_grad():
        F = ae.encode(X[keep])
    mask = (F > 0)
    A = mask.float() @ mask.float().T          # [N, N]: # of features each token pair shares
    A.fill_diagonal_(0)                        # drop self-loops
    A = A * (A >= 3)                           # drop weak single-feature links so the graph stays sparse

    G = nx.from_numpy_array(A.cpu().numpy())
    communities = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=4.0)
    crime = next(c for c in communities if robber_idx in c)
    crime_token_ids = {keep_ids[i] for i in crime}
    top_latents = mask[list(crime)].sum(0).topk(10).indices.tolist()
    return crime_token_ids, top_latents

results = {}
for seed in range(5):
    ae = AutoEncoderTopK.from_pretrained(f"sae_w8192_k32_seed{seed}.pt", device=device)
    crime_ids, top_latents = extract_crime_community(ae)
    results[seed] = {"crime_ids": sorted(crime_ids), "top_latents": top_latents}
    print(f"seed {seed}: |crime|={len(crime_ids)}  top latents={top_latents[:5]}")

import json
json.dump(results, open("crime_communities.json", "w"))
json.dump(keep_ids, open("keep_ids.json", "w"))

