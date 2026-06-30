import os
import sys
import torch as t
import networkx as nx
from itertools import combinations
# dictionary_learning is cloned next to this script (in SAEs/) by default;
# set the DL_PATH env var to override if your clone lives elsewhere.
DL_PATH = os.environ.get("DL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary_learning"))
sys.path.insert(0, DL_PATH)
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from transformer_lens import HookedTransformer

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print("using device:", device)

model = HookedTransformer.from_pretrained("gpt2")
X = model.W_E.detach().to(device)    # [50257, 768]

d_model = 768
X = X * (d_model ** 0.5) / X.norm(dim=-1).mean()

def is_filter(i):
    s = model.tokenizer.decode([i])
    return s.startswith(" ") and s[1:].isalpha() and len(s) >= 4

keep_ids = [i for i in range(X.shape[0]) if is_filter(i)]
keep = t.tensor(keep_ids, device=device)
robber_idx = keep_ids.index(model.tokenizer.encode(" robber")[0])

# --- build the token graph ONCE from a single fixed SAE ---
# the SAE and the graph G never change below, so any disagreement between runs
# is purely Louvain's own randomness, not the SAE's.
ae = AutoEncoderTopK.from_pretrained("sae_w8192_k32_seed0.pt", device=device)
with t.no_grad():
    F = ae.encode(X[keep])
mask = (F > 0)
A = mask.float() @ mask.float().T          # [N, N]: # of features each token pair shares
A.fill_diagonal_(0)                        # drop self-loops
A = A * (A >= 3)                           # drop weak single-feature links
G = nx.from_numpy_array(A.cpu().numpy())

# --- vary ONLY the louvain seed; SAE + graph are held fixed ---
def crime_set(louvain_seed):
    communities = nx.community.louvain_communities(G, weight="weight", seed=louvain_seed, resolution=4.0)
    crime = next(c for c in communities if robber_idx in c)
    return {keep_ids[i] for i in crime}

crime_sets = {s: crime_set(s) for s in range(5)}
for s in range(5):
    print(f"louvain seed {s}: |crime|={len(crime_sets[s])}")

def jaccard(a, b):
    return len(a & b) / len(a | b)

scores = []
for i, j in combinations(range(5), 2):
    r = jaccard(crime_sets[i], crime_sets[j])
    scores.append(r)
    print(f"louvain seeds {i},{j}:  jaccard={r:.3f}")

print(f"\nmean jaccard (louvain-only) = {sum(scores)/len(scores):.3f}")
