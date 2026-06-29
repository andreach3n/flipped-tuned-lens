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

ae = AutoEncoderTopK.from_pretrained("sae.pt", device=device)

def is_filter(i):
    s = model.tokenizer.decode([i])
    return s.startswith(" ") and s[1:].isalpha() and len(s) >= 4

keep_ids = []
for i in range(X.shape[0]):
    if is_filter(i):
        keep_ids.append(i)
keep = t.tensor(keep_ids, device=device)
print(len(keep_ids))

# build the token graph over a manageable slice of the vocab (graph only — the
# SAE was trained on all 50k embeddings above)
with t.no_grad():
    F = ae.encode(X[keep])           # [N, 8192]
mask = (F > 0)
A = mask.float() @ mask.float().T          # [N, N]: # of features each token pair shares
A.fill_diagonal_(0)                        # drop self-loops
A = A * (A >= 3)                           # drop weak single-feature links so the graph stays sparse

G = nx.from_numpy_array(A.cpu().numpy())
communities = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=4.0)
print(len(communities))
print(sorted([len(c) for c in communities], reverse=True))

robber_token = model.tokenizer.encode(" robber")
robber_idx = keep_ids.index(robber_token[0])
crime = next(c for c in communities if robber_idx in c)

crime_tokens = mask[list(crime)] # selects the rows of tokens that are contained in crime
latent_count = crime_tokens.sum(dim=0)
top_latents = latent_count.topk(10)
print(len(crime), top_latents)

for j in top_latents.indices:
    fire = mask[:, j].nonzero().flatten().tolist()
    decoded_fire = [model.tokenizer.decode([keep_ids[pos]]) for pos in fire]
