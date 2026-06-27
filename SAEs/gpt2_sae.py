import os
import sys
import torch as t
import networkx as nx
sys.path.insert(0, "/Users/andrea/Documents/dictionary_learning")
from dictionary_learning.trainers.top_k import TopKTrainer
from transformer_lens import HookedTransformer

# pick the best available device: CUDA GPU (e.g. on RunPod) > Apple MPS > CPU.
# same script runs anywhere with no edits.
device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print("using device:", device)

model = HookedTransformer.from_pretrained("gpt2")
X = model.W_E.detach().to(device)    # [50257, 768]

# normalize with a single global scalar so the average embedding has L2 norm
# sqrt(d_model). One scalar for the whole matrix keeps angles + relative norms
# intact and just fixes the overall scale, which is what the trainer's lr /
# threshold / AuxK machinery expects. No mean-centering: the SAE's b_dec absorbs
# the mean (initialized to the geometric median at step 0).
d_model = 768
X = X * (d_model ** 0.5) / X.norm(dim=-1).mean()

STEPS = 20000
trainer = TopKTrainer(steps=STEPS, activation_dim=768, dict_size=8192, k=32,
                      layer=0, lm_name="gpt2", device=device)
trainer.dead_feature_threshold = 1_000_000

for step in range(STEPS):
    idx = t.randint(0, X.shape[0], (4096,), device=device)
    loss = trainer.update(step, X[idx])

# compute reconstruction error
with t.no_grad():
    X_hat = trainer.ae(X)                      # [50257, 768], one shot
    sq_err = (X - X_hat).pow(2).sum(-1)        # per-token squared error
    mse = sq_err.mean().item()
    fvu = (sq_err.sum() / (X - X.mean(0)).pow(2).sum(-1).sum()).item()
print(mse, fvu)
print(trainer.dead_features, trainer.num_tokens_since_fired)
print(trainer.effective_l0)

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
    F = trainer.ae.encode(X[keep])           # [N, 8192]
mask = (F > 0)
A = mask.float() @ mask.float().T          # [N, N]: # of features each token pair shares
A.fill_diagonal_(0)                        # drop self-loops
A = A * (A >= 3)                           # drop weak single-feature links so the graph stays sparse

G = nx.from_numpy_array(A.cpu().numpy())
communities = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=4.0)
print(len(communities))
print(sorted([len(c) for c in communities], reverse=True))

# look at mid-sized communities (not the giant generic hubs, not singletons) —
# this band is where coherent themes like "months" or "country names" live
MIN_SIZE, MAX_SIZE = 30, 300
band = [c for c in communities if MIN_SIZE <= len(c) <= MAX_SIZE]

out_path = os.path.join(os.path.dirname(__file__), "communities_30_300.txt")
with open(out_path, "w") as f:
    for c in sorted(band, key=len, reverse=True):
        words = [model.tokenizer.decode([keep_ids[i]]) for i in c]
        f.write(f"{len(c)} {words}\n\n")   # blank line between communities
print(f"saved {len(band)} communities (size {MIN_SIZE}-{MAX_SIZE}) to {out_path}")
