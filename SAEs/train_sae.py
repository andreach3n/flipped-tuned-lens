import os
import sys
import torch as t
import networkx as nx
DL_PATH = os.environ.get("DL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary_learning"))
sys.path.insert(0, DL_PATH)
from dictionary_learning.trainers.top_k import TopKTrainer
from transformer_lens import HookedTransformer

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print("using device:", device)

model = HookedTransformer.from_pretrained("gpt2")
X = model.W_E.detach().to(device)    # [50257, 768]

d_model = 768
X = X * (d_model ** 0.5) / X.norm(dim=-1).mean()

STEPS = 20000

def train_sae(seed, k=32, dict_size=8192):
    t.manual_seed(seed)
    trainer = TopKTrainer(steps=STEPS, activation_dim=768, dict_size=dict_size,
                          k=k, layer=0, lm_name="gpt2", device=device)
    trainer.dead_feature_threshold = 1_000_000
    for step in range(STEPS):
        idx = t.randint(0, X.shape[0], (4096,), device=device)
        trainer.update(step, X[idx])
    return trainer.ae

for seed in range(5):
    ae = train_sae(seed)
    with t.no_grad():
        X_hat = ae(X)
        fvu = ((X - X_hat).pow(2).sum() / (X - X.mean(0)).pow(2).sum()).item()
    t.save(ae.state_dict(), f"sae_w8192_k32_seed{seed}.pt")
    print(f"saved seed {seed}  fvu={fvu:.4f}")
