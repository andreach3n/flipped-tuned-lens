import os
import torch as t
from transformer_lens import HookedTransformer
import numpy as np
import matplotlib.pyplot as plt

model = HookedTransformer.from_pretrained("gpt2")
W = model.W_E.detach().float().cpu()            # [50257, 768]

svals = t.linalg.svdvals(W)
stable_rank = (svals.pow(2).sum() / svals[0].pow(2)).item()
print("stable rank:", stable_rank)

Wc = W - W.mean(0, keepdim=True)               # remove the mean embedding
svals_c = t.linalg.svdvals(Wc)
print("stable rank (centered):", (svals_c.pow(2).sum() / svals_c[0].pow(2)).item())

p = svals.pow(2) / svals.pow(2).sum()
eff_rank = t.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()
print("effective rank (entropy):", eff_rank)

import matplotlib.pyplot as plt
plt.semilogy(svals_c.numpy())      # centered singular values, descending
plt.xlabel("index"); plt.ylabel("singular value (log)")
plt.show()
