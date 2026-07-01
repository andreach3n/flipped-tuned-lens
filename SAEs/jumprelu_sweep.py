import os
import torch as t
from transformer_lens import HookedTransformer
import numpy as np
import matplotlib.pyplot as plt

# SAELens v6 (6.44.x). All training classes are top-level exports.
from sae_lens import (
    JumpReLUTrainingSAE,         # the SAE module (encoder/decoder + jumprelu)
    JumpReLUTrainingSAEConfig,   # arch + sparsity knobs: l0_coefficient,
                                 # jumprelu_sparsity_loss_mode ("tanh" = Anthropic),
                                 # jumprelu_init_threshold, jumprelu_bandwidth,
                                 # l0_warm_up_steps, pre_act_loss_coefficient
    SAETrainer,                  # training loop: SAETrainer(cfg, sae, data_provider).fit()
)
from sae_lens.config import SAETrainerConfig  # lr, total_training_samples, logging, etc.

# pick the best available device: CUDA > Apple MPS > CPU
device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print("using device:", device)

# the data you're decomposing: GPT-2's embedding matrix, same as your other scripts
model = HookedTransformer.from_pretrained("gpt2")
X = model.W_E.detach().to(device)            # [50257, 768]
d_model = 768
X = X * (d_model ** 0.5) / X.norm(dim=-1).mean()

# --- experiment goes below ---
# data_provider is just an Iterator[torch.Tensor] yielding [batch, 768] tensors.
# sweep l0_coefficient over a geometric range; set jumprelu_sparsity_loss_mode="tanh".

def data_provider_fn():
    while True:
        idx = t.randint(0, X.shape[0], (4096,), device=device)
        yield X[idx]

# total_steps = total_training_samples / batch_size
# l0_warm_up_steps = 0.1 * total_steps

coefficients = [1]
# np.geomspace(1e-2, 1e1, 12)
results = []
BATCH = 4096
total_training_samples = 300_000_000
total_steps = total_training_samples/BATCH
l0_warmup_steps = int(0.1 * total_steps)

for coeff in coefficients:
    config = JumpReLUTrainingSAEConfig(d_in=768, d_sae=8192, device=device, jumprelu_sparsity_loss_mode="tanh", l0_coefficient=coeff, l0_warm_up_steps=l0_warmup_steps)
    sae = JumpReLUTrainingSAE(config, use_error_term=False)

    trainer_config = SAETrainerConfig(total_training_samples=total_training_samples, train_batch_size_samples=BATCH, device=device, lr_scheduler_name="cosineannealing", lr_end=3e-5)
    trained_sae = SAETrainer(trainer_config, sae, data_provider_fn()).fit()

    with t.no_grad():
        acts = trained_sae.encode(X)
        L0 = (acts>0).sum(dim=1).float().mean().item()
        X_hat = trained_sae(X)
        fvu = ((X - X_hat).pow(2).sum() / (X - X.mean(0)).pow(2).sum()).item()

        dead_mask = (acts == 0).all(dim=0)
        dead_fraction = dead_mask.float().mean().item()
    results.append((coeff, L0, fvu, dead_fraction))
    print(f"l0_coefficient={config.l0_coefficient}  L0={L0:.2f}  FVU={fvu:.4f}  dead={dead_fraction:.2%}")

# --- save raw results first, so a plotting bug can't cost you the runs ---
results_arr = np.array(results)                       # [n_runs, 5]: coeff, seed, L0, fvu, dead
out_name = f"sweep_results_{total_training_samples}.npy"   # e.g. sweep_results_300000000.npy
np.save(out_name, results_arr)
print(f"saved raw results to {out_name}")

# --- aggregate + plot: commented out for the convergence check (1 coeff, no seeds).
#     it's the seeded 5-column version; re-enable + restore the seed loop for the full sweep.
# coeffs_all = results_arr[:, 0]                         # every run's coeff (repeats per seed)
# L0_all     = results_arr[:, 2]
# fvu_all    = results_arr[:, 3]
# dead_all   = results_arr[:, 4]
#
# uniq = np.unique(coeffs_all)                           # the 12 distinct coefficients
# mean_by_coeff = lambda vals: np.array([vals[coeffs_all == c].mean() for c in uniq])
# L0_mean, fvu_mean, dead_mean = mean_by_coeff(L0_all), mean_by_coeff(fvu_all), mean_by_coeff(dead_all)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# # Panel 1: the plateau plot. Dots = individual seeds, line = mean.
# # Flat band in the mean AND tight dots = a real "sticky" L0; scattered dots = noise.
# ax1.scatter(coeffs_all, L0_all, color="C0", alpha=0.35, s=25, label="individual seeds")
# ax1.plot(uniq, L0_mean, "o-", color="C0", label="mean across seeds")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlabel("l0_coefficient (sparsity penalty)")
# ax1.set_ylabel("L0 (avg active features)")
# ax1.set_title("L0 vs sparsity penalty — flat band = sticky L0")
# ax1.legend()
# ax1.grid(True, which="both", alpha=0.3)
#
# # Panel 2: diagnostics — rules out "plateau is really feature death / bad recon".
# ax2.scatter(coeffs_all, fvu_all, color="C1", alpha=0.35, s=25)
# ax2.plot(uniq, fvu_mean, "s-", color="C1", label="FVU (reconstruction error)")
# ax2.scatter(coeffs_all, dead_all, color="C2", alpha=0.35, s=25)
# ax2.plot(uniq, dead_mean, "^-", color="C2", label="dead fraction")
# ax2.set_xscale("log")
# ax2.set_xlabel("l0_coefficient (sparsity penalty)")
# ax2.set_ylabel("fraction")
# ax2.set_title("reconstruction & dead features")
# ax2.set_ylim(0, 1)
# ax2.legend()
# ax2.grid(True, which="both", alpha=0.3)
#
# fig.tight_layout()
# fig.savefig("jumprelu_sweep_convergence.png", dpi=150)
# print("saved plot to jumprelu_sweep_convergence.png")
# plt.show()
