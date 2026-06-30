import os
import torch as t
from transformer_lens import HookedTransformer

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

config = JumpReLUTrainingSAEConfig(d_in=768, d_sae=8192, device=device, jumprelu_sparsity_loss_mode="tanh", l0_coefficient=1.0, l0_warm_up_steps=0)
sae = JumpReLUTrainingSAE(config, use_error_term=False)

total_training_samples=82_000_000
trainer_config = SAETrainerConfig(total_training_samples=total_training_samples, train_batch_size_samples=4096, device=device, lr_end=3e-5)
trained_sae = SAETrainer(trainer_config, sae, data_provider_fn()).fit()

with t.no_grad():
    acts = trained_sae.encode(X)
    L0 = (acts>0).sum(dim=1).float().mean().item()
    X_hat = trained_sae(X)
    fvu = ((X - X_hat).pow(2).sum() / (X - X.mean(0)).pow(2).sum()).item()

    dead_mask = (acts == 0).all(dim=0)
    dead_fraction = dead_mask.float().mean().item()

print(f"l0_coefficient={config.l0_coefficient}  L0={L0:.2f}  FVU={fvu:.4f}  dead={dead_fraction:.2%}")
