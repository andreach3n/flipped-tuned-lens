import torch as t
import torch.nn as nn
import glob
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset
import matplotlib.pyplot as plt

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS)+1
MAX_TOKENS = 100000

ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
ds = ds.skip(5000)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()

layer_activations = {l: None for l in LAYERS}
names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in layer_activations]

linear_layers = sorted(glob.glob(f"/workspace/linear_map_layer_*.pt"))
linear_map = {}
for layer in linear_layers:
    l = int(layer.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(model.cfg.d_model, model.cfg.d_model).to(device)
    linear_map[l].load_state_dict(t.load(layer, weights_only=False))

position_error_sum = {l: t.zeros(512).to(device) for l in LAYERS}
position_error_count = {l: t.zeros(512).to(device) for l in LAYERS}
token_error_sum = {l: t.zeros(model.cfg.vocab_size).to(device) for l in LAYERS}
token_error_count = {l: t.zeros(model.cfg.vocab_size).to(device) for l in LAYERS}
count = 0
ones = t.ones(512, device=device)

for data in ds:
    tokens = model.to_tokens(data["text"]).to(device)
    tokens = tokens[:, :512]

    if tokens.shape[1] < 10:
        continue

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    embd = cache["hook_embed"].squeeze(0).to(device).float()
    for l in layer_activations:
        layer_activations[l] = cache[f"blocks.{l}.hook_resid_post"].squeeze(0).to(device).float()
        pred = linear_map[l](embd)
        err = t.linalg.vector_norm(pred - layer_activations[l], dim=1) # pred - layer_activations[l] is [T, d_model], so you shrink down to [T]

        T = err.shape[0]
        position_error_sum[l][:T] += err
        position_error_count[l][:T] += 1

        token_ids = tokens.squeeze(0)
        token_error_sum[l].index_add_(0, token_ids, err)
        token_error_count[l].index_add_(0, token_ids, ones[:T])

    del cache, logits
    t.cuda.empty_cache()

    count += tokens.shape[1]
    if count > MAX_TOKENS:
        break

position_mean = {l: None for l in LAYERS}
token_mean = {l: None for l in LAYERS}
for l in LAYERS:
    position_mean[l] = t.where(position_error_count[l] > 0, position_error_sum[l] / position_error_count[l], t.zeros_like(position_error_sum[l]))
    token_mean[l] = t.where(token_error_count[l] > 0, token_error_sum[l] / token_error_count[l], t.zeros_like(token_error_sum[l]))

del position_error_sum, position_error_count, token_error_sum, token_error_count
t.cuda.empty_cache()

# plot position mean
plt.figure(figsize=(10, 6))
for l in LAYERS:
    plt.plot(position_mean[l].cpu().numpy(), label=f"Layer {l}")
plt.xlabel("Position in sequence")
plt.ylabel("Mean error magnitude")
plt.title("Error by position in sequence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/error_by_position.png", dpi=150)
plt.close()

TOP_N = 30

plt.figure(figsize=(14, 6))
for l in LAYERS:
    top_values, top_indices = t.topk(token_mean[l], TOP_N)
    top_tokens = [model.to_string(idx.unsqueeze(0)) for idx in top_indices]
    plt.bar(top_tokens, top_values.cpu().numpy(), label=f"Layer {l}", alpha=0.5)
plt.xlabel("Token")
plt.ylabel("Mean error magnitude")
plt.title("Top tokens by error magnitude")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("/workspace/error_by_token.png", dpi=150)
plt.close()
