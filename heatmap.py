import torch as t
import torch.nn as nn
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformer_lens import HookedTransformer
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1

# ── Sentences to analyze ───────────────────────────────────────────────────────
# Edit this list freely — one sentence per entry.
# Try to include the same word in different senses across sentences.

SENTENCES = [
    # bank
    "She spent the afternoon sitting on the muddy bank, watching the river flow past.",
    "She spent the afternoon standing outside the bank, waiting for it to open.",

    # light
    "At dusk, the warm light from the lamp cast long shadows across the wall.",
    "At dusk, the surprisingly light suitcase was easy to lift into the overhead bin.",

    # run
    "Each morning before breakfast, her run through the park helped clear her mind.",
    "Each morning before breakfast, her run of the financial models took about an hour.",

    # lead
    "The inspector confirmed that the pipes were made of lead and posed a health risk.",
    "The inspector was chosen to lead the team through the hazardous building safely.",
]

# ── Setup ──────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
log(f"Using device: {device}")

log("Loading model...")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
log("Model loaded.")

names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in LAYERS]

log("Loading linear maps...")
linear_map = {}
for path in sorted(glob.glob("/workspace/linear_map_layer_*.pt")):
    l = int(path.split("layer_")[1][:-3])
    linear_map[l] = nn.Linear(model.cfg.d_model, model.cfg.d_model).to(device)
    linear_map[l].load_state_dict(t.load(path, weights_only=False, map_location=device))
log(f"Loaded maps for layers: {sorted(linear_map.keys())}")

# ── Error computation ──────────────────────────────────────────────────────────

def compute_token_errors(sentence):
    """
    Returns:
        token_strs: list of string tokens, length T
        errors: tensor of shape [num_layers, T] — L2 error magnitude per token per layer
    """
    tokens = model.to_tokens(sentence).to(device)  # [1, T]

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    embd = cache["hook_embed"].squeeze(0)[1:].to(device).float()  # [T-1, d_model], skip BOS

    errors = []
    for l in LAYERS:
        h = cache[f"blocks.{l}.hook_resid_post"].squeeze(0)[1:].to(device).float()  # [T-1, d_model], skip BOS
        pred = linear_map[l](embd)                                                    # [T-1, d_model]
        err = t.linalg.vector_norm(pred - h, dim=1)                                  # [T-1]
        errors.append(err.cpu())

    del cache, logits
    t.cuda.empty_cache()

    errors = t.stack(errors, dim=0)  # [num_layers, T-1]

    # convert token ids back to strings, skip BOS
    token_ids = tokens.squeeze(0)[1:]  # [T-1]
    token_strs = [model.to_string(tid.unsqueeze(0)) for tid in token_ids]

    return token_strs, errors

# ── Heatmap plotting ───────────────────────────────────────────────────────────

def plot_heatmap(token_strs, errors, title, save_path, normalize=True):
    """
    token_strs: list of T strings
    errors: tensor [num_layers, T]
    normalize: if True, normalize each row to [0, 1] independently
    """
    num_layers = len(LAYERS)
    T = len(token_strs)

    fig, ax = plt.subplots(figsize=(max(10, T * 0.5), num_layers * 0.8 + 1.5))

    err_np = errors.detach().numpy()  # [num_layers, T]

    if normalize:
        data = err_np.copy()
        for i in range(num_layers):
            row = err_np[i]
            row_min, row_max = row.min(), row.max()
            if row_max > row_min:
                data[i] = (row - row_min) / (row_max - row_min)
            else:
                data[i] = 0.0
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        cbar_label = "Normalized error (within layer)"
    else:
        im = ax.imshow(err_np, cmap="RdYlGn_r", aspect="auto")
        cbar_label = "Error magnitude (L2)"

    # x-axis: token strings
    ax.set_xticks(range(T))
    ax.set_xticklabels(token_strs, rotation=45, ha="right", fontsize=9)

    # y-axis: layer numbers
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)

    ax.set_title(title, fontsize=12, pad=10)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {save_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

import os
os.makedirs("/workspace/heatmaps_normalized", exist_ok=True)
os.makedirs("/workspace/heatmaps_not_normalized", exist_ok=True)

for idx, sentence in enumerate(SENTENCES):
    log(f"Processing sentence {idx+1}/{len(SENTENCES)}: {sentence[:60]}...")
    token_strs, errors = compute_token_errors(sentence)
    plot_heatmap(token_strs, errors, title=sentence,
                 save_path=f"/workspace/heatmaps_normalized/heatmap_{idx+1:02d}.png",
                 normalize=True)
    plot_heatmap(token_strs, errors, title=sentence,
                 save_path=f"/workspace/heatmaps_not_normalized/heatmap_{idx+1:02d}.png",
                 normalize=False)

log("Done. Heatmaps saved to /workspace/heatmaps_normalized/ and /workspace/heatmaps_not_normalized/")
