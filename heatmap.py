import torch as t
import torch.nn as nn
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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
    "She walked along the bank of the river, watching the water rush past.",
    "The next morning she visited the bank to deposit her paycheck.",
    "After the crash, the plane had to bank sharply to the left.",

    "He struck a match to light the candle on the table.",
    "The bag was surprisingly light, easy to carry with one hand.",
    "The evening light filtered through the curtains and cast long shadows.",

    "She decided to go for a run before breakfast every morning.",
    "The candidate chose to run for office despite the long odds.",
    "After a long run in the theater, the play finally closed on Saturday.",

    "The scientist warned that lead pipes were still common in older buildings.",
    "She would lead the expedition through the mountains in early spring.",
    "The detective followed every lead until the case was finally solved.",
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

def plot_heatmap(token_strs, errors, title, save_path):
    """
    token_strs: list of T strings
    errors: tensor [num_layers, T]
    """
    num_layers = len(LAYERS)
    T = len(token_strs)

    fig, ax = plt.subplots(figsize=(max(10, T * 0.5), num_layers * 0.8 + 1.5))

    # normalize each layer independently so colors are comparable within a layer
    err_np = errors.detach().numpy()  # [num_layers, T]
    err_normalized = np.zeros_like(err_np)
    for i in range(num_layers):
        row = err_np[i]
        row_min, row_max = row.min(), row.max()
        if row_max > row_min:
            err_normalized[i] = (row - row_min) / (row_max - row_min)
        else:
            err_normalized[i] = 0.0

    im = ax.imshow(err_normalized, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # x-axis: token strings
    ax.set_xticks(range(T))
    ax.set_xticklabels(token_strs, rotation=45, ha="right", fontsize=9)

    # y-axis: layer numbers
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=9)

    ax.set_title(title, fontsize=12, pad=10)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Normalized error (within layer)", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {save_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

for idx, sentence in enumerate(SENTENCES):
    log(f"Processing sentence {idx+1}/{len(SENTENCES)}: {sentence[:60]}...")
    token_strs, errors = compute_token_errors(sentence)
    save_path = f"/workspace/heatmap_{idx+1:02d}.png"
    plot_heatmap(token_strs, errors, title=sentence, save_path=save_path)

log("All heatmaps saved to /workspace/heatmap_*.png")
