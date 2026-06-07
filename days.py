import torch as t
import torch.nn as nn
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [1, 5, 9, 13, 17, 21, 25]
STOP_LAYER = max(LAYERS) + 1
HITS_PER_DAY_TARGET = 200

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
print(f"Layers: {model.cfg.n_layers}")

# loading openwebtext
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

days = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
days_token = {}
for day in days:
    day_token = model.to_tokens(day, prepend_bos=False)
    assert day_token.shape[1] == 1, f"{day!r} tokenized to {day_token.shape[1]} tokens: {day_token}"
    days_token[day_token[0][0].item()] = day


names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in LAYERS]
token_ids = t.tensor(list(days_token.keys()), device=device)
embeddings = []
layer_activations = {l: [] for l in LAYERS}
matched_ids_list = []

for i, data in enumerate(ds):
    tokens = model.to_tokens(data["text"]).to(device)
    tokens = tokens[:, :512]

    mask = t.isin(tokens, token_ids)
    mask[:, 0] = False
    positions = mask.nonzero(as_tuple=False)

    if tokens.shape[1] < 10 or positions.shape[0] == 0:
        continue

    matched_ids = tokens[positions[:, 0], positions[:, 1]]
    matched_ids_list.append(matched_ids.cpu())

    with t.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, stop_at_layer=STOP_LAYER)

    embd = cache["hook_embed"]
    embd_days = embd[positions[:, 0], positions[:, 1]]
    embeddings.append(embd_days.cpu())

    for l in LAYERS:
        h = cache[f"blocks.{l}.hook_resid_post"]          # [1, 512, 2304]
        h_days = h[positions[:, 0], positions[:, 1]]
        layer_activations[l].append(h_days.cpu())

    del cache, logits
    t.cuda.empty_cache()

    if i % 100 == 0:
        all_ids_so_far = t.cat(matched_ids_list)
        counts = {tid: (all_ids_so_far == tid).sum().item() for tid in days_token}
        min_count = min(counts.values())
        print(f"step {i}, min hits per day: {min_count}, total hits: {len(all_ids_so_far)}")
        if min_count >= HITS_PER_DAY_TARGET:
            print(f"Reached target. Final counts: {counts}")
            break

print("Concatenating and saving...")

all_embd = t.cat(embeddings, dim=0)          # [N, 2304]
all_ids = t.cat(matched_ids_list, dim=0)     # [N]

t.save(all_embd, "/workspace/day_embeddings.pt")
t.save(all_ids, "/workspace/day_token_ids.pt")

for l in LAYERS:
    all_h = t.cat(layer_activations[l], dim=0)   # [N, 2304]
    t.save(all_h, f"/workspace/day_layer_{l}.pt")
    print(f"  saved layer {l}: shape {all_h.shape}")

# Also save the days_token dict so you can recover labels later
t.save(days_token, "/workspace/days_token_map.pt")

print("Done.")
