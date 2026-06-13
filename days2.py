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

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
model.eval()
print(f"Layers: {model.cfg.n_layers}")

days = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
days_token = {}
for day in days:
    day_token = model.to_tokens(day, prepend_bos=False)
    assert day_token.shape[1] == 1, f"{day!r} tokenized to {day_token.shape[1]} tokens: {day_token}"
    days_token[day_token[0][0].item()] = day

# templates = [
#     "Today is {day}.",
#     "Tomorrow is {day}.",
#     "Yesterday was {day}.",
#     "The meeting is on {day}.",
#     "She was born on a {day}.",
#     "The event takes place on {day}.",
#     "I have an appointment on {day}.",
#     "The deadline is {day}.",
#     "We will meet on {day}.",
#     "The package arrives on {day}.",
#     "The store is closed on {day}.",
#     "He called me on {day}.",
#     "The class is every {day}.",
#     "It happened last {day}.",
#     "See you next {day}.",
#     "The flight departs on {day}.",
#     "She left on {day}.",
#     "The report is due {day}.",
#     "It rained on {day}.",
#     "The party is on {day}.",
# ]

templates = ["The day after {day} is",
"The day after {day},",
"The day after {day} she",
"The day after {day} he",
"The day after {day} the",
"It was the day after {day}",
"This was the day after {day}.",
"That happened the day after {day}.",
"The morning after {day} was",
"One day after {day},",
]

texts = [tmpl.format(day=day.strip()) for tmpl in templates for day in days]
labels = [day for tmpl in templates for day in days]

names_filter = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in LAYERS]
token_ids = t.tensor(list(days_token.keys()), device=device)
embeddings = []
layer_activations = {l: [] for l in LAYERS}
matched_ids_list = []

for i, text in enumerate(texts):
    tokens = model.to_tokens(text).to(device)
    tokens = tokens[:, :512]

    mask = t.isin(tokens, token_ids)
    mask[:, 0] = False
    positions = mask.nonzero(as_tuple=False)

    if positions.shape[0] == 0:
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

print("Concatenating and saving...")

all_embd = t.cat(embeddings, dim=0)          # [N, 2304]
all_ids = t.cat(matched_ids_list, dim=0)     # [N]

t.save(all_embd, "/workspace/day_embeddings_simple.pt")
t.save(all_ids, "/workspace/day_token_ids_simple.pt")

for l in LAYERS:
    all_h = t.cat(layer_activations[l], dim=0)   # [N, 2304]
    t.save(all_h, f"/workspace/day_layer_{l}_simple.pt")
    print(f"  saved layer {l}: shape {all_h.shape}")

# Also save the days_token dict so you can recover labels later
t.save(days_token, "/workspace/days_token_map_simple.pt")

print("Done.")
