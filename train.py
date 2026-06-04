import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import glob
import matplotlib.pyplot as plt

LAYERS = [1, 5, 9, 13, 17, 21, 25]

# get all the embedding data out first, hold the middle layers and extract them individually to save memory
# embd = t.load("/workspace/embeddings.pt", weights_only=False)
print("Loading embedding chunks...")
embd_chunks = sorted(glob.glob("/workspace/embeddings_chunk_*.pt"))
embd = []
for chunk_path in embd_chunks:
    print(f"Loading {chunk_path}...")
    embd.extend(t.load(chunk_path))
print("Concatenating embeddings...")
embd_cat = t.cat(embd, dim=0)
del embd
print(f"embd_cat shape: {embd_cat.shape}")

# randomly shuffle the concatenated tensors
random_ind = t.randperm(len(embd_cat))
embd_shuffled = embd_cat[random_ind]

# train test split
train_index = int(embd_shuffled.shape[0] * 0.9)
embd_train = embd_shuffled[:train_index, :]
embd_test = embd_shuffled[train_index:, :]

# linear map
device = t.device("cuda" if t.cuda.is_available() else "cpu")
embd_train = embd_train
embd_test = embd_test
del embd_shuffled
del embd_cat

BATCH_SIZE = 128
N_TRAIN = embd_train.shape[0]
total_steps = N_TRAIN // BATCH_SIZE
warmup_steps = int(0.05 * total_steps)

# LR_CANDIDATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
lr = 1e-2
# results = {}

# training loop
# for lr in LR_CANDIDATES:

for_plotting = {}
for l in LAYERS:
    mid_lay_chunks = sorted(glob.glob(f"/workspace/layer_{l}_chunk_*.pt"))
    mid_lay = []
    for chunk_path in mid_lay_chunks:
        mid_lay.extend(t.load(chunk_path))

    mid_lay_cat = t.cat(mid_lay, dim=0)
    del mid_lay
    mid_lay_shuffled = mid_lay_cat[random_ind]

    # train test split
    mid_lay_train = mid_lay_shuffled[:train_index, :]
    mid_lay_test = mid_lay_shuffled[train_index:, :]
    del mid_lay_shuffled
    del mid_lay_cat

    linear_layer = nn.Linear(embd_train.shape[1], mid_lay_train.shape[1]).to(device)
    optimizer = optim.Adam(linear_layer.parameters(), lr=lr)

    # implementing learning rate scheduler, but need to check if warmup steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_steps)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

    total_train_loss = 0

    for step in range(total_steps):
        batch_start = step * BATCH_SIZE
        batch_end = (step + 1) * BATCH_SIZE

        embd_batch = embd_train[batch_start: batch_end].to(device).float()
        target_batch = mid_lay_train[batch_start:batch_end].to(device).float()

        optimizer.zero_grad()
        train_loss = t.nn.functional.mse_loss(linear_layer(embd_batch), target_batch)
        train_loss.backward()
        total_train_loss += train_loss.item()
        optimizer.step()
        scheduler.step()

    with t.no_grad():
        total_train_se = 0
        for i in range(0, embd_train.shape[0], BATCH_SIZE):
            embd_batch = embd_train[i:i+BATCH_SIZE].to(device).float()
            target_batch = mid_lay_train[i:i+BATCH_SIZE].to(device).float()
            se = ((linear_layer(embd_batch) - target_batch) ** 2).sum()
            total_train_se += se.item()
        train_loss_final = total_train_se / (embd_train.shape[0] * mid_lay_train.shape[1])

    with t.no_grad():
        total_se = 0
        for i in range(0, embd_test.shape[0], BATCH_SIZE):
            embd_batch = embd_test[i:i+BATCH_SIZE].to(device).float()
            mid_lay_batch = mid_lay_test[i:i+BATCH_SIZE].to(device).float()
            se = ((linear_layer(embd_batch)-mid_lay_batch)**2).sum() # compute squared error
            total_se += se.item()
        test_loss = total_se / (embd_test.shape[0] * mid_lay_test.shape[1])
        var = mid_lay_test.float().var()
        r2 = 1 - test_loss / var

        for_plotting[l] = {"train_loss": train_loss_final, "test_loss": test_loss, "r2": r2.item()}
        # test_loss = t.nn.functional.mse_loss(linear_layer(embd_test), mid_lay_test)
        # var = mid_lay_test.var()
        # r2 = 1 - test_loss / var

    avg_train_loss = total_train_loss / total_steps
    # results[lr] = {"train_loss": avg_train_loss, "test_loss": test_loss, "r2": r2.item()}
    print(f"Layer {l} — train loss: {train_loss_final}, test loss: {test_loss}, R²: {r2}")
    t.save(linear_layer.state_dict(), f"/workspace/linear_map_layer_{l}.pt")

# for testing learning rates
# for lr, metrics in results.items():
#     print(f"LR: {lr} — train: {metrics['train_loss']}, test: {metrics['test_loss']}, R²: {metrics['r2']}")

layers = list(for_plotting.keys())
r2_values = [for_plotting[l]["r2"] for l in layers]

plt.figure(figsize=(8, 5))
plt.plot(layers, r2_values, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel("Layer", fontsize=13)
plt.ylabel("Test $R^2$", fontsize=13)
plt.title("Linear Map from Embeddings → Layer Activations", fontsize=14)
plt.xticks(layers)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/r2_by_layer.png", dpi=150)
print("Saved to /workspace/r2_by_layer.png")
