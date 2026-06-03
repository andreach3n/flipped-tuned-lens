import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

LAYERS = [1, 5, 9, 13, 17, 21, 25]

# get all the embedding data out first, hold the middle layers and extract them individually to save memory
embd = t.load("/workspace/embeddings.pt", weights_only=False)
mid_lays = {l: t.load(f"/workspace/layer_{l}.pt", weights_only=False) for l in LAYERS}

embd_cat = t.cat(embd, dim=0).float()

# randomly shuffle the concatenated tensors
random_ind = t.randperm(len(embd_cat))
embd_shuffled = embd_cat[random_ind]

# train test split
train_index = int(embd_shuffled.shape[0] * 0.9)
embd_train = embd_shuffled[:train_index, :]
embd_test = embd_shuffled[train_index:, :]

# linear map
device = t.device("cuda" if t.cuda.is_available() else "cpu")
embd_train = embd_train.to(device)
embd_test = embd_test.to(device)

BATCH_SIZE = 128
N_TRAIN = embd_train.shape[0]
total_steps = N_TRAIN // BATCH_SIZE
warmup_steps = int(0.05 * total_steps)

# training loop
for l in LAYERS:
    mid_lay_cat = t.cat(mid_lays[l], dim=0).float()
    mid_lay_shuffled = mid_lay_cat[random_ind]

    # train test split, put on cuda
    mid_lay_train = mid_lay_shuffled[:train_index, :].to(device)
    mid_lay_test = mid_lay_shuffled[train_index:, :].to(device)

    linear_layer = nn.Linear(embd_train.shape[1], mid_lay_train.shape[1]).to(device)
    optimizer = optim.Adam(linear_layer.parameters(), lr=0.001)

    # implementing learning rate scheduler, but need to check if warmup steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_steps)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

    total_train_loss = 0
    for step in range(total_steps):
        batch_start = step * BATCH_SIZE
        batch_end = (step + 1) * BATCH_SIZE

        embd_batch = embd_train[batch_start: batch_end]
        target_batch = mid_lay_train[batch_start:batch_end]

        optimizer.zero_grad()
        train_loss = t.nn.functional.mse_loss(linear_layer(embd_batch), target_batch)
        train_loss.backward()
        total_train_loss += train_loss.item()
        optimizer.step()
        scheduler.step()

    with t.no_grad():
        test_loss = t.nn.functional.mse_loss(linear_layer(embd_test), mid_lay_test)
        var = mid_lay_test.var()
        r2 = 1 - test_loss / var

    avg_train_loss = total_train_loss / total_steps
    print(f"Layer {l} — train loss: {avg_train_loss}, test loss: {test_loss.item()}, R²: {r2}")
    t.save(linear_layer.state_dict(), f"/workspace/linear_map_layer_{l}.pt")
