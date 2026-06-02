import torch as t
import torch.nn as nn
import torch.optim as optim

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

# training loop
for l in LAYERS:
    mid_lay_cat = t.cat(mid_lays[l], dim=0).float()
    mid_lay_shuffled = mid_lay_cat[random_ind]

    # train test split, put on cuda
    mid_lay_train = mid_lay_shuffled[:train_index, :].to(device)
    mid_lay_test = mid_lay_shuffled[train_index:, :].to(device)

    linear_layer = nn.Linear(embd_train.shape[1], mid_lay_train.shape[1]).to(device)
    optimizer = optim.Adam(linear_layer.parameters(), lr=0.001)

    optimizer.zero_grad()
    train_loss = t.nn.functional.mse_loss(linear_layer(embd_train), mid_lay_train)
    train_loss.backward()
    optimizer.step()

    with t.no_grad():
        test_loss = t.nn.functional.mse_loss(linear_layer(embd_test), mid_lay_test)
        var = mid_lay_test.var()
        r2 = 1 - test_loss / var

    print(f"train loss: {train_loss.item()}, test loss: {test_loss.item()}, R^2: {r2}")

t.save(linear_layer.state_dict(), "/workspace/linear_map.pt")
