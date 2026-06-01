import torch as t
import torch.nn as nn
import torch.optim as optim

embd = t.load("./embeddings.pt")
mid_lay = t.load("./middle_layer.pt")

embd_cat = t.cat(embd, dim=0)
mid_lay_cat = t.cat(mid_lay, dim=0)

# randomly shuffle the concatenated tensors
random_ind = t.randperm(len(embd_cat))
embd_shuffled = embd_cat[random_ind]
mid_lay_shuffled = mid_lay_cat[random_ind]

# train test split
train_index = int(embd_shuffled.shape[0] * 0.9)
embd_train = embd_shuffled[:train_index, :]
mid_lay_train = mid_lay_shuffled[:train_index, :]
embd_test = embd_shuffled[train_index:, :]
mid_lay_test = mid_lay_shuffled[train_index:, :]

# linear map
linear_layer = nn.Linear(embd_train.shape[1], mid_lay_train.shape[1])

# training loop
optimizer = optim.Adam(linear_layer.parameters(), lr=0.001)
for epoch in range(10):
    optimizer.zero_grad()
    train_loss = t.nn.functional.mse_loss(linear_layer(embd_train), mid_lay_train)
    train_loss.backward()
    optimizer.step()

    with t.no_grad():
        test_loss = t.nn.functional.mse_loss(linear_layer(embd_test), mid_lay_test)
        var = mid_lay_test.var()
        r2 = 1 - test_loss / var

    print(f"Epoch {epoch}, train loss: {train_loss.item()}, test loss: {test_loss.item()}, R^2: {r2}")

t.save(linear_layer.state_dict(), "./linear_map.pt")
