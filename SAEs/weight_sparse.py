import io
import os
import torch
from tiktoken.load import read_file_cached
from circuit_sparsity.registries import MODEL_BASE_DIR
from tiktoken import Encoding
from circuit_sparsity.tiktoken_ext import tinypython

sd = torch.load(io.BytesIO(read_file_cached(f"{MODEL_BASE_DIR}/models/csp_yolo1/final_model.pt")), weights_only=True, map_location="cpu")
embd = sd["transformer.wte.weight"]
pos = sd["transformer.wpe.weight"]

enc = Encoding(**tinypython.tinypython_2k())

mask = embd != 0
hypergraph = {}
sizes = mask.sum(0)
for j in range(mask.shape[1]):
    if sizes[j] == 0:
        continue
    hypergraph[j] = mask[:, j].nonzero().flatten().tolist()

def decoding(hypergraph, j):
    return [enc.decode([i]) for i in hypergraph[j]]

sorted_hypergraph = sorted(hypergraph, key=lambda j: len(hypergraph[j]))
out_path = os.path.join(os.path.dirname(__file__), "hypergraph_output.txt")
with open(out_path, "w") as f:
    for dim in sorted_hypergraph:
        if 5 <= len(hypergraph[dim]) <= 40:
            print(dim, len(hypergraph[dim]), decoding(hypergraph, dim), file=f)
