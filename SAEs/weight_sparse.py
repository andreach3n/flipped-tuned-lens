import io
import os
import torch
from tiktoken.load import read_file_cached
from circuit_sparsity.registries import MODEL_BASE_DIR
from tiktoken import Encoding
from circuit_sparsity.tiktoken_ext import tinypython
import networkx as nx
from tabulate import tabulate

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


def safe_decode(i):
    try:
        return enc.decode([i])
    except KeyError:
        return None
def decoding(hypergraph, j):
    return [safe_decode(i) for i in hypergraph[j]]

sorted_hypergraph = sorted(hypergraph, key=lambda j: len(hypergraph[j]))
out_path = os.path.join(os.path.dirname(__file__), "hypergraph_output.txt")
with open(out_path, "w") as f:
    for dim in sorted_hypergraph:
        if 5 <= len(hypergraph[dim]) <= 40:
            print(dim, len(hypergraph[dim]), decoding(hypergraph, dim), file=f)

labeled = [
    (888, "graph/vertices"),
    (973, "paths / search"),
    (898, "exceptions"),
    (625, "numeric literals"),
    (66, "loop variables"),
    (272, "capital letters"),
]


def clean_tokens(dim):
    # decoded tokens for a dim: drop undecodable/byte-fragment ones, repr() to keep
    # whitespace on one line, escape | so it can't break the markdown table
    return ", ".join(
        repr(t).replace("|", "\\|")
        for t in decoding(hypergraph, dim)
        if t and "\ufffd" not in t
    )


rows = [(dim, label, clean_tokens(dim)) for dim, label in labeled]
table_md = tabulate(rows, headers=["Dim", "Category", "Tokens"], tablefmt="github")

md_path = os.path.join(os.path.dirname(__file__), "clusters_table.md")
with open(md_path, "w") as f:
    f.write("# Per-dimension token clusters (csp_yolo1 embedding)\n\n")
    f.write(table_md + "\n")


keep = sizes <= 50
filtered = mask[:, keep]
A = filtered.float() @ filtered.float().T
A.fill_diagonal_(0)
# A_thresh = A * (A >= 3)

# removing fragment tokens
bad = [i for i in range(embd.shape[0]) if safe_decode(i) is None or "\ufffd" in safe_decode(i)]

G = nx.from_numpy_array(A.numpy())
G.remove_nodes_from(bad)
communities = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=2.0)
print(len(communities))
print(sorted([len(c) for c in communities], reverse=True))

for c in sorted(communities, key=len, reverse=True)[:7]:
    print(len(c), [enc.decode([i]) for i in c])
