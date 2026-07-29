"""Run the Temporal-SAE paper's PROBING PROTOCOL (arXiv:2511.05541, Fig 3) on YOUR sae_lens SAEs.

This faithfully replicates their src/experiments/probing.py:
  - `quiver_arrows` feature selection: top-K features PER CLASS by SIGNED mean(class)-mean(rest),
    then UNIONed across classes,
  - a SINGLE MULTICLASS logistic regression on that union,
  - plain multiclass accuracy (chance = 1/n_classes), NOT one-vs-rest balanced accuracy,
  - their ~3000-train / rest-test split.

The ONLY adaptation (per Nathan) is the feature source: instead of their `load_dictionary` SAE on
raw `h`, we use your sae_lens SAEs and do the skip-embed `h - P[token]` step, so all four variants
(topk / skip-embed x3) are computed correctly. This gives your SAEs in THEIR metric -> directly
comparable to the paper, unlike probe_v2.py (which uses balanced accuracy).

    DATASET=finefineweb python -u probe_paper.py   # the paper's default (10 web domains), fast
    DATASET=mmlu       python -u probe_paper.py    # 57 subjects (slower: bigger feature unions)

GPU (for encoding) + deps:  python -m pip install scikit-learn spacy && python -m spacy download en_core_web_sm
"""
import os
import json
import warnings
import numpy as np
import torch as t
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activations import load_model, HOOK, LAYER, SEQ_LEN
from hf_io import pull, push
from sae_lens import BatchTopKTrainingSAE

K_SAE     = int(os.environ.get("K", 64))
DATASET   = os.environ.get("DATASET", "finefineweb")      # paper default; or "mmlu"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 1500))
CONTEXT_N = int(os.environ.get("CONTEXT_N", 100))         # # texts for the contextual probe
TRAIN_N   = int(os.environ.get("TRAIN_N", 3000))          # their train size; test = the rest
KS        = [1, 5, 10, 20]                                # per-class K in quiver_arrows (their Fig-3 sweep)
OUT_DIR   = os.environ.get("OUT_DIR", "/workspace/out")
SEED      = 0
os.makedirs(OUT_DIR, exist_ok=True)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

METHOD_SPEC = [
    ("raw-h",   "raw residual"),                          # = their baseline_model (probe the dense latents)
    ("full",    "topk SAE"),
    ("resid",   "skip-embed (pretrained skip)"),
    ("hybrid",  "skip-embed (residual encoder)"),
    ("outbias", "skip-embed (raw encoder)"),
]
COLORS = {"raw residual": "#888888", "topk SAE": "#4553c9",
          "skip-embed (pretrained skip)": "#b5762e",
          "skip-embed (residual encoder)": "#2c885f",
          "skip-embed (raw encoder)": "#a0439c"}

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])
    HAVE_SPACY = True
except Exception as e:
    print(f"spaCy unavailable ({e}) -> skipping SYNTACTIC panel"); HAVE_SPACY = False

model = load_model(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)
USE_OFFSETS = HAVE_SPACY and tokenizer.is_fast

# ---- SAEs from HF + rebuilt P tables ----
def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"]); sae.load_state_dict(ckpt["sae"])
    return sae.to(device).eval(), ckpt["scale"], ckpt.get("linear_map")

with t.no_grad():
    embed_table = model.embed(t.arange(model.cfg.d_vocab, device=device)).float()
_lm = t.nn.Linear(2304, 2304).to(device)
_lm.load_state_dict(t.load(pull("linear_map_layer_13.pt"), weights_only=False))
with t.no_grad():
    P = _lm(embed_table)

saes = {}
for stem in ("full", "resid", "hybrid", "outbias"):
    try:
        path = pull(f"sae_{stem}_k{K_SAE}_final.pt")
    except Exception:
        print(f"  {stem}: not on HF -> skipping"); continue
    sae, scale, lmap = load_sae(path)
    Ptab = P if stem == "resid" else None
    if stem == "hybrid":
        _hm = t.nn.Linear(2304, 2304).to(device); _hm.load_state_dict(lmap)
        with t.no_grad(): Ptab = _hm(embed_table)
    saes[stem] = (sae, scale, stem, Ptab)

# ---- dataset -> list of (text, topic_label) ----
def load_probe_data():
    if DATASET == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(N_SAMPLES))
        return [(r["question"], r["subject"]) for r in ds]
    if DATASET == "finefineweb":
        DOMAINS = ["christianity", "law", "literature", "economics", "food",
                   "drama_and_film", "health", "mathematics", "medical", "history"]
        ds = load_dataset("m-a-p/FineFineWeb-test", split="train")
        ds = ds.filter(lambda x: x["domain"] in DOMAINS).shuffle(seed=SEED).select(range(N_SAMPLES))
        return [(r.get("text") or r.get("content") or "", r["domain"]) for r in ds]
    raise ValueError(f"unknown DATASET={DATASET}")

data = load_probe_data()

@t.no_grad()
def encode_text(text):
    kw = dict(truncation=True, max_length=SEQ_LEN)
    if USE_OFFSETS: kw["return_offsets_mapping"] = True
    enc = tokenizer(text, **kw)
    ids = enc["input_ids"]
    if len(ids) < 3: return None
    _, cache = model.run_with_cache(t.tensor([ids], device=device),
                                    names_filter=[HOOK], stop_at_layer=LAYER + 1)
    h    = cache[HOOK][0, 1:].cpu()
    toks = t.tensor(ids[1:])
    pos  = ["X"] * h.shape[0]
    if USE_OFFSETS:
        char_pos = {}
        for w in nlp(text):
            for c in range(w.idx, w.idx + len(w.text)): char_pos[c] = w.pos_
        pos = [char_pos.get(a, "X") for (a, b) in enc["offset_mapping"][1:]]
    return h, toks, pos

topics = sorted({lab for _, lab in data})
topic_to_id = {s: i for i, s in enumerate(topics)}
H_l, TOK_l, TOP_l, SID_l, POS_l = [], [], [], [], []
for i, (text, topic) in enumerate(data):
    out = encode_text(text)
    if out is None: continue
    h, toks, pos = out; n = h.shape[0]
    H_l.append(h); TOK_l.append(toks)
    TOP_l.append(np.full(n, topic_to_id[topic])); SID_l.append(np.full(n, i)); POS_l.extend(pos)
    if i % 200 == 0: print(f"  encoded {i}/{len(data)} texts")

H = t.cat(H_l).float(); TOK = t.cat(TOK_l)
TOPIC = np.concatenate(TOP_l); SID = np.concatenate(SID_l).astype(np.int64); POS = np.array(POS_l)
N = H.shape[0]
print(f"{DATASET}: {N} tokens over {len(data)} texts, {len(topics)} topics")

def _ids(labels, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx.get(x, -1) for x in labels], dtype=np.int64)

_sem_c = sorted(set(TOPIC.tolist()))
label_types = {                                # name -> (int label ids, #classes, token mask)
    "semantic":   (_ids(TOPIC, _sem_c), len(_sem_c), np.ones(N, bool)),
    "contextual": (SID,                 CONTEXT_N,   SID < CONTEXT_N),
}
if USE_OFFSETS:
    valid = POS != "X"
    _syn_c = sorted(set(POS[valid].tolist()))
    label_types["syntactic"] = (_ids(POS, _syn_c), len(_syn_c), valid)

@t.no_grad()
def feature_acts(sae, scale, mode, Ptab, bs=8192):
    outs = []
    for i in range(0, N, bs):
        hh = H[i:i+bs].to(device); tt = TOK[i:i+bs].to(device)
        x = (hh - Ptab[tt]) if mode in ("resid", "hybrid") else hh
        outs.append(sae.encode(x / scale).cpu())
    return t.cat(outs).numpy()                 # numpy (N, F) for sklearn

# ---- the paper's protocol (their src/experiments/probing.py) ----
def quiver_arrows(labels, feats, K):
    """Top-K features PER CLASS by SIGNED mean(class)-mean(rest), UNIONed. Verbatim from their code."""
    top = set()
    for c in np.unique(labels):
        m = labels == c
        diff = feats[m].mean(0) - feats[~m].mean(0)      # SIGNED (their code, not abs)
        top.update(np.argsort(diff)[-K:].tolist())       # features that fire MORE for the class
    return np.array(sorted(top))

def paper_probe(feats, labels, mask, K, rng):
    """quiver_arrows -> one MULTICLASS logistic regression -> multiclass accuracy on the held-out rest."""
    idx = rng.permutation(np.flatnonzero(mask))
    n_tr = min(TRAIN_N, int(0.7 * len(idx)))             # their 3000-train when the pool is big enough
    tr, te = idx[:n_tr], idx[n_tr:]
    sel = quiver_arrows(labels[tr], feats[tr], K)
    if len(sel) == 0 or len(te) == 0: return float("nan")
    clf = LogisticRegression(max_iter=1000)              # their default: plain multiclass LR
    clf.fit(feats[tr][:, sel], labels[tr])
    return float(clf.score(feats[te][:, sel], labels[te]))

def methods_iter():
    yield "raw residual", H.numpy()            # = baseline_model (probe the raw residual latents)
    label_by_stem = dict(METHOD_SPEC)
    for stem, (sae, scale, mode, Ptab) in saes.items():
        yield label_by_stem[stem], feature_acts(sae, scale, mode, Ptab)

results = {}
for mlabel, A in methods_iter():
    results[mlabel] = {}
    for lname, (labels, C, mask) in label_types.items():
        # fresh RandomState(SEED) per call -> identical train/test split for every method and K
        acc = {str(K): paper_probe(A, labels, mask, K, np.random.RandomState(SEED)) for K in KS}
        results[mlabel][lname] = acc
        print(f"{mlabel:32s} {lname:11s} " + "  ".join(f"K{k}={acc[str(k)]:.3f}" for k in KS))
    del A

# ---- plot: grouped bars, multiclass accuracy vs K, chance = 1/n_classes per panel ----
present = [l for _, l in METHOD_SPEC if l in results]
panels  = [p for p in ("semantic", "contextual", "syntactic") if p in next(iter(results.values()))]
n_classes = {p: label_types[p][1] for p in panels}
x = np.arange(len(KS)); width = 0.8 / len(present)
offs = (np.arange(len(present)) - (len(present) - 1) / 2) * width
fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), squeeze=False)
for ax, panel in zip(axes[0], panels):
    for i, m in enumerate(present):
        ys = [results[m][panel][str(k)] for k in KS]
        ax.bar(x + offs[i], ys, width, color=COLORS.get(m, "#333"), label=m, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0 / n_classes[panel], color="#ccc", lw=.8, ls=":")   # chance = 1/n_classes
    ax.set_title(f"{panel}  ({n_classes[panel]} classes)")
    ax.set_xlabel("K  (features per class, quiver_arrows)"); ax.set_ylabel("multiclass accuracy")
    ax.set_xticks(x); ax.set_xticklabels(KS); ax.set_ylim(0, 1.0)
axes[0][0].legend(fontsize=6.5, loc="upper left")
fig.suptitle(f"paper protocol (quiver_arrows + multiclass)  —  Gemma-2-2b L{LAYER}, {DATASET}  ({N} tokens)")
fig.tight_layout()
png = f"{OUT_DIR}/probe_paper_{DATASET}.png"; jsn = f"{OUT_DIR}/probe_paper_{DATASET}.json"
fig.savefig(png, dpi=140, bbox_inches="tight")
with open(jsn, "w") as f: json.dump(results, f, indent=2)
try:
    push(png); push(jsn)
except Exception as e:
    print(f"(not pushed to HF: {e})")
print(f"\nsaved -> {png}  and  {jsn}")
