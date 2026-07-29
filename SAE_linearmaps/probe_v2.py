"""k-sparse + DENSE probing of SAE features -- GPU/SGD version, standardized (paper-aligned) names.

Kept SEPARATE from probe_sparse.py so it never clobbers the k-sparse outputs. Differences vs v1:
  - PROBES RUN ON GPU: each probe is a tiny PyTorch one-vs-rest logistic regression (a single linear
    layer trained with Adam), fitting ALL classes at once. This replaces the ~3000 separate sklearn/CPU
    fits that made the old version take hours -> now minutes. Same logistic objective, so ~same numbers.
  - Features are STANDARDIZED (per Nathan) so the probe trains cleanly -- this is what fixes the earlier
    "dense < k-sparse" bug (an unscaled probe underfits). Dense should now be a real ceiling (>= sparse).
  - Train tokens subsampled to SUBSAMPLE (same subsample for every method -> fair comparison).
  - adds a DENSE bar (all features); standardized names; switchable dataset; new output names.

Metric: one-vs-rest BALANCED accuracy averaged over classes (chance 0.5).

    python -u probe_v2.py                     # MMLU, 57 subjects, k-sparse + dense (default)
    DATASET=finefineweb python -u probe_v2.py # FineFineWeb, 10 web domains (paper default)
    DENSE=0 python -u probe_v2.py             # k-sparse only
    SUBSAMPLE=40000 python -u probe_v2.py     # bigger train set -> check the bars are stable

GPU + deps:  python -m pip install spacy && python -m spacy download en_core_web_sm
"""
import os
import json
import numpy as np
import torch as t
import torch.nn.functional as F_nn
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activations import load_model, HOOK, LAYER, SEQ_LEN
from hf_io import pull, push
from sae_lens import BatchTopKTrainingSAE

K_SAE     = int(os.environ.get("K", 64))
DATASET   = os.environ.get("DATASET", "mmlu")             # "mmlu" | "finefineweb"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 1500))        # texts scanned (questions or passages)
CONTEXT_N = int(os.environ.get("CONTEXT_N", 50))          # # texts used for the contextual probe
DENSE     = os.environ.get("DENSE", "1") == "1"           # include the all-features (dense) probe
SUBSAMPLE = int(os.environ.get("SUBSAMPLE", 15000))       # cap train tokens per probe (same for all methods)
EPOCHS    = int(os.environ.get("EPOCHS", 300))            # Adam steps per probe (full-batch)
PROBE_LR  = float(os.environ.get("PROBE_LR", 0.05))
PROBE_WD  = float(os.environ.get("PROBE_WD", 1e-3))       # L2 on the probe weights (regularization)
KS        = [1, 5, 10, 20]                                # sparse probe budgets (+ a "dense" bar)
OUT_DIR   = os.environ.get("OUT_DIR", "/workspace/out")
SEED      = 0
os.makedirs(OUT_DIR, exist_ok=True)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# standardized terminology: (HF filename stem, display label). raw-h has no SAE file.
METHOD_SPEC = [
    ("raw-h",   "raw residual"),
    ("full",    "topk SAE"),
    ("resid",   "skip-embed (pretrained skip)"),
    ("hybrid",  "skip-embed (residual encoder)"),
    ("outbias", "skip-embed (raw encoder)"),
]
COLORS = {"raw residual": "#888888", "topk SAE": "#4553c9",
          "skip-embed (pretrained skip)": "#b5762e",
          "skip-embed (residual encoder)": "#2c885f",
          "skip-embed (raw encoder)": "#a0439c"}

# spaCy optional -> syntactic panel
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

saes = {}   # stem -> (sae, scale, mode, P_table)
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
    h    = cache[HOOK][0, 1:].cpu()            # (T-1, 2304), drop BOS
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

H = t.cat(H_l).float(); TOK = t.cat(TOK_l)     # H stays a CPU tensor (also the raw-residual "features")
TOPIC = np.concatenate(TOP_l); SID = np.concatenate(SID_l).astype(np.int64); POS = np.array(POS_l)
N = H.shape[0]
print(f"{DATASET}: {N} tokens over {len(data)} texts, {len(topics)} topics")

rng = np.random.default_rng(SEED)
is_test = rng.random(N) < 0.2

def _ids(labels, classes):                     # map arbitrary labels -> contiguous ints [0, C)
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
    return t.cat(outs)                         # CPU tensor (N, F)

def probe_gpu(A, label_ids, C, tr, te):
    """One-vs-rest logistic-regression probe on GPU, ALL classes batched. Returns {k: acc, 'dense': acc}.
    A: CPU float tensor (N, F). label_ids: int (N,) in [0,C). Metric: mean per-class balanced accuracy."""
    tri = np.flatnonzero(tr)
    if len(tri) > SUBSAMPLE:
        tri = np.random.default_rng(SEED).choice(tri, SUBSAMPLE, replace=False)   # same subsample for all methods
    tei = np.flatnonzero(te)
    Xtr = A[tri].float().to(device); Xte = A[tei].float().to(device)              # (n, F)
    ytr = t.as_tensor(label_ids[tri], device=device)
    yte = t.as_tensor(label_ids[tei], device=device)
    cols = t.arange(C, device=device)
    Ytr = (ytr[:, None] == cols).float()                                          # (n_tr, C) one-vs-rest targets
    Yte = (yte[:, None] == cols).float()
    n_pos = Ytr.sum(0); n_neg = Xtr.shape[0] - n_pos
    pos_w = (n_neg / n_pos.clamp_min(1)).clamp(max=1e4)                           # balance the BCE per class

    # per-class top-k ranking by |mean(pos) - mean(neg)| on RAW train features
    cls_sum = Ytr.T @ Xtr                                                         # (C, F)
    mean_pos = cls_sum / n_pos.clamp_min(1)[:, None]
    mean_neg = (Xtr.sum(0, keepdim=True) - cls_sum) / n_neg.clamp_min(1)[:, None]
    order = (mean_pos - mean_neg).abs().argsort(dim=1, descending=True)           # (C, F)

    # standardize features (fit on train); leave near-constant features unscaled
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True)
    sd = t.where(sd < 1e-6, t.ones_like(sd), sd)
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Fdim = Xtr.shape[1]

    def fit_eval(mask):
        W = t.zeros(Fdim, C, device=device, requires_grad=True)
        b = t.zeros(C, device=device, requires_grad=True)
        opt = t.optim.Adam([W, b], lr=PROBE_LR, weight_decay=PROBE_WD)
        for _ in range(EPOCHS):
            logits = Xtr @ (W * mask if mask is not None else W) + b
            loss = F_nn.binary_cross_entropy_with_logits(logits, Ytr, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        with t.no_grad():
            pred = (Xte @ (W * mask if mask is not None else W) + b) > 0          # (n_te, C)
        Yb = Yte.bool()
        tpr = (pred & Yb).sum(0).float() / Yb.sum(0).clamp_min(1)
        tnr = (~pred & ~Yb).sum(0).float() / (~Yb).sum(0).clamp_min(1)
        bal = 0.5 * (tpr + tnr)
        keep = (Yb.sum(0) >= 2) & ((~Yb).sum(0) >= 5)                             # classes with enough test data
        return bal[keep].mean().item() if keep.any() else float("nan")

    accs = {}
    for k in KS:
        mask = t.zeros(Fdim, C, device=device)
        mask.scatter_(0, order[:, :k].T.contiguous(), 1.0)                        # top-k features per class
        accs[str(k)] = fit_eval(mask)
    accs["dense"] = fit_eval(None) if DENSE else float("nan")
    return accs

def methods_iter():
    yield "raw residual", H                    # H is already the raw-residual feature tensor
    label_by_stem = dict(METHOD_SPEC)
    for stem, (sae, scale, mode, Ptab) in saes.items():
        yield label_by_stem[stem], feature_acts(sae, scale, mode, Ptab)

results = {}
for mlabel, A in methods_iter():
    results[mlabel] = {}
    for lname, (label_ids, C, mask) in label_types.items():
        acc = probe_gpu(A, label_ids, C, (~is_test) & mask, is_test & mask)
        results[mlabel][lname] = acc
        cols_p = [*map(str, KS)] + (["dense"] if DENSE else [])
        print(f"{mlabel:32s} {lname:11s} " + "  ".join(f"{c}={acc[c]:.3f}" for c in cols_p))
    del A

# ---- plot: grouped bars, x = [1,5,10,20,(dense)], one bar per method ----
xt = [str(k) for k in KS] + (["dense"] if DENSE else [])
present = [l for _, l in METHOD_SPEC if l in results]
panels  = [p for p in ("semantic", "contextual", "syntactic") if p in next(iter(results.values()))]
x = np.arange(len(xt)); width = 0.8 / len(present)
offs = (np.arange(len(present)) - (len(present) - 1) / 2) * width
fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), squeeze=False)
for ax, panel in zip(axes[0], panels):
    for i, m in enumerate(present):
        ys = [results[m][panel].get(k, np.nan) for k in xt]
        ax.bar(x + offs[i], ys, width, color=COLORS.get(m, "#333"), label=m, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="#ccc", lw=.8, ls=":")
    ax.set_title(panel); ax.set_xlabel("probe sparsity  ('dense' = all features)")
    ax.set_ylabel("balanced accuracy"); ax.set_xticks(x); ax.set_xticklabels(xt); ax.set_ylim(0.5, 1.0)
axes[0][0].legend(fontsize=6.5, loc="lower left")
fig.suptitle(f"k-sparse + dense probing (GPU)  —  Gemma-2-2b L{LAYER}, {DATASET}  ({N} tokens, {SUBSAMPLE} train)")
fig.tight_layout()
tag = f"{DATASET}_sub{SUBSAMPLE}"                     # subsample in the name -> 15k and 40k runs stay separate
png = f"{OUT_DIR}/probe_v2_{tag}.png"; jsn = f"{OUT_DIR}/probe_v2_{tag}.json"
fig.savefig(png, dpi=140, bbox_inches="tight")
with open(jsn, "w") as f: json.dump(results, f, indent=2)
try:
    push(png); push(jsn)
except Exception as e:
    print(f"(not pushed to HF: {e})")
print(f"\nsaved -> {png}  and  {jsn}")
