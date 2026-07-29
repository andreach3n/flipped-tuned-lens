"""k-sparse + DENSE probing of SAE features, with standardized (paper-aligned) method names.

v2 of probe_sparse.py -- kept as a SEPARATE file so it never clobbers the k-sparse outputs:
  - adds a DENSE probe (logistic regression on ALL features) as a "dense" ceiling bar per method,
  - relabels methods with the standardized terminology (topk SAE / skip-embed variants),
  - switchable dataset: MMLU `subject` (default) or FineFineWeb `domain` (matches the T-SAE paper),
  - writes to NEW names: probe_v2_<dataset>.{json,png}  (old ksparse_probe.* untouched).

Metric matches probe_sparse.py -- one-vs-rest BALANCED accuracy averaged over classes (chance 0.5),
so the dense bar sits on the same axis as the k=1..20 bars.

    python -u probe_v2.py                     # MMLU, 57 subjects, k-sparse + dense (default)
    DATASET=finefineweb python -u probe_v2.py # FineFineWeb, 10 web domains (paper default)
    DENSE=0 python -u probe_v2.py             # k-sparse only (fastest)
    SUBSAMPLE=40000 python -u probe_v2.py     # bigger dense train set -> check the dense bars are stable

GPU + deps:  python -m pip install scikit-learn scipy spacy && python -m spacy download en_core_web_sm
"""
import os
import json
import numpy as np
import torch as t
import torch.nn as nn
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # probes hit the iter cap; harmless
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
SUBSAMPLE = int(os.environ.get("SUBSAMPLE", 15000))       # cap train tokens for the dense fit (speed)
DENSE_ITER = int(os.environ.get("DENSE_ITER", 300))       # max_iter for the dense fit
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

# ---- SAEs from HF + rebuilt P tables (same as eval_trivial / probe_sparse) ----
def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"]); sae.load_state_dict(ckpt["sae"])
    return sae.to(device).eval(), ckpt["scale"], ckpt.get("linear_map")

with t.no_grad():
    embed_table = model.embed(t.arange(model.cfg.d_vocab, device=device)).float()
_lm = nn.Linear(2304, 2304).to(device)
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
        _hm = nn.Linear(2304, 2304).to(device); _hm.load_state_dict(lmap)
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
    if USE_OFFSETS:                            # per-token POS via char-span alignment
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
TOPIC = np.concatenate(TOP_l); SID = np.concatenate(SID_l); POS = np.array(POS_l)
N = H.shape[0]
print(f"{DATASET}: {N} tokens over {len(data)} texts, {len(topics)} topics")

rng = np.random.default_rng(SEED)
is_test = rng.random(N) < 0.2
label_types = {
    "semantic":   (TOPIC, sorted(set(TOPIC.tolist())), np.ones(N, bool)),
    "contextual": (SID,   list(range(CONTEXT_N)),      SID < CONTEXT_N),
}
if USE_OFFSETS:
    valid = POS != "X"
    label_types["syntactic"] = (POS, sorted(set(POS[valid].tolist())), valid)

@t.no_grad()
def feature_acts(sae, scale, mode, Ptab, bs=8192):
    outs = []
    for i in range(0, N, bs):
        hh = H[i:i+bs].to(device); tt = TOK[i:i+bs].to(device)
        x = (hh - Ptab[tt]) if mode in ("resid", "hybrid") else hh
        outs.append(sae.encode(x / scale).cpu())
    return t.cat(outs).numpy()

def probe_sparse_k(A, labels, classes, tr, te):
    accs = {k: [] for k in KS}
    Atr, Ate = A[tr], A[te]
    for c in classes:
        ytr = labels[tr] == c; yte = labels[te] == c
        if ytr.sum() < 5 or yte.sum() < 2 or (~ytr).sum() < 5: continue
        order = np.argsort(-np.abs(Atr[ytr].mean(0) - Atr[~ytr].mean(0)))
        for k in KS:
            sel = order[:k]
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(Atr[:, sel], ytr)
            accs[k].append(balanced_accuracy_score(yte, clf.predict(Ate[:, sel])))
    return {str(k): (float(np.mean(v)) if v else float("nan")) for k, v in accs.items()}

def probe_dense(A, labels, classes, tr, te):
    # DENSE = logistic regression on ALL features (the decodability ceiling). Two speedups keep it
    # from taking hours: (1) SAE features are ~99.6% zero -> a sparse matrix; (2) subsample the TRAIN
    # tokens to SUBSAMPLE. Subsampling is safe here: the ~83k tokens come from only ~1.5k questions
    # (heavily correlated -> effective n is far smaller, learning curve is flat past ~15k), and every
    # method uses the SAME subsample (fixed seed + method-independent train indices), so the RELATIVE
    # comparison is unbiased. Bump SUBSAMPLE to confirm the bars barely move. Test set stays full.
    Am = csr_matrix(A) if A.shape[1] > 3000 else A
    tri = np.flatnonzero(tr)
    if len(tri) > SUBSAMPLE:
        tri = np.random.default_rng(SEED).choice(tri, SUBSAMPLE, replace=False)   # same across methods
    tei = np.flatnonzero(te)
    Atr, Ate = Am[tri], Am[tei]
    ytr_all, yte_all = labels[tri], labels[tei]
    accs = []
    for c in classes:
        ytr, yte = ytr_all == c, yte_all == c
        if ytr.sum() < 5 or yte.sum() < 2 or (~ytr).sum() < 5: continue
        clf = LogisticRegression(max_iter=DENSE_ITER, class_weight="balanced", solver="liblinear")
        clf.fit(Atr, ytr)
        accs.append(balanced_accuracy_score(yte, clf.predict(Ate)))
    return float(np.mean(accs)) if accs else float("nan")

def methods_iter():
    yield "raw residual", H.numpy()
    label_by_stem = dict(METHOD_SPEC)
    for stem, (sae, scale, mode, Ptab) in saes.items():
        yield label_by_stem[stem], feature_acts(sae, scale, mode, Ptab)

results = {}
for mlabel, A in methods_iter():
    results[mlabel] = {}
    for lname, (labels, classes, mask) in label_types.items():
        tr, te = (~is_test) & mask, is_test & mask
        acc = probe_sparse_k(A, labels, classes, tr, te)
        acc["dense"] = probe_dense(A, labels, classes, tr, te) if DENSE else float("nan")
        results[mlabel][lname] = acc
        cols = [*map(str, KS)] + (["dense"] if DENSE else [])
        print(f"{mlabel:32s} {lname:11s} " + "  ".join(f"{c}={acc[c]:.3f}" for c in cols))
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
fig.suptitle(f"k-sparse + dense probing  —  Gemma-2-2b L{LAYER}, {DATASET}  ({N} tokens)")
fig.tight_layout()
png = f"{OUT_DIR}/probe_v2_{DATASET}.png"; jsn = f"{OUT_DIR}/probe_v2_{DATASET}.json"
fig.savefig(png, dpi=140, bbox_inches="tight")
with open(jsn, "w") as f: json.dump(results, f, indent=2)
try:
    push(png); push(jsn)
except Exception as e:
    print(f"(not pushed to HF: {e})")
print(f"\nsaved -> {png}  and  {jsn}")
