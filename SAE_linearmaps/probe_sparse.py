"""k-sparse probing of SAE features -- the Figure 3 check from arXiv:2511.05541 ("Temporal SAEs").

Question: how well can a k-SPARSE logistic probe recover a token's label using only k of a method's
features? We compare the raw layer-13 residual and the four SAEs (full/resid/hybrid/outbias) across
three label types, following Kantamneni et al. / Gurnee "Finding Neurons in a Haystack":

    semantic   = MMLU subject         (e.g. "high_school_european_history")  <- the money panel
    contextual = which question a token came from (sequence-clustering proxy)
    syntactic  = part-of-speech tag    (spaCy)                               <- the control panel

Procedure per (method, label class, k):  select the top-k features by |mean(pos) - mean(neg)| on the
TRAIN split, fit logistic regression on just those k, report BALANCED accuracy on TEST. Average over
classes. Sweep k in {1,5,10,20}.

Interpretation for THIS project: if hybrid/outbias recover SEMANTIC info at LOWER k than full while
matching on SYNTACTIC, that's functional evidence the de-trivialized features are more semantic --
the probing counterpart to the word-count / LLM-judge results.

Needs a GPU (encode pass) plus, for the syntactic panel:
    python -m pip install scikit-learn spacy && python -m spacy download en_core_web_sm
Run: python probe_sparse.py    (env: K, N_QUESTIONS, CONTEXT_Q, OUT_DIR)
"""
import os
import json
import numpy as np
import torch as t
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activations import load_model, HOOK, LAYER, SEQ_LEN
from hf_io import pull, push
from sae_lens import BatchTopKTrainingSAE

K_SAE       = int(os.environ.get("K", 64))
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", 1500))    # MMLU questions to scan
CONTEXT_Q   = int(os.environ.get("CONTEXT_Q", 50))        # # questions used for the contextual probe
KS          = [1, 5, 10, 20]                              # probe sparsities (x-axis)
OUT_DIR     = os.environ.get("OUT_DIR", "/workspace/out")
SEED        = 0
os.makedirs(OUT_DIR, exist_ok=True)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# spaCy is optional; the syntactic panel is skipped without it (semantic + contextual still run)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])
    HAVE_SPACY = True
except Exception as e:
    print(f"spaCy unavailable ({e}) -> skipping the SYNTACTIC panel")
    HAVE_SPACY = False

model = load_model(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)
USE_OFFSETS = HAVE_SPACY and tokenizer.is_fast   # POS alignment needs char offsets (fast tokenizer)

# ---- pull SAEs from HF + rebuild the P tables (same construction as eval_trivial.py) ----
def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"]); sae.load_state_dict(ckpt["sae"])
    return sae.to(device).eval(), ckpt["scale"], ckpt.get("linear_map")

with t.no_grad():
    embed_table = model.embed(t.arange(model.cfg.d_vocab, device=device)).float()
_lm = nn.Linear(2304, 2304).to(device)
_lm.load_state_dict(t.load(pull("linear_map_layer_13.pt"), weights_only=False))
with t.no_grad():
    P = _lm(embed_table)                          # frozen map table -> resid encoder

saes = {}   # name -> (sae, scale, mode, P_table or None)
for name in ("full", "resid", "hybrid", "outbias"):
    try:
        path = pull(f"sae_{name}_k{K_SAE}_final.pt")
    except Exception:
        print(f"  {name}: not on HF -> skipping"); continue
    sae, scale, lmap = load_sae(path)
    Ptab = P if name == "resid" else None
    if name == "hybrid":                          # rebuild P_hybrid from the ckpt's jointly-trained map
        _hm = nn.Linear(2304, 2304).to(device); _hm.load_state_dict(lmap)
        with t.no_grad(): Ptab = _hm(embed_table)
    saes[name] = (sae, scale, name, Ptab)

# ---- collect per-token layer-13 activations + labels over MMLU ----
ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(N_QUESTIONS))
subjects  = sorted(set(ds["subject"]))
subj_to_id = {s: i for i, s in enumerate(subjects)}

@t.no_grad()
def encode_question(text):
    kw = dict(truncation=True, max_length=SEQ_LEN)
    if USE_OFFSETS: kw["return_offsets_mapping"] = True
    enc = tokenizer(text, **kw)
    ids = enc["input_ids"]
    _, cache = model.run_with_cache(t.tensor([ids], device=device),
                                    names_filter=[HOOK], stop_at_layer=LAYER + 1)
    h    = cache[HOOK][0, 1:].cpu()               # (T-1, 2304), drop BOS
    toks = t.tensor(ids[1:])                       # (T-1,) token ids for P[token] indexing
    pos  = ["X"] * h.shape[0]
    if USE_OFFSETS:                                # per-token POS via char-span alignment with spaCy
        char_pos = {}
        for w in nlp(text):
            for c in range(w.idx, w.idx + len(w.text)): char_pos[c] = w.pos_
        pos = [char_pos.get(a, "X") for (a, b) in enc["offset_mapping"][1:]]
    return h, toks, pos

H_l, TOK_l, SUBJ_l, QID_l, POS_l = [], [], [], [], []
for qi in range(len(ds)):
    row = ds[qi]
    h, toks, pos = encode_question(row["question"])
    n = h.shape[0]
    if n == 0: continue
    H_l.append(h); TOK_l.append(toks)
    SUBJ_l.append(np.full(n, subj_to_id[row["subject"]]))
    QID_l.append(np.full(n, qi)); POS_l.extend(pos)
    if qi % 200 == 0: print(f"  encoded {qi}/{len(ds)} questions")

H   = t.cat(H_l).float()                           # (N, 2304) on CPU
TOK = t.cat(TOK_l)                                 # (N,)
SUBJ = np.concatenate(SUBJ_l); QID = np.concatenate(QID_l); POS = np.array(POS_l)
N = H.shape[0]
print(f"collected {N} tokens over {len(ds)} questions")

rng = np.random.default_rng(SEED)                  # token-level 80/20 split (shared across methods)
is_test = rng.random(N) < 0.2

# label type -> (per-token labels, classes to probe, token mask)
label_types = {
    "semantic":   (SUBJ, sorted(set(SUBJ.tolist())),      np.ones(N, bool)),
    "contextual": (QID,  list(range(CONTEXT_Q)),          QID < CONTEXT_Q),
}
if USE_OFFSETS:
    valid = POS != "X"
    label_types["syntactic"] = (POS, sorted(set(POS[valid].tolist())), valid)

# ---- one method's feature-activation matrix (N, F) ----
@t.no_grad()
def feature_acts(sae, scale, mode, Ptab, bs=8192):
    outs = []
    for i in range(0, N, bs):
        hh = H[i:i+bs].to(device); tt = TOK[i:i+bs].to(device)
        x = (hh - Ptab[tt]) if mode in ("resid", "hybrid") else hh
        outs.append(sae.encode(x / scale).cpu())
    return t.cat(outs).numpy()                     # (N, F)

# ---- k-sparse probe: top-k features by |mean-diff|, logistic regression, balanced accuracy ----
def probe_label(A, labels, classes, tr, te, ks):
    accs = {k: [] for k in ks}
    Atr, Ate = A[tr], A[te]
    for c in classes:
        ytr = labels[tr] == c; yte = labels[te] == c
        if ytr.sum() < 5 or yte.sum() < 2 or (~ytr).sum() < 5:
            continue                               # need enough pos AND neg examples
        diff = np.abs(Atr[ytr].mean(0) - Atr[~ytr].mean(0))   # discriminativeness per feature
        order = np.argsort(-diff)
        for k in ks:
            sel = order[:k]
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(Atr[:, sel], ytr)
            accs[k].append(balanced_accuracy_score(yte, clf.predict(Ate[:, sel])))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in accs.items()}

# process one method at a time to keep only one (N, F) matrix in RAM
def methods():
    yield "raw-h", H.numpy()                       # baseline: probe the raw residual dims directly
    for name, (sae, scale, mode, Ptab) in saes.items():
        yield name, feature_acts(sae, scale, mode, Ptab)

results = {}
for mname, A in methods():
    results[mname] = {}
    for lname, (labels, classes, mask) in label_types.items():
        acc = probe_label(A, labels, classes, (~is_test) & mask, is_test & mask, KS)
        results[mname][lname] = acc
        print(f"{mname:9s} {lname:11s} " + "  ".join(f"k{k}={acc[k]:.3f}" for k in KS))
    del A

# ---- plot: one panel per label type, balanced accuracy vs k, one line per method ----
COLORS = {"raw-h": "#888888", "full": "#4553c9", "resid": "#b5762e",
          "hybrid": "#2c885f", "outbias": "#a0439c"}
panels = [p for p in ("semantic", "contextual", "syntactic") if p in label_types]
fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
for ax, lname in zip(axes[0], panels):
    for mname in results:
        ys = [results[mname][lname][k] for k in KS]
        ax.plot(KS, ys, ("--o" if mname == "raw-h" else "-o"),
                color=COLORS.get(mname, "#333"), label=mname)
    ax.axhline(0.5, color="#ccc", lw=.8, ls=":")   # chance for balanced one-vs-rest accuracy
    ax.set_title(lname); ax.set_xlabel("k (probe sparsity)"); ax.set_ylabel("balanced accuracy")
    ax.set_xticks(KS); ax.set_ylim(0.45, 1.0); ax.legend(fontsize=8)
fig.suptitle(f"k-sparse probing of SAE features  —  MMLU, Gemma-2-2b L{LAYER}  ({N} tokens)")
fig.tight_layout()
out = f"{OUT_DIR}/ksparse_probe.png"
fig.savefig(out, dpi=140, bbox_inches="tight")

with open(f"{OUT_DIR}/ksparse_probe.json", "w") as f:
    json.dump(results, f, indent=2)
push(out); push(f"{OUT_DIR}/ksparse_probe.json")
print(f"\nsaved + pushed -> {out}  and  ksparse_probe.json")
