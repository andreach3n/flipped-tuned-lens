"""Collect everything the standard-autointerp + rubric runs need, for ONE arm per invocation.

Four passes (PASS env var), in order:

  PASS=freq      (pod, per arm)   count per-feature firings for full+resid -> autointerp_freq.pt
  PASS=sample    (once, HF_REPO=trained repo)  pull BOTH arms' freq files, stratified-sample
                 features so all 4 cells have identical firing-rate marginals -> autointerp_selection.json
  PASS=collect   (pod, per arm)   document-ordered scan; per selected feature gather
                 explain/detect/fuzz/rubric example blocks -> autointerp_examples_{arm}.json
  PASS=finalize  (once, HF_REPO=trained repo)  merge both arms; build the Mac-facing
                 autointerp_features.json + the blinded rubric files (autointerp_blind/key.json)

Arm env (freq/collect): VARIANT, INIT_SEED, HF_REPO, OUT_DIR — as for eval_fvu.py.
Cross-arm env (sample/finalize): TRAINED_REPO, RAND_REPO (defaults below).

WHY DOCUMENT-ORDERED: activation_stream shuffles tokens (right for SAE training, fatal for
context windows) — PASS=collect iterates raw documents via activations._docs/_forward so
windows decode as real text. PASS=freq uses the fast shuffled stream (counts don't care).

Sampling caps (documented deviations from exhaustive collection):
  - peak examples: at most 2 kept per document per feature -> the explain block spans >=10
    distinct documents (delphi also diversifies contexts; also bounds the python loop).
  - reservoir: at most 8 candidate firings per document per feature (slight bias against
    bursty documents; uniform otherwise).
"""
import json
import os
import heapq
import torch as t
from transformers import AutoTokenizer

import sys
# this experiment lives in SAE_linearmaps/randomized/ -- the shared modules
# (activations, hf_io, sae_lens loaders) live one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activations import load_model, activation_stream, _docs, _forward, D_IN, VARIANT, INIT_SEED
from hf_io import push, pull
from autointerp_common import (CELLS, cell_name, stratified_sample, print_plan,
                               render_marked, render_anchor, plant_wrong_marks,
                               split_examples)

PASS         = os.environ.get("PASS", "freq")
MODES        = os.environ.get("MODES", "full,resid").split(",")   # e.g. MODES=full for the cheap pass
K_SAE        = int(os.environ.get("K", 64))
SUFFIX       = os.environ.get("SUFFIX", "")   # tagged runs, e.g. _d73728_100M (see train_sae_res.py)
TRAINED_REPO = os.environ.get("TRAINED_REPO", "andreayhchen/gemma2-2b-linearmap-saes-trained-20m")
RAND_REPO    = os.environ.get("RAND_REPO",    "andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0")
OUT_DIR      = os.environ.get("OUT_DIR", "/workspace/out")
SEED         = int(os.environ.get("SEED", 0))

FREQ_TOKENS  = int(os.environ.get("FREQ_TOKENS", 2_000_000))    # PASS=freq scan length
SCAN_TOKENS  = int(os.environ.get("SCAN_TOKENS", 5_000_000))    # PASS=collect scan length
TARGET_PER_BIN = int(os.environ.get("TARGET_PER_BIN", 60))      # ~250-400 feats/cell over bins
MIN_FIRINGS  = int(os.environ.get("MIN_FIRINGS", 20))           # alive threshold in the freq scan
W            = int(os.environ.get("WINDOW", 16))                # context tokens each side (delphi ~32-token windows)
W_RUBRIC     = 8                                                # the validated rubric format's window
N_EXPLAIN    = int(os.environ.get("N_EXPLAIN", 20))             # peak examples for the explainer
N_SCORE      = int(os.environ.get("N_SCORE", 24))               # held-out reservoir (split 12 detect / 12 fuzz)
N_DISTRACT   = int(os.environ.get("N_DISTRACT", 12))            # detection negatives per feature
N_POOL       = int(os.environ.get("N_POOL", 3000))              # shared random-window pool
K_RUBRIC     = 12                                               # rubric peak/typical block size

os.makedirs(OUT_DIR, exist_ok=True)
ARM = "trained" if VARIANT == "trained" else VARIANT
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"[autointerp_collect] PASS={PASS} VARIANT={VARIANT} INIT_SEED={INIT_SEED} arm={ARM}")

# EVERY artifact carries SUFFIX. Without this a k=32/d73728 run would overwrite the 20M/16k
# run's freq/selection/blind/key files in the SAME repos -- destroying the provenance of the
# very result the new SAEs are being compared against. Default "" reproduces historical names.
FREQ_FILE = f"autointerp_freq{SUFFIX}.pt"
SEL_FILE  = f"autointerp_selection{SUFFIX}.json"


def load_sae(name):
    from sae_lens import BatchTopKTrainingSAE
    ckpt = t.load(pull(name), weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, float(ckpt["scale"])


def load_arm_saes(model):
    """This arm's full+resid SAEs and the frozen P table, exactly as eval_fvu.py builds them."""
    import torch.nn as nn
    saes = {m: load_sae(f"sae_{m}_k{K_SAE}{SUFFIX}_final.pt") for m in MODES}
    lm = nn.Linear(D_IN, D_IN).to(device)
    lm.load_state_dict(t.load(pull("linear_map_layer_13.pt"), weights_only=False))
    lm.eval()
    with t.no_grad():
        P = lm(model.embed(t.arange(model.cfg.d_vocab, device=device)).float())
    return saes, P


def encode_mode(saes, P, h, tok, mode):
    sae, scale = saes[mode]
    x = (h - P[tok]) if mode == "resid" else h
    return sae.encode(x / scale)


# ------------------------------------------------------------------ PASS=freq
def pass_freq():
    model = load_model(device)
    saes, P = load_arm_saes(model)
    counts = {m: t.zeros(saes[m][0].cfg.d_sae, dtype=t.long, device=device) for m in MODES}
    seen = 0
    with t.no_grad():
        for step, (h, tok) in enumerate(activation_stream(model, device, batch=4096,
                                                          seed=SEED, max_tokens=FREQ_TOKENS)):
            for m in MODES:
                counts[m] += (encode_mode(saes, P, h, tok, m) > 0).sum(0)
            seen += h.shape[0]
            if step % 100 == 0:
                print(f"  {seen:,}/{FREQ_TOKENS:,} tokens")
    out = {m: counts[m].cpu() for m in MODES}
    out["n_tokens"] = seen
    path = f"{OUT_DIR}/{FREQ_FILE}"
    t.save(out, path)
    push(path)
    for m in MODES:
        alive = int((out[m] >= MIN_FIRINGS).sum())
        print(f"  {ARM}/{m}: {alive:,} features with >= {MIN_FIRINGS} firings in {seen:,} tokens")


# ---------------------------------------------------------------- PASS=sample
def pass_sample():
    cell_rates = {}
    for arm, repo in (("trained", TRAINED_REPO), ("rand_all", RAND_REPO)):
        fr = t.load(pull(FREQ_FILE, repo=repo), weights_only=False)
        n = fr["n_tokens"]
        for m in MODES:
            c = fr[m]
            ids = (c >= MIN_FIRINGS).nonzero(as_tuple=False).squeeze(1).tolist()
            cell_rates[cell_name(arm, m)] = {int(i): float(c[i]) / n for i in ids}
    selected, plan = stratified_sample(cell_rates, TARGET_PER_BIN, SEED)
    print_plan(plan, CELLS)
    path = f"{OUT_DIR}/{SEL_FILE}"
    with open(path, "w") as f:
        json.dump({"selected": selected, "seed": SEED, "target_per_bin": TARGET_PER_BIN,
                   "min_firings": MIN_FIRINGS}, f)
    push(path)   # lives in HF_REPO — run this pass with HF_REPO = the trained repo


# --------------------------------------------------------------- PASS=collect
class FeatState:
    """Bounded per-feature accumulators for one streaming scan."""
    __slots__ = ("peak", "res", "n_seen", "ctr")

    def __init__(self):
        self.peak = []      # heap of (val, ctr, record)
        self.res = []       # heap of (rand_key, ctr, record); keep LARGEST keys
        self.n_seen = 0
        self.ctr = 0

    def offer_peak(self, val, rec):
        self.ctr += 1
        if len(self.peak) < N_EXPLAIN:
            heapq.heappush(self.peak, (val, self.ctr, rec))
        elif val > self.peak[0][0]:
            heapq.heapreplace(self.peak, (val, self.ctr, rec))

    def offer_res(self, key, rec):
        self.ctr += 1
        if len(self.res) < N_SCORE:
            heapq.heappush(self.res, (key, self.ctr, rec))
        elif key > self.res[0][0]:
            heapq.heapreplace(self.res, (key, self.ctr, rec))


def make_record(tok_ids, acts, p, lo):
    """A window record around global-stream position p (window starts at stream index lo)."""
    return {"ids": tok_ids, "acts": acts, "anchor": p - lo, "pos": p}


def pass_collect():
    sel = json.loads(open(pull(SEL_FILE, repo=TRAINED_REPO)).read())["selected"]
    model = load_model(device)
    saes, P = load_arm_saes(model)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    gen = t.Generator().manual_seed(SEED)

    sel_ids = {m: sorted(sel[cell_name(ARM, m)]) for m in MODES}
    sel_t = {m: t.tensor(sel_ids[m], device=device) for m in MODES}
    state = {m: [FeatState() for _ in sel_ids[m]] for m in MODES}
    # shared random-window pool: (record, {mode: max-act vector over the window})
    pool, pool_ctr = [], 0

    docs, gpos, seen = _docs(SEED + 31), 0, 0
    with t.no_grad():
        while seen < SCAN_TOKENS:
            out = _forward(model, device, next(docs))
            if out is None:
                continue
            h, tok = out
            h = h.float().to(device)
            tokd = tok.to(device)
            T = h.shape[0]
            acts = {m: encode_mode(saes, P, h, tokd, m)[:, sel_t[m]].cpu() for m in MODES}
            tok_l = tok.tolist()

            for m in MODES:
                a = acts[m]
                fired = (a > 0).any(0).nonzero(as_tuple=False).squeeze(1).tolist()
                for j in fired:
                    col = a[:, j]
                    rows = (col > 0).nonzero(as_tuple=False).squeeze(1)
                    st = state[m][j]
                    st.n_seen += int(rows.numel())
                    vals = col[rows]
                    # peak: at most 2 per doc (context diversity + bounded work)
                    for r in vals.topk(min(2, vals.numel())).indices.tolist():
                        p = int(rows[r])
                        lo, hi = max(0, p - W), min(T, p + W + 1)
                        rec = make_record(tok_l[lo:hi], col[lo:hi].tolist(), p + gpos, lo + gpos)
                        st.offer_peak(float(vals[r]), rec)
                    # reservoir: at most 8 random firings per doc
                    take = min(8, vals.numel())
                    for r in t.randperm(vals.numel(), generator=gen)[:take].tolist():
                        p = int(rows[r])
                        lo, hi = max(0, p - W), min(T, p + W + 1)
                        rec = make_record(tok_l[lo:hi], col[lo:hi].tolist(), p + gpos, lo + gpos)
                        st.offer_res(float(t.rand((), generator=gen)), rec)

            # distractor pool: 2 random windows per doc, with per-feature max-acts
            for _ in range(2):
                p = int(t.randint(0, T, (), generator=gen))
                lo, hi = max(0, p - W), min(T, p + W + 1)
                maxes = {m: acts[m][lo:hi].max(0).values for m in MODES}
                rec = {"ids": tok_l[lo:hi],
                       "max": {m: maxes[m].numpy().astype("float16") for m in MODES}}
                key = float(t.rand((), generator=gen))
                pool_ctr += 1
                if len(pool) < N_POOL:
                    heapq.heappush(pool, (key, pool_ctr, rec))
                else:
                    heapq.heapreplace(pool, (key, pool_ctr, rec))

            gpos += T
            seen += T
            if seen // 500_000 != (seen - T) // 500_000:
                print(f"  scanned {seen:,}/{SCAN_TOKENS:,} tokens", flush=True)

    emit(sel_ids, state, pool, tokenizer)


def toks(ids, tokenizer):
    return [tokenizer.decode([i]) for i in ids]


def emit(sel_ids, state, pool, tokenizer):
    """Render every block type and write autointerp_examples_{ARM}.json."""
    out = {}
    for m in MODES:
        cell = cell_name(ARM, m)
        out[cell] = {}
        for j, fid in enumerate(sel_ids[m]):
            st = state[m][j]
            peak = sorted(st.peak, key=lambda x: -x[0])
            res = [r for _, _, r in st.res]
            peak_pos = {rec["pos"] for _, _, rec in peak}
            res = [r for r in res if r["pos"] not in peak_pos]        # held-out: disjoint from explain
            detect, fuzz = split_examples(res, N_SCORE // 2, N_SCORE // 2, SEED + fid)

            def marked(rec):
                return render_marked(toks(rec["ids"], tokenizer), rec["acts"])

            def plain(rec):
                return "".join(toks(rec["ids"], tokenizer))

            def rubric(rec):
                tt = toks(rec["ids"], tokenizer)
                a = rec["anchor"]
                lo, hi = max(0, a - W_RUBRIC), min(len(tt), a + W_RUBRIC + 1)
                return render_anchor(tt[lo:hi], a - lo)

            n_marks = max(1, int(sorted(sum(1 for x in r["acts"] if x > 0) for r in fuzz)
                                 [len(fuzz) // 2]) if fuzz else 1)
            # distractors: pool windows where this feature never fires
            silent = [rec for _, _, rec in pool if float(rec["max"][m][j]) == 0.0]
            dis = silent[:N_DISTRACT]
            fuzz_neg = []
            for i, rec in enumerate(silent[N_DISTRACT:N_DISTRACT * 2]):
                txt = plant_wrong_marks(toks(rec["ids"], tokenizer),
                                        [0.0] * len(rec["ids"]), n_marks, SEED + fid + i)
                if txt:
                    fuzz_neg.append(txt)

            out[cell][str(fid)] = {
                "n_firings_scan": st.n_seen,
                "explain":    [{"act": round(v, 2), "text": marked(rec)} for v, _, rec in peak],
                "detect_pos": [plain(r) for r in detect],
                "fuzz_pos":   [marked(r) for r in fuzz],
                "fuzz_neg":   fuzz_neg,
                "distract":   ["".join(toks(rec["ids"], tokenizer)) for rec in dis],
                "rubric_peak":    [{"act": round(v, 2), "text": rubric(rec)} for v, _, rec in peak[:K_RUBRIC]],
                "rubric_typical": [{"act": round(r["acts"][r["anchor"]], 2), "text": rubric(r)}
                                   for r in (detect + fuzz)[:K_RUBRIC]],
            }
        n_thin = sum(1 for f in out[cell].values() if len(f["detect_pos"]) < 6)
        print(f"  {cell}: {len(out[cell])} features emitted ({n_thin} with <6 detection positives)")

    path = f"{OUT_DIR}/autointerp_examples_{ARM}{SUFFIX}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    push(path)


# -------------------------------------------------------------- PASS=finalize
def pass_finalize():
    import random
    merged = {}
    for arm, repo in (("trained", TRAINED_REPO), ("rand_all", RAND_REPO)):
        merged.update(json.loads(open(pull(f"autointerp_examples_{arm}{SUFFIX}.json", repo=repo)).read()))

    feat_path = f"{OUT_DIR}/autointerp_features{SUFFIX}.json"
    with open(feat_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False)

    # blinded rubric files, in judge_features.py's exact input format
    records = [(cell, fid, d) for cell, feats in merged.items() for fid, d in feats.items()
               if d["rubric_peak"]]
    rng = random.Random(SEED)
    rng.shuffle(records)
    blind, key = [], {}
    for i, (cell, fid, d) in enumerate(records):
        aid = f"A{i:04d}"
        blind.append({"id": aid, "peak": d["rubric_peak"], "typical": d["rubric_typical"]})
        key[aid] = {"sae": cell, "feat": int(fid), "freq": d["n_firings_scan"],
                    "freq_bin": "", "nd": None, "nd_peak": None}
    b_path, k_path = f"{OUT_DIR}/autointerp_blind{SUFFIX}.json", f"{OUT_DIR}/autointerp_key{SUFFIX}.json"
    with open(b_path, "w") as f:
        json.dump(blind, f, ensure_ascii=False)
    with open(k_path, "w") as f:
        json.dump(key, f, ensure_ascii=False)
    for p in (feat_path, b_path, k_path):
        push(p)
    print(f"finalized: {len(records)} features across {len(merged)} cells -> pushed to HF")


if __name__ == "__main__":
    {"freq": pass_freq, "sample": pass_sample,
     "collect": pass_collect, "finalize": pass_finalize}[PASS]()
