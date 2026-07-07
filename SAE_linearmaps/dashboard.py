"""Neuronpedia-style feature dashboards for the trained SAEs.

For each chosen feature it prints:
  - firing rate + activation histogram (from a small streamed sample)
  - logit effects (which tokens the feature's decoder direction promotes/suppresses)
  - top activating examples with surrounding context

Run this instead of re-running the full 4M eval; feature selection reads the
per-feature stats saved by eval_trivial.py (stats_*.pt).
"""
import glob
import torch as t
from transformers import AutoTokenizer
from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

CACHE_DIR   = "/workspace/sae_cache_layer13"
FULL_PATH   = f"{CACHE_DIR}/sae_full_final.pt"
RESID_PATH  = f"{CACHE_DIR}/sae_resid_final.pt"
P_PATH      = f"{CACHE_DIR}/P.pt"
MODEL_NAME  = "google/gemma-2-2b"
N_TOKENS    = 1_000_000     # tokens to scan for top activations (small -> fast)
TOPK        = 20          # top activating examples to show
WINDOW      = 8           # context tokens on each side
SHOW_LOGITS = False        # load the model's unembedding for logit effects (~5 GB, ~30 s)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
P = t.load(P_PATH, map_location=device)   # (V, 2304) per-token linear prediction

def load_sae(path):
    ckpt = t.load(path, weights_only=False)
    sae = BatchTopKTrainingSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["sae"])
    sae.to(device).eval()
    return sae, ckpt["scale"]

sae_full,  scale_full  = load_sae(FULL_PATH)
sae_resid, scale_resid = load_sae(RESID_PATH)

# Unembedding for logit effects (optional). Direct logit attribution: W_dec[f] @ W_U.
W_U = None
if SHOW_LOGITS:
    from transformer_lens import HookedTransformer
    _m = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, dtype=t.bfloat16).to(device)
    W_U = _m.W_U.float()          # (d_model, d_vocab)
    del _m
    t.cuda.empty_cache()

def hist_text(x, bins=10, width=40):
    """Crude ASCII histogram of a 1-D tensor."""
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        return
    counts = t.histc(x.float(), bins=bins, min=lo, max=hi)
    mx = counts.max().item()
    for b in range(bins):
        edge = lo + (hi - lo) * b / bins
        bar = "#" * (int(width * counts[b].item() / mx) if mx else 0)
        print(f"    {edge:7.2f} | {bar} {int(counts[b].item())}")

def dashboard(sae, scale, mode, feat_ids, n_tokens=N_TOKENS, bs=8192):
    fids = t.tensor(feat_ids, device=device)
    # --- collect the chosen features' activations + token ids over a small sample ---
    acts_parts, toks_parts, seen = [], [], 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)
                tt = tc[start:start+bs].to(device)
                x = (hh - P[tt]) if mode == "resid" else hh
                a = sae.encode(x / scale)[:, fids]       # only the chosen features
                acts_parts.append(a.cpu()); toks_parts.append(tt.cpu())
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    acts = t.cat(acts_parts, dim=0)   # (N, n_feats)
    toks = t.cat(toks_parts, dim=0)   # (N,)

    # --- one dashboard per feature ---
    for j, f in enumerate(feat_ids):
        col = acts[:, j]
        nz = col[col > 0]
        print(f"\n{'=' * 72}\n### {mode} feature {f} ###")
        print(f"firing rate: {nz.numel()}/{col.numel()} ({100 * nz.numel() / max(1, col.numel()):.3f}%)")
        if nz.numel() > 0:
            print(f"activation: min {nz.min():.2f} | median {nz.median():.2f} | "
                  f"p90 {nz.quantile(0.9):.2f} | max {nz.max():.2f}")
            print("  histogram (nonzero activations):")
            hist_text(nz)
        if W_U is not None:
            eff = sae.W_dec[f].float() @ W_U             # (vocab,) logit push per token
            pos = eff.topk(10).indices.tolist()
            neg = (-eff).topk(10).indices.tolist()
            print("  promotes: ", [tokenizer.decode([i]) for i in pos])
            print("  suppresses:", [tokenizer.decode([i]) for i in neg])
        vals, idx = col.topk(min(TOPK, col.numel()))
        print("  top activating examples:")
        for p, v in zip(idx.tolist(), vals.tolist()):
            lo, hi = max(0, p - WINDOW), p + WINDOW + 1
            ctx = tokenizer.decode(toks[lo:hi].tolist())
            focus = tokenizer.decode([toks[p].item()])
            print(f"    [{v:6.1f}] ...{ctx}...  <<{focus!r}>>")

def find_word_feature(sae, scale, mode, words, n_tokens=500_000, topn=3, max_rate=0.10, bs=8192):
    """Find the feature(s) that fire most on a given word (locates the same concept across
    SAEs). Excludes always-on/dense features (firing rate > max_rate) that fire on everything
    and would otherwise dominate the ranking."""
    target = set()
    for w in words:                                   # collect token ids for the surface forms
        target.update(tokenizer(w, add_special_tokens=False).input_ids)
    target = t.tensor(sorted(target), device=device)
    acc = t.zeros(sae.cfg.d_sae, device=device)       # sum activation on target-word positions
    fire = t.zeros(sae.cfg.d_sae, device=device)      # total firings per feature (for rate)
    cnt = 0; seen = 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device)
                tt = tc[start:start+bs].to(device)
                x = (hh - P[tt]) if mode == "resid" else hh
                a = sae.encode(x / scale)              # (b, F)
                fire += (a > 0).sum(0)
                m = t.isin(tt, target)                 # positions sitting on the target word
                if m.any():
                    acc += a[m].sum(0); cnt += int(m.sum())
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    mean_act = acc / max(1, cnt)
    mean_act[(fire / max(1, seen)) > max_rate] = -1e9  # drop always-on features
    top = mean_act.topk(topn)
    print(f"[{mode}] features most active on {words} ({cnt} occurrences): "
          f"{[(i, round(v, 2)) for i, v in zip(top.indices.tolist(), top.values.tolist())]}")
    return top.indices.tolist()

# ------------------------------------------------------------------
# Individual-feature dashboards (uncomment to eyeball specific features):
# s = t.load(f"{CACHE_DIR}/stats_resid.pt")
# band = s["alive"] & (s["freq"] >= 100) & (s["freq"] < 1000) & (s["nd"] == 1)
# dashboard(sae_resid, scale_resid, "resid", band.nonzero().squeeze(1)[:8].tolist())
# dashboard(sae_full, scale_full, "full", find_word_feature(sae_full, scale_full, "full", [" district", " District"]))
# dashboard(sae_full, scale_full, "full", find_word_feature(sae_full, scale_full, "full", [" health"]))

# ================= matched-pairs study =================
# Question: do the FULL SAE's single-token features stay single-token in the RESID SAE?
# For n single-token full features: find each feature's top token -> find the RESID feature
# that detects that same token -> classify it (single-token / multi-word / absent).

def _norm_map():
    """token id -> normalized-word id (collapses case/space variants)."""
    raw = tokenizer.convert_ids_to_tokens(list(range(P.shape[0])))
    d, rows = {}, []
    for s in raw:
        rows.append(d.setdefault(s.replace("▁", "").strip().lower(), len(d)))
    return t.tensor(rows, device=device)

def top_token_of(sae, scale, mode, feat_ids, n_tokens=500_000, bs=8192):
    """Token id of each feature's single strongest activation over a sample."""
    fids = t.tensor(feat_ids, device=device)
    best_val = t.full((len(feat_ids),), -1e9, device=device)
    best_tok = t.zeros(len(feat_ids), dtype=t.long, device=device)
    seen = 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device); tt = tc[start:start+bs].to(device)
                x = (hh - P[tt]) if mode == "resid" else hh
                a = sae.encode(x / scale)[:, fids]           # (b, nf)
                mx, arg = a.max(dim=0)                        # per feature: strongest activation this batch
                upd = mx > best_val
                best_val = t.where(upd, mx, best_val)
                best_tok = t.where(upd, tt[arg], best_tok)   # token id at that position
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    return best_tok

def best_resid_feature_per_word(word_tokid_sets, dense_mask, n_tokens=500_000, bs=8192):
    """One resid pass: per word (a set of token ids), the non-dense feature with highest mean activation."""
    W = len(word_tokid_sets); F = sae_resid.cfg.d_sae
    acc = t.zeros(W, F, device=device); cnt = t.zeros(W, device=device)
    tgts = [t.tensor(sorted(s), device=device) for s in word_tokid_sets]
    seen = 0
    with t.no_grad():
        for lf in sorted(glob.glob(f"{CACHE_DIR}/layer_13_chunk_*.pt")):
            tf = lf.replace("layer_13_chunk_", "tokens_chunk_")
            hc = t.cat(t.load(lf), dim=0); tc = t.cat(t.load(tf), dim=0)
            for start in range(0, hc.shape[0], bs):
                hh = hc[start:start+bs].float().to(device); tt = tc[start:start+bs].to(device)
                a = sae_resid.encode((hh - P[tt]) / scale_resid)   # resid input
                for wi, tg in enumerate(tgts):
                    m = t.isin(tt, tg)
                    if m.any():
                        acc[wi] += a[m].sum(0); cnt[wi] += int(m.sum())
                seen += hh.shape[0]
                if seen >= n_tokens: break
            if seen >= n_tokens: break
    out = []
    for wi in range(W):
        mean_act = acc[wi] / max(1, cnt[wi].item())
        mean_act[dense_mask] = -1e9                          # exclude always-on features
        out.append((int(mean_act.argmax()), float(mean_act.max()), int(cnt[wi].item())))
    return out

def matched_pairs(n=25):
    sf = t.load(f"{CACHE_DIR}/stats_full.pt"); sr = t.load(f"{CACHE_DIR}/stats_resid.pt")
    nm = _norm_map()
    # sample n single-token, alive FULL features (spread across the index range)
    cand = (sf["alive"] & (sf["nd"] == 1)).nonzero().squeeze(1)
    full_feats = cand[t.linspace(0, len(cand) - 1, min(n, len(cand))).long()].tolist()
    # each full feature's top token -> normalized word -> all token-id variants of that word
    toks = top_token_of(sae_full, scale_full, "full", full_feats).tolist()
    words = [tokenizer.decode([tk]).strip().lower() for tk in toks]
    word_ids = [set((nm == nm[tk].item()).nonzero().squeeze(1).tolist()) for tk in toks]
    # resid always-on features to exclude (rate = freq / approx-#tokens, avg L0 = 64)
    freq = sr["freq"].to(device).float()
    dense = (freq / (freq.sum() / 64.0)) > 0.10
    # find the resid counterpart for each word, then classify
    res = best_resid_feature_per_word(word_ids, dense)
    nd_resid = sr["nd"].to(device)
    counts = {"single-token": 0, "multi-word": 0, "absent": 0}
    print(f"\n=== matched pairs: {len(full_feats)} single-token FULL features -> RESID counterpart ===")
    for ff, w, (rf, ract, occ) in zip(full_feats, words, res):
        if occ < 20 or ract < 1.0:                           # resid has no real detector for this word
            cls, rnd = "absent", -1
        else:
            rnd = int(nd_resid[rf]); cls = "single-token" if rnd == 1 else "multi-word"
        counts[cls] += 1
        print(f"  {w!r:18} full#{ff:<6} -> resid#{rf:<6} nd={rnd:>2}  {cls}")
    print(f"\nsummary: {counts}")

matched_pairs(n=25)
