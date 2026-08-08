"""Stage 3: detection + fuzzing scoring -> per-latent AUROC -> the 2x2 figure.

Per latent, the scorer LLM sees ONLY the stage-2 explanation (never the peak examples):
  detection: 12 held-out activating examples (unmarked) + 12 never-fires distractors,
             shuffled; rate 0-10 "does this neuron activate somewhere in the example?"
  fuzzing:   12 held-out examples with CORRECT <<marks>> + up to 12 with marks planted on
             non-activating windows; rate 0-10 "are the marked tokens the right ones?"
AUROC over ratings vs ground truth, averaged per cell. Fuzzing is the headline (matches
Heap et al. Fig 1); detection is the secondary. One API call per (latent, task).

  python autointerp_score.py estimate | pilot | submit | collect | analyze

analyze also joins the rubric ratings (RUBRIC_RATINGS + autointerp_key.json, if present)
for the per-latent "high AUROC, zero abstractness" scatter. Runs on the Mac (.env key).
"""
import json
import os
import sys
import time
import random

# this experiment lives in SAE_linearmaps/randomized/ -- the shared modules
# (activations, hf_io, sae_lens loaders) live one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from autointerp_common import (CELLS, cell_name, DETECT_SYSTEM, FUZZ_SYSTEM,
                               numbered_block, ratings_schema, auroc, mean)

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.environ.get("FEATURES_FILE", os.path.join(BASE, "autointerp_features.json"))
EXPLANATIONS = os.path.join(BASE, "autointerp_explanations.json")
MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.6-terra")
REASONING = os.environ.get("REASONING", "low")
MAX_OUT = int(os.environ.get("MAX_OUT", 1200))     # 24 ints + low reasoning; 2500 was overkill
CHUNK = int(os.environ.get("CHUNK", 300))          # batch-queue cap: see autointerp_explain.py
RUN_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autointerp_score_run_state.txt")
SEED = int(os.environ.get("SEED", 0))
SCORE_KEY = os.path.join(BASE, "autointerp_score_key.json")     # custom_id -> true labels
SCORES_OUT = os.path.join(BASE, "autointerp_scores.json")
BATCH_ID_FILE = os.path.join(BASE, "autointerp_score_batch_id.txt")
BATCH_INPUT = os.path.join(BASE, "autointerp_score_batch.jsonl")

PRICES = {"gpt-5.6-sol": (5.00e-6, 30.0e-6), "gpt-5.6-terra": (2.50e-6, 15.0e-6),
          "gpt-5.6-luna": (1.00e-6, 6.0e-6), "gpt-5.4-mini": (0.75e-6, 4.5e-6)}


def load_inputs():
    with open(FEATURES) as f:
        feats = json.load(f)
    with open(EXPLANATIONS) as f:
        expl = json.load(f)
    return feats, expl


def build_requests(feats, expl):
    """Yield (custom_id, system, examples, labels). Examples shuffled per-latent, labels
    stored separately so the scorer output joins back without re-deriving the shuffle."""
    for cell, d in sorted(feats.items()):
        for fid, blocks in sorted(d.items(), key=lambda kv: int(kv[0])):
            cid = f"{cell}|{fid}"
            if cid not in expl:
                continue
            # NOTE: "no clear pattern" explanations are still scored — they should earn
            # ~0.5 AUROC, and that chance-level score is DATA (the random/skip-embed cell
            # is predicted to be full of them), not a case to filter out.
            e = expl[cid]
            rng = random.Random(SEED + int(fid))
            det = [(t_, 1) for t_ in blocks["detect_pos"]] + [(t_, 0) for t_ in blocks["distract"]]
            fuz = [(t_, 1) for t_ in blocks["fuzz_pos"]] + [(t_, 0) for t_ in blocks["fuzz_neg"]]
            rng.shuffle(det)
            rng.shuffle(fuz)
            if len(det) >= 8 and sum(l for _, l in det) and sum(1 - l for _, l in det):
                yield (f"{cid}|detect", DETECT_SYSTEM.format(explanation=e),
                       [t_ for t_, _ in det], [l for _, l in det])
            if len(fuz) >= 8 and sum(l for _, l in fuz) and sum(1 - l for _, l in fuz):
                yield (f"{cid}|fuzz", FUZZ_SYSTEM.format(explanation=e),
                       [t_ for t_, _ in fuz], [l for _, l in fuz])


def req_body(system, examples):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": numbered_block(examples)}],
            "response_format": ratings_schema(len(examples)),
            "max_completion_tokens": MAX_OUT}
    if REASONING != "none":
        body["reasoning_effort"] = REASONING
    return body


def estimate():
    reqs = list(build_requests(*load_inputs()))
    client = OpenAI()
    n_pilot = int(os.environ.get("N_PILOT", 6))
    sample = reqs[:: max(1, len(reqs) // n_pilot)][:n_pilot]
    pin = pout = 0
    for _, system, examples, _ in sample:
        r = client.chat.completions.create(**req_body(system, examples))
        pin += r.usage.prompt_tokens
        pout += r.usage.completion_tokens
    ain, aout = pin / len(sample), pout / len(sample)
    print(f"scoring requests: {len(reqs)}  ({len(reqs)//2} latents x ~2 tasks)")
    print(f"pilot: {ain:.0f} in / {aout:.0f} out tok/req")
    if MODEL in PRICES:
        p_in, p_out = PRICES[MODEL]
        total = len(reqs) * (ain * p_in + aout * p_out)
        print(f"est TOTAL — Batch (50% off): ~${total * 0.5:.2f}   standard: ~${total:.2f}")


def pilot():
    reqs = list(build_requests(*load_inputs()))
    client = OpenAI()
    for cid, system, examples, labels in reqs[:int(os.environ.get("N_PILOT", 4))]:
        r = client.chat.completions.create(**req_body(system, examples))
        ratings = json.loads(r.choices[0].message.content)["ratings"]
        print(f"{cid:40s} AUROC={auroc(labels, ratings):.3f}  (n={len(labels)})")


def submit():
    reqs = list(build_requests(*load_inputs()))
    client = OpenAI()
    keys = {}
    with open(BATCH_INPUT, "w") as fh:
        for cid, system, examples, labels in reqs:
            keys[cid] = labels
            fh.write(json.dumps({"custom_id": cid, "method": "POST",
                                 "url": "/v1/chat/completions",
                                 "body": req_body(system, examples)}, ensure_ascii=False) + "\n")
    with open(SCORE_KEY, "w") as f:
        json.dump(keys, f)
    up = client.files.create(file=open(BATCH_INPUT, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    open(BATCH_ID_FILE, "w").write(batch.id)
    print(f"uploaded {len(reqs)} scoring requests -> batch {batch.id}")


def collect():
    bid = open(BATCH_ID_FILE).read().strip()
    with open(SCORE_KEY) as f:
        keys = json.load(f)
    client = OpenAI()
    while True:
        b = client.batches.retrieve(bid)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"status={b.status}  {b.request_counts}", flush=True)
        time.sleep(30)
    print(f"batch ended: status={b.status}  {b.request_counts}")
    if getattr(b, "errors", None):
        print(f"  batch-level errors: {b.errors}")
    if b.error_file_id:
        errtext = client.files.content(b.error_file_id).text
        print(f"  {len(errtext.strip().splitlines())} requests in the error file; first line:")
        print("  " + errtext.strip().splitlines()[0][:400])
    scores, errs = {}, 0
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            rec = json.loads(line)
            resp = rec.get("response")
            cid = rec["custom_id"]
            if resp and resp.get("status_code") == 200:
                content = resp["body"]["choices"][0]["message"].get("content")
                if content and cid in keys:
                    ratings = json.loads(content)["ratings"]
                    labels = keys[cid]
                    if len(ratings) == len(labels):
                        scores[cid] = {"auroc": auroc(labels, ratings), "n": len(labels)}
                        continue
            errs += 1
    with open(SCORES_OUT, "w") as f:
        json.dump(scores, f)
    print(f"wrote {len(scores)} AUROCs ({errs} unusable) -> {SCORES_OUT}")


def analyze():
    with open(SCORES_OUT) as f:
        scores = json.load(f)
    by_cell = {cell_name(*c): {"fuzz": [], "detect": []} for c in CELLS}
    for cid, s in scores.items():
        cell, _fid, task = cid.rsplit("|", 2)     # "trained/full|123|fuzz"
        if cell in by_cell and s["auroc"] is not None:
            by_cell[cell][task].append(s["auroc"])

    print(f"{'cell':>16} | {'fuzz AUROC':>10} | {'detect AUROC':>12} | n")
    for c in CELLS:
        cell = cell_name(*c)
        fz, dt = mean(by_cell[cell]["fuzz"]), mean(by_cell[cell]["detect"])
        print(f"{cell:>16} | {fz if fz is None else round(fz,3)!s:>10} | "
              f"{dt if dt is None else round(dt,3)!s:>12} | {len(by_cell[cell]['fuzz'])}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#0072B2", "#E69F00", "#1a1a1a", "#666666"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    centers = [0.0, 1.0, 2.5, 3.5]
    for cx, c in zip(centers, CELLS):
        cell = cell_name(*c)
        fz, dt = mean(by_cell[cell]["fuzz"]), mean(by_cell[cell]["detect"])
        if fz is not None:
            ax.bar(cx - 0.17, fz, 0.34, color=BLUE, zorder=3)
            ax.text(cx - 0.17, fz + 0.012, f"{fz:.2f}", ha="center", fontsize=8, color=MUTED)
        if dt is not None:
            ax.bar(cx + 0.17, dt, 0.34, color=ORANGE, zorder=3)
            ax.text(cx + 0.17, dt + 0.012, f"{dt:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.text(3.95, 0.505, "chance", fontsize=8, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(centers)
    ax.set_xticklabels(["plain\n(topk)", "skip-embed\n(resid)", "plain\n(topk)", "skip-embed\n(resid)"],
                       fontsize=9, color=INK)
    for x, lbl in [(0.5, "trained Gemma-2-2b"), (3.0, "re-randomized (incl. emb, seed 0)")]:
        ax.text(x, -0.14, lbl, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("auto-interpretability AUROC", fontsize=9, color=INK)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              ["fuzzing", "detection"], loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title("Explain-then-classify autointerp (following Paulo et al. 2024)",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    out = os.path.join(BASE, "plots", "autointerp_auroc.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def _poll(client, bid):
    while True:
        b = client.batches.retrieve(bid)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b
        print(f"  status={b.status}  {b.request_counts}", flush=True)
        time.sleep(30)


def _harvest(client, b, keys, scores):
    n_new = n_bad = 0
    if getattr(b, "errors", None):
        print(f"  batch-level errors: {b.errors}")
    if b.error_file_id:
        lines = client.files.content(b.error_file_id).text.strip().splitlines()
        print(f"  {len(lines)} requests errored; first: {lines[0][:300]}")
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            rec = json.loads(line)
            resp = rec.get("response")
            cid = rec["custom_id"]
            if resp and resp.get("status_code") == 200:
                content = resp["body"]["choices"][0]["message"].get("content")
                if content and cid in keys:
                    ratings = json.loads(content)["ratings"]
                    labels = keys[cid]
                    if len(ratings) == len(labels):
                        scores[cid] = {"auroc": auroc(labels, ratings), "n": len(labels)}
                        n_new += 1
                        continue
            n_bad += 1
    return n_new, n_bad


def run():
    """Chunked submit+collect under the batch-queue cap. Resumable: rerun after any crash."""
    reqs = list(build_requests(*load_inputs()))
    keys = {cid: labels for cid, _, _, labels in reqs}
    with open(SCORE_KEY, "w") as f:
        json.dump(keys, f)                              # full label sidecar up front (resume needs it)
    client = OpenAI()
    scores = {}
    if os.path.exists(SCORES_OUT):
        with open(SCORES_OUT) as f:
            scores = json.load(f)
        if scores:
            print(f"resuming: {len(scores)} scores already collected")
    if os.path.exists(RUN_STATE):
        bid = open(RUN_STATE).read().strip()
        if bid:
            print(f"polling in-flight batch {bid} from a previous run")
            n_new, n_bad = _harvest(client, _poll(client, bid), keys, scores)
            print(f"  recovered {n_new} results ({n_bad} bad)")
        os.remove(RUN_STATE)

    todo = [(cid, s, e, l) for cid, s, e, l in reqs if cid not in scores]
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    print(f"{len(scores)} done, {len(todo)} to go in {len(chunks)} chunks of <= {CHUNK}")
    for k, ch in enumerate(chunks):
        with open(BATCH_INPUT, "w") as fh:
            for cid, system, examples, _ in ch:
                fh.write(json.dumps({"custom_id": cid, "method": "POST",
                                     "url": "/v1/chat/completions",
                                     "body": req_body(system, examples)}, ensure_ascii=False) + "\n")
        up = client.files.create(file=open(BATCH_INPUT, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
        open(RUN_STATE, "w").write(batch.id)
        print(f"chunk {k + 1}/{len(chunks)}: {len(ch)} requests -> {batch.id}")
        b = _poll(client, batch.id)
        n_new, n_bad = _harvest(client, b, keys, scores)
        with open(SCORES_OUT, "w") as f:
            json.dump(scores, f)                        # incremental save after every chunk
        os.remove(RUN_STATE)
        print(f"  chunk ended: {b.status}; +{n_new} ({n_bad} bad); total {len(scores)}/{len(reqs)}")
        if b.status == "failed":
            print("  chunk FAILED — if token_limit_exceeded, lower CHUNK and rerun `run`")
            return
    print(f"all done: {len(scores)}/{len(reqs)} AUROCs -> {SCORES_OUT}; next: analyze")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "estimate"
    {"estimate": estimate, "pilot": pilot, "submit": submit,
     "collect": collect, "run": run, "analyze": analyze}[cmd]()
