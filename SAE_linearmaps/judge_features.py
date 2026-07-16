"""LLM abstractness judge for the feature-complexity experiment (steps 3-4) — OpenAI.

Reads judge_blind.json (blinded, frequency-matched features), asks an OpenAI model
to rate each feature on three axes — breadth / coherence / abstractness — for BOTH
its peak and typical activation blocks, and writes judge_ratings.json keyed by the
anonymous feature id. The model never sees which SAE a feature came from; join back
to full/hybrid/outbias afterwards via judge_key.json.

Auth: uses the OpenAI SDK's normal credential resolution — OPENAI_API_KEY. No key
is stored in this file. Runs on any machine with internet (no GPU); the blind file
is just text.

Subcommands (run in this order):
  python judge_features.py estimate   # small PAID pilot (~8 calls) -> real cost projection
  python judge_features.py submit     # upload + create the Batch job, save its id
  python judge_features.py collect    # poll, download, write judge_ratings.json

Env knobs: OPENAI_JUDGE_MODEL (default gpt-5.6-terra), EX_PER_BLOCK (examples shown
per block, default 10), REASONING (low|minimal|medium|none, default low), N_PILOT.
"""
import json
import os
import sys
import time

from openai import OpenAI

try:                       # optional: load OPENAI_API_KEY from a .env file if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
BLIND = os.path.join(BASE, "judge_blind.json")
MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.6-terra")
EX_PER_BLOCK = int(os.environ.get("EX_PER_BLOCK", 10))   # examples shown per block (blind file has 12)
REASONING = os.environ.get("REASONING", "low")           # low|minimal|medium|high|none
MAX_OUT = int(os.environ.get("MAX_OUT", 3000))           # cap (reasoning tokens count against this)
# Optional strong-judge robustness run: judge only a seeded, SAE-balanced SUBSET.
# Same SUBSET_SEED -> identical features across model/reasoning configs, so the ratings
# are directly comparable to the full terra/low run. Output files are auto-tagged.
SUBSET_N = int(os.environ.get("SUBSET_N", 0))            # 0 = all; e.g. 300 = 100 per SAE
SUBSET_SEED = int(os.environ.get("SUBSET_SEED", 0))
_sfx = f"_{MODEL.replace('.', '').replace('-', '')}_{REASONING}_sub{SUBSET_N}" if SUBSET_N else ""
RATINGS = os.path.join(BASE, f"judge_ratings{_sfx}.json")
BATCH_ID_FILE = os.path.join(BASE, f"judge_batch_id{_sfx}.txt")
BATCH_INPUT = os.path.join(BASE, f"judge_batch_input{_sfx}.jsonl")

# $/token (input, output); Batch API is 50% off both directions.
PRICES = {
    "gpt-5.6-sol":  (5.00e-6, 30.0e-6),
    "gpt-5.6-terra": (2.50e-6, 15.0e-6),
    "gpt-5.6-luna": (1.00e-6,  6.0e-6),
    "gpt-5.5":      (5.00e-6, 30.0e-6),
    "gpt-5.4":      (2.50e-6, 15.0e-6),
    "gpt-5.4-mini": (0.75e-6,  4.5e-6),
    "gpt-5.4-nano": (0.20e-6,  1.25e-6),
}

SYSTEM = """\
You are an expert interpretability researcher rating features from a sparse autoencoder (SAE) trained on a language model's residual stream. A feature "fires" on certain tokens; you are shown the token it fired on wrapped in 《》, with surrounding context, and the activation strength in brackets.

You get TWO blocks per feature:
- PEAK: the feature's strongest activations (where it fires hardest).
- TYPICAL: a random sample of the feature's firings (its everyday behavior).
A feature can look narrow at its peak but broad in typical firing (or vice versa). Rate the two blocks INDEPENDENTLY, on what each block alone shows.

Rate each block on three axes, each an integer 1-5:

BREADTH — how many distinct words / tokens / contexts the feature responds to.
  1 = a single token or word, one surface form only.
  3 = a handful of related words.
  5 = many varied words across many contexts.

COHERENCE — do the activating contexts share one consistent theme or meaning?
  1 = no shared theme (unrelated hits / looks like noise).
  3 = loosely related.
  5 = a single tightly-unified theme.

ABSTRACTNESS — is the thing detected a surface property or an abstract concept?
  1 = pure lexical / surface: one token regardless of meaning, or spelling / morphology.
  2 = one specific word across its different senses.
  3 = a topic or subject area (e.g. healthcare, elections, sports).
  4 = a semantic relation, role, or frame spanning many different words (e.g. contenders/favorites, negation, cause-and-effect).
  5 = a highly abstract concept (e.g. uncertainty, sycophancy, formality).

Also give `label` (2-6 words naming what the feature detects) and a one-sentence `rationale`. Judge only from the evidence shown; if a block is too sparse to tell, rate conservatively and note it in the rationale."""

# OpenAI strict json_schema: every property required, additionalProperties:false.
_AXIS = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
_PROPS = {f"{blk}_{ax}": _AXIS for blk in ("peak", "typical")
          for ax in ("breadth", "coherence", "abstractness")}
_PROPS["label"] = {"type": "string"}
_PROPS["rationale"] = {"type": "string"}
SCHEMA = {"type": "object", "properties": _PROPS,
          "required": list(_PROPS), "additionalProperties": False}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "feature_rating", "strict": True, "schema": SCHEMA}}


def render(feat, ex=EX_PER_BLOCK):
    """The user-turn text: the peak block then the typical block."""
    def block(name, items):
        lines = [f"{name} activations (fired token in 《》):"]
        for e in items[:ex]:
            lines.append(f"  [{e['act']:.1f}] {e['text']}")
        return "\n".join(lines)
    return block("PEAK", feat["peak"]) + "\n\n" + block("TYPICAL", feat["typical"])


def req_body(feat):
    """The chat.completions body — reused for the pilot call and each batch line."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render(feat)},
        ],
        "response_format": RESPONSE_FORMAT,
        "max_completion_tokens": MAX_OUT,   # reasoning models: NOT max_tokens
    }
    if REASONING != "none":
        body["reasoning_effort"] = REASONING
    return body


def load_features():
    with open(BLIND) as f:
        feats = json.load(f)
    if SUBSET_N:                       # seeded, SAE-balanced subset for the robustness run
        import random
        with open(os.path.join(BASE, "judge_key.json")) as fh:
            key = json.load(fh)
        by_sae = {}
        for x in feats:
            by_sae.setdefault(key[x["id"]]["sae"], []).append(x)
        per = SUBSET_N // len(by_sae)
        rng = random.Random(SUBSET_SEED)
        pick = set()
        for s in sorted(by_sae):
            lst = sorted(by_sae[s], key=lambda x: x["id"])     # deterministic pool
            pick.update(x["id"] for x in rng.sample(lst, min(per, len(lst))))
        feats = [x for x in feats if x["id"] in pick]          # keep blind shuffle order
    return feats


# ------------------------------------------------------------------ estimate
def estimate():
    """OpenAI has no server-side token-count endpoint, and reasoning-token usage is
    hard to predict — so measure it directly with a small real pilot, then project."""
    feats = load_features()
    client = OpenAI()
    n = len(feats)
    n_pilot = int(os.environ.get("N_PILOT", 8))
    sample = feats[:: max(1, n // n_pilot)][:n_pilot]
    pin = pout = 0
    for f in sample:
        r = client.chat.completions.create(**req_body(f))
        pin += r.usage.prompt_tokens
        pout += r.usage.completion_tokens   # includes reasoning tokens
    ain, aout = pin / len(sample), pout / len(sample)

    if MODEL not in PRICES:
        print(f"(no price on file for {MODEL} — showing token counts only)")
        p_in = p_out = None
    else:
        p_in, p_out = PRICES[MODEL]

    print(f"features            : {n}")
    print(f"model               : {MODEL}   reasoning={REASONING}   ex/block={EX_PER_BLOCK}")
    print(f"pilot ({len(sample)} real calls): {ain:.0f} input tok/req, {aout:.0f} output tok/req (incl. reasoning)")
    if p_in is not None:
        batch = (n * ain * p_in + n * aout * p_out) * 0.5
        std = (n * ain * p_in + n * aout * p_out)
        print(f"projected input   : {n * ain / 1e6:.2f}M tok")
        print(f"projected output  : {n * aout / 1e6:.2f}M tok")
        print(f"est TOTAL — Batch (50% off): ~${batch:.2f}")
        print(f"est TOTAL — standard        : ~${std:.2f}")
    print("(pilot itself cost a few cents at standard rates.)")


# -------------------------------------------------------------------- submit
def submit():
    feats = load_features()
    client = OpenAI()
    with open(BATCH_INPUT, "w") as fh:
        for f in feats:
            fh.write(json.dumps({
                "custom_id": f["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": req_body(f),
            }, ensure_ascii=False) + "\n")
    up = client.files.create(file=open(BATCH_INPUT, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    with open(BATCH_ID_FILE, "w") as fh:
        fh.write(batch.id)
    print(f"uploaded {len(feats)} requests -> batch {batch.id} (status {batch.status})")
    print(f"batch id saved to {BATCH_ID_FILE}; run `collect` to poll + retrieve.")


# ------------------------------------------------------------------- collect
def collect():
    with open(BATCH_ID_FILE) as fh:
        bid = fh.read().strip()
    client = OpenAI()
    while True:
        b = client.batches.retrieve(bid)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"status={b.status}  {b.request_counts}", flush=True)
        time.sleep(30)
    print(f"batch ended: status={b.status}  {b.request_counts}")

    out, errs = {}, 0
    if b.output_file_id:
        text = client.files.content(b.output_file_id).text
        for line in text.splitlines():
            rec = json.loads(line)
            resp = rec.get("response")
            if resp and resp.get("status_code") == 200:
                msg = resp["body"]["choices"][0]["message"]
                content = msg.get("content")
                if content:
                    out[rec["custom_id"]] = json.loads(content)
                else:                       # refusal or empty
                    errs += 1
            else:
                errs += 1
    if b.error_file_id:
        n_err = len(client.files.content(b.error_file_id).text.strip().splitlines())
        print(f"  {n_err} requests in the error file")
    with open(RATINGS, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"wrote {len(out)} ratings ({errs} unusable) -> {RATINGS}")


def pilot():
    """Rate a few features and PRINT the full ratings, joined to their true SAE
    (via judge_key.json — for our inspection only; the model call is still blind).
    Use this to eyeball judge quality before committing to the full batch."""
    feats = load_features()
    with open(os.path.join(BASE, "judge_key.json")) as fh:
        key = json.load(fh)
    client = OpenAI()
    n = int(os.environ.get("N_PILOT", 6))
    for f in feats[:n]:
        r = client.chat.completions.create(**req_body(f))
        v = json.loads(r.choices[0].message.content)
        k = key[f["id"]]
        print(f"\n{f['id']}  [{k['sae']} #{k['feat']}  {k['freq_bin']}]  \"{v['label']}\"")
        print(f"  peak    : breadth {v['peak_breadth']}  coherence {v['peak_coherence']}  abstract {v['peak_abstractness']}")
        print(f"  typical : breadth {v['typical_breadth']}  coherence {v['typical_coherence']}  abstract {v['typical_abstractness']}")
        print(f"  → {v['rationale']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "estimate"
    {"estimate": estimate, "pilot": pilot, "submit": submit, "collect": collect}[cmd]()
