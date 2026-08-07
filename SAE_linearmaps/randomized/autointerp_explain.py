"""Stage 2 of the standard-autointerp run: generate ONE explanation per latent (OpenAI Batch).

Mirrors judge_features.py's machinery (same auth, same subcommands, same Batch flow):
  python autointerp_explain.py estimate   # paid mini-pilot -> cost projection
  python autointerp_explain.py pilot      # print a few explanations for eyeballing
  python autointerp_explain.py submit     # upload the Batch job
  python autointerp_explain.py collect    # poll + write autointerp_explanations.json

Input: autointerp_features.json (from autointerp_collect.py PASS=finalize; pulled from HF
if missing). Runs on the Mac — OPENAI_API_KEY from the gitignored .env, never on the pod.
Env: OPENAI_JUDGE_MODEL (default gpt-5.6-terra), REASONING (default low), N_PILOT.
"""
import json
import os
import sys
import time

# this experiment lives in SAE_linearmaps/randomized/ -- the shared modules
# (activations, hf_io, sae_lens loaders) live one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from autointerp_common import EXPLAIN_SYSTEM, EXPLANATION_SCHEMA

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.environ.get("FEATURES_FILE", os.path.join(BASE, "autointerp_features.json"))
MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.6-terra")
REASONING = os.environ.get("REASONING", "low")
MAX_OUT = int(os.environ.get("MAX_OUT", 2000))
EXPL_OUT = os.path.join(BASE, "autointerp_explanations.json")
BATCH_ID_FILE = os.path.join(BASE, "autointerp_explain_batch_id.txt")
BATCH_INPUT = os.path.join(BASE, "autointerp_explain_batch.jsonl")

PRICES = {"gpt-5.6-sol": (5.00e-6, 30.0e-6), "gpt-5.6-terra": (2.50e-6, 15.0e-6),
          "gpt-5.6-luna": (1.00e-6, 6.0e-6), "gpt-5.4-mini": (0.75e-6, 4.5e-6)}


def load_features():
    if not os.path.exists(FEATURES):
        from hf_io import pull   # fall back to HF (HF_TOKEN in .env works here too)
        repo = os.environ.get("TRAINED_REPO", "andreayhchen/gemma2-2b-linearmap-saes-trained-20m")
        path = pull("autointerp_features.json", repo=repo)
        with open(path) as f:
            return json.load(f)
    with open(FEATURES) as f:
        return json.load(f)


def requests_iter(feats):
    for cell, d in sorted(feats.items()):
        for fid, blocks in sorted(d.items(), key=lambda kv: int(kv[0])):
            if not blocks["explain"]:
                continue
            lines = [f"[{e['act']}] {e['text']}" for e in blocks["explain"]]
            yield f"{cell}|{fid}", "\n".join(lines)


def req_body(user_text):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": EXPLAIN_SYSTEM},
                         {"role": "user", "content": user_text}],
            "response_format": EXPLANATION_SCHEMA,
            "max_completion_tokens": MAX_OUT}
    if REASONING != "none":
        body["reasoning_effort"] = REASONING
    return body


def estimate():
    reqs = list(requests_iter(load_features()))
    client = OpenAI()
    n_pilot = int(os.environ.get("N_PILOT", 6))
    sample = reqs[:: max(1, len(reqs) // n_pilot)][:n_pilot]
    pin = pout = 0
    for _, text in sample:
        r = client.chat.completions.create(**req_body(text))
        pin += r.usage.prompt_tokens
        pout += r.usage.completion_tokens
    ain, aout = pin / len(sample), pout / len(sample)
    print(f"latents: {len(reqs)}   model: {MODEL} reasoning={REASONING}")
    print(f"pilot: {ain:.0f} in / {aout:.0f} out tok/req")
    if MODEL in PRICES:
        p_in, p_out = PRICES[MODEL]
        total = len(reqs) * (ain * p_in + aout * p_out)
        print(f"est TOTAL — Batch (50% off): ~${total * 0.5:.2f}   standard: ~${total:.2f}")


def pilot():
    reqs = list(requests_iter(load_features()))
    client = OpenAI()
    for cid, text in reqs[:int(os.environ.get("N_PILOT", 6))]:
        r = client.chat.completions.create(**req_body(text))
        v = json.loads(r.choices[0].message.content)
        print(f"{cid:30s} -> {v['explanation']}")


def submit():
    reqs = list(requests_iter(load_features()))
    client = OpenAI()
    with open(BATCH_INPUT, "w") as fh:
        for cid, text in reqs:
            fh.write(json.dumps({"custom_id": cid, "method": "POST",
                                 "url": "/v1/chat/completions",
                                 "body": req_body(text)}, ensure_ascii=False) + "\n")
    up = client.files.create(file=open(BATCH_INPUT, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    open(BATCH_ID_FILE, "w").write(batch.id)
    print(f"uploaded {len(reqs)} explanation requests -> batch {batch.id}")


def collect():
    bid = open(BATCH_ID_FILE).read().strip()
    client = OpenAI()
    while True:
        b = client.batches.retrieve(bid)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        print(f"status={b.status}  {b.request_counts}", flush=True)
        time.sleep(30)
    out, errs = {}, 0
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            rec = json.loads(line)
            resp = rec.get("response")
            if resp and resp.get("status_code") == 200:
                content = resp["body"]["choices"][0]["message"].get("content")
                if content:
                    out[rec["custom_id"]] = json.loads(content)["explanation"]
                    continue
            errs += 1
    with open(EXPL_OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {len(out)} explanations ({errs} unusable) -> {EXPL_OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "estimate"
    {"estimate": estimate, "pilot": pilot, "submit": submit, "collect": collect}[cmd]()
