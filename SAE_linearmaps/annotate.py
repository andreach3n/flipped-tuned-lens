"""Validation harness for the LLM judge (the step Nathan asked for: hand-label a
handful of features and confirm the judge agrees before trusting the full batch).

Works on a small, SEEDED, SAE-balanced subset of the blind features so your labels
and the judge's labels are on EXACTLY the same features. Blind: while labeling you
never see which SAE a feature came from, nor the judge's rating.

Run on the Mac (no GPU). Three subcommands, in order:
  python3 annotate.py label    # interactive: you rate each feature -> human_val.json (resumable)
  python3 annotate.py judge     # run the NEW-rubric judge on the same features -> judge_val.json
  python3 annotate.py agree     # compare human vs judge: agreement per axis + the incoherence gate

Env knobs:
  VAL_PER_SAE (default 7)  features drawn per SAE  -> total = 3 * VAL_PER_SAE
  VAL_SEED    (default 1)  selection seed (same seed -> same features across runs)
  EX_SHOW     (default 10) examples shown per block while labeling
"""
import json
import os
import random
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
BLIND = os.path.join(BASE, "judge_blind.json")
KEY = os.path.join(BASE, "judge_key.json")
HUMAN = os.path.join(BASE, "human_val.json")
JUDGE = os.path.join(BASE, "judge_val.json")

VAL_PER_SAE = int(os.environ.get("VAL_PER_SAE", 7))
VAL_SEED = int(os.environ.get("VAL_SEED", 1))
EX_SHOW = int(os.environ.get("EX_SHOW", 10))

# breadth is measured in Python (eff_words), not the LLM — so it is not labeled/validated here
AXES = ["coherence", "abstractness"]
BLOCKS = ["peak", "typical"]


def select():
    """Deterministic SAE-balanced subset: same seed -> identical feature ids.
    Returns the list of blind feature dicts (peak/typical), blind order preserved."""
    feats = json.load(open(BLIND))
    key = json.load(open(KEY))
    by_sae = {}
    for f in feats:
        by_sae.setdefault(key[f["id"]]["sae"], []).append(f)
    rng = random.Random(VAL_SEED)
    pick = set()
    for s in sorted(by_sae):
        pool = sorted(by_sae[s], key=lambda x: x["id"])   # deterministic pool order
        pick.update(x["id"] for x in rng.sample(pool, min(VAL_PER_SAE, len(pool))))
    return [f for f in feats if f["id"] in pick]           # keep blind shuffle order


def show_block(name, items):
    print(f"\n  {name} activations (fired token in 《》):")
    for e in items[:EX_SHOW]:
        print(f"    [{e['act']:.1f}] {e['text']}")


# ----------------------------------------------------------------- label (you)
# The FULL anchored rubric — identical to the judge's SYSTEM prompt, so your labels
# and the judge's are scored against the same definitions. Rate coherence FIRST; it
# gates abstractness. Judge from the 《》 token itself: what property of THAT token
# fires? (surrounding words are only context).
RUBRIC = """\
  ── RUBRIC (rate the 《》 token; rate COHERENCE first — it gates abstractness) ──
  SUBWORDS: the 《》 token is often just a PIECE of a word (Cr|umble, Vie|ja). Read the whole word
    and the small construction it completes, not the bare fragment — the shared property may live there
    (umble/Dip/Soup/Cr all = "final piece of a dish name in a recipe title" -> coherent). But don't infer
    a property the token or its word doesn't itself carry (a generic word in a topical paragraph isn't a topic).

  COHERENCE (1-5) — do the 《》 tokens share one consistent property that explains firing?
     1 none: highlighted tokens look unrelated; no property explains them (noise)
     2 weak: a minority share a property; most do not
     3 mixed: a clear property holds for most, with several outliers
     4 strong: nearly all firings share one clear property
     5 exact: every firing has the same clear property, no exceptions

  ABSTRACTNESS (0-5) — given a coherent property, how abstract is it?
     0 no real pattern — USE THIS WHENEVER coherence <= 2 (don't invent a concept from noise)
     1 the exact same word or spelling every time: one specific token or letter-pattern, regardless of
       meaning (always "the"; or always words ending in "-ing")
     2 one word or its close variants: a single word across its senses, or a few synonyms for one thing
       (big / large / huge)
     3 one topic: many different words, but all from the same subject area (healthcare, elections, basketball)
     4 a role or relationship, not a topic: many unrelated words linked by what they DO in the sentence,
       not what they are about (negation words, comparisons, or "an organization being founded" which
       shows up in sports, business, and charities alike)
     5 an abstract idea no word list could capture: spans many topics and roles at once
       (uncertainty, formality, politeness)
     3-vs-4 TEST: look at what the passages are ABOUT. Locked to one subject (sports words never appear
       outside sports) -> 3. Same relation/action/role recurring across many DIFFERENT subjects -> 4.
       A noun you could name a subject -> topic (3); a verb-y action/relation in any subject -> frame (4).

  (type 'r' at any prompt to reprint this rubric · 'skip' to stop and save)"""


def ask_int(prompt, lo, hi):
    while True:
        r = input(prompt).strip().lower()
        if r == "skip":
            return None
        if r == "r":
            print(RUBRIC)
            continue
        if r.isdigit() and lo <= int(r) <= hi:
            return int(r)
        print(f"    (enter an integer {lo}-{hi}, 'r' to reprint the rubric, or 'skip' to quit)")


def label():
    feats = select()
    done = json.load(open(HUMAN)) if os.path.exists(HUMAN) else {}
    todo = [f for f in feats if f["id"] not in done]
    labeled_here = len(feats) - len(todo)   # count within THIS selection, not the whole file
    print(f"{len(feats)} features in the validation set; {labeled_here} already labeled, {len(todo)} to go.")
    print("Rate each block from the evidence ALONE. Type 'skip' at any prompt to stop and save.\n")
    print(RUBRIC)
    for i, f in enumerate(todo, 1):
        print("\n" + "=" * 72)
        print(f"feature {i}/{len(todo)}   (id {f['id']} — SAE hidden)")
        rec = {}
        for b in BLOCKS:
            show_block(b.upper(), f[b])
        print(RUBRIC)
        for b in BLOCKS:
            print(f"\n  -- your ratings for the {b.upper()} block --")
            for a in AXES:
                lo = 0 if a == "abstractness" else 1
                v = ask_int(f"    {a} ({lo}-5): ", lo, 5)
                if v is None:
                    json.dump(done, open(HUMAN, "w"))
                    print(f"\nsaved {len(done)} labels -> {HUMAN}. Re-run `label` to resume.")
                    return
                rec[f"{b}_{a}"] = v
        done[f["id"]] = rec
        json.dump(done, open(HUMAN, "w"))             # save after every feature
    print(f"\nAll done. {len(done)} labels -> {HUMAN}")


# --------------------------------------------------------------- judge (model)
def judge():
    from judge_features import req_body            # reuse the exact NEW-rubric prompt + schema
    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    feats = select()
    client = OpenAI()
    out = json.load(open(JUDGE)) if os.path.exists(JUDGE) else {}
    todo = [f for f in feats if f["id"] not in out]
    print(f"judging {len(todo)} features (of {len(feats)}) with the new rubric...")
    for i, f in enumerate(todo, 1):
        r = client.chat.completions.create(**req_body(f))
        out[f["id"]] = json.loads(r.choices[0].message.content)
        json.dump(out, open(JUDGE, "w"))
        print(f"  {i}/{len(todo)}  {f['id']}  {out[f['id']].get('label','')!r}")
    print(f"wrote {len(out)} judge ratings -> {JUDGE}")


# --------------------------------------------------------------------- agree
def agree():
    H = json.load(open(HUMAN))
    J = json.load(open(JUDGE))
    ids = sorted(set(H) & set(J))
    if not ids:
        print("no overlap between human_val.json and judge_val.json — run `label` and `judge` first.")
        return
    print(f"comparing {len(ids)} features (human vs judge)\n")

    def corr(a, b):                                  # Pearson, no scipy dependency
        n = len(a)
        if n < 2:
            return float("nan")
        ma, mb = sum(a) / n, sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((x - mb) ** 2 for x in b)
        if va == 0 or vb == 0:
            return float("nan")
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return cov / (va ** 0.5 * vb ** 0.5)

    print(f"{'axis':22} {'exact%':>7} {'within1%':>9} {'meanAbsD':>9} {'corr':>6}")
    for b in BLOCKS:
        for a in AXES:
            hv = [H[i][f"{b}_{a}"] for i in ids]
            jv = [J[i][f"{b}_{a}"] for i in ids]
            exact = 100 * sum(x == y for x, y in zip(hv, jv)) / len(ids)
            near = 100 * sum(abs(x - y) <= 1 for x, y in zip(hv, jv)) / len(ids)
            mad = sum(abs(x - y) for x, y in zip(hv, jv)) / len(ids)
            print(f"{b + '/' + a:22} {exact:6.0f}% {near:8.0f}% {mad:9.2f} {corr(hv, jv):6.2f}")

    # the coherence gate, as a binary: do we agree on which features are "no pattern"?
    print("\nincoherence gate (abstractness == 0):")
    for b in BLOCKS:
        hz = [H[i][f"{b}_abstractness"] == 0 for i in ids]
        jz = [J[i][f"{b}_abstractness"] == 0 for i in ids]
        agree_n = sum(x == y for x, y in zip(hz, jz))
        both = sum(x and y for x, y in zip(hz, jz))
        print(f"  {b:8}  agree {100 * agree_n / len(ids):3.0f}%   "
              f"(you flagged {sum(hz)}, judge flagged {sum(jz)}, both {both})")


def diff():
    """Show the features where you and the judge disagreed MOST on one axis/block,
    with the judge's label + rationale, so we can see if it's a rubric gap or noise.
      python3 annotate.py diff [peak|typical] [abstractness|coherence]
    (defaults: peak abstractness — the axis that matters most)."""
    H = json.load(open(HUMAN))
    J = json.load(open(JUDGE))
    K = json.load(open(KEY))
    b = sys.argv[2] if len(sys.argv) > 2 else "peak"
    a = sys.argv[3] if len(sys.argv) > 3 else "abstractness"
    ids = sorted(set(H) & set(J))
    rows = sorted(((abs(H[i][f"{b}_{a}"] - J[i][f"{b}_{a}"]), i) for i in ids), reverse=True)
    print(f"{b}/{a} disagreements, largest first (Δ = |you − judge|):\n")
    for d, i in rows:
        k, r = K[i], J[i]
        flag = "   <-- disagree" if d >= 2 else ""
        print(f"Δ{d}  {i}  [{k['sae']} #{k['feat']} · {k['freq_bin']}]  "
              f"you={H[i][f'{b}_{a}']}  judge={J[i][f'{b}_{a}']}{flag}")
        print(f"     judge says {r.get('label','')!r}: {r.get('rationale','')}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "label"
    {"label": label, "judge": judge, "agree": agree, "diff": diff}[cmd]()
