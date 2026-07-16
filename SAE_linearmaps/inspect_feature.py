"""Ground-truth the judge: show the actual activating examples behind a feature
next to the judge's rating, so you can decide for yourself if the call is right.

Usage:
  python3 inspect_feature.py F0000 F0003 F0004     # by anonymous judge id
  python3 inspect_feature.py --sae hybrid --feat 577   # by real SAE + feature id
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
blind = {b["id"]: b for b in json.load(open(os.path.join(BASE, "judge_blind.json")))}
key = json.load(open(os.path.join(BASE, "judge_key.json")))
rp = os.path.join(BASE, "judge_ratings.json")
ratings = json.load(open(rp)) if os.path.exists(rp) else {}

args = sys.argv[1:]
if "--sae" in args:
    sae = args[args.index("--sae") + 1]
    feat = int(args[args.index("--feat") + 1])
    ids = [aid for aid, m in key.items() if m["sae"] == sae and m["feat"] == feat]
else:
    ids = [a for a in args if a.startswith("F")]

for aid in ids:
    m, b, r = key[aid], blind[aid], ratings.get(aid)
    print("=" * 78)
    print(f"{aid}   {m['sae']} #{m['feat']}   freq_bin {m['freq_bin']}")
    if r:
        print(f"  JUDGE  peak: breadth {r['peak_breadth']} coherence {r['peak_coherence']} "
              f"abstract {r['peak_abstractness']}  |  typical: breadth {r['typical_breadth']} "
              f"coherence {r['typical_coherence']} abstract {r['typical_abstractness']}")
        print(f"  LABEL  \"{r['label']}\"")
        print(f"  WHY    {r['rationale']}")
    print("  --- PEAK (strongest activations) ---")
    for e in b["peak"]:
        print(f"    [{e['act']:6.1f}] {e['text']}")
    print("  --- TYPICAL (random firings) ---")
    for e in b["typical"]:
        print(f"    [{e['act']:6.1f}] {e['text']}")
