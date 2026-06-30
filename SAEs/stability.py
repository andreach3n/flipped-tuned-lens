import json
from itertools import combinations
import random

results = json.load(open("crime_communities.json"))
universe = json.load(open("keep_ids.json"))

communities = {int(s) : set(d["crime_ids"]) for s, d in results.items()}

def jaccard(a, b):
    return (a & b) / (a | b)

def containment(a, b):
    return (a & b) / (min(len(a), len(b)))

for i, j in combinations(range(5), 2):
    a, b = communities[i], communities[j]
    print(f"seeds {i},{j}:  jaccard={jaccard(a, b):.3f}  containment={containment(a, b):.3f}")

# baseline
def null_jaccard(size_a, size_b, trials=1000):
    total = 0.0
    for _ in range(trials):
        a = set(random.sample(universe, size_a))
        b = set(random.sample(universe, size_b))
        total += jaccard(a, b)
    return total / trials

real_scores, null_scores = [], []
for i, j in combinations(range(5), 2):
    a, b = communities[i], communities[j]
    r = jaccard(a, b)
    n = null_jaccard(len(a), len(b))
    real_scores.append(r)
    null_scores.append(n)
    print(f"seeds {i},{j}:  real={r:.3f}  null={n:.3f}  containment={containment(a, b):.3f}")

print(f"\nmean real jaccard = {sum(real_scores)/len(real_scores):.3f}")
print(f"mean null jaccard = {sum(null_scores)/len(null_scores):.3f}")
