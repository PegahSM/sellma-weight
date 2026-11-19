import json, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
pos = np.array([r["score"] for r in rows if r["label"] == "correct"])
neg = np.array([r["score"] for r in rows if r["label"] == "incorrect"])

def auc(p, n):
    return ((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

rng = np.random.default_rng(42)
boots = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(2000)]
print("AUC", round(auc(pos, neg), 3), "CI95", np.percentile(boots, [2.5, 97.5]).round(3))