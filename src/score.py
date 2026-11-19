import json, re, sys
import numpy as np

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8")]

def parse(reply):
    m = re.match(r"^\W*(ja|nej|yes|no)\b", reply.strip().lower())
    return m.group(1) if m else None

pos, neg = {"ja", "yes"}, {"nej", "no"}
for r in rows:
    r["ans"] = parse(r["reply"])
    r["valid"] = r["ans"] is not None
    r["pred"] = None if not r["valid"] else ("correct" if r["ans"] in pos else "incorrect")
    r["right"] = r["valid"] and r["pred"] == r["label"]

rng = np.random.default_rng(42)

def boot_ci(x, B=2000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    means = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return tuple(np.percentile(means, [2.5, 97.5]).round(3))

n = len(rows)
valid = [r for r in rows if r["valid"]]
valid_flags = [r["valid"] for r in rows]
acc_flags = [r["right"] for r in valid]
ja_share = [r["pred"] == "correct" for r in valid]
swedish_share = [r["ans"] in {"ja", "nej"} for r in valid]

summary = {
    "file": path, "n": n, "n_valid": len(valid),
    "valid_rate": round(float(np.mean(valid_flags)), 3), "valid_rate_ci95": boot_ci(valid_flags),
    "accuracy_among_valid": round(float(np.mean(acc_flags)), 3) if valid else None,
    "accuracy_ci95": boot_ci(acc_flags),
    "ja_share_among_valid": round(float(np.mean(ja_share)), 3) if valid else None,
    "swedish_answer_share_among_valid": round(float(np.mean(swedish_share)), 3) if valid else None,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
json.dump(summary, open(path.replace(".jsonl", "_summary.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)