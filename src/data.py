import json, random, yaml
from datasets import load_dataset

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ds = load_dataset(cfg["data"]["dataset_id"], cfg["data"]["dataset_config"])
print(ds)                          # check split names and sizes
print(ds["val"].features)          # check column names
print(set(ds["val"]["label"]))     # check exact label values
for r in ds["val"].select(range(5)):
    print(r)

random.seed(cfg["seed"])

def sample(split, n):
    # equal numbers per label, fixed seed
    labels = sorted(set(split["label"]))
    per = n // len(labels)
    out = []
    for lab in labels:
        idx = [i for i, l in enumerate(split["label"]) if l == lab]
        assert len(idx) >= per, f"only {len(idx)} examples with label {lab}"
        out += random.sample(idx, per)
    random.shuffle(out)
    return [dict(split[i], id=i) for i in out]

val = sample(ds["val"], cfg["data"]["n_val"])
test = sample(ds["test"], cfg["data"]["n_test"])
json.dump(val, open("results/val_subset.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(test, open("results/test_subset.json", "w", encoding="utf-8"), ensure_ascii=False)