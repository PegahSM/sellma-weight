import argparse, json, re, torch
from datasets import load_dataset
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForCausalLM

DetectorFactory.seed = 0
ap = argparse.ArgumentParser()
ap.add_argument("--model_id")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--q_col")
ap.add_argument("--a_col")
ap.add_argument("--out")
ap.add_argument("--inspect", action="store_true")
args = ap.parse_args()

ds = load_dataset("liu-nlp/swedish-facts-v1")
split = ds[list(ds.keys())[0]]
if args.inspect:
    print(ds); print(split.features); print(split[0])
    raise SystemExit

split = split.shuffle(seed=42).select(range(min(args.n, len(split))))
tok = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).eval()
MAX_NEW = 64

rows = []
for ex in split:
    q, gold = ex[args.q_col], str(ex[args.a_col])
    if tok.chat_template:
        inputs = tok.apply_chat_template([{"role": "user", "content": q}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True)
    else:
        inputs = tok(q, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1]:]
    reply = tok.decode(new, skip_special_tokens=True).strip()
    try:
        lang = detect(reply) if reply else "empty"
    except Exception:
        lang = "unk"
    toks = re.findall(r"\w+", reply.lower())
    repetitive = len(toks) >= 15 and len(set(toks)) / len(toks) < 0.5
    rows.append({"q": q, "gold": gold, "reply": reply, "lang": lang,
                 "empty": not reply, "repetitive": repetitive,
                 "hit_limit": len(new) >= MAX_NEW,
                 "contains_gold": gold.lower() in reply.lower()})

with open(args.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(rows)
fail = [r["empty"] or r["lang"] != "sv" or r["repetitive"] for r in rows]
ok = [r for r, f in zip(rows, fail) if not f]
print("n", n)
print("empty", sum(r["empty"] for r in rows) / n)
print("not Swedish", sum(r["lang"] != "sv" for r in rows) / n)
print("repetitive", sum(r["repetitive"] for r in rows) / n)
print("hit token limit", sum(r["hit_limit"] for r in rows) / n)
print("any failure", sum(fail) / n)
print("contains gold among non-failure replies",
      (sum(r["contains_gold"] for r in ok) / len(ok)) if ok else None)