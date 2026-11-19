import argparse, json, time, yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")        # val or test
ap.add_argument("--n", type=int, default=None)  
ap.add_argument("--model_id", default=None)
ap.add_argument("--revision", default=None) # use first n examples (for timing)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
model_id = args.model_id or cfg["model"]["id"]
revision = args.revision if args.model_id else cfg["model"]["revision"]
revision = revision or None
torch.manual_seed(cfg["seed"])

tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
model.eval()
ja_id = tok.encode("ja", add_special_tokens=False)[0]
nej_id = tok.encode("nej", add_special_tokens=False)[0]
assert ja_id != nej_id
data = json.load(open(f"results/{args.split}_subset.json", encoding="utf-8"))
if args.n:
    data = data[: args.n]

with open(args.out, "w", encoding="utf-8") as f:
    for ex in data:
        prompt = cfg["prompt"].format(text=ex["text"])
        if tok.chat_template:
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                 add_generation_prompt=True, return_tensors="pt", return_dict=True,
         )
        else:
            inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        score = (logits[ja_id] - logits[nej_id]).item()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        sec = round(time.time() - t0, 2)
        rec = {"id": ex["id"], "label": ex["label"], "text": ex["text"], "reply": reply, "seconds": sec, "score": round(score, 4)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(sec, "s |", ex["label"], "|", reply)