import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_texts(split, n):
    rows = json.load(open(f"results/{split}_subset.json", encoding="utf-8"))
    return [r["text"] for r in rows if r["label"] == "correct"][:n]


def mean_loss(model, tok, texts):
    total, count = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt")
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss.item()
        n = enc["input_ids"].shape[1] - 1
        total += loss * n
        count += n
    return total / count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).eval()
    texts = get_texts(args.split, args.n)
    res = {
        "model_id": args.model_id, "split": args.split, "n": len(texts),
        "swedish_loss": round(mean_loss(model, tok, texts), 4),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)