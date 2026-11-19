"""Alternatives to the targeted weight edit (run after edit.py works).

A. prompt changes, weights unchanged
B. inference-time logit bias (subtract lam*b from the ja-minus-nej score; no weights change)
C. supervised fine-tuning of the same two output-embedding rows on val labels (conventional baseline)

Statistics and training use val only. Usage: python src/alternatives.py --split val|test
"""
import argparse, json
import numpy as np, torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from loss import get_texts, mean_loss

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=1e-2)
args = ap.parse_args()

MODEL = "jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk"
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
ja_id = tok.encode("ja", add_special_tokens=False)[0]
nej_id = tok.encode("nej", add_special_tokens=False)[0]

PROMPTS = {
    "original": cfg["prompt"],
    "nej_first": 'Är följande mening grammatiskt korrekt? Svara endast "nej" eller "ja".\n\nMening: {text}',
    "neutral": 'Bedöm meningen. Svara "ja" om den är grammatiskt korrekt och "nej" om den innehåller ett grammatiskt fel.\n\nMening: {text}',
}


def load(split):
    return json.load(open(f"results/{split}_subset.json", encoding="utf-8"))


def make_inputs(prompt_tpl, text):
    prompt = prompt_tpl.format(text=text)
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return tok(prompt, return_tensors="pt")


@torch.no_grad()
def scores(rows, prompt_tpl, want_h=False):
    S, H = [], []
    for r in rows:
        out = model(**make_inputs(prompt_tpl, r["text"]), output_hidden_states=want_h)
        lg = out.logits[0, -1]
        S.append((lg[ja_id] - lg[nej_id]).item())
        if want_h:
            H.append(out.hidden_states[-1][0, -1].clone())
    return (np.array(S), torch.stack(H)) if want_h else np.array(S)


def auc(p, n):
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def evaluate(rows, S):
    y = np.array([r["label"] == "correct" for r in rows])
    pred = S > 0
    rng = np.random.default_rng(42)
    accs, aucs = [], []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        accs.append((pred[i] == y[i]).mean())
        if y[i].any() and (~y[i]).any():
            aucs.append(auc(S[i][y[i]], S[i][~y[i]]))
    return {"ja_share": float(pred.mean()),
            "acc": float((pred == y).mean()), "acc_ci": np.percentile(accs, [2.5, 97.5]).round(3).tolist(),
            "auc": round(auc(S[y], S[~y]), 3), "auc_ci": np.percentile(aucs, [2.5, 97.5]).round(3).tolist()}


W = model.get_output_embeddings().weight
W0 = W.detach().clone()
dev, rows = load("val"), load(args.split)
texts = get_texts(args.split, 50)
results = []


def log(name, m, with_loss=False):
    m["condition"] = name
    if with_loss:
        m["swedish_loss"] = round(mean_loss(model, tok, texts), 4)
    results.append(m)
    print(json.dumps(m))


# A. prompt changes (original weights)
for name, tpl in PROMPTS.items():
    log(f"prompt={name}", evaluate(rows, scores(rows, tpl)))

# B. logit bias: shift the decision threshold by the val mean score
S_dev, H = scores(dev, PROMPTS["original"], want_h=True)
b = float(S_dev.mean())
S_test = scores(rows, PROMPTS["original"])
log("logit_bias lam=1.0", evaluate(rows, S_test - b))

# C. supervised tuning of the two rows (logistic regression on cached final hidden states)
y_dev = torch.tensor([r["label"] == "correct" for r in dev], dtype=torch.float32)
u = torch.zeros(H.shape[1], requires_grad=True)
opt = torch.optim.Adam([u], lr=args.lr)
S0 = torch.tensor(S_dev, dtype=torch.float32)
for _ in range(args.steps):
    opt.zero_grad()
    z = S0 + H @ u
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_dev)
    loss.backward()
    opt.step()
print("train loss", round(float(loss), 4))
with torch.no_grad():
    W[ja_id] += u / 2
    W[nej_id] -= u / 2
log(f"finetune_2rows steps={args.steps} lr={args.lr}", evaluate(rows, scores(rows, PROMPTS["original"])), with_loss=True)

json.dump(results, open(f"results/alternatives_{args.split}.json", "w", encoding="utf-8"), indent=2)
