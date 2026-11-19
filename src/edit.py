import argparse, json
import numpy as np, torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from loss import get_texts, mean_loss

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--lams", default="0.25,0.5,0.75,1.0")
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

MODEL = "jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk"
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
ja_id = tok.encode("ja", add_special_tokens=False)[0]
nej_id = tok.encode("nej", add_special_tokens=False)[0]
assert ja_id != nej_id


def load(split):
    return json.load(open(f"results/{split}_subset.json", encoding="utf-8"))


def make_inputs(text):
    prompt = cfg["prompt"].format(text=text)
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return tok(prompt, return_tensors="pt")


@torch.no_grad()
def scores(rows, want_h=False):
    S, H = [], []
    for r in rows:
        out = model(**make_inputs(r["text"]), output_hidden_states=want_h)
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
print("input/output embeddings tied:", W.data_ptr() == model.get_input_embeddings().weight.data_ptr())

# development statistics, always from val
dev = load("val")
S_dev, H = scores(dev, want_h=True)
check = (H @ (W0[ja_id] - W0[nej_id])).numpy()
if not np.allclose(check, S_dev, atol=1e-2, rtol=1e-2):
    raise SystemExit("hidden state is not the final one: sanity check failed")
mu = H.mean(0)
v = mu / (mu @ mu)
b = float(S_dev.mean())
print("bias b =", round(b, 3))


def apply(c, direction):
    with torch.no_grad():
        W.copy_(W0)
        W[ja_id] -= (c / 2) * direction
        W[nej_id] += (c / 2) * direction


rows = load(args.split)
texts = get_texts(args.split, 50)
results = []


def run(name, c, direction):
    apply(c, direction)
    m = evaluate(rows, scores(rows))
    m["swedish_loss"] = round(mean_loss(model, tok, texts), 4)
    m["condition"] = name
    results.append(m)
    print(json.dumps(m))


run("unchanged", 0.0, v)
for lam in [float(x) for x in args.lams.split(",")]:
    run(f"targeted lam={lam}", lam * b, v)
for s in range(args.seeds):
    g = torch.Generator().manual_seed(s)
    r = torch.randn(v.shape, generator=g)
    r = r * (v.norm() / r.norm())
    run(f"random seed={s} lam=1.0", 1.0 * b, r)

json.dump(results, open(f"results/edit_{args.split}.json", "w", encoding="utf-8"), indent=2)