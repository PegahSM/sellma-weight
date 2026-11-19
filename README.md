# Targeted weight edit for answer bias in a Swedish language model

Candidate assignment for SE.LLMA (after reading "Sparks of Weight-Space Programming").

**Research question:** can a targeted weight edit correct an undesirable behavior without degrading other capabilities?

**Setting:** Swedish grammatical-acceptability judgments (ScaLA, Swedish) on a Swedish-adapted Gemma 3 1B that answers "ja" to every sentence.

**Hypothesis:** a rank-1 edit of the output-embedding rows for "ja" and "nej" moves the share of "ja" answers toward 50%, beyond what a random edit of the same size does, without changing the model's ranking ability (AUC) or its Swedish language-modeling loss. Because the model's AUC is near 0.5, the edit is expected to remove a failure (the bias), not to add capability.

Research note: `NOTE.md`. Longer report: `report.pdf`. AI tool usage: `AI_USAGE.md`.

## Repository layout

```
config.yaml          270M model id + revision, seed, subset sizes, max_new_tokens, prompt
requirements.txt     pinned packages (pip freeze); the facts pilot also needs `langdetect`
src/data.py          loads ScaLA Swedish, writes balanced val/test subsets (seed 42)
src/run.py           generates replies and the ja-minus-nej score for a model on a subset
src/score.py         valid-answer rate, ja share, accuracy (bootstrap CIs)
src/auc.py           AUC of the ja-minus-nej score (bootstrap CI)
src/loss.py          Swedish language-modeling loss on the `correct` sentences
src/edit.py          the targeted edit, random-edit control, evaluation
src/facts_pilot.py   exploratory pilot: open-ended Swedish questions (not part of the final design)
results/             subsets, per-example replies, summaries, edit results
```

## Data

- Dataset: `mteb/multilingual-scala-classification`, config `Swedish` (ScaLA; CC-BY-SA-4.0). Sentences are labeled `correct` or `incorrect` (a word deleted or two neighbours swapped).
- `src/data.py` draws balanced subsets with seed 42: 100 examples from `val` (development: choosing the edit) and 100 from `test` (final run only); each has 50 `correct` and 50 `incorrect`. The exact examples are in `results/val_subset.json` and `results/test_subset.json`. These files contain ScaLA sentences and remain under CC-BY-SA-4.0 (Nielsen, 2023, ScandEval).
- The pilot in `src/facts_pilot.py` uses `liu-nlp/swedish-facts-v1` (Swedish question answering, 50 random questions, seed 42).

## Models

| Model | Hugging Face id | Revision |
|---|---|---|
| Gemma 3 270M (pilot) | `google/gemma-3-270m-it` | `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` |
| Gemma 3 1B (pilot) | `google/gemma-3-1b-it` | TODO (not pinned in the runs) |
| Swedish-adapted Gemma 3 1B (main) | `jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk` | TODO (not pinned in the runs) |

Gemma models are gated: accept the license on the model page and log in with `hf auth login`. Weights are not redistributed here. `src/edit.py` and `src/loss.py` use the Swedish-adapted 1B model (the id is set at the top of `edit.py`).

## Reproduce

Windows, Git Bash, Python 3.10+, CPU only (no GPU needed).

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 1. data subsets (seed 42)
python src/data.py

# 2. baseline pilots on val (one run per model; greedy decoding)
python src/run.py --split val --out results/baseline_val.jsonl                       # 270M, from config.yaml
python src/run.py --split val --model_id google/gemma-3-1b-it --out results/baseline_val_1b.jsonl
python src/run.py --split val --model_id jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk --out results/baseline_val_sv1b.jsonl
python src/score.py results/baseline_val_sv1b.jsonl     # repeat for each file
python src/auc.py results/baseline_val_sv1b.jsonl       # repeat for each file

# 3. development run of the edit on val (strength grid, random control)
python src/edit.py --split val

# 4. final run on test (once): strength fixed at 1.0, five random seeds
python src/edit.py --split test --lams 1.0 --seeds 5

# optional: open-ended Swedish facts pilot
pip install langdetect
python src/facts_pilot.py --model_id google/gemma-3-1b-it --q_col "Question (Swedish)" --a_col "Answer (Swedish)" --out results/facts_1b.jsonl
python src/facts_pilot.py --model_id jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk --q_col "Question (Swedish)" --a_col "Answer (Swedish)" --out results/facts_sv1b.jsonl
```

Decoding is greedy, so replies are deterministic. Random seeds: 42 for data sampling, bootstrap intervals and the facts sample; seeds 0 to 4 for the random edit directions. `src/loss.py` is imported by `edit.py` (it can also be run alone with `--model_id`, `--split`, `--out`).

## The edit

`src/edit.py` computes, on the `val` subset, the mean ja-minus-nej score `b` and the mean final hidden state `mu`. It changes only two rows of the (tied) output embedding matrix: `E[ja] -= (c/2) v` and `E[nej] += (c/2) v`, with `v = mu / (mu . mu)` and `c = lambda * b`. This shifts the ja-minus-nej score by about `c` for every input. The random control uses a random vector with the same norm as `v`. A sanity check stops the script if the final hidden state does not reproduce the logits. Edit parameters (`b`, `mu`) always come from `val`, including in the `test` run.

## Results

### Baselines on `val` (n = 100, balanced; 95% bootstrap intervals)

| Model | "ja" share | Accuracy | AUC |
|---|---|---|---|
| Gemma 3 270M | 0.02 | 0.50 (0.40-0.60) | 0.517 (0.403-0.630) |
| Gemma 3 1B | 0.58 | 0.64 (0.54-0.73) | 0.689 (0.582-0.788) |
| Swedish-adapted 1B | 1.00 | 0.50 (0.40-0.60) | 0.540 (0.432-0.656) |

All models answered in Swedish with a valid ja/nej in 100% of cases.

### Edit on `val` (Swedish-adapted 1B; development)

| Condition | "ja" share | Accuracy | AUC | Swedish loss |
|---|---|---|---|---|
| Unchanged | 1.00 | 0.50 | 0.540 | 3.1428 |
| Targeted, lambda 0.25 / 0.5 | 1.00 | 0.50 | 0.536 | 3.1428 |
| Targeted, lambda 0.75 | 0.94 | 0.54 | 0.533 | 3.1428 |
| Targeted, lambda 1.0 | 0.58 | 0.50 | 0.533 | 3.1427 |
| Random, lambda 1.0 (3 seeds) | 1.00 | 0.50 | 0.537-0.542 | 3.1428 |

Lambda 1.0, the value that brings the "ja" share closest to 0.5, was fixed before the test run.

### Final run on `test` (n = 100, balanced; run once)

| Condition | "ja" share | Accuracy | AUC | Swedish loss |
|---|---|---|---|---|
| Unchanged | 1.00 | 0.50 (0.40-0.60) | 0.414 (0.305-0.529) | 3.1359 |
| Targeted edit, lambda 1.0 | 0.67 | 0.43 (0.33-0.53) | 0.416 (0.306-0.528) | 3.1358 |
| Random edit, lambda 1.0 (5 seeds) | 1.00 | 0.50 (0.40-0.60) | 0.412-0.415 | 3.1359 |

Output: `results/edit_test.json`.

### Open-ended Swedish facts pilot (50 questions, 64 new tokens)

| Model | not Swedish | repetitive | hit token limit | gold answer in well-formed replies |
|---|---|---|---|---|
| Gemma 3 1B | 0.02 | 0.00 | 0.44 | 1 of 49 |
| Swedish-adapted 1B | 0.02 | 0.00 | 0.56 | 1 of 49 |

Failures are rare and accuracy is at the floor, so this setup could not test the hypothesis and was dropped.

## Findings

- The targeted edit reduces the "ja" bias (100% to 67% on `test`); random edits of the same size do not change it at all.
- Accuracy does not improve (0.43 versus 0.50; intervals overlap, so not a significant change). The model's AUC is near 0.5 (0.54 on `val`, 0.41 on `test`, both intervals include 0.5), so balancing the answers only redistributes errors: failure removal without added capability.
- AUC and Swedish loss are unchanged by the edit.

## Limitations

- Small samples (100 examples per subset): intervals are about +/-10 points. AUC differs between `val` (0.54) and `test` (0.41), which shows the noise.
- "ja" share, accuracy and AUC in `src/edit.py` use the sign of the ja-minus-nej score, not generated replies.
- The edit is close to a calibration implemented in the weights.
- The loss check is weak evidence of no regression: only two embedding rows change and the test sentences rarely contain "ja" or "nej". Nothing outside Swedish was checked.
- The edit strength was chosen by feedback on `val`; no LLM proposed or wrote the edit. This tests whether a targeted edit works, not the article's claim that an LLM can program weights by reasoning.
- Single model, single task; ScaLA's corruptions are automatic and simple; 1B model revisions were not pinned.
