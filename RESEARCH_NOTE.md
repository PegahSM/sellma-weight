# Can a targeted weight edit remove an answer bias in a Swedish language model?

## Hypothesis

**Question.** Can a targeted weight edit correct an undesirable behavior in a Swedish language model without degrading other capabilities?

**Narrowed to.** The Swedish-adapted Gemma 3 1B (`jekunz/Gemma-3-1B-pt-sv-CPT-sv-SmolTalk`) answers "ja" to every sentence in a Swedish grammatical-acceptability task (ScaLA). A rank-1 edit of the "ja" and "nej" output-embedding rows should move the share of "ja" answers toward 50%, more than a random edit of the same size does, while leaving the model's ranking ability (AUC) and Swedish language-modeling loss unchanged. Because the model's AUC is near 0.5, the edit should remove a failure (the bias), not add grammatical ability.

**Why it matters.** A Swedish-adapted model can show a systematic answer bias that makes it unreliable on Swedish evaluation tasks. A small, inspectable weight change would be a cheap alternative to retraining. This is the narrow, testable part of the blog's weight-space-programming claim.

**Challenged by:** random edits moving the "ja" share as much as the targeted edit; a drop in AUC or a rise in loss; prompting or a logit bias fixing the bias equally well; an effect that disappears from val to test.

## Experiment

- **Data.** ScaLA Swedish (`mteb/multilingual-scala-classification`). Two balanced subsets of 100 sentences (seed 42): `val` to design the edit and choose its strength, `test` used once for the final result.
- **Model.** A pilot on val compared three models. Gemma 3 270M has no ranking ability, Gemma 3 1B has ability but little bias, and the Swedish-adapted 1B answers "ja" every time (accuracy 0.50, AUC 0.54), so it was used.
- **Measures.** Target behavior: share of "ja" answers (sign of the first-token score logit("ja") − logit("nej")). Ability: AUC of that score. Accuracy with bootstrap 95% CI. Regression check: Swedish language-modeling loss on 50 correct sentences.
- **Edit.** From val, take the mean final hidden state `mu` and mean score `b`. Change only two output-embedding rows: `E[ja] -= (c/2)·v`, `E[nej] += (c/2)·v`, with `v = mu/(mu·mu)` and `c = λ·b`. λ = 1.0 was chosen on val.
- **Controls and alternatives.** Unchanged model; random edit of the same norm (5 seeds); two prompt changes; an inference-time logit bias (subtract `b`); supervised fine-tuning of the same two rows on val labels.

## Results (test, n = 100, balanced)

| Condition | "ja" share | Accuracy (95% CI) | AUC (95% CI) | Swedish loss |
|---|---|---|---|---|
| Unchanged | 1.00 | 0.50 (0.40–0.60) | 0.414 (0.305–0.529) | 3.1359 |
| Random edit (5 seeds) | 1.00 | 0.50 | 0.412–0.415 | 3.1359 |
| **Targeted edit** | **0.67** | 0.43 (0.33–0.53) | 0.416 (0.306–0.528) | 3.1358 |
| Prompt: option order swapped | 0.96 | 0.54 (0.44–0.64) | 0.441 (0.322–0.552) | 3.1359 |
| Prompt: different wording | 1.00 | 0.50 (0.40–0.60) | 0.464 (0.344–0.574) | 3.1359 |
| Logit bias | 0.61 | 0.45 (0.35–0.55) | 0.414 (0.305–0.529) | 3.1359 |
| Fine-tune of the 2 rows | 0.85 | 0.49 (0.39–0.59) | 0.513 (0.394–0.632) | 3.1408 |

Prompt and logit-bias rows change no weights, so their loss equals the unchanged model's. On val, the targeted edit gave a "ja" share of 0.58 and the logit bias 0.53. Fine-tuning reached accuracy 0.76 on val, which is its training data.

## Interpretation

- The targeted edit removes much of the bias (1.00 to 0.67). Random edits of equal size do not move it, so the effect depends on the direction.
- AUC and Swedish loss are unchanged, so the checked abilities are not degraded.
- Accuracy does not improve (0.43 versus 0.50, intervals overlap). The model cannot separate correct from incorrect sentences (unchanged AUC near 0.5), so shifting every score lowers the "ja" share without reordering sentences. This is failure removal, not new capability.
- Prompt changes do not remove the bias. Fine-tuning the two rows fits val but does not generalize (test accuracy 0.49, AUC 0.513) and raises loss slightly.
- A logit bias gives a similar result (0.61) with identical AUC and no weight change. The edit is a calibration written into the weights, so for this kind of bias it has no advantage over a simpler method.

## Recommendation

SE.LLMA should investigate further, in a narrower form. A scripted weight edit for answer bias is not worth pursuing, because a logit bias does the same job. The claim worth testing is whether an LLM can write edits for failures that a constant shift cannot fix, such as language switching, repetition or failure to stop in generation. The most informative next experiment: give a frontier LLM such a failure and a small validation set, compare its edits with and without forward-pass feedback against steering and LoRA baselines, and use real regression checks (Swedish benchmarks, general knowledge) with at least 500 examples per split, several seeds, and pinned model revisions.

## Limitations

- Small samples (100 per split): intervals are about ±10 points. AUC differs between val (0.54) and test (0.41), and both intervals include 0.5.
- "ja" share, accuracy and AUC come from the first-token score, not from generated replies.
- The loss check is weak: only two embedding rows change, and the sentences rarely contain "ja" or "nej". Nothing outside Swedish was checked.
- The edit strength was chosen on val, and only λ = 1.0 was tested on test.
- One model, one task, one prompt family. The fine-tuning baseline was lightly configured.
- The edit was suggested and coded by an AI assistant (Claude Sonnet 5.5) and applied as a fixed script. An LLM did not read or write the weights itself or iterate on forward-pass feedback, as in the blog, so this does not test LLM-driven weight programming.

Reproduction commands, model revisions and seeds are in `README.md`. AI tool use is described in `AI_USAGE.md`.
