# Conditioning Experiment Protocol

## Frozen Comparison

The synthetic dataset fingerprint is retained in `results/research-conditioning/plan.json` before the multi-seed experiment. The predefined methods are character prefix/output loss, character prefix/full loss, word decoder/flat memory, and word decoder/field memory. Each uses seeds 42, 43, and 44 on the strict compositional split, validation checkpoint selection, a 100-epoch maximum, patience 15, and greedy decoding. Retrieval is computed for each matching split.

The word models use identical architecture and parameter counts: two Transformer decoder layers, embedding size 64, four attention heads, feed-forward size 256, and 96-token position tables. The flat model represents the canonical state as word/punctuation memory tokens. The field model represents each of the five fields by the mean of its value-token embeddings, with a fixed position identifying the field. Cross-attention gives the output decoder access to the condition at every step.

Character models retain their original embedding size 128 and encoder-style causal architecture. Comparing character models to word models changes tokenization, architecture, capacity, and conditioning together; it cannot isolate the effect of field structure. The matched word-flat versus word-slots comparison isolates memory construction more directly.

## Evaluation

Primary diagnostics are exact explicit NPC/player marker accuracy and token overlap F1. A marker check succeeds only when the desired value occurs and no competing value from the training domain occurs. Missing markers fail. These are conservative lexical checks, not full semantic consistency or human ratings. High F1 can still accompany the wrong actor. NLL and conditioning gaps use character units for the original model and word units for the new model; do not compare their magnitudes as if they had the same units.

The word implementation first evaluates validation predictions only. The runner finishes all training/validation runs before its separate test phase. This dataset's earlier test results have already been inspected, so it remains an exploratory synthetic benchmark, not a pristine final evaluation. Targets share templates and have AI review, not independent human review.

## Execution in WSL

```bash
source .venv-wsl/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/run_conditioning.py --phase train
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/run_conditioning.py --phase test
```

The runner reuses only matching completed runs and rejects incomplete checkpoints or changed frozen plans. It preserves per-run training/evaluation logs and generates a comparison with sample standard deviation across three seeds. Reuse does not imply test-set novelty. All source files and results remain in English.

## Preliminary Validation Check

Before the full comparison, seed 42 was run in validation-only mode for both word-memory variants. The flat-memory model achieved validation overlap F1=0.91709 and exact match=0.58333; the field-memory model achieved F1=0.91115 and exact match=0.54167. Both had explicit NPC and player marker accuracy of 1.0. These validation results were not used to change the predefined architecture or training settings, and both methods remain in the comparison. They do not establish a field-memory advantage.

Across the three seeds, both initialization and the held-out pairs vary. Reported standard deviations therefore combine training and split variation; they are not a clean estimate of initialization-only variance or statistical significance. Marker metrics require explicit words and can fail a reasonable paraphrase. Template-derived targets can favor reusable language patterns, so an independent human-authored evaluation remains necessary for broader claims.

## Supplementary Synthetic Context Markers

For this generator only, additional diagnostics detect forest trail/village square/cave entrance/riverbank, morning/lantern-light contexts, and clear-sky/travel-canopy contexts. A diagnostic fails when a competing context or no context is found. These phrase checks are tied to the synthetic target grammar; they are not semantic judgments about arbitrary text. The frozen primary metrics remain NPC/player explicit accuracy and F1. These supplemental checks were specified before the multi-seed test phase.

## Generating an Event with a Trained Word Model

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/conditional_words.py \
  --phase generate --conditioning slots --seed 42 \
  --out results/research-words-slots-42 \
  --state "Location: forest; Time: morning; Player: healthy; NPC: traveler; Weather: clear"
```

Generation loads the validation-selected checkpoint and verifies the dataset fingerprint. This demonstration rejects field values outside the training domain; it does not claim arbitrary-world coverage.
