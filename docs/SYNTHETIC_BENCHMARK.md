# AI-Reviewed Synthetic Benchmark

## Completed Collection

At the user's request, the assistant completed a separate AI-reviewed version of the 192-state plan:

- `data/dense_ai_reviewed.csv`: original suggestions, revised targets, and review provenance.
- `data/dense_ai_reviewed.json`: the training-ready synthetic benchmark.
- `results/synthetic_collection.json`: collection fingerprint, counts, context lengths, and coverage.
- `results/dense_ai_audit.json`: complete dataset audit.

Every target is labeled `source=rule_template_ai_reviewed`, `review_kind=ai`, and `reviewer=Codex`. There are 192 AI-reviewed targets and zero human-reviewed targets. The original `dense_annotation.csv` remains available for independent human review and is not relabeled or overwritten.

The assistant reviewed the reusable action rules and their combinations with all planned states, then applied consistent revisions programmatically. This is not an independent human annotation study or a separate evaluator model. Rule-based checks verify identifiers, NPC/player mentions, maximum context length, duplicate outputs, and split coverage; they do not establish all semantic properties of an event.

## Revisions

Location phrases were corrected to “Along a forest trail”, “In the village square”, “At the cave entrance”, and “On the riverbank”. Night events have explicit lantern illumination, and rainy events have a temporary travel canopy. Healthy hunters at night show tracks using a lantern instead of starting an unexplained night hunt. Healthy-player healer encounters in darkness or rain refer to herbs in a pouch rather than immediate outdoor gathering.

Lanterns and canopies are plausible new details introduced by events; they are not fields observed in the input. This assumption is part of the benchmark definition. If future game rules disallow introducing items, these targets will need revision.

All 192 outputs are distinct. The longest complete sequence is 214 tokens, within the 256-token context. At seed 42 the strict pair split contains 144 train, 24 validation, and 24 test examples. No held-out single field value is absent from training, and no held-out Location/NPC pair is shared with training.

## Interpretation Limits

Targets share reusable templates across subsets. A successful model on this dataset demonstrates learning within this controlled distribution, not generalization to independently authored stories or open-ended creativity. AI review cannot be presented as human review, and it does not replace the planned two-reviewer assessment of model predictions. Original and synthetic results use different data and should not be directly compared as a model improvement claim.

## Reproduction

The dataset is frozen for the initial run. `prepare_synthetic.py` refuses to overwrite existing artifacts. For a fresh collection reproduction, preserve existing artifacts and regenerate in an isolated checkout rather than deleting the benchmark.

```bash
source .venv-wsl/bin/activate
python model/prepare_synthetic.py
python model/audit.py --data data/dense_ai_reviewed.json --out results/dense_ai_audit.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/train.py \
  --data data/dense_ai_reviewed.json --split compositional_seen \
  --seed 42 --epochs 100 --out results/research-synthetic-output-42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/evaluate.py \
  --data data/dense_ai_reviewed.json --run results/research-synthetic-output-42 \
  --out results/research-synthetic-output-42/evaluation
```

Use a fresh run directory when repeating training. Collection provenance and the first completed training run are recorded below.

## First Full Run: 2026-10-12

The seed-42 output-only model completed training on the strict compositional split. Epoch 27 was selected by validation event NLL (1.41235); early stopping occurred at epoch 42. The test set contains 24 examples. Architecture and optimizer settings match the original initial experiment: emb=128, depth=2, context=256, AdamW lr=3e-4, batch_size=16, and patience=15. Evaluation uses greedy decoding.

| Metric | Transformer | Retrieval |
|---|---:|---:|
| Token overlap F1 | 0.29946 | 0.81803 |
| Exact match | 0.00 | 0.00 |
| Training copy rate | 0.00 | 1.00 |
| Distinct-2 | 0.69547 | 0.21087 |
| Mean word count | 21.25 | 20.17 |

Test event NLL is 1.51840 and character perplexity is 4.56494. Shuffled-state event NLL is 1.42183, giving a conditioning NLL gap of -0.09658. This run does not show a likelihood advantage for the correct conditions.

A forest/morning/healthy/traveler/clear state produced the Transformer output “Ing a rilage tre in inthe t morng under clear skies, a healer hans thelthealtheay player player her play pl.” This illustrates incomplete words, repetition, and an incorrect actor. Retrieval returned a fluent event with a healer rather than the requested traveler. Its high overlap F1 is therefore not proof of state consistency: much of the score comes from the shared contextual template. Likewise, high distinct-2 in corrupted model text does not establish creativity.

Ten tests pass, including a test that rejects AI provenance at the human export gate. The dataset collection has been completed as an AI-reviewed synthetic benchmark, but neither human target review nor independent human evaluation has been performed.

The complete run remains in `results/research-synthetic-output-42/`. Trackable manifest, history, metrics, and unscored review-sheet copies are in `docs/synthetic_results/`. Do not alter this frozen dataset based on its test errors and then present a rerun as an untouched final test. Use validation for method changes; any new dataset or settings should be documented separately.

The next research steps are the predefined loss/seed comparisons, an explicit check of actor/player consistency alongside word overlap, and a better conditioning method assessed on validation data. Results on this template benchmark cannot substitute for an independently authored test set.
