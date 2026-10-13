# Balanced Dataset Collection Protocol

## Purpose

Separate generalization to new Location/NPC pairings from generalization to entirely new field values. The original 135-example dataset cannot provide the requested holdout proportions while keeping all field values in training. At seed 42, 17 of 20 original random-test examples contain at least one unseen field value.

## Annotation Plan

`data/dense_annotation.csv` contains 192 planned states, not 192 completed events:

All rows now have rule-template suggestions in `draft_event`, with `draft_source=rule_template_v1`. Final outputs and review approvals remain empty. Start with the first 24 rows using the [review instructions](ANNOTATION_REVIEW.md). These drafts are a controlled template distribution, not independently authored narrative data.

| Field | Values |
|---|---|
| Location | forest, village, cave, river |
| Time | morning, night |
| Player | healthy, injured, hungry |
| NPC | hunter, merchant, healer, traveler |
| Weather | clear, rainy |

All combinations are included: 4 × 2 × 3 × 4 × 2 = 192. Each of the 16 Location/NPC pairs has 12 examples. The seed-42 `compositional_seen` plan assigns 144 training, 24 validation, and 24 test examples. Every individual value occurs in training; Location/NPC pairs are disjoint across subsets. This is a controlled small-domain benchmark, not a broad game-world dataset or evidence of improved model performance.

The collection sheet displays seed-42 roles for planning. Seeds 43 and 44 generate their own reproducible pair holdouts over the same completed dataset; `planned_split` does not fix roles across all seeds. Keep authorship and rating procedures consistent across subsets.

## Filling the Sheet

- Keep `example_id`, `planned_split`, and `input` unchanged. Row order can change.
- Use `draft_event` for proposed text. It is never exported as a target automatically.
- Write the accepted English event in `output`. Aim for one clear sentence of roughly 12–25 words. Ensure it fits the player condition, NPC, location, time, and weather without mechanically listing every field.
- Vary phrasing and actions. Do not reuse a single sentence frame throughout the dataset. Avoid direct contradictions, such as serving an injured player as though the player were healthy.
- Set `source` to `human_written`, `ai_assisted_human_reviewed`, or `rule_template_human_reviewed`, reflecting actual provenance.
- Set `reviewed` to `yes` only after a person checks the final event; enter the reviewer's name or stable identifier in `reviewer`.
- Record ambiguity, edits, or limitations in `notes`.

No row is currently claimed to be human-reviewed. Human review of collection targets is separate from the two-person evaluation of model outputs. Do not label AI-assisted or template-generated text as human-written.

## Export and Verify

In WSL, from the project root:

```bash
source .venv-wsl/bin/activate
python model/annotation.py export
python model/audit.py --data data/dense_reviewed.json --out results/dense_audit.json
python -m unittest discover -s model -p "test_*.py"
```

Export requires all 192 states, explicit review, valid provenance, and a nonempty final event. It rejects modified states/split labels, duplicate IDs, incomplete rows, and an existing output file. Source metadata is retained in the exported JSON. This is a completeness check, not automatic proof of semantic quality.

After collection and validation, freeze the data before training. Check character coverage and exact event duplicates using the audit. The state plan alone does not guarantee diverse event language or enough examples to train a fluent model.

## Training After Review

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/train.py \
  --data data/dense_reviewed.json --split compositional_seen \
  --seed 42 --out results/research-dense-output-42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/evaluate.py \
  --data data/dense_reviewed.json --run results/research-dense-output-42 \
  --out results/research-dense-output-42/evaluation
```

Use fresh output directories. Tune only with validation data, then freeze settings before test evaluation. Run the predefined output/full-loss comparisons and seeds 42, 43, and 44, followed by blinded human evaluation. The original dataset remains unchanged for historical comparison.

## Delegated AI Review

A separate [AI-reviewed synthetic benchmark](SYNTHETIC_BENCHMARK.md) is now available at the user's request. Use data/dense_ai_reviewed.json for that controlled experiment. Human collection requirements above continue to apply to data/dense_reviewed.json; AI review has not been recorded as human review.
