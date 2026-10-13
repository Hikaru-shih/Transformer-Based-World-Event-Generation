# World State → Event Generation

A research prototype for game-event generation conditioned on world state, with character-prefix and word-level conditional Transformers. The original version was a final project for the 2025 ML course.

Start with the [complete research report](docs/RESEARCH_REPORT.md), updated October 13, 2026. The controlled synthetic study is complete: four model configurations across three seeds, matching retrieval baselines, and traceable evaluation artifacts. Seventeen tests pass. Independent human references and ratings remain pending.

## Research Questions

1. With only 135 handcrafted examples, can a Transformer generate better events than a baseline that retrieves events using world-state fields?
2. How does the model generalize to unseen Location × NPC combinations?
3. Is an event-only training loss more effective than a full-sequence loss?

The original 135-example experiment did not outperform retrieval. The separate 192-state AI-reviewed synthetic study shows higher overlap and NPC-marker scores for word-level models; it does not establish human-rated quality or a field-memory advantage. The broader original-data matrix remains an optional extension.

## Data and Method

`data/dataset.json` contains 135 examples, each with Location, Time, Player, NPC, Weather, and an English event. There are no duplicate complete input states. The longest training sequence, including EOS, contains 170 tokens. Each state has only one reference event, and this small dataset does not represent a complete game world.

The research pipeline adds atomic PAD/UNK/EOS tokens, a vocabulary fitted only on training data, event-only loss, validation-based checkpoint selection, early stopping, gradient clipping, random seeds, and dataset fingerprints. It retains the original PyTorch TransformerEncoder architecture with a causal attention mask and autoregressive generation.

## Setup and Execution

Requires Python 3.10+ and PyTorch. Run the commands from the project root. On Windows, `py` can replace `python`. In WSL, activate the existing environment with `source .venv-wsl/bin/activate` first.

```bash
python -m pip install -r requirements.txt
python model/audit.py
python -m unittest discover -s model -p "test_*.py"
python model/evaluate.py --out results/baseline-random
python model/train.py --out results/research-random-42 --seed 42 --split random
python model/evaluate.py --run results/research-random-42 --out results/research-random-42/evaluation
python model/generate.py --run results/research-random-42
```

The original `results/model.pt`, `tokenizer.json`, and `loss_curve.png` are retained as historical course artifacts. They are incompatible with the new tokenizer. New experiments write to separate directories. `Report.pdf` is the English edition of the course report; its historical claims are distinguished from the current experiment results. The editable translation is in [Course Report](docs/COURSE_REPORT_ENGLISH.md).

## Experiment Matrix

The `random` split groups examples by complete input state. The `compositional` split groups examples by Location/NPC pair, preventing a pair from appearing in multiple subsets. Approximately 70/15/15 of the groups are assigned to train/validation/test; sample proportions may differ. Unseen combinations may also contain unseen individual field values, so this split does not necessarily isolate compositional generalization.

```bash
python model/train.py --out results/research-compositional-42 --seed 42 --split compositional
python model/evaluate.py --run results/research-compositional-42 --out results/research-compositional-42/evaluation
python model/train.py --out results/research-full-42 --seed 42 --objective full
python model/evaluate.py --run results/research-full-42 --out results/research-full-42/evaluation
```

For each split, compare retrieval, the output-loss Transformer, and the full-loss Transformer using seeds 42, 43, and 44. Use a separate output directory for each run, compare methods with matching seeds, and report individual results alongside the mean and standard deviation. Select models using validation data; do not tune on test results.

## Evaluation and Artifacts

- `manifest.json`: dataset SHA-256, split indices, seed, model configuration, parameter count, and environment.
- `history.json` and `best.pt`: training/validation history and the best validation checkpoint.
- `metrics.json`: event NLL, character perplexity, shuffled-state NLL gap, token overlap F1, exact match, training copy rate, distinct-2, empty-output rate, and word count.
- `human_review.csv` and `system_key.json`: a review sheet with neutral system labels and a separate method mapping.

Word overlap does not establish semantic correctness, and a low copy rate does not establish creativity. Retrieval is expected to have a copy rate of 1. See the [research plan and research experience wording](docs/RESEARCH_PLAN.md).

## Verification Status

Seventeen current tests pass in WSL. Historically, the first four tests and a three-epoch smoke run verified the initial pipeline. In that smoke run, validation NLL decreased from 2.929 to 2.530; test token overlap F1 was 0.064 for the Transformer and 0.345 for retrieval, with a conditioning NLL gap of -0.00244. These historical checks should not be presented as evidence of generation quality.

The first full training run and evaluation are also complete; see the [initial experiment results](docs/INITIAL_RESULTS.md). With seed 42, early stopping occurred at epoch 29 and selected epoch 14. Generation quality remains inadequate, and the model has not outperformed retrieval in this comparison.

Author: Pin Guang Shih, National Yang Ming Chiao Tung University.

## Balanced Dataset Collection

The [balanced collection protocol](docs/DENSE_DATASET.md) provides a 192-state human annotation sheet and an exporter that requires explicit human review and provenance. The `compositional_seen` split retains every field value in training while holding out Location/NPC pairs. The original sparse dataset cannot satisfy this constraint at the requested holdout proportions. Human annotation remains pending; a separately labeled AI-reviewed synthetic dataset has already been used for the completed study.

Rule-template drafts are now filled for all 192 states, without final answers or human-review flags. See [Reviewing the Event Drafts](docs/ANNOTATION_REVIEW.md) to review them in batches of 24. Run `python model/draft_events.py` to fill missing suggestions while preserving existing work.

## AI-Reviewed Synthetic Benchmark

At the user's request, a separate 192-example AI-reviewed benchmark is ready in data/dense_ai_reviewed.json. Its source is rule_template_ai_reviewed, with review_kind=ai and zero human-reviewed targets. The original human annotation sheet remains unchanged. See [Synthetic Benchmark](docs/SYNTHETIC_BENCHMARK.md) for provenance, reproducible commands, and interpretation limits.

## Completed Conditioning Experiments

The [conditioning results](docs/CONDITIONING_RESULTS.md) include four model settings across three seeds, with matched retrieval baselines. Word-flat achieved F1 0.904 ± 0.014 versus retrieval 0.728 ± 0.079 on the AI-reviewed synthetic benchmark; both word models achieved 100% explicit NPC markers. These are controlled lexical diagnostics, not independent human ratings or evidence of open-ended narrative ability. The [experiment protocol](docs/CONDITIONING_EXPERIMENT.md) includes reproducible training, evaluation, and event-generation commands. All seventeen current tests pass.

## Final Validation Workflow

See [Final Validation](docs/FINAL_VALIDATION.md) for the prepared independent-reference collection, frozen-model evaluation, paired human-rating validation, and honest completion status. Human targets and ratings remain blank until real participants supply them.
