# Initial Experiment on the Original Dataset: 2026-10-10

Historical record: results and verification counts below describe this original-data run only. For the completed three-seed synthetic study and current status, see the [complete research report](RESEARCH_REPORT.md). The next steps below record the plan at that time; synthetic experiments are now complete and human validation remains pending.

Purpose: complete the first planned training run after smoke testing and record the limitations of the original dataset. This is an initial result from one seed, not the complete experiment matrix or a human evaluation.

## Configuration and Verification

- WSL with a Python 3.10 virtual environment, `.venv-wsl`; PyTorch 2.14.1+cu130, running on CPU.
- All four unit tests passed, with no skipped tests.
- Original 135-example dataset, random split, seed 42: 95 training, 20 validation, and 20 test examples.
- Character-level Transformer with 1,230,895 parameters; emb=128, depth=2, block_size=256.
- Output-only loss, AdamW, lr=3e-4, batch_size=16, a maximum of 100 epochs, and patience=15.
- Validation selected epoch 14 (validation event NLL=2.21937); early stopping occurred at epoch 29.
- Model evaluation used greedy decoding with at most 150 new tokens.

## Test Results

| Metric | Transformer | Retrieval |
|---|---:|---:|
| Token overlap F1 | 0.1190 | 0.3453 |
| Distinct-2 | 0.1386 | 0.6744 |
| Mean word count | 36.00 | 11.75 |
| Training copy rate | 0.00 | 1.00 |
| Exact match | 0.00 | 0.00 |
| Empty-output rate | 0.00 | 0.00 |

Transformer test event NLL=2.31793 and character perplexity=10.15462. Shuffled-state NLL=2.31107, with a conditioning NLL gap of -0.00686.

This run showed no likelihood advantage from the correct conditions, and model outputs contained many incomplete words and repeated fragments. For example, the output for desert/thirsty/traveler/hot began with “A therer the there the the the player ...”. Not copying a complete training sentence does not imply good generation quality. Retrieval outputs were more fluent but also included unsuitable events, such as pairing a desert state with a lava-edge event; human review is still needed.

After epoch 14, training NLL continued to decrease while validation NLL increased, indicating overfitting under this configuration. The current results do not support claims that the Transformer outperforms retrieval or has learned world-state consistency. F1 is a word-overlap proxy and cannot establish semantic quality on its own. The shuffled-state result is also a diagnostic, not a conclusion about statistical significance.

## Next Steps

1. Design a denser handcrafted dataset: reduce the Location/NPC sets and include repeated examples for individual field values and multiple pairings in training, holding out only specific pairings. Save the new dataset separately and retain the original baseline.
2. Use validation data for error analysis and method changes; do not repeatedly select settings using the current test scores. This test set has now been inspected and should be treated as an initial benchmark that has already been seen. Evaluate a new test set only after its design is frozen.
3. After freezing the new dataset and settings, run the planned two-loss comparison across three seeds, compare with retrieval, and obtain ratings from two reviewers.

There is no need to run every combination immediately, and longer training or a larger model should not be the only improvement strategy.

## Reproduction and Artifacts

Run from the project root in WSL. For a new reproduction, change `--out` to a fresh directory to avoid overwriting existing records.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-wsl/bin/python model/train.py \
  --epochs 100 --seed 42 --split random --out results/research-random-42-initial
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-wsl/bin/python model/evaluate.py \
  --run results/research-random-42-initial \
  --out results/research-random-42-initial/evaluation
```

The complete manifest, history, checkpoint, metrics, and human-review CSV are stored in `results/research-random-42-initial/`, which is currently ignored by Git. Trackable copies of the JSON records are stored in `docs/initial_results/`.

During execution, PyTorch reported a GPU driver compatibility warning and fell back to CPU. It also warned that NumPy was not installed. All tests, training, and evaluation completed successfully despite these warnings.
