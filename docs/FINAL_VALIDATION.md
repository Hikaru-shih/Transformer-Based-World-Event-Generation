# Final Validation and Delivery Status

The implemented synthetic study is complete: twelve model runs across three seeds, matched retrieval comparisons, frozen manifests, predictions, diagnostics, and a reproducible results report. Independent human evidence remains pending. Software preparation does not count as completed human evaluation.

## Independent Reference Collection

`data/independent_test.csv` contains 24 predefined seed-42 held-out states with blank targets. Ask an author to write one natural English event per state without viewing synthetic references, model predictions, or example drafts. Ask a different person to check state consistency. Fill `output`, `author`, `reviewer`, `reviewed=yes`, and `source=human_written`. Preserve all IDs and states. Do not use AI drafts and then label them human-written. Author/reviewer identifiers are provenance attestations, not independently verified identities.

This collection tests new human-written references for existing benchmark states, not unseen environments. It is independent in authorship only if the collection instructions are followed; the states and synthetic test results have already been inspected. Avoid reusing this test for tuning. A broader generalization claim needs separately designed new scenarios and independent authors.

After all 24 rows are complete, run in WSL from the project root:

```bash
source .venv-wsl/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python model/independent_test.py --phase evaluate
```

The command checks human provenance fields, seen field values, held-out Location/NPC pairs, the complete predefined collection, and matching frozen seed-42 checkpoints. It evaluates word-flat, word-slots, and retrieval without retraining. It records reference vocabulary gaps, overlap diagnostics, predictions, and dataset fingerprints in `results/independent-test`. A fresh output directory is required. Novel reference wording may lower overlap even for plausible predictions; supplement these results with genuine human judgments. Contextual template-phrase checks do not apply to human-written references.

## Independent Ratings

Use the existing [review instructions](human_evaluation/README.md). Each of two independent reviewers completes their own 360-item CSV. Keep `canonical_items.csv` unchanged as the original item inventory and keep the method key away from reviewers. These sheets evaluate the synthetic study; they are separate from independent reference collection.

```bash
python model/analyze_human_review.py \
  --a docs/human_evaluation/reviewer_a.csv \
  --b docs/human_evaluation/reviewer_b.csv \
  --template docs/human_evaluation/canonical_items.csv \
  --key docs/human_evaluation/system_key.json \
  --out docs/human_evaluation/status.json
```

The analyzer rejects duplicate/missing/changed items, invalid scores, and identical reviewer IDs. Incomplete sheets produce progress counts only, with no selected-subset averages. Complete sheets produce per-system reviewer means, paired means, exact agreement, within-one-point agreement, and mean absolute disagreement. Raw sheets and notes remain intact. Repeated states across seeds preclude treating rows as independent evidence for a significance claim. Reviewer identity and independence still require honest collection practices.

## What Is Complete and What Remains

| Deliverable | Status |
|---|---|
| Controlled synthetic benchmark and provenance | Complete; AI-reviewed templates |
| Four model configurations across three seeds | Complete |
| Predictions, baseline comparison, conditioning diagnostics | Complete |
| Research-experience wording with measured results | Complete in CONDITIONING_RESULTS.md |
| Independent-reference collection and evaluation tooling | Prepared; human targets pending |
| Two-reviewer sheets and agreement analysis tooling | Prepared; real ratings pending |
| Broader original-data split matrix | Outside the completed synthetic study; pending |
| Novelty claim or publication | Not established |

The project can be described as a completed controlled synthetic research prototype now. Do not claim human-validated quality, open-ended creativity, statistical superiority, or a completed independent human study until the corresponding evidence exists.
