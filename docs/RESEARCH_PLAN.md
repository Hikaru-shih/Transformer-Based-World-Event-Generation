# Research Plan and Research Experience Wording

## Current Scope: October 13, 2026

The [complete research report](RESEARCH_REPORT.md) is the main current account. Four model configurations across three seeds and matching retrieval baselines are complete on the 192-state AI-reviewed synthetic benchmark. Seventeen tests pass. The word pipeline improves automated lexical diagnostics, but field memory alone has no clear advantage. Independent human validation remains pending. The original-data matrix below is an optional extension outside the completed synthetic study.

## Project Review

The original project implemented a small causal Transformer with 135 handcrafted input/event pairs, but trained on the entire dataset and reported only training loss. It had no holdout evaluation, baseline comparison, ablation, or human review. The README's original count of 150 examples has been corrected. The original EOS marker consisted of five characters, while inference stopped at the first `<`; the new tokenizer uses a dedicated EOS ID. Paths are now resolved relative to the script location, allowing execution from the project root.

`Report.pdf` has an English edition with historical claims and audit corrections distinguished. Its original results belong to the old pipeline. Initial WSL checks and the first full single-seed original-data run are documented in [Initial Results](INITIAL_RESULTS.md). The completed synthetic comparison is reported separately; improvements there concern lexical diagnostics on that benchmark, not human-rated narrative quality.

## Experimental Design

| Split | Methods | Seeds | Purpose |
|---|---|---|---|
| random | retrieval / output-loss / full-loss | 42, 43, 44 | Low-data performance and loss ablation |
| compositional | retrieval / output-loss / full-loss | 42, 43, 44 | Unseen-combination generalization and loss ablation |

Keep the architecture, learning rate, epoch limit, validation-based selection, and greedy decoding consistent. Each evaluation automatically computes retrieval results on the same test set. Retain individual metrics, predictions, failure examples, and human ratings rather than reporting only the best seed. Means and standard deviations across three seeds describe variation within these experiments; they do not replace large-scale statistical evidence.

Splits are reproducible for a fixed seed, although different PyTorch versions or devices may introduce floating-point differences. Select hyperparameters on validation data and reserve test data for final evaluation. The random and compositional test sets differ, so compare each method with the baseline for its own split. Check for unseen individual field values to avoid conflating compositional generalization with generalization to new field values.

## Conditioning Diagnostic

Evaluation also shuffles test states using a fixed seed while retaining the original target events. It computes shuffled-state NLL minus normal NLL. A positive gap means the model assigns higher likelihood to events under the correct conditions; a gap near zero may indicate that it ignores the state. This is a diagnostic of conditioning, not direct evidence of event quality. Small test sets and differences in input length can affect the result.

## Human Review and Error Analysis

Use at least two reviewers to independently score all test outputs. Hide `system_key`, randomize row order before review, and retain `example_id`. A reference event is one reasonable answer, not the only acceptable answer.

- State consistency, 1–5: 1 means a clear contradiction; 3 means generally plausible but missing an important condition; 5 means plausible and responsive to important conditions.
- Narrative coherence, 1–5: 1 means difficult to understand; 3 means understandable but unnatural; 5 means clear and complete.
- Interestingness, 1–5: 1 means empty or repetitive; 3 means plausible but ordinary; 5 means specific and promising for game interaction.

Retain raw ratings and report method averages and reviewer differences. One simple agreement measure is the proportion of ratings whose difference is at most one point. Record written reasons for contradictions. Reasonably omitting a field is not itself a contradiction. Classify failures involving character/state conflicts, copied training events, incomplete sentences, unconditional templates, unknown characters, and unseen field values. Keep representative examples from each category.

## Research Experience Wording for the Current Project

“Developed a research prototype for world-state-conditioned game event generation using a character-level causal Transformer and 135 handcrafted examples; implemented grouped holdout protocols, event-only language modeling, validation-based checkpoint selection, and retrieval-based evaluation with a human-review workflow.”

This older wording describes the original implementation. Current measured resume and research-statement versions are in [Research Experience](RESEARCH_EXPERIENCE.md). Present the project as an extension of a course project or research prototype, with synthetic provenance disclosed and no publication claim.

## Wording to Add After Completing the Experiments

“On [number] test examples under the [split] protocol, state-consistency ratings changed from [A] for retrieval to [B] for the Transformer (two reviewers, three seeds); compared output-only and full-sequence losses and analyzed [failure categories].”

If the model does not outperform retrieval, report the limitations of low-data generation and the generalization gap honestly. Such findings can still be valuable research results. Do not describe this project as a large-scale game deployment, a world model, reinforcement learning, or a new Transformer architecture. Related literature has not yet been reviewed, so methodological novelty is not claimed.

## Data Sparsity

The [Balanced Dataset Collection](DENSE_DATASET.md) includes a 192-state human annotation plan, review-gated export, strict `compositional_seen` splitting, and coverage diagnostics. Its human targets remain incomplete. A separate AI-reviewed synthetic collection has been completed and evaluated; the two sources must remain distinct.

Direct inspection found 102 Location values, 49 NPC values, and 124 Location/NPC combinations in just 135 examples. Most combinations occur only once. Test degradation may reflect entirely new field values and cannot be treated as clean evidence about compositional generalization. Prioritize a smaller, consistent set of field values and manually add examples covering each value and multiple pairings. Design the split so individual values occur in training while specific pairings are held out. Preserve the original dataset for separate comparisons and do not mix augmentation into the test set.

## Future Extensions

After completing these experiments, consider collecting more human-written data, splitting by author, adding multiple reference events, and evaluating paired counterfactual states. Label synthetic data by source and maintain an independent human-written test set to prevent template overlap from producing misleading improvements.

## Completed Conditioning Comparison

The [three-seed conditioning comparison](CONDITIONING_RESULTS.md) is complete: four model settings, twelve training runs, matching retrieval baselines, actor/player marker checks, contextual phrase diagnostics, and traceable predictions. The word-flat and word-field models substantially improve the controlled synthetic benchmark over the character-prefix models. Field memory alone has not shown a clear advantage over flat word memory. The original broader dataset/split matrix and independent human evaluation remain unfinished. Current measured research-experience wording is included in that report.

## Current Delivery Status

The controlled synthetic study and final validation tooling are complete. See [Final Validation](FINAL_VALIDATION.md) for the current checklist and WSL commands. Independent human targets and two-reviewer ratings remain pending; earlier pending-experiment statements describe the historical original-data plan.

