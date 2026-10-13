# Conditioning Comparison Results

Four model settings were evaluated across seeds 42, 43, and 44 on the frozen 192-example AI-reviewed synthetic benchmark. Each strict compositional test subset contains 24 examples. All single field values are seen in training; Location/NPC pairs are held out. Results below are mean ± sample standard deviation across three runs.

## Primary Metrics

| Method | Parameters | NPC explicit accuracy | Player explicit accuracy | Token overlap F1 |
|---|---:|---:|---:|---:|
| char-output | 1230381 | 0.000 ± 0.000 | 0.083 ± 0.000 | 0.430 ± 0.116 |
| char-full | 1230381 | 0.000 ± 0.000 | 0.083 ± 0.110 | 0.456 ± 0.129 |
| words-flat | 160755 | 1.000 ± 0.000 | 0.986 ± 0.024 | 0.904 ± 0.014 |
| words-slots | 160755 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.894 ± 0.024 |
| retrieval | N/A | 0.667 ± 0.289 | 1.000 ± 0.000 | 0.728 ± 0.079 |

## Supplementary Context Markers

| Method | Location | Time | Weather |
|---|---:|---:|---:|
| char-output | 0.708 ± 0.220 | 0.444 ± 0.354 | 0.278 ± 0.173 |
| char-full | 0.639 ± 0.337 | 0.528 ± 0.347 | 0.375 ± 0.331 |
| words-flat | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| words-slots | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| retrieval | 0.333 ± 0.289 | 1.000 ± 0.000 | 1.000 ± 0.000 |

These checks require explicit vocabulary or generator-specific phrases. They do not establish causal plausibility, interestingness, or full semantic consistency. Missing markers or reasonable paraphrases may fail. High overlap F1 does not compensate for an incorrect actor.

## Per-Run Checkpoints and Conditioning Diagnostics

| Method | Seed | Selected epoch | Conditioning NLL gap | NLL unit |
|---|---:|---:|---:|---|
| char-output | 42 | 27 | -0.0966 | character |
| char-output | 43 | 39 | -0.3271 | character |
| char-output | 44 | 39 | 0.1809 | character |
| char-full | 42 | 39 | -0.0774 | character |
| char-full | 43 | 47 | -0.3852 | character |
| char-full | 44 | 56 | 0.2488 | character |
| words-flat | 42 | 100 | 1.1245 | word |
| words-flat | 43 | 100 | 0.8947 | word |
| words-flat | 44 | 100 | 1.1786 | word |
| words-slots | 42 | 100 | 1.2577 | word |
| words-slots | 43 | 100 | 0.9953 | word |
| words-slots | 44 | 99 | 1.2367 | word |

NLL units differ between tokenizers. Compare conditioning gaps within a method; do not compare character and word magnitudes directly. A positive gap indicates a likelihood advantage for the correct state in this diagnostic, not proof of general narrative quality.

## Interpretation and Limits

- Word-flat and word-slots share architecture, capacity, optimizer, and seed choices. Their comparison examines memory construction. A character-to-word comparison also changes tokenization, architecture, and capacity, so it cannot isolate a field-encoding effect.
- The dataset shares rule templates across subsets and has AI review only. It has zero independent human-reviewed targets and no completed human ratings of generated events.
- Earlier test results on this benchmark were inspected. The current settings were frozen before the multi-seed test phase, but this remains an exploratory synthetic benchmark rather than a pristine final test.
- Split membership and initialization both vary with seed. Three-run standard deviations summarize these runs, not statistical significance or initialization-only variance.
- Each state has one target. Exact-match differences can reflect alternate valid phrasing or actions; no reference-overlap metric alone proves semantic quality.

## Reproduction and Evidence

See [Conditioning Experiment Protocol](CONDITIONING_EXPERIMENT.md) for the frozen setup, runner commands, and generation demo. The complete comparison, frozen plan, run manifests, histories, predictions, and unscored review sheets are retained in `docs/conditioning_results/`. Checkpoints and logs remain in the original `results/` run directories. No human review scores have been filled automatically.

## Measured Takeaways and Current Research Experience Wording

The matched word models both substantially outperform retrieval on overlap F1 and explicit NPC markers within this controlled benchmark. Across three seeds, word-flat has F1 0.904 ± 0.014 and NPC marker accuracy 1.000; word-slots has F1 0.894 ± 0.024 and NPC/player marker accuracy 1.000. Retrieval has F1 0.728 ± 0.079 and NPC marker accuracy 0.667 ± 0.289. These are observed benchmark results, not human semantic ratings.

Field memory did not provide a clear general advantage over flat word memory: flat memory has a higher mean F1, while slots have perfect player markers versus 0.986 for flat memory. These small differences across three runs do not establish statistical superiority. The strongest supported finding is that the word-level conditional decoder pipeline handles this low-data template task much more reliably than the original character-prefix models. Tokenization, decoder architecture, and capacity all change in that comparison, so the gain cannot be attributed to field encoding alone.

A suitable research-experience statement is:

“Designed a leakage-audited 192-state synthetic benchmark for compositional game-event generation and compared retrieval, character-prefix language models, and word-level conditional Transformers across three seeds. Achieved 0.904 ± 0.014 token-overlap F1 with a word-level decoder versus 0.728 ± 0.079 for retrieval, while measuring explicit actor/state consistency and analyzing conditioning and loss ablations.”

Disclose that targets are template-derived and AI-reviewed, and that no independent human ratings have been completed. The separate original 135-example course dataset is retained. This supports a research-project description, not a claim of a new architecture, a published result, or general open-ended narrative ability.
