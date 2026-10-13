# Research Experience: Current Evidence

## Resume Version

**World-State-Conditioned Game Event Generation — Research Prototype**

- Developed a PyTorch evaluation pipeline for game-event generation using grouped holdout splits, event-only loss, validation-based checkpoint selection, and reproducible experiment manifests.
- Built a 192-state template-derived, AI-reviewed synthetic benchmark; compared four Transformer configurations with retrieval baselines across three seeds and twelve training runs.
- Obtained mean token-overlap F1 of 0.904 (sample SD 0.014) with a word-level conditional decoder versus 0.728 (SD 0.079) for retrieval; analyzed lexical state consistency, conditioning sensitivity, and loss ablations.

## Research Statement Version

I extended a course project into a controlled study of world-state-conditioned game-event generation. I identified sparse field coverage and limitations in the original character-level generation pipeline, then implemented grouped holdout protocols, validation-based checkpoint selection, retrieval comparisons, and conditioning diagnostics. On a 192-state template-derived, AI-reviewed synthetic benchmark, I compared four model configurations across three seeds. A word-level conditional decoder achieved mean token-overlap F1 of 0.904 (sample SD 0.014), compared with 0.728 (SD 0.079) for retrieval. The matched word-model comparison did not establish a clear advantage for explicit field memory over flat state text. I also prepared independent-reference collection and two-reviewer evaluation workflows; those human studies remain pending.

## Interview Explanation

The main contribution is a reproducible evaluation and analysis pipeline, rather than a new Transformer architecture. The strongest observed improvement is for the word-level conditional decoder pipeline on this controlled synthetic task. Changes in tokenization, architecture, and capacity prevent attributing that improvement solely to field encoding. Token overlap and explicit state markers are automated diagnostics, not human quality or creativity ratings. Seeds vary both splitting and initialization; their sample standard deviations do not establish statistical significance.

The project currently supports a research-prototype experience entry. It does not establish publication, deployment, methodological novelty, or independent human-validated narrative quality. See [the measured results](CONDITIONING_RESULTS.md) and [remaining validation](FINAL_VALIDATION.md).
