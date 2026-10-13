## Measured Takeaways and Current Research Experience Wording

The matched word models both substantially outperform retrieval on overlap F1 and explicit NPC markers within this controlled benchmark. Across three seeds, word-flat has F1 0.904 ± 0.014 and NPC marker accuracy 1.000; word-slots has F1 0.894 ± 0.024 and NPC/player marker accuracy 1.000. Retrieval has F1 0.728 ± 0.079 and NPC marker accuracy 0.667 ± 0.289. These are observed benchmark results, not human semantic ratings.

Field memory did not provide a clear general advantage over flat word memory: flat memory has a higher mean F1, while slots have perfect player markers versus 0.986 for flat memory. These small differences across three runs do not establish statistical superiority. The strongest supported finding is that the word-level conditional decoder pipeline handles this low-data template task much more reliably than the original character-prefix models. Tokenization, decoder architecture, and capacity all change in that comparison, so the gain cannot be attributed to field encoding alone.

A suitable research-experience statement is:

“Designed a leakage-audited 192-state synthetic benchmark for compositional game-event generation and compared retrieval, character-prefix language models, and word-level conditional Transformers across three seeds. Achieved 0.904 ± 0.014 token-overlap F1 with a word-level decoder versus 0.728 ± 0.079 for retrieval, while measuring explicit actor/state consistency and analyzing conditioning and loss ablations.”

Disclose that targets are template-derived and AI-reviewed, and that no independent human ratings have been completed. The separate original 135-example course dataset is retained. This supports a research-project description, not a claim of a new architecture, a published result, or general open-ended narrative ability.
