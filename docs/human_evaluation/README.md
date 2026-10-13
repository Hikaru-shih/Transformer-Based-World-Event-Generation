# Independent Human Evaluation

Two unscored sheets, `reviewer_a.csv` and `reviewer_b.csv`, contain the same 360 items in different randomized orders: five systems, three seeds, and 24 examples per seed. No human scores or reviewer identities have been filled. These are synthetic game scenarios, not independently authored target data.

Give each reviewer one sheet and these instructions. Keep `system_key.json` separate until the ratings have been finalized. Reviewers should work independently and avoid viewing the method comparison or discussing scores before completion.

For each item, read the world state and generated prediction. The reference is one possible event, not the only correct answer. Rate:

- `state_consistency_1_to_5`: 1 for a clear contradiction, 3 for broadly plausible text that misses important conditions, and 5 for a plausible event that respects important conditions.
- `coherence_1_to_5`: 1 for unreadable or incomplete text, 3 for understandable but unnatural text, and 5 for a clear complete event.
- `interestingness_1_to_5`: 1 for empty or repetitive content, 3 for plausible but ordinary content, and 5 for a specific event with interaction potential.

Enter a stable reviewer identifier and explanatory notes for contradictions or ambiguity. Do not treat a reasonable omission of a field as a contradiction, or reward word overlap with the reference automatically. Lanterns and temporary travel canopies are permissible newly introduced event details in this benchmark.

Save UTF-8 CSV with item IDs and all columns preserved. Complete all items or document a predefined sampling plan before inspecting system outputs. Do not select a convenient subset after seeing which method performs well. Preserve raw ratings and report reviewer disagreement alongside averages. Repeated states across seeds are possible; ratings are not independent samples for a simple significance claim.

Collection-target AI review and these human ratings are different forms of evidence. The assistant prepared the sheets and computed lexical diagnostics but has not supplied human scores or contacted any reviewers.
