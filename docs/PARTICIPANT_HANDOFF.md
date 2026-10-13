# Participant Handoff

## Coordinator Checklist

1. Recruit an event author and a different event checker for the 24-state independent-reference collection.
2. Recruit two independent output raters. To limit exposure to target writing and method information, use different participants for rating when feasible; disclose any overlap.
3. Send authors only the collection sheet and author instructions below. Send each rater only their own rating sheet and the rating instructions. Do not send method results, the repository, or system keys to participants before they finish.
4. Preserve original files. Ask participants to return UTF-8 CSV copies; keep author/checker IDs and rater IDs stable, using pseudonyms if preferred.
5. Run the commands in [Final Validation](FINAL_VALIDATION.md) after receiving the completed files. Preserve raw returns before copying scores into the project sheets.

The existing 24 states and 360-item sheets are predefined. Do not drop difficult examples or choose a subset after viewing results. No completion-time estimate has been measured; participants may work in multiple sessions.

## Invitation: Event Author

> I am evaluating a research prototype that generates short game events from a world state. Could you independently write one English event for each of 24 states? You will receive a CSV containing the state and blank answer fields. Please work without AI assistance, model predictions, or existing reference events. The events should be natural, plausible, and consistent with the specified state. Participation is voluntary; please let me know whether you can complete the collection before starting.

Provide `data/independent_test.csv` as a copy. Fill only `output` and `author`; preserve `example_id` and `input`. Use one stable author identifier.

For each row, write a short, complete English game event involving the given NPC and player. Respect Location, Time, Player, NPC, and Weather; do not invent a conflicting condition. New interactions or objects are allowed when plausible. There is no required wording or template. Each event may be one or two sentences; avoid unnecessarily long text. Do not use the same sentence with only field words substituted throughout the collection.

Save as UTF-8 CSV. Do not change headers, delete rows, or add model-generated drafts. If a state is unclear, leave a separate note for the coordinator rather than changing the state.

## Invitation: Event Checker

> Could you independently check 24 human-written game events against their supplied world states? Please verify that each event is understandable and does not contradict the location, time, player condition, NPC, or weather. Please work without AI assistance or model outputs, and return unclear or conflicting items to the author for correction.

Use an identifier different from the author. Once a row is acceptable, fill `reviewer`, `reviewed=yes`, and `source=human_written`. Only use this source label when the author confirms independent human writing. Preserve the author's identity and final wording; return corrections to the author instead of silently replacing their events. Retain revision notes separately.

These provenance entries attest to the collection process; software cannot independently verify authorship.

## Invitation: Output Rater

> I am comparing five systems for short game-event generation. Could you independently rate a shuffled sheet of 360 outputs? Each item includes its world state, one possible reference, and a generated prediction. Please rate state consistency, coherence, and interestingness from 1 to 5, using the accompanying rubric. You may complete the sheet across several sessions. Please avoid AI assistance, method results, or discussing ratings with the other rater until both sheets are finalized. Participation is voluntary; please confirm whether the full sheet is feasible before starting.

Send one participant a copy of `docs/human_evaluation/reviewer_a.csv`, and the other a copy of `reviewer_b.csv`. Include only [the rating instructions](human_evaluation/README.md). Do not include `system_key.json`, aggregate results, or the other participant's sheet.

Each rater fills the three score columns and a stable `reviewer` identifier on every row. Notes are encouraged for conflicts and ambiguity. Keep all other columns unchanged. The reference is one possible answer, not a required phrase match. The complete rating rubric is in the linked instructions.

## Current Evidence

The synthetic comparison is complete. Independent references and human ratings have not been supplied. These invitations have been prepared for the coordinator to send; no participants have been contacted by the assistant.
