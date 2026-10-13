# Reviewing the Event Drafts

## Current Status

All 192 planned states have an English rule-template draft in `data/dense_annotation.csv`. These are generated suggestions, not independently human-written events. No final output, reviewer, or review approval has been filled automatically. `draft_source=rule_template_v1` records how each suggestion was produced.

The reusable rules intentionally make the draft task easy to inspect. They also limit language diversity: a benchmark based entirely on these drafts measures learning of a controlled template distribution, not open-ended narrative creativity. Human review checks plausibility but does not remove this limitation. For a broader narrative claim, separately collect independently written examples and keep their provenance distinct.

## Start with One Batch

Review `dense-000` through `dense-023` first: the 24 forest/morning states. Then work in batches of 24. There are eight batches in total, one for each Location/Time combination. Do not train on unreviewed drafts.

For each row:

1. Read `input` and compare the suggestion in `draft_event` with all five fields.
2. Check that the correct NPC acts, the action fits the player condition, and the location/time/weather make sense. A cave entrance is used to make outdoor weather plausible in cave scenarios. Ask whether a healthy player's activities are safe in rain and whether a night activity needs a light source.
3. Check that the English sentence is natural, understandable, and not just a list of input fields. Edit repetitive phrasing and vague actions where appropriate.
4. Copy an accepted draft into `output`, or write an improved final event there. Keep `draft_event` and `draft_source` as provenance records.
5. For an accepted or adapted template draft, set `source` to `rule_template_human_reviewed`. Use `human_written` only for an independently written event, and `ai_assisted_human_reviewed` for an actual AI-assisted event.
6. Set `reviewed` to `yes` and enter your name or stable reviewer ID in `reviewer` only after checking the final event. Record uncertainty or important edits in `notes`.

Keep `example_id`, `planned_split`, and `input` unchanged. Do not blanket-fill review approvals before reading the events. Completing a batch does not imply all 192 rows are ready for export.

## Column Example

| Column | Example value after actual review |
|---|---|
| output | At the forest trail on a clear morning, a hunter invites the healthy player to track deer. |
| source | rule_template_human_reviewed |
| reviewed | yes |
| reviewer | your stable reviewer ID |
| notes | Optional explanation of an edit or ambiguity |

This example demonstrates the format; it is not an assertion that the corresponding row has been reviewed.

A spreadsheet editor is usually easier for this CSV. Save as UTF-8 CSV with the column names unchanged. If editing in VS Code, preserve CSV quoting: fields containing commas must be enclosed in double quotes.

## After All Rows Have Been Reviewed

In WSL, from the project root:

```bash
source .venv-wsl/bin/activate
python model/annotation.py export
python model/audit.py --data data/dense_reviewed.json --out results/dense_audit.json
```

Export will fail until all 192 rows have valid final events, provenance, and explicit review. Do not work around the review requirement by copying every draft and marking every row without inspection. After export, use the training commands in `DENSE_DATASET.md` and freeze settings before examining new test results.

## Regenerating Missing Drafts

```bash
python model/draft_events.py
```

This fills only empty draft rows without final outputs or review flags. Existing drafts and reviewed work are preserved. It never populates final answers or claims human review.
