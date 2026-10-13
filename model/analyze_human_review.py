"""Validate paired independent ratings; report descriptive agreement, never impute scores."""
import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = ('state_consistency_1_to_5', 'coherence_1_to_5', 'interestingness_1_to_5')
IDENTITY = ('item_id', 'seed', 'example_id', 'system', 'state', 'reference', 'prediction')


def read_sheet(path, expected):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        key = row['item_id']
        if key in indexed or key not in expected:
            raise ValueError('Duplicate or unexpected item: ' + key)
        if any(row[field] != expected[key][field] for field in IDENTITY):
            raise ValueError('Review item was modified: ' + key)
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError('Missing review items')
    completed, reviewers = {}, set()
    for key, row in indexed.items():
        values = [row[m].strip() for m in METRICS]
        if not any(values) and not row['reviewer'].strip():
            continue
        if not all(v in ('1', '2', '3', '4', '5') for v in values) or not row['reviewer'].strip():
            raise ValueError('Every rated item needs three integer scores 1-5 and a reviewer: ' + key)
        reviewers.add(row['reviewer'].strip())
        completed[key] = row | {m: int(row[m]) for m in METRICS}
    if len(reviewers) > 1:
        raise ValueError('Each sheet must belong to one reviewer')
    return completed, reviewers


def analyze(a, b, key_path, template):
    with Path(template).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {r['item_id']: r for r in rows}
    if len(expected) != len(rows) or not rows:
        raise ValueError('Invalid canonical review template')
    mapping = json.loads(Path(key_path).read_text(encoding='utf-8'))
    left, ra = read_sheet(a, expected)
    right, rb = read_sheet(b, expected)
    if ra & rb:
        raise ValueError('Two distinct reviewer identifiers are required')
    complete = len(left) == len(right) == len(expected)
    report = dict(status='complete' if complete else 'awaiting_human_ratings',
                  expected_per_reviewer=len(expected), completed_a=len(left), completed_b=len(right),
                  paired_items=len(set(left) & set(right)), systems={},
                  limitation='Descriptive item averages only; repeated states and seeds are not independent samples. No significance or semantic claims are inferred.')
    # Incomplete sheets get a progress report only, preventing selective-subset conclusions.
    if complete:
        for label in sorted({r['system'] for r in rows}):
            ids = [r['item_id'] for r in rows if r['system'] == label]
            metrics = {}
            for metric in METRICS:
                differences = [abs(left[i][metric] - right[i][metric]) for i in ids]
                metrics[metric] = dict(reviewer_a_mean=statistics.mean(left[i][metric] for i in ids),
                    reviewer_b_mean=statistics.mean(right[i][metric] for i in ids),
                    paired_mean=statistics.mean((left[i][metric]+right[i][metric])/2 for i in ids),
                    exact_agreement=sum(d == 0 for d in differences)/len(ids),
                    within_one_agreement=sum(d <= 1 for d in differences)/len(ids),
                    mean_absolute_difference=statistics.mean(differences))
            report['systems'][mapping[label]] = dict(items=len(ids), metrics=metrics)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True,
                   help='Preserved unscored sheet; do not edit this canonical copy')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.a, args.b, args.key, args.template)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
