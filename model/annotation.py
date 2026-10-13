"""Build a balanced annotation plan and export only explicitly reviewed events."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from protocol import ROOT, FIELDS, canonical, fingerprint, split_data, coverage_report

VALUES = {
    'Location': ('forest', 'village', 'cave', 'river'),
    'Time': ('morning', 'night'),
    'Player': ('healthy', 'injured', 'hungry'),
    'NPC': ('hunter', 'merchant', 'healer', 'traveler'),
    'Weather': ('clear', 'rainy'),
}
COLUMNS = ['example_id', 'planned_split', 'input', 'draft_event', 'draft_source', 'output',
           'source', 'reviewed', 'reviewer', 'notes']
SOURCES = {'human_written', 'ai_assisted_human_reviewed', 'rule_template_human_reviewed'}

def planned_data():
    return [dict(input=canonical('; '.join(f'{k}: {v}' for k,v in zip(FIELDS, values))),
                 output='Annotation pending.')
            for values in itertools.product(*(VALUES[k] for k in FIELDS))]

def create_sheet(path):
    if path.exists():
        raise ValueError('Annotation sheet already exists; refusing to overwrite it.')
    data = planned_data()
    splits = split_data(data, 42, 'compositional_seen')
    roles = {i: name for name, indices in splits.items() for i in indices}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for i,row in enumerate(data):
            writer.writerow(dict(example_id=f'dense-{i:03d}', planned_split=roles[i], input=row['input']))
    return {'examples': len(data), 'counts': {k:len(v) for k,v in splits.items()},
            'coverage': coverage_report(data, splits)}

def read_reviewed(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    expected = planned_data()
    if len(rows) != len(expected):
        raise ValueError('Complete all 192 planned rows; partial export changes the benchmark.')
    roles = {i: name for name, indices in split_data(expected, 42, 'compositional_seen').items()
             for i in indices}
    indexed = {}
    for row in rows:
        identifier = row.get('example_id', '')
        if identifier in indexed:
            raise ValueError('Duplicate example ID: ' + identifier)
        indexed[identifier] = row
    data = []
    for i, planned in enumerate(expected):
        identifier = f'dense-{i:03d}'
        row = indexed.get(identifier)
        if row is None:
            raise ValueError('Missing row: ' + identifier)
        if canonical(row['input']) != planned['input'] or row['planned_split'] != roles[i]:
            raise ValueError('Planned input/split was changed: ' + identifier)
        if row.get('reviewed', '').strip().lower() != 'yes' or not row.get('reviewer', '').strip():
            raise ValueError('Human review is incomplete: ' + identifier)
        if row.get('source', '').strip() not in SOURCES:
            raise ValueError('Specify an allowed provenance source: ' + identifier)
        output = row.get('output', '').strip()
        if not output or output == 'Annotation pending.':
            raise ValueError('Missing reviewed event: ' + identifier)
        if any('\u3400' <= char <= '\u9fff' for char in output):
            raise ValueError('Events must be written in English: ' + identifier)
        data.append(dict(input=planned['input'], output=output, example_id=identifier,
                         source=row['source'].strip(), reviewer=row['reviewer'].strip(),
                         reviewed=True, draft_source=row.get('draft_source', ''), notes=row.get('notes', '')))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'export'])
    parser.add_argument('--sheet', type=Path, default=ROOT/'data/dense_annotation.csv')
    parser.add_argument('--out', type=Path, default=ROOT/'data/dense_reviewed.json')
    args = parser.parse_args()
    if args.action == 'create':
        print(json.dumps(create_sheet(args.sheet), indent=2))
    else:
        if args.out.exists():
            raise ValueError('Reviewed dataset already exists; choose a new --out path.')
        data = read_reviewed(args.sheet)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(json.dumps({'examples':len(data), 'data_sha256':fingerprint(data)}, indent=2))

if __name__ == '__main__':
    main()
