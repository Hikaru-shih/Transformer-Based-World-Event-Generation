"""Prepare a seed-42 held-pair human collection and evaluate frozen word models."""
import argparse
import csv
import json
from pathlib import Path
from protocol import ROOT, load_data, fingerprint, parse_state, canonical, FIELDS

COLUMNS = ('example_id', 'input', 'output', 'author', 'reviewer', 'reviewed', 'source')


def validate(rows, train):
    if not rows:
        raise ValueError('No independent examples')
    seen = {f: {parse_state(r['input'])[f] for r in train} for f in FIELDS}
    pairs = {(parse_state(r['input'])['Location'], parse_state(r['input'])['NPC']) for r in train}
    ids, states = set(), set()
    for row in rows:
        state = parse_state(row['input'])
        text = canonical(row['input'])
        if not row['example_id'] or row['example_id'] in ids or text in states:
            raise ValueError('Missing/duplicate ID or state')
        ids.add(row['example_id'])
        states.add(text)
        if any(state[f] not in seen[f] for f in FIELDS) or (state['Location'], state['NPC']) in pairs:
            raise ValueError('Test must use seen values and held-out Location/NPC pairs')
        if (not row['output'].strip() or not row['author'].strip() or not row['reviewer'].strip()
                or row['author'].strip() == row['reviewer'].strip()
                or row['reviewed'].strip().lower() != 'yes' or row['source'] != 'human_written'):
            raise ValueError('Human event, independent author/reviewer, reviewed=yes and source=human_written required')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'evaluate'], required=True)
    p.add_argument('--sheet', type=Path, default=ROOT/'data/independent_test.csv')
    p.add_argument('--out', type=Path, default=ROOT/'results/independent-test')
    args = p.parse_args()
    data = load_data(ROOT/'data/dense_ai_reviewed.json')
    run = ROOT/'results/research-words-flat-42'
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['data_sha256'] != fingerprint(data):
        raise ValueError('Training dataset changed')
    train = [data[i] for i in manifest['indices']['train']]
    if args.phase == 'prepare':
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        with args.sheet.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for i in manifest['indices']['test']:
                writer.writerow(dict(example_id=f'independent-{i:03d}', input=data[i]['input']))
        print('Prepared 24 blank human-authored targets; no references or model outputs supplied.')
        return
    with args.sheet.open(encoding='utf-8-sig', newline='') as stream:
        rows = validate(list(csv.DictReader(stream)), train)
    expected = {canonical(data[i]['input']) for i in manifest['indices']['test']}
    if {canonical(r['input']) for r in rows} != expected:
        raise ValueError('Complete the predefined 24-state collection; no selective subset')
    import torch
    from conditional_words import WordTokenizer, ConditionalDecoder, generate, tokens
    from evaluate import summarize, retrieve
    from state_metrics import slot_metrics
    outputs = {'retrieval': [retrieve(r['input'], train) for r in rows]}
    unknown = {}
    for condition in ('flat', 'slots'):
        path = ROOT/f'results/research-words-{condition}-42'
        m = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if m['data_sha256'] != manifest['data_sha256'] or m['indices'] != manifest['indices'] or m['conditioning'] != condition or m['seed'] != 42:
            raise ValueError('Frozen models must share training data and seed-42 split')
        t = WordTokenizer([])
        t.itos = m['vocabulary']
        t.stoi = {v: i for i, v in enumerate(t.itos)}
        model = ConditionalDecoder(len(t.itos), conditioning=condition, **m['config'])
        model.load_state_dict(torch.load(path/'best.pt', map_location='cpu', weights_only=True))
        outputs['words-'+condition] = [generate(model, t, row) for row in rows]
        unknown[condition] = sorted({w for r in rows for w in tokens(r['output']) if w not in t.stoi})
    report = dict(test_sha256=fingerprint(rows), training_sha256=manifest['data_sha256'], seed=42,
        examples=len(rows), unknown_reference_words=unknown, metrics={},
        limitation='Human authorship is attested in the CSV, not independently verified. Single seed; new references on predefined synthetic states, not new environments. No retraining or test-based tuning permitted.')
    for method, predictions in outputs.items():
        report['metrics'][method] = summarize(predictions, [r['output'] for r in rows], {r['output'] for r in train}) | slot_metrics(predictions, rows, train)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.out/'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['example_id', 'input', 'reference', *outputs])
        writer.writerows([r['example_id'], r['input'], r['output'], *[preds[i] for preds in outputs.values()]] for i, r in enumerate(rows))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
