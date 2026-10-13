"""Create unscored, shuffled review sheets; never invent human ratings."""
import csv
import json
import random
from protocol import ROOT
from run_conditioning import METHODS, SEEDS, OUT, run_path


def main():
    result = json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
    destination = ROOT/'docs/human_evaluation'
    if any((destination/name).exists() for name in ('reviewer_a.csv','reviewer_b.csv','system_key.json')):
        raise ValueError('Review artifacts already exist; refusing to overwrite potential human ratings.')
    destination.mkdir(parents=True,exist_ok=True)
    methods = [*METHODS,'retrieval']
    random.Random(20261012).shuffle(methods)
    mapping = {method:f'S{i+1:02d}' for i,method in enumerate(methods)}
    rows = []
    for seed in SEEDS:
        for method in methods:
            path = run_path(method,seed) if method != 'retrieval' else run_path('char-output',seed)
            indices = json.loads((path/'manifest.json').read_text(encoding='utf-8'))['indices']['test']
            if method.startswith('words-'):
                with (path/'test_predictions.csv').open(newline='',encoding='utf-8-sig') as stream:
                    predictions = list(csv.DictReader(stream))
                items = [(r['input'],r['reference'],r['word_transformer']) for r in predictions]
            else:
                name = 'char-output' if method=='retrieval' else method
                folder = OUT/f'{name}-{seed}-evaluation'
                key = json.loads((folder/'system_key.json').read_text(encoding='utf-8'))
                system = next(label for label,value in key.items() if value == ('retrieval' if method=='retrieval' else 'transformer'))
                with (folder/'human_review.csv').open(newline='',encoding='utf-8-sig') as stream:
                    predictions = [r for r in csv.DictReader(stream) if r['system']==system]
                items = [(r['state'],r['reference'],r['prediction']) for r in predictions]
            assert len(items)==len(indices)==24
            for index,(state,reference,prediction) in zip(indices,items):
                rows.append(dict(item_id=f'{seed}-dense-{index:03d}-{mapping[method]}',seed=seed,
                    example_id=f'dense-{index:03d}',system=mapping[method],state=state,
                    reference=reference,prediction=prediction,state_consistency_1_to_5='',
                    coherence_1_to_5='',interestingness_1_to_5='',reviewer='',notes=''))
    assert len(rows)==360 and len({r['item_id'] for r in rows})==360
    for name,salt in [('reviewer_a',1201),('reviewer_b',1202)]:
        shuffled = rows.copy()
        random.Random(salt).shuffle(shuffled)
        with (destination/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(shuffled)
    (destination/'system_key.json').write_text(json.dumps({label:method for method,label in mapping.items()},indent=2),encoding='utf-8')
    print({'rows_per_reviewer':len(rows),'completed_human_ratings':0,'systems':5,'seeds':3})

if __name__ == '__main__':
    main()
