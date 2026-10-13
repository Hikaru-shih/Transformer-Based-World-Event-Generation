"""Held-out comparison; overlap metrics are proxies, not semantic correctness."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import random
from state_metrics import slot_metrics
from protocol import ROOT, fingerprint, load_data, parse_state, split_data, coverage_report

def words(text):
    return re.findall(r"[a-z]+", text.lower())

def overlap_f1(prediction, reference):
    p, r = Counter(words(prediction)), Counter(words(reference))
    matched = sum((p & r).values())
    return 2 * matched / (sum(p.values()) + sum(r.values())) if p or r else 1.

def retrieve(state, train):
    query = parse_state(state)
    # A deterministic state-slot nearest-neighbor baseline; sees training targets only.
    return max(train, key=lambda row: sum(query[k] == parse_state(row['input'])[k] for k in query))['output']

def summarize(predictions, references, training_outputs):
    tokens = [words(p) for p in predictions]
    bigrams = [tuple(t[i:i+2]) for t in tokens for i in range(len(t)-1)]
    return dict(token_overlap_f1=sum(overlap_f1(p,r) for p,r in zip(predictions,references))/len(predictions),
        exact_match=sum(p.strip()==r.strip() for p,r in zip(predictions,references))/len(predictions),
        training_copy_rate=sum(p.strip() in training_outputs for p in predictions)/len(predictions),
        empty_rate=sum(not p.strip() for p in predictions)/len(predictions),
        distinct_2=len(set(bigrams))/len(bigrams) if bigrams else 0.,
        mean_words=sum(map(len,tokens))/len(tokens))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset.json')
    p.add_argument('--run', type=Path, help='Optional trained run; omit for baseline-only evaluation')
    p.add_argument('--out', type=Path, default=ROOT/'results/baseline')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['random','compositional','compositional_seen'], default='random')
    p.add_argument('--max-new', type=int, default=150)
    a = p.parse_args()
    data = load_data(a.data)
    if a.run:
        manifest = json.loads((a.run/'manifest.json').read_text(encoding='utf-8'))
        if fingerprint(data) != manifest['data_sha256']:
            raise ValueError('Dataset differs from training manifest')
        splits = manifest['indices']
    else:
        splits = split_data(data, a.seed, a.split)
    train = [data[i] for i in splits['train']]
    test = [data[i] for i in splits['test']]
    outputs = {'retrieval': [retrieve(r['input'], train) for r in test]}
    report = {'data_sha256': fingerprint(data), 'indices': splits,
        'protocol': {'seed': manifest['seed'] if a.run else a.seed, 'split': manifest['split'] if a.run else a.split},
        'test_examples': len(test), 'coverage': coverage_report(data, splits)}
    if a.run:
        import torch
        from torch.utils.data import DataLoader
        from generate import load_model, generate
        from train import collate, run_epoch
        model, tokenizer = load_model(a.run)
        outputs['transformer'] = [generate(model,tokenizer,r['input'],a.max_new) for r in test]
        loader = DataLoader(test,batch_size=16,collate_fn=lambda rows: collate(rows,tokenizer,model.block_size))
        nll = run_epoch(model,loader,'cpu')
        report['event_nll'] = nll
        report['character_perplexity'] = math.exp(nll)
        # Rotate a seeded permutation: every event receives a different test state.
        if len(test) > 1:
            order = list(range(len(test)))
            random.Random(manifest['seed']).shuffle(order)
            wrong_states = {}
            for pos, index in enumerate(order):
                wrong_states[index] = test[order[(pos+1) % len(order)]]['input']
            shuffled = [dict(input=wrong_states[i], output=row['output']) for i,row in enumerate(test)]
            shuffled_loader = DataLoader(shuffled,batch_size=16,collate_fn=lambda rows: collate(rows,tokenizer,model.block_size))
            shuffled_nll = run_epoch(model,shuffled_loader,'cpu')
            report['shuffled_state_event_nll'] = shuffled_nll
            report['conditioning_nll_gap'] = shuffled_nll - nll
    report['metrics'] = {name: summarize(preds,[r['output'] for r in test], {r['output'].strip() for r in train}) | slot_metrics(preds,test,train) for name,preds in outputs.items()}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.out/'human_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['example_id','state','reference','system','prediction','consistency_1_to_5','coherence_1_to_5','interestingness_1_to_5','contradiction_notes'])
        # Neutral labels; label mapping is separate to support blinded review.
        for j, row in enumerate(test):
            for label, (name,preds) in enumerate(outputs.items()):
                writer.writerow([splits['test'][j],row['input'],row['output'],f'system_{label}',preds[j],'','','',''])
    (a.out/'system_key.json').write_text(json.dumps({f'system_{i}':name for i,name in enumerate(outputs)},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
