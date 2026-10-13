"""Frozen multi-seed comparison; all training/validation finishes before test evaluation."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from protocol import ROOT, load_data, fingerprint

METHODS = ('char-output','char-full','words-flat','words-slots')
SEEDS = (42,43,44)
DATA = ROOT/'data/dense_ai_reviewed.json'
OUT = ROOT/'results/research-conditioning'

def run_path(method,seed):
    if seed == 42 and method == 'char-output':
        return ROOT/'results/research-synthetic-output-42'
    if seed == 42 and method.startswith('words-'):
        return ROOT/f'results/research-words-{method[6:]}-42'
    return OUT/f'{method}-{seed}'

def call(arguments,log):
    with log.open('w',encoding='utf-8') as stream:
        result = subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed; inspect {log}')

def verify_existing(path,method,seed,sha):
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest['data_sha256'] != sha or manifest['seed'] != seed:
        raise ValueError('Existing run differs from frozen plan: '+str(path))
    if manifest.get('split','compositional_seen') != 'compositional_seen':
        raise ValueError('Unexpected split')
    if method.startswith('words-'):
        if manifest['conditioning'] != method[6:] or manifest['epochs'] != 100 or manifest['config'] != dict(emb=64,depth=2,block_size=96):
            raise ValueError('Unexpected word model configuration')
    else:
        expected = 'output' if method == 'char-output' else 'full'
        if manifest['objective'] != expected or manifest['arguments']['epochs'] != 100 or manifest['model_config']['emb'] != 128:
            raise ValueError('Unexpected character model configuration')
    history = json.loads((path/'history.json').read_text(encoding='utf-8'))
    metric = 'validation_word_nll' if method.startswith('words-') else 'validation_event_nll'
    best = min(history,key=lambda row:row[metric])['epoch']
    if history[-1]['epoch'] < 100 and history[-1]['epoch']-best < 15:
        raise ValueError('Incomplete run; use a fresh directory rather than silently reusing it')
    if not (path/'best.pt').exists():
        raise ValueError('Missing checkpoint')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['train','test'],required=True)
    args = parser.parse_args()
    sha = fingerprint(load_data(DATA))
    plan = dict(dataset_sha256=sha,seeds=list(SEEDS),methods=list(METHODS),split='compositional_seen',
                epochs=100,patience=15,lr=3e-4,greedy_decoding=True,
                word_config=dict(emb=64,depth=2,block_size=96),
                char_config=dict(emb=128,depth=2,block_size=256),
                limitation='Previously inspected synthetic benchmark; not a pristine human-authored final test.',
                primary_metrics=['npc_explicit_accuracy','player_explicit_accuracy','token_overlap_f1'])
    OUT.mkdir(parents=True,exist_ok=True)
    plan_path = OUT/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Frozen plan changed')
    elif args.phase == 'test':
        raise ValueError('Freeze and train the plan first')
    else:
        plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if args.phase == 'train':
        for method in METHODS:
            for seed in SEEDS:
                path = run_path(method,seed)
                if (path/'manifest.json').exists():
                    verify_existing(path,method,seed,sha)
                    print(f'Reuse completed {method} seed={seed}',flush=True)
                    continue
                print(f'Train {method} seed={seed}',flush=True)
                if method.startswith('words-'):
                    command = ['model/conditional_words.py','--data',DATA,'--conditioning',method[6:],
                               '--seed',seed,'--out',path,'--epochs',100,'--phase','validation']
                else:
                    command = ['model/train.py','--data',DATA,'--split','compositional_seen','--seed',seed,
                               '--objective','output' if method=='char-output' else 'full','--out',path,'--epochs',100]
                call(command,OUT/f'{method}-{seed}-training.log')
                verify_existing(path,method,seed,sha)
        print('All twelve model runs are complete. The test phase can now execute.',flush=True)
        return
    for method in METHODS:
        for seed in SEEDS:
            verify_existing(run_path(method,seed),method,seed,sha)
    records = []
    for method in METHODS:
        for seed in SEEDS:
            path = run_path(method,seed)
            print(f'Evaluate {method} seed={seed}',flush=True)
            if method.startswith('words-'):
                call(['model/conditional_words.py','--data',DATA,'--conditioning',method[6:],
                      '--seed',seed,'--out',path,'--phase','test'],OUT/f'{method}-{seed}-evaluation.log')
                result = json.loads((path/'test_metrics.json').read_text(encoding='utf-8'))
                values = result['metrics']['word_transformer']
            else:
                call(['model/evaluate.py','--data',DATA,'--run',path,'--out',OUT/f'{method}-{seed}-evaluation'],
                     OUT/f'{method}-{seed}-evaluation.log')
                result = json.loads((OUT/f'{method}-{seed}-evaluation/metrics.json').read_text(encoding='utf-8'))
                values = result['metrics']['transformer']
            records.append(dict(method=method,seed=seed,metrics=values,parameters=json.loads((path/'manifest.json').read_text(encoding='utf-8'))['parameters'],
                                conditioning_nll_gap=result['conditioning_nll_gap']))
            if method == 'char-output':
                records.append(dict(method='retrieval',seed=seed,metrics=result['metrics']['retrieval']))
    summary = {}
    for method in (*METHODS,'retrieval'):
        rows = [r for r in records if r['method']==method]
        summary[method] = {key:dict(mean=statistics.mean(r['metrics'][key] for r in rows),
                                   sd=statistics.stdev(r['metrics'][key] for r in rows))
                           for key in rows[0]['metrics']}
    report = dict(plan=plan,runs=records,summary=summary,
                  notes='NLL units differ across character/word models; do not compare NLL gaps across tokenizers.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
