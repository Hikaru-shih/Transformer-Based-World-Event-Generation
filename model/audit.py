import argparse
from collections import Counter
import json
from pathlib import Path
from protocol import ROOT, FIELDS, fingerprint, load_data, parse_state, split_data, prompt, coverage_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset.json')
    p.add_argument('--out', type=Path, default=ROOT/'results/data_audit.json')
    a = p.parse_args()
    data = load_data(a.data)
    states = [parse_state(r['input']) for r in data]
    result = dict(examples=len(data), data_sha256=fingerprint(data),
        duplicate_states=len(data)-len({r['input'] for r in data}),
        duplicate_events=len(data)-len({r['output'] for r in data}),
        max_sequence_tokens=max(len(prompt(r['input'])+r['output'])+1 for r in data),
        fields={k: dict(sorted(Counter(s[k] for s in states).items())) for k in FIELDS},
        splits={mode: {k:len(v) for k,v in split_data(data,42,mode).items()} for mode in ['random','compositional']})
    result['coverage'] = {mode: coverage_report(data, split_data(data, 42, mode))
                          for mode in ['random', 'compositional']}
    try:
        strict = split_data(data, 42, 'compositional_seen')
        result['strict_compositional'] = {'indices': strict, 'coverage': coverage_report(data, strict)}
    except ValueError as exc:
        result['strict_compositional'] = {'available': False, 'reason': str(exc)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
