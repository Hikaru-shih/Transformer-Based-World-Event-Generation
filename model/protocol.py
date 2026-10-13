"""Dependency-free audit and leakage-safe, deterministic experiment splits."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('Location', 'Time', 'Player', 'NPC', 'Weather')

def parse_state(text):
    state = {}
    for part in text.split(';'):
        key, value = part.strip().split(':', 1)
        if key in state:
            raise ValueError('Duplicate field: ' + key)
        state[key] = value.strip()
    if set(state) != set(FIELDS) or not all(state.values()):
        raise ValueError('Expected five nonempty world-state fields')
    return state

def canonical(text):
    state = parse_state(text)
    return '; '.join(f'{key}: {state[key]}' for key in FIELDS)

def load_data(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    for row in data:
        row['input'] = canonical(row['input'])
        if not isinstance(row['output'], str) or not row['output'].strip():
            raise ValueError('Empty event')
    return data

def split_data(data, seed=42, mode='random'):
    if mode not in ('random', 'compositional', 'compositional_seen'):
        raise ValueError('Unknown split mode: ' + mode)
    # Keep identical states together; compositional split holds out Location/NPC pairs.
    groups = {}
    for i, row in enumerate(data):
        s = parse_state(row['input'])
        key = row['input'] if mode == 'random' else (s['Location'], s['NPC'])
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('At least three groups required')
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * .15))
    n_val = max(1, round(len(keys) * .15))
    if mode == 'compositional_seen':
        # Only remove a group when all of its field values remain in training.
        # Retry deterministically to avoid an unlucky greedy ordering.
        states = [parse_state(row['input']) for row in data]
        for attempt in range(100):
            candidates = sorted(groups)
            random.Random(seed + attempt * 1009).shuffle(candidates)
            remaining = set(range(len(data)))
            held = []
            for key in candidates:
                proposed = remaining - set(groups[key])
                seen = {field: {states[i][field] for i in proposed} for field in FIELDS}
                if all(states[i][field] in seen[field] for i in groups[key] for field in FIELDS):
                    remaining = proposed
                    held.append(key)
                if len(held) == n_test + n_val:
                    result = dict(test=sorted(i for k in held[:n_test] for i in groups[k]),
                                  validation=sorted(i for k in held[n_test:] for i in groups[k]),
                                  train=sorted(remaining))
                    return result
        raise ValueError('Cannot hold out enough pairs while keeping every field value in training. '
                         'Collect denser data or use the legacy compositional split.')
    cuts = {'test': keys[:n_test], 'validation': keys[n_test:n_test+n_val], 'train': keys[n_test+n_val:]}
    return {name: sorted(i for key in selected for i in groups[key]) for name, selected in cuts.items()}


def coverage_report(data, splits):
    states = [parse_state(row['input']) for row in data]
    train = [states[i] for i in splits['train']]
    values = {field: {s[field] for s in train} for field in FIELDS}
    pairs = {(s['Location'], s['NPC']) for s in train}
    train_chars = set(''.join(prompt(data[i]['input']) + data[i]['output'] for i in splits['train']))
    report = {}
    for name in ('validation', 'test'):
        indices = splits[name]
        report[name] = {
            'examples': len(indices),
            'unseen_values': {field: sorted({states[i][field] for i in indices} - values[field])
                              for field in FIELDS},
            'examples_with_unseen_values': sum(any(states[i][f] not in values[f] for f in FIELDS)
                                              for i in indices),
            'location_npc_pairs_shared_with_train': len({(states[i]['Location'], states[i]['NPC'])
                                                       for i in indices} & pairs),
            'unseen_input_characters': sorted(set(''.join(prompt(data[i]['input']) for i in indices)) - train_chars),
            'unseen_output_characters': sorted(set(''.join(data[i]['output'] for i in indices)) - train_chars),
        }
    return report

def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def prompt(state):
    return 'IN:' + canonical(state) + '\nOUT:'
