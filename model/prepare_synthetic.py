"""Create a separate AI-reviewed synthetic benchmark, never human review evidence."""
import csv
import json
from pathlib import Path
from annotation import planned_data, COLUMNS
from draft_events import ACTIONS
from protocol import ROOT, parse_state, split_data, fingerprint, coverage_report

PLACES = {'forest': 'Along a forest trail', 'village': 'In the village square',
          'cave': 'At the cave entrance', 'river': 'On the riverbank'}

def revised_event(state):
    s = parse_state(state)
    variant = (list(PLACES).index(s['Location']) + (s['Time'] == 'night') + (s['Weather'] == 'rainy')) % 2
    action = ACTIONS[(s['NPC'], s['Player'])][variant]
    if s['NPC'] == 'hunter' and s['Player'] == 'healthy' and s['Time'] == 'night':
        action = 'uses a lantern to show the healthy player animal tracks'
    if s['NPC'] == 'healer' and s['Player'] == 'healthy' and (s['Time'] == 'night' or s['Weather'] == 'rainy'):
        action = 'shows the healthy player medicinal herbs from a pouch'
    # Give night actions explicit illumination; rainy encounters use temporary cover.
    if s['Time'] == 'night':
        context = 'by lantern light'
    else:
        context = 'in the morning'
    if s['Weather'] == 'rainy':
        context += ' beneath a travel canopy'
    else:
        context += ' under clear skies'
    return f'{PLACES[s["Location"]]} {context}, a {s["NPC"]} {action}.'

def main():
    sheet = ROOT/'data/dense_ai_reviewed.csv'
    out = ROOT/'data/dense_ai_reviewed.json'
    if sheet.exists() or out.exists():
        raise ValueError('Synthetic artifact exists; do not overwrite a frozen dataset.')
    with (ROOT/'data/dense_annotation.csv').open(newline='', encoding='utf-8-sig') as stream:
        original = {r['example_id']: r for r in csv.DictReader(stream)}
    planned = planned_data()
    splits = split_data(planned, 42, 'compositional_seen')
    roles = {i: name for name, ids in splits.items() for i in ids}
    rows, data = [], []
    for i, example in enumerate(planned):
        identifier = f'dense-{i:03d}'
        old = original[identifier]
        state = parse_state(example['input'])
        text = revised_event(example['input'])
        assert state['NPC'] in text and state['Player'] in text
        assert len('IN:' + example['input'] + '\nOUT:' + text) + 1 <= 256
        notes = ('AI review of rule-generated targets, not independent human review. '
                 'Corrected location prepositions; supplied illumination at night and temporary cover in rain. '
                 'Travel canopies and lanterns are new event details, not observed input fields.')
        row = dict(example_id=identifier, planned_split=roles[i], input=example['input'],
                   draft_event=old['draft_event'], draft_source=old.get('draft_source', 'rule_template_v1'),
                   output=text, source='rule_template_ai_reviewed', reviewed='yes', reviewer='Codex',
                   review_kind='ai', notes=notes)
        rows.append(row)
        data.append({k:row[k] for k in ('example_id','input','output','source','reviewer','review_kind','draft_source','notes')} | {'reviewed':True})
    with sheet.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS + ['review_kind'])
        writer.writeheader()
        writer.writerows(rows)
    out.write_text(json.dumps(data, indent=2), encoding='utf-8')
    report = dict(examples=len(data), human_reviewed_examples=0, ai_reviewed_examples=len(data),
        data_sha256=fingerprint(data), review_kind='ai', source='rule_template_ai_reviewed',
        unique_outputs=len({r['output'] for r in data}),
        max_sequence_tokens=max(len('IN:'+r['input']+'\nOUT:'+r['output'])+1 for r in data),
        seed42_coverage=coverage_report(data,splits),
        limitations=['Shared rule templates across subsets', 'No independent human review',
                     'Not evidence of open-ended narrative creativity'])
    (ROOT/'results/synthetic_collection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
