"""Deterministic rule-template drafts; never mark events as human-reviewed."""
import argparse
import csv
import json
from pathlib import Path
from protocol import ROOT, parse_state

PLACES = {'forest': 'the forest trail', 'village': 'the village square',
          'cave': 'the cave entrance', 'river': 'the riverbank'}
# Two wording variants per NPC/player condition. These are reusable rules,
# not independently authored events or a natural-language generalization benchmark.
ACTIONS = {
    ('hunter', 'healthy'): ('invites the healthy player to track deer', 'shows the healthy player fresh animal tracks'),
    ('hunter', 'injured'): ("bandages the injured player's wound", 'helps the injured player find a resting spot'),
    ('hunter', 'hungry'): ('shares dried meat with the hungry player', 'offers the hungry player a packed meal'),
    ('merchant', 'healthy'): ('offers the healthy player a supply trade', 'asks the healthy player to inspect new goods'),
    ('merchant', 'injured'): ('gives the injured player a clean bandage', 'offers the injured player a healing salve'),
    ('merchant', 'hungry'): ('sells the hungry player a loaf of bread', 'offers the hungry player a small food parcel'),
    ('healer', 'healthy'): ('asks the healthy player to gather herbs', 'teaches the healthy player to identify medicinal plants'),
    ('healer', 'injured'): ("cleans the injured player's wound", 'wraps the injured player in a fresh bandage'),
    ('healer', 'hungry'): ('shares a simple meal with the hungry player', 'gives the hungry player bread from a satchel'),
    ('traveler', 'healthy'): ('invites the healthy player to exchange route advice', 'asks the healthy player about nearby paths'),
    ('traveler', 'injured'): ('helps the injured player sit down safely', 'offers the injured player a spare walking stick'),
    ('traveler', 'hungry'): ('shares trail rations with the hungry player', 'offers the hungry player fruit from a pack'),
}

def draft_event(state):
    s = parse_state(state)
    location_index = list(PLACES).index(s['Location'])
    time_index = ('morning', 'night').index(s['Time'])
    weather_index = ('clear', 'rainy').index(s['Weather'])
    variant = (location_index + time_index + weather_index) % 2
    action = ACTIONS[(s['NPC'], s['Player'])][variant]
    place = PLACES[s['Location']]
    time = ('morning', 'night')[time_index]
    weather = ('clear', 'rainy')[weather_index]
    if location_index % 2:
        return f'On a {weather} {time} at {place}, a {s["NPC"]} {action}.'
    return f'At {place} on a {weather} {time}, a {s["NPC"]} {action}.'

def fill_drafts(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if 'draft_event' not in columns or 'input' not in columns:
        raise ValueError('Expected an annotation sheet')
    if 'draft_source' not in columns:
        columns.insert(columns.index('draft_event') + 1, 'draft_source')
    count = 0
    for row in rows:
        if row.get('draft_event', '').strip() or row.get('output', '').strip() or row.get('reviewed', '').strip():
            continue
        row['draft_event'] = draft_event(row['input'])
        row['draft_source'] = 'rule_template_v1'
        count += 1
    if count:
        # Complete generation before writing; replace the file only after a full write.
        temporary = path.with_suffix(path.suffix + '.draft-tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return {'new_drafts': count, 'total_rows': len(rows),
            'reviewed_rows': sum(r.get('reviewed', '').strip().lower() == 'yes' for r in rows)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sheet', type=Path, default=ROOT/'data/dense_annotation.csv')
    args = parser.parse_args()
    print(json.dumps(fill_drafts(args.sheet), indent=2))

if __name__ == '__main__':
    main()
