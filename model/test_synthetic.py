import csv
import tempfile
import unittest
from pathlib import Path
from annotation import COLUMNS, create_sheet, read_reviewed, planned_data
from prepare_synthetic import revised_event
from protocol import parse_state, prompt

class SyntheticTests(unittest.TestCase):
    def test_all_revised_targets_fit_and_match_explicit_actor_and_condition(self):
        outputs = []
        for row in planned_data():
            s = parse_state(row['input'])
            text = revised_event(row['input'])
            self.assertIn('a '+s['NPC']+' ', text)
            self.assertIn(s['Player']+' player', text)
            self.assertLessEqual(len(prompt(row['input'])+text)+1, 256)
            if s['Time'] == 'night':
                self.assertIn('lantern', text)
            if s['Weather'] == 'rainy':
                self.assertIn('canopy', text)
            outputs.append(text)
        self.assertEqual(len(set(outputs)), 192)

    def test_ai_source_cannot_pass_the_human_export_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sheet.csv'
            create_sheet(path)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row.update(output=revised_event(row['input']), source='rule_template_ai_reviewed',
                           reviewed='yes', reviewer='Codex')
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'provenance source'):
                read_reviewed(path)

if __name__ == '__main__':
    unittest.main()
