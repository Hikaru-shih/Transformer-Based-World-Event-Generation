"""Fixtures below are artificial test scores, never research observations."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from analyze_human_review import analyze, METRICS
from independent_test import validate


class HumanWorkflowTests(unittest.TestCase):
    def test_pending_complete_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(item_id='x', seed='42', example_id='e', system='S01', state='state', reference='ref', prediction='pred', reviewer='', **{m: '' for m in METRICS})
            def write(name, item):
                with (root/name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(item)
            write('template.csv', row)
            write('a.csv', row)
            write('b.csv', row)
            (root/'key.json').write_text(json.dumps({'S01': 'fixture'}))
            def run():
                return analyze(root/'a.csv', root/'b.csv', root/'key.json', root/'template.csv')
            self.assertEqual(run()['status'], 'awaiting_human_ratings')
            write('a.csv', row | dict(reviewer='A', **{m: '4' for m in METRICS}))
            write('b.csv', row | dict(reviewer='B', **{m: '3' for m in METRICS}))
            self.assertEqual(run()['systems']['fixture']['metrics'][METRICS[0]]['paired_mean'], 3.5)
            write('b.csv', row | dict(reviewer='A', **{m: '3' for m in METRICS}))
            with self.assertRaises(ValueError):
                run()
            write('b.csv', row | dict(prediction='changed'))
            with self.assertRaises(ValueError):
                run()

    def test_independent_collection_gate(self):
        train = [{'input': 'Location: forest; Time: night; Player: healthy; NPC: healer; Weather: clear'},
                 {'input': 'Location: river; Time: night; Player: healthy; NPC: traveler; Weather: clear'}]
        row = dict(example_id='fixture', input='Location: forest; Time: night; Player: healthy; NPC: traveler; Weather: clear', output='Test fixture event.', author='A', reviewer='B', reviewed='yes', source='human_written')
        self.assertEqual(len(validate([row], train)), 1)
        for changed in (dict(output=''), dict(source='ai_written'), dict(reviewer='A'), dict(input=train[0]['input'])):
            with self.assertRaises(ValueError):
                validate([row | changed], train)


if __name__ == '__main__':
    unittest.main()
