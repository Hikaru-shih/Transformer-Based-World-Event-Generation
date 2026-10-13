import unittest
from protocol import ROOT, load_data, parse_state, split_data, coverage_report
from tokenizer import CharTokenizer
from evaluate import overlap_f1, retrieve

class ProtocolTests(unittest.TestCase):
    def test_draft_generation_preserves_review_and_existing_work(self):
        import csv
        import tempfile
        from pathlib import Path
        from annotation import create_sheet, read_reviewed, COLUMNS
        from draft_events import fill_drafts
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sheet.csv'
            create_sheet(path)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['draft_event'] = 'A manually edited draft.'
            rows[1].update(output='An accepted event.', reviewed='yes', reviewer='test', source='human_written')
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(fill_drafts(path)['new_drafts'], 190)
            self.assertEqual(fill_drafts(path)['new_drafts'], 0)
            with path.open(newline='', encoding='utf-8') as stream:
                filled = list(csv.DictReader(stream))
            self.assertEqual(filled[0]['draft_event'], rows[0]['draft_event'])
            self.assertEqual(filled[1]['output'], rows[1]['output'])
            self.assertTrue(all(not r['output'] and not r['reviewed'] and not r['reviewer'] for r in filled[2:]))
            self.assertTrue(all(r['draft_source'] == 'rule_template_v1' for r in filled[2:]))
            with self.assertRaises(ValueError):
                read_reviewed(path)

    def test_strict_compositional_split_keeps_values_seen(self):
        from annotation import planned_data
        data = planned_data()
        for seed in (42, 43, 44):
            splits = split_data(data, seed, 'compositional_seen')
            self.assertEqual(splits, split_data(data, seed, 'compositional_seen'))
            self.assertEqual(sorted(i for v in splits.values() for i in v), list(range(len(data))))
            pairs = [{(parse_state(data[i]['input'])['Location'], parse_state(data[i]['input'])['NPC'])
                      for i in indices} for indices in splits.values()]
            for i in range(3):
                for j in range(i):
                    self.assertFalse(pairs[i] & pairs[j])
            for subset in coverage_report(data, splits).values():
                self.assertEqual(subset['examples_with_unseen_values'], 0)
                self.assertEqual(subset['location_npc_pairs_shared_with_train'], 0)

    def test_strict_split_rejects_sparse_values_and_unknown_modes(self):
        from annotation import planned_data
        data = planned_data()[:3]
        for i,row in enumerate(data):
            row['input'] = row['input'].replace('forest', f'unique{i}')
        with self.assertRaises(ValueError):
            split_data(data, 42, 'compositional_seen')
        with self.assertRaises(ValueError):
            split_data(data, 42, 'typo')

    def test_annotation_review_gate_and_reordering(self):
        import csv
        import tempfile
        from pathlib import Path
        from annotation import create_sheet, read_reviewed, COLUMNS
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sheet.csv'
            create_sheet(path)
            with self.assertRaises(ValueError):
                read_reviewed(path)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row.update(output='A test fixture event.', source='human_written', reviewed='yes', reviewer='test')
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(reversed(rows))
            data = read_reviewed(path)
            self.assertEqual(len(data), 192)
            self.assertEqual(data[0]['example_id'], 'dense-000')
            self.assertTrue(all(row['reviewed'] for row in data))
            rows[0]['output'] = '\u4e00'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'English'):
                read_reviewed(path)

    def test_split_is_reproducible_disjoint_and_exhaustive(self):
        data = load_data(ROOT/'data/dataset.json')
        for mode in ['random','compositional']:
            split = split_data(data,42,mode)
            self.assertEqual(split,split_data(data,42,mode))
            self.assertEqual(sorted(i for v in split.values() for i in v),list(range(len(data))))
            keys = [{r['input'] if mode=='random' else (parse_state(r['input'])['Location'],parse_state(r['input'])['NPC']) for r in (data[i] for i in v)} for v in split.values()]
            for i in range(3):
                for j in range(i):
                    self.assertFalse(keys[i] & keys[j])

    def test_eos_is_distinct_from_literal_angle_bracket(self):
        t = CharTokenizer(['abc<'])
        self.assertNotEqual(t.encode('<')[0],t.stoi['<EOS>'])
        self.assertEqual(t.encode('z'),[t.stoi['<UNK>']])
        self.assertEqual(t.decode(t.encode('abc')+[t.stoi['<EOS>']]),'abc')

    def test_metrics_and_baseline(self):
        self.assertEqual(overlap_f1('a a b','a b b'),2/3)
        self.assertEqual(overlap_f1('','x'),0)
        data = load_data(ROOT/'data/dataset.json')
        self.assertEqual(retrieve(data[0]['input'],data[:2]),data[0]['output'])

    def test_output_mask_and_causality(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is required for model tests')
        from train import collate
        from protocol import prompt
        from transformer import TinyTransformer
        row = load_data(ROOT/'data/dataset.json')[0]
        t = CharTokenizer([prompt(row['input'])+row['output']])
        x,y = collate([row],t,256)
        boundary = len(t.encode(prompt(row['input'])))-1
        self.assertTrue((y[0,:boundary]==-100).all())
        self.assertEqual(y[0,boundary].item(),t.encode(row['output'])[0])
        self.assertEqual(y[0,-1].item(),t.stoi['<EOS>'])
        model = TinyTransformer(len(t.stoi),emb=16,nhead=4,depth=1,block_size=256).eval()
        changed = x.clone()
        changed[0,-1] = (changed[0,-1]+1)%len(t.stoi)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[:,:-1],model(changed)[:,:-1],atol=1e-6))
        with self.assertRaises(ValueError):
            collate([row],t,10)

if __name__ == '__main__':
    unittest.main()
