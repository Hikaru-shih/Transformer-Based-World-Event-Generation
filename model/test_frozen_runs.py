import json
from pathlib import Path
import tempfile
import unittest
from run_conditioning import verify_existing

class FrozenRunTests(unittest.TestCase):
    def test_incomplete_changed_and_complete_run_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = dict(data_sha256='fixture',seed=43,split='compositional_seen',objective='output',
                            arguments={'epochs':100},model_config={'emb':128})
            (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            (path/'history.json').write_text(json.dumps([dict(epoch=1,validation_event_nll=1.)]),encoding='utf-8')
            (path/'best.pt').write_bytes(b'test checkpoint marker')
            with self.assertRaisesRegex(ValueError,'Incomplete run'):
                verify_existing(path,'char-output',43,'fixture')
            history = [dict(epoch=i,validation_event_nll=1. if i==1 else 2.) for i in range(1,17)]
            (path/'history.json').write_text(json.dumps(history),encoding='utf-8')
            verify_existing(path,'char-output',43,'fixture')
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_existing(path,'char-output',44,'fixture')
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_existing(path,'char-output',43,'changed-data')

if __name__ == '__main__':
    unittest.main()
