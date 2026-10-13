import unittest
import torch
from protocol import load_data, ROOT, parse_state
from annotation import planned_data
from state_metrics import slot_metrics
from conditional_words import WordTokenizer, ConditionalDecoder, collate, memory_batch

class ConditionalTests(unittest.TestCase):
    def test_synthetic_context_markers_detect_wrong_time_and_weather(self):
        data = load_data(ROOT/'data/dense_ai_reviewed.json')
        row = data[0]
        correct = slot_metrics([row['output']],[row],data)
        self.assertEqual(correct['location_context_marker_accuracy'],1.)
        self.assertEqual(correct['time_context_marker_accuracy'],1.)
        self.assertEqual(correct['weather_context_marker_accuracy'],1.)
        altered = row['output'].replace('in the morning','by lantern light').replace('under clear skies','beneath a travel canopy')
        wrong = slot_metrics([altered],[row],data)
        self.assertEqual(wrong['time_context_marker_accuracy'],0.)
        self.assertEqual(wrong['weather_context_marker_accuracy'],0.)

    def test_word_unknown_and_eos(self):
        tokenizer = WordTokenizer([{'input':'forest', 'output':'A hunter helps.'}])
        self.assertEqual(tokenizer.encode('unseenword'),[1])
        self.assertEqual(tokenizer.decode([2]+tokenizer.encode('A hunter helps.')+[3]), 'A hunter helps.')
    def test_slot_markers_reject_wrong_or_missing_actor(self):
        rows = planned_data()
        row = rows[0]
        domain = rows
        self.assertEqual(slot_metrics(['a hunter helps the healthy player'],[row],domain)['npc_explicit_accuracy'],1.)
        self.assertEqual(slot_metrics(['a hunter and merchant help'],[row],domain)['npc_explicit_accuracy'],0.)
        self.assertEqual(slot_metrics(['a merchant helps'],[row],domain)['npc_explicit_accuracy'],0.)
        self.assertEqual(slot_metrics(['unreadable'],[row],domain)['player_explicit_accuracy'],0.)
    def test_memory_causality_padding_and_conditioning(self):
        rows = load_data(ROOT/'data/dense_ai_reviewed.json')[:2]
        tokenizer = WordTokenizer(rows)
        for condition in ('flat','slots'):
            model = ConditionalDecoder(len(tokenizer.stoi),emb=16,depth=1,conditioning=condition).eval()
            x,y,memory,pad = collate(rows,tokenizer,condition)
            changed = x.clone()
            changed[:,-1] = (changed[:,-1]+1)%len(tokenizer.stoi)
            with torch.no_grad():
                original = model(x,memory,pad)
                self.assertTrue(torch.allclose(original[:,:-1], model(changed,memory,pad)[:,:-1],atol=1e-6))
                altered = memory.clone()
                altered[0] = memory[1]
                altered_pad = pad.clone()
                altered_pad[0] = pad[1]
                self.assertFalse(torch.allclose(original[0],model(x,altered,altered_pad)[0]))
            self.assertEqual(y[0,(y[0]!=-100).sum()-1].item(),3)
            model.zero_grad()
            model(x,memory,pad).sum().backward()
            self.assertIsNotNone(model.memory_pos.weight.grad)

if __name__ == '__main__':
    unittest.main()
