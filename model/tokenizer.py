"""Character vocabulary fitted on training text, with atomic special tokens."""
import json

class CharTokenizer:
    def __init__(self, texts):
        tokens = ['<PAD>', '<UNK>', '<EOS>'] + sorted(set(''.join(texts)))
        self.stoi = {c: i for i, c in enumerate(tokens)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(c, self.stoi['<UNK>']) for c in text]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i not in (0, self.stoi['<EOS>']))

    def save(self, path):
        path.write_text(json.dumps(self.stoi, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = cls([])
        obj.stoi = json.loads(path.read_text(encoding='utf-8'))
        obj.itos = {i: c for c, i in obj.stoi.items()}
        if '<EOS>' not in obj.stoi or '<UNK>' not in obj.stoi:
            raise ValueError('Legacy tokenizer: retrain with the research pipeline.')
        return obj
