import argparse
from pathlib import Path
import torch
from tokenizer import CharTokenizer
from transformer import TinyTransformer
from protocol import ROOT, prompt

@torch.no_grad()
def generate(model, tokenizer, state, max_new=150, temperature=0., top_k=0):
    if temperature < 0 or max_new < 1 or top_k < 0:
        raise ValueError('Invalid decoding parameters')
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt(state))
    if len(ids) + max_new > model.block_size:
        raise ValueError('Prompt plus generation exceeds context; increase training --block-size or reduce --max-new.')
    generated = []
    for _ in range(max_new):
        x = torch.tensor([ids], device=device)
        logits = model(x)[0, -1].clone()
        logits[0] = logits[tokenizer.stoi['<UNK>']] = -float('inf')
        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            logits /= temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, len(logits))).values[-1]
                logits[logits < threshold] = -float('inf')
            next_id = torch.multinomial(logits.softmax(-1), 1).item()
        if next_id == tokenizer.stoi['<EOS>']:
            break
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated)

def load_model(directory):
    tokenizer = CharTokenizer.load(directory/'tokenizer.json')
    checkpoint = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
    model = TinyTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, tokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=ROOT/'results/research')
    p.add_argument('--state', default='Location: forest; Time: night; Player: injured; NPC: hunter; Weather: foggy')
    p.add_argument('--max-new', type=int, default=150)
    p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    model, tokenizer = load_model(a.run)
    print(generate(model, tokenizer, a.state, a.max_new, a.temperature, a.top_k))

if __name__ == '__main__':
    main()
