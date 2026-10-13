"""Output-only causal language modeling with held-out checkpoint selection."""
import argparse
import json
import math
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tokenizer import CharTokenizer
from transformer import TinyTransformer
from protocol import ROOT, load_data, split_data, fingerprint, prompt, coverage_report

def collate(rows, tokenizer, block_size, objective='output'):
    samples = []
    for row in rows:
        prefix = tokenizer.encode(prompt(row['input']))
        ids = prefix + tokenizer.encode(row['output']) + [tokenizer.stoi['<EOS>']]
        if len(ids) - 1 > block_size:
            raise ValueError('Sequence exceeds block size; increase --block-size (no silent truncation).')
        x, y = ids[:-1], ids[1:]
        if objective == 'output':
            y[:len(prefix)-1] = [-100] * (len(prefix)-1)
        samples.append((x, y))
    length = max(len(x) for x, _ in samples)
    return (torch.tensor([x + [0]*(length-len(x)) for x, _ in samples]),
            torch.tensor([y + [-100]*(length-len(y)) for _, y in samples]))

def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum')
            tokens = (y != -100).sum().item()
            if optimizer is not None:
                optimizer.zero_grad()
                (loss / tokens).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += loss.item()
            count += tokens
    return total / count

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset.json')
    p.add_argument('--out', type=Path, default=ROOT/'results/research')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['random', 'compositional', 'compositional_seen'], default='random')
    p.add_argument('--objective', choices=['output', 'full'], default='output')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--emb', type=int, default=128)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    a = p.parse_args()
    if min(a.epochs, a.patience, a.batch_size) < 1:
        p.error('epochs, patience and batch-size must be positive')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_data(a.data)
    splits = split_data(data, a.seed, a.split)
    train = [data[i] for i in splits['train']]
    tokenizer = CharTokenizer([prompt(r['input']) + r['output'] for r in train])
    config = dict(vocab_size=len(tokenizer.stoi), emb=a.emb, depth=a.depth, block_size=a.block_size)
    model = TinyTransformer(**config).to(device)
    # Validation always measures event-only likelihood, including the full-loss ablation.
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
        collate_fn=lambda rows: collate(rows, tokenizer, a.block_size, a.objective))
    val_loader = DataLoader([data[i] for i in splits['validation']], batch_size=a.batch_size,
        collate_fn=lambda rows: collate(rows, tokenizer, a.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'manifest.json').exists() or (a.out/'best.pt').exists():
        raise ValueError('Output directory already contains a run; choose a fresh --out directory.')
    tokenizer.save(a.out/'tokenizer.json')
    metadata = dict(seed=a.seed, split=a.split, objective=a.objective, data_sha256=fingerprint(data),
        indices=splits, model_config=config, parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__, device=device, coverage=coverage_report(data, splits),
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    (a.out/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, a.epochs+1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError('Non-finite loss')
        history.append(dict(epoch=epoch, train_nll=train_loss, validation_event_nll=val_loss))
        print(f'{epoch}: train={train_loss:.4f} validation={val_loss:.4f}', flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(state_dict=model.state_dict(), config=config, epoch=epoch), a.out/'best.pt')
        else:
            stale += 1
        (a.out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if stale >= a.patience:
            break

if __name__ == '__main__':
    main()
