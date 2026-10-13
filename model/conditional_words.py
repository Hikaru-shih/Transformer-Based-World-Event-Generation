"""Word-level conditional Transformer: flat text versus explicit field memory."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from protocol import ROOT, FIELDS, load_data, parse_state, split_data, fingerprint, coverage_report
from evaluate import summarize, retrieve
from state_metrics import slot_metrics


def tokens(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text.lower())

class WordTokenizer:
    def __init__(self, rows):
        vocabulary = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted({t for row in rows for text in (row['input'],row['output']) for t in tokens(text)})
        self.stoi = {t:i for i,t in enumerate(vocabulary)}
        self.itos = vocabulary
    def encode(self,text):
        return [self.stoi.get(t,1) for t in tokens(text)]
    def decode(self,ids):
        text = ' '.join(self.itos[i] for i in ids if i >= 4)
        text = re.sub(r'\s+([.,;:!?])',r'\1',text)
        return text[:1].upper()+text[1:]

class ConditionalDecoder(nn.Module):
    def __init__(self,vocab_size,emb=64,depth=2,conditioning='slots',block_size=96):
        super().__init__()
        self.conditioning = conditioning
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size,emb,padding_idx=0)
        self.pos_emb = nn.Embedding(block_size,emb)
        self.memory_pos = nn.Embedding(block_size,emb)
        layer = nn.TransformerDecoderLayer(emb,4,dim_feedforward=emb*4,dropout=.1,batch_first=True)
        self.decoder = nn.TransformerDecoder(layer,depth,norm=nn.LayerNorm(emb))
        self.head = nn.Linear(emb,vocab_size)
    def forward(self,x,memory_ids,memory_padding):
        # Both representations use the same model and parameters. Only memory construction differs.
        if memory_ids.ndim == 3:
            mask = (memory_ids != 0).unsqueeze(-1)
            memory = (self.token_emb(memory_ids)*mask).sum(2)/mask.sum(2).clamp_min(1)
        else:
            memory = self.token_emb(memory_ids)
        memory = memory + self.memory_pos(torch.arange(memory.shape[1],device=x.device))[None]
        target = self.token_emb(x)+self.pos_emb(torch.arange(x.shape[1],device=x.device))[None]
        causal = torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)
        output = self.decoder(target,memory,tgt_mask=causal,tgt_key_padding_mask=x.eq(0),memory_key_padding_mask=memory_padding)
        return self.head(output)

def memory_batch(rows,t,conditioning):
    if conditioning == 'slots':
        values = [[t.encode(parse_state(row['input'])[field]) for field in FIELDS] for row in rows]
        width = max(len(v) for row in values for v in row)
        ids = torch.tensor([[v+[0]*(width-len(v)) for v in row] for row in values])
        return ids,torch.zeros(len(rows),len(FIELDS),dtype=torch.bool)
    values = [t.encode(row['input']) for row in rows]
    length = max(map(len,values))
    if length > 96:
        raise ValueError('Input exceeds memory context')
    ids = torch.tensor([v+[0]*(length-len(v)) for v in values])
    return ids,ids.eq(0)

def collate(rows,t,conditioning):
    values = [[2]+t.encode(r['output'])+[3] for r in rows]
    length = max(map(len,values))-1
    if length > 96:
        raise ValueError('Event exceeds decoder context')
    x = torch.tensor([v[:-1]+[0]*(length-len(v)+1) for v in values])
    y = torch.tensor([v[1:]+[-100]*(length-len(v)+1) for v in values])
    memory,padding = memory_batch(rows,t,conditioning)
    return x,y,memory,padding

def epoch(model,loader,optimizer=None):
    model.train(optimizer is not None)
    total,count = 0.,0
    with torch.set_grad_enabled(optimizer is not None):
        for x,y,m,p in loader:
            logits = model(x,m,p)
            loss = nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')
            n = y.ne(-100).sum().item()
            if optimizer is not None:
                optimizer.zero_grad()
                (loss/n).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            total += loss.item()
            count += n
    return total/count

@torch.no_grad()
def generate(model,t,row):
    model.eval()
    memory,padding = memory_batch([row],t,model.conditioning)
    ids = [2]
    for _ in range(60):
        logits = model(torch.tensor([ids]),memory,padding)[0,-1].clone()
        logits[:3] = -float('inf')
        next_id = logits.argmax().item()
        if next_id == 3:
            break
        ids.append(next_id)
    return t.decode(ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/dense_ai_reviewed.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--conditioning',choices=['flat','slots'],default='slots')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--phase',choices=['validation','test','generate'],default='validation')
    parser.add_argument('--state',default='Location: forest; Time: morning; Player: healthy; NPC: traveler; Weather: clear')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    data = load_data(args.data)
    splits = split_data(data,args.seed,'compositional_seen')
    train = [data[i] for i in splits['train']]
    validation = [data[i] for i in splits['validation']]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.phase == 'validation':
        if args.out.exists():
            raise ValueError('Use a fresh run directory')
        args.out.mkdir(parents=True)
        t = WordTokenizer(train)
        model = ConditionalDecoder(len(t.stoi),conditioning=args.conditioning)
        loader = DataLoader(train,batch_size=16,shuffle=True,collate_fn=lambda rows:collate(rows,t,args.conditioning))
        val_loader = DataLoader(validation,batch_size=16,collate_fn=lambda rows:collate(rows,t,args.conditioning))
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
        history,best,stale = [],float('inf'),0
        manifest = dict(data_sha256=fingerprint(data),seed=args.seed,conditioning=args.conditioning,
            indices=splits,coverage=coverage_report(data,splits),vocabulary=t.itos,
            parameters=sum(p.numel() for p in model.parameters()),torch_version=str(torch.__version__),
            config=dict(emb=64,depth=2,block_size=96),epochs=args.epochs,patience=15,lr=3e-4,
            dataset_sources=dict(Counter(r.get('source','unspecified') for r in data)))
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        for i in range(args.epochs):
            loss = epoch(model,loader,optimizer)
            val = epoch(model,val_loader)
            if not math.isfinite(loss+val):
                raise ValueError('Nonfinite loss')
            history.append(dict(epoch=i+1,train_word_nll=loss,validation_word_nll=val))
            if val < best:
                best,stale = val,0
                torch.save(model.state_dict(),args.out/'best.pt')
            else:
                stale += 1
            if (i+1)%10 == 0:
                print(f'{args.conditioning} seed={args.seed} epoch={i+1} validation={val:.4f}',flush=True)
            if stale >= 15:
                break
        (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    manifest = json.loads((args.out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['data_sha256'] != fingerprint(data) or manifest['seed'] != args.seed or manifest['conditioning'] != args.conditioning:
        raise ValueError('Run/data/arguments mismatch')
    t = WordTokenizer([])
    t.itos = manifest['vocabulary']
    t.stoi = {v:i for i,v in enumerate(t.itos)}
    model = ConditionalDecoder(len(t.stoi),conditioning=args.conditioning,**manifest['config'])
    model.load_state_dict(torch.load(args.out/'best.pt',map_location='cpu',weights_only=True))
    if args.phase == 'generate':
        state = parse_state(args.state)
        for field in FIELDS:
            known = {parse_state(row['input'])[field] for row in train}
            if state[field] not in known:
                raise ValueError('Unseen field value outside this controlled benchmark: '+field+'='+state[field])
        print(generate(model,t,{'input':args.state}),flush=True)
        return
    rows = [data[i] for i in manifest['indices'][args.phase]]
    predictions = [generate(model,t,r) for r in rows]
    retrieval = [retrieve(r['input'],train) for r in rows]
    result = dict(phase=args.phase,seed=args.seed,conditioning=args.conditioning,data_sha256=fingerprint(data),
        examples=len(rows),parameters=manifest['parameters'],metrics={})
    for name,preds in [('retrieval',retrieval),('word_transformer',predictions)]:
        result['metrics'][name] = summarize(preds,[r['output'] for r in rows],{r['output'] for r in train}) | slot_metrics(preds,rows,train)
    reference_tokens = [token for row in rows for token in tokens(row['output'])]
    result['unknown_reference_words'] = sorted({token for token in reference_tokens if token not in t.stoi})
    result['unknown_reference_word_rate'] = sum(token not in t.stoi for token in reference_tokens)/len(reference_tokens)
    loader = DataLoader(rows,batch_size=16,collate_fn=lambda batch:collate(batch,t,args.conditioning))
    result['word_nll'] = epoch(model,loader)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    wrong = {idx:rows[order[(pos+1)%len(order)]]['input'] for pos,idx in enumerate(order)}
    shuffled = [dict(input=wrong[i],output=r['output']) for i,r in enumerate(rows)]
    shuffled_loader = DataLoader(shuffled,batch_size=16,collate_fn=lambda batch:collate(batch,t,args.conditioning))
    result['shuffled_word_nll'] = epoch(model,shuffled_loader)
    result['conditioning_nll_gap'] = result['shuffled_word_nll']-result['word_nll']
    (args.out/(args.phase+'_metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (args.out/(args.phase+'_predictions.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['input','reference','retrieval','word_transformer'])
        writer.writerows((r['input'],r['output'],baseline,pred) for r,baseline,pred in zip(rows,retrieval,predictions))
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    main()
