"""Budgeted, two-GPU discrete embedding inversion; no text-generation model or LLM API.
Input: support_inputs.json from scripts/bounds/support_probe.py, uploaded as the private
Kaggle dataset named in kernel-metadata.json. Outputs are append-only witness JSONL.
The objective is Qwen document cosine, explicit EOS, last-token pooling, <=512 tokens.
Search is a lower-bound experiment. Every proposed token edit is decoded and re-tokenised.
"""
import glob
import json
import multiprocessing as mp
import os
import re
from pathlib import Path
import subprocess
import sys
import time

MODEL='Qwen/Qwen3-Embedding-0.6B'
EOS=151643
SEED=20261024


def worker(gpu, jobs, deadline):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    torch.cuda.set_device(gpu)
    torch.manual_seed(SEED+gpu)
    rng=np.random.default_rng(SEED+gpu)
    dev=f'cuda:{gpu}'
    print(f'GPU {gpu} loading {MODEL}',flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModel.from_pretrained(MODEL,torch_dtype=torch.float16,
                                   attn_implementation='sdpa').to(dev).eval()
    for p in model.parameters():p.requires_grad_(False)
    weights=model.get_input_embeddings().weight
    allowed=[i for i in range(len(tok)) if i not in tok.all_special_ids and
             (s:=tok.decode([i])).isprintable() and '\ufffd' not in s and s.strip()]
    allowed=torch.tensor(allowed,device=dev)

    def ids(text, limit=512):
        a=tok.encode(text,add_special_tokens=False)
        if a and a[-1]==EOS:a=a[:-1]
        return a[:limit-1]+[EOS]

    def canonical(a):
        return ids(tok.decode(a,skip_special_tokens=True),128)

    def forward(seqs, grad=False):
        n=max(map(len,seqs)); batch=len(seqs)
        x=torch.full((batch,n),EOS,device=dev,dtype=torch.long)
        mask=torch.zeros_like(x)
        for b,a in enumerate(seqs):
            x[b,:len(a)]=torch.tensor(a,device=dev);mask[b,:len(a)]=1
        emb=model.get_input_embeddings()(x).detach()
        emb.requires_grad_(grad)
        h=model(inputs_embeds=emb,attention_mask=mask,use_cache=False).last_hidden_state
        e=F.normalize(h[torch.arange(batch,device=dev),mask.sum(1)-1].float(),dim=-1)
        return e,emb

    def scores(seqs,t):
        result=[]
        with torch.no_grad():
            for b in range(0,len(seqs),8):
                e,_=forward(seqs[b:b+8]);result.extend((e@t).cpu().tolist())
        return result

    dest=Path(f'/kaggle/working/witness_gpu{gpu}.jsonl')
    print(f'GPU {gpu} ready; {len(jobs)} targets; vocabulary {len(allowed)}',flush=True)
    for j in jobs:
        if time.monotonic()>deadline:break
        start=time.monotonic(); stop=min(deadline,start+45)
        t=F.normalize(torch.tensor(j['target'],device=dev,dtype=torch.float32),dim=0)
        seed_texts=j['seeds']+j.get('decoder_seeds',[])
        seeds=[ids(s) for s in seed_texts]
        sc=scores(seeds,t); best=max(range(len(sc)),key=sc.__getitem__)
        bestids=seeds[best]; bestscore=sc[best]
        # A separate sentence-deletion branch supplies plausible coherent witnesses.
        # It never assigns coherence automatically; save these even if token search wins.
        phrase=[]
        for source in [seed_texts[z] for z in sorted(range(len(sc)),key=sc.__getitem__,reverse=True)[:3]]:
            sentences=re.split(r'(?<=[.!?])\s+',source)
            for parity in (0,1):
                keep=[sentences[0]]+[v for z,v in enumerate(sentences[1:]) if z%2==parity]
                phrase.append(ids(' '.join(keep)))
        phrase_scores=scores(phrase,t)
        shortlist=sorted(range(len(phrase)),key=lambda z:phrase_scores[z],reverse=True)[:3]
        coherent_candidates=[tok.decode(phrase[z][:-1],skip_special_tokens=True) for z in shortlist]
        if phrase_scores[shortlist[0]]>bestscore:
            bestscore,bestids=phrase_scores[shortlist[0]],phrase[shortlist[0]]
        # A short best available seed plus an unrelated token restart explore different basins.
        randomids=canonical(rng.choice(allowed.cpu().numpy(),32).tolist())
        starts=[ids(seed_texts[best],128),randomids]
        trace=[]; evaluations=len(seeds)+len(phrase)
        for restart,current in enumerate(starts):
            restart_stop=min(stop,start+22.5*(restart+1))
            value=scores([current],t)[0];evaluations+=1
            if value>bestscore:bestscore,bestids=value,current[:]
            for step in range(24):
                if time.monotonic()>restart_stop:break
                e,emb=forward([current],grad=True)
                (e@t).sum().backward()
                # Minimise first-order negative score using one-token replacements.
                positions=list(range(max(0,len(current)-33),len(current)-1))
                if not positions:break
                take=rng.choice(positions,min(8,len(positions)),replace=False)
                grad=emb.grad[0,take].float()
                gains=grad@weights[allowed].float().T
                top=torch.topk(gains,8,dim=1).indices
                proposals=[]
                for row,pos in enumerate(take):
                    for replacement in allowed[top[row]].tolist():
                        a=current[:];a[int(pos)]=replacement;proposals.append(canonical(a[:-1]))
                del e,emb,grad,gains
                vals=scores(proposals,t);evaluations+=len(vals)
                b=max(range(len(vals)),key=vals.__getitem__)
                if vals[b]>value:current,value=proposals[b],vals[b]
                if value>bestscore:bestscore,bestids=value,current[:]
                trace.append(dict(restart=restart,step=step,best=bestscore,current=value))
                print(f'GPU {gpu} {j["domain"]}/{j["arm"]}/{j["k"]} '
                      f'block {restart}:{step} best={bestscore:.5f} evals={evaluations}',flush=True)
        text=tok.decode(bestids[:-1],skip_special_tokens=True)
        # Authoritative score is the actual round-tripped string, not an unreachable token array.
        canonical_ids=ids(text)
        final=scores([canonical_ids],t)[0]
        row={k:j[k] for k in ('domain','arm','k','corpus_best')}
        row.update(text=text,token_ids=canonical_ids,score=final,seed_scores=sc,
                   coherence_candidates=coherent_candidates,phrase_scores=phrase_scores,
                   evaluations=evaluations,seconds=time.monotonic()-start,trace=trace,
                   coherence='unadjudicated',model=MODEL,precision='float16',seed=SEED+gpu)
        with dest.open('a') as f:f.write(json.dumps(row)+'\n')
        print(f'GPU {gpu} saved {j["domain"]}/{j["arm"]}/{j["k"]}: {final:.5f}',flush=True)


def main():
    began=time.monotonic()
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.1',
                    'sentencepiece','accelerate'],check=True,timeout=300)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError('Select Kaggle GPU T4 x2')
    inputs=glob.glob('/kaggle/input/**/support_inputs.json',recursive=True)
    if len(inputs)!=1:raise RuntimeError(f'Expected one support_inputs.json; got {inputs}')
    jobs=json.loads(Path(inputs[0]).read_text())
    # Interleave arms/domains so a timeout still covers each stratum.
    jobs.sort(key=lambda j:(j['k'],j['domain'],j['arm']))
    deadline=began+100*60
    ctx=mp.get_context('spawn')
    ps=[ctx.Process(target=worker,args=(i,jobs[i::2],deadline)) for i in range(2)]
    for p in ps:p.start()
    for p in ps:p.join(max(1,began+110*60-time.monotonic()))
    for p in ps:
        if p.is_alive():p.terminate();p.join();print('Watchdog stopped worker; partial witnesses saved',flush=True)
        elif p.exitcode:raise RuntimeError(f'GPU worker exited {p.exitcode}')
    print('Finished. Re-embed saved text with LocalQwen before reporting bounds; judge coherence blind.',flush=True)

if __name__=='__main__':main()
