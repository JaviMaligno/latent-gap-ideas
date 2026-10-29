"""Re-embed downloaded Kaggle witnesses with the canonical local ONNX encoder.
Usage: uv run python scripts/bounds/verify_support.py /path/to/witness_gpu*.jsonl
Optional --judgments CSV has domain,sha256,rater1,rater2 (1=coherent domain description,
0=not coherent, blank=unjudged). Judges see exact text/domain but no arm/score.
No LLM calls. A GPU score is never accepted as a canonical numerical lower bound.
"""
import argparse
import hashlib
import json
import time
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
from latent_gap.embedder import LocalQwen, MODEL_DIR
from support_probe import DATA


def main():
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path)
    ap.add_argument('--judgments',type=Path)
    ap.add_argument('--minutes',type=float,default=5,help='Resumable wall-clock budget for local inference')
    args=ap.parse_args();deadline=time.monotonic()+60*args.minutes
    root=DATA/'runs/support-search'
    jobs=json.loads((root/'support_inputs.json').read_text())
    lookup={(j['domain'],j['arm'],j['k']):j for j in jobs}
    model=LocalQwen();rows=[];blind={};stopped=False
    signature=hashlib.sha256((inspect.getsource(LocalQwen)+str([
        (str(p.relative_to(MODEL_DIR)),p.stat().st_size,p.stat().st_mtime_ns)
        for p in sorted(MODEL_DIR.rglob('*')) if p.is_file()])).encode()).hexdigest()[:16]
    cache=root/'verified_cache'/signature;cache.mkdir(parents=True,exist_ok=True)
    for path in args.files:
        if stopped:break
        for line in path.read_text().splitlines():
            r=json.loads(line);j=lookup[r['domain'],r['arm'],r['k']]
            # Include seeds so a degraded GPU proposal cannot lower the observed bound.
            for text in [r['text']]+r.get('coherence_candidates',[])+j['seeds']+j.get('decoder_seeds',[]):
                original=model.tok.encode(text).ids
                if not original or original[-1]!=151643:original=original+[151643]
                while len(original)>512:
                    text=model.tok.decode(original[:511],skip_special_tokens=True)
                    original=model.tok.encode(text).ids
                    if not original or original[-1]!=151643:original=original+[151643]
                assert len(original)<=512
                key=hashlib.sha256(text.encode()).hexdigest()
                cached=cache/(key+'.npy')
                if not cached.exists():
                    if time.monotonic()>=deadline:
                        stopped=True;break
                    emb=model.embed_one(text);np.save(cached,emb)
                else:emb=np.load(cached)
                if (r['domain'],key) not in blind:
                    blind[r['domain'],key]=dict(sha256=key,domain=r['domain'],text=text,rater1='',rater2='')
                t=np.array(j['target']);t/=np.linalg.norm(t)
                score=float(emb@t)
                rows.append(dict(domain=r['domain'],arm=r['arm'],k=r['k'],sha256=key,score=score))
            if stopped:break
            print(r['domain'],r['arm'],r['k'],'verified',flush=True)
    if not rows:raise RuntimeError('No witness verified; increase --minutes to cover model startup')
    print('Partial coverage; rerun to resume from cache' if stopped else 'All supplied witnesses verified',flush=True)
    df=pd.DataFrame(rows)
    queue=pd.DataFrame(blind.values()).sample(frac=1,random_state=20261024)
    if args.judgments:
        votes=pd.read_csv(args.judgments)
        assert not votes.duplicated(['domain','sha256']).any()
        queue=queue.drop(columns=['rater1','rater2']).merge(
            votes[['domain','sha256','rater1','rater2']],on=['domain','sha256'],how='left',validate='one_to_one')
        df=df.merge(votes[['domain','sha256','rater1','rater2']],on=['domain','sha256'],how='left',validate='many_to_one')
        df['accepted']=(df.rater1==1)&(df.rater2==1)
    else:df['accepted']=False
    queue.to_csv(root/'coherence_blind.csv',index=False)
    df.to_csv(root/'verified_witnesses.csv',index=False)
    summary=df.groupby(['domain','arm','k']).score.max().rename('h_any_lower').to_frame()
    summary['h_coh_lower']=df[df.accepted].groupby(['domain','arm','k']).score.max()
    summary.to_csv(root/'support_bounds.csv')
    print(summary.to_string())

if __name__=='__main__':main()
