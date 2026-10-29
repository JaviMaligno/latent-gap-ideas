"""Target-grouped OOF decoder distribution prediction; no LLM calls or FAR battery.
Primary model: candidate-level round means and a symmetrised 12-dimensional Gaussian
covariance; no final maxima enter training. Empirical trajectory transfer is a sensitivity
check (it does reuse historical maxima implicitly); compare iid first-round.
All historical scores belonging to a held-out k (all arms/generators) are excluded.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry import DATA, unit
from bounds.analyse import RUNS

ARMS=('REAL','INTERP','NOISE','CROWD','FAR','GAP')


def main():
    out=DATA/'runs/steam/bounds2';out.mkdir(parents=True,exist_ok=True)
    root=DATA/'prepared/steam'
    meta=pd.read_parquet(root/'corpus.parquet',columns=['adult','split'])
    p=unit(np.load(root/'emb_qwen3e06.npy')[(~meta.adult & (meta.split=='past')).to_numpy()].astype(float))
    tg=np.load(DATA/'runs/steam/pilot/targets.npz',allow_pickle=True)
    targets={a:unit(tg[a].astype(float)) for a in ARMS}
    ceilings={}
    for a,ts in targets.items():
        s=p@ts.T
        if a+'_hidden' in tg:
            for k,hidden in enumerate(tg[a+'_hidden']):s[np.asarray(hidden,dtype=int),k]=-np.inf
        ceilings[a]=s.max(0)
        print('Target features ready:',a,flush=True)
    rows=[];trajs=[]
    for run,gen in RUNS.items():
        for line in (DATA/f'runs/steam/{run}/items.jsonl').read_text().splitlines():
            r=json.loads(line);a=r['arm'];k=r['k']
            if a not in ARMS or not r.get('best'):continue
            t=targets[a][k]; e=unit(np.asarray(r['best']['emb'],float))
            c=unit(p[r['neighbours'][:8]].mean(0))
            trajectory=[np.array(h.get('candidate_sims',[]),float) for h in r['history']]
            finite=[x for x in trajectory if len(x)]
            if not finite:raise ValueError('Missing candidate scores')
            allscores=np.concatenate(finite)
            # saved scores used original fp32 targets; the unit-vector recomputation differs slightly.
            assert abs(allscores.max()-e@t)<.002
            rows.append(dict(gen=gen,arm=a,k=k,landing=e@t,centroid=c@t,
                             anchor=e@c,ceiling=ceilings[a][k],candidates=len(allscores)))
            trajs.append(trajectory)
    df=pd.DataFrame(rows)
    coeff=[]
    for gen in df.gen.unique():
        print('Cross-fitting:',gen,flush=True)
        for fold in range(5):
            trgen=(df.gen==gen)&(df.k%5!=fold)
            tegen=(df.gen==gen)&(df.k%5==fold)
            real=trgen&(df.arm=='REAL')
            kap=np.median(df.loc[real,'landing']/df.loc[real,'ceiling'])
            df.loc[tegen,'kappa']=kap*df.loc[tegen,'ceiling']
            for arm in ARMS:
                tr=np.flatnonzero(trgen&(df.arm==arm));te=np.flatnonzero(tegen&(df.arm==arm))
                if not len(te):continue
                assert not set(df.loc[tr,'k'])&set(df.loc[te,'k'])
                df.loc[te,'anchoring']=df.loc[tr,'anchor'].mean()*df.loc[te,'centroid']
                # Two fixed features; equal target weights, candidate-level squared error.
                # No tuning against any test outcome, no final-score offset fitted.
                xs=[];ys=[];ws=[]
                for i in tr:
                    y=np.concatenate([v for v in trajs[i] if len(v)])
                    xs.extend([[1,df.loc[i,'centroid']]]*len(y));ys.extend(y);ws.extend([1/len(y)]*len(y))
                x=np.array(xs);y=np.array(ys);w=np.array(ws)
                beta=np.linalg.solve(x.T@(w[:,None]*x)+np.diag([0,1e-3]),x.T@(w*y))
                residuals=[];first=[];independent=[]
                for i in tr:
                    loc=beta@[1,df.loc[i,'centroid']]
                    values=np.concatenate([v for v in trajs[i] if len(v)])-loc
                    residuals.append(values.max())
                    if len(trajs[i][0]):first.extend(trajs[i][0]-loc)
                    independent.extend(values)
                # Exact E[max] under discrete empirical iid CDF, no Monte Carlo noise.
                def emax(values,budget):
                    v=np.sort(values);f=np.arange(1,len(v)+1)/len(v)
                    return v@(f**budget-(f-1/len(v))**budget)
                loc=np.column_stack([np.ones(len(te)),df.loc[te,'centroid']])@beta
                df.loc[te,'trajectory']=loc+np.mean(residuals)
                df.loc[te,'iid12']=loc+emax(first,12)
                df.loc[te,'iid3']=loc+emax(first,3)
                df.loc[te,'one']=loc+np.mean(first)
                df.loc[te,'trajectory_lo']=loc+np.quantile(residuals,.1)
                df.loc[te,'trajectory_hi']=loc+np.quantile(residuals,.9)
                # Primary model: fit only candidate means and second moments, NOT maxima.
                gx=[];gy=[];gw=[]
                for i in tr:
                    count=sum(map(len,trajs[i]))
                    for stage,vs in enumerate(trajs[i]):
                        for score in vs:
                            gx.append([1,df.loc[i,'centroid']]+[float(stage==j) for j in (1,2,3)])
                            gy.append(score);gw.append(1/count)
                gx=np.array(gx);gy=np.array(gy);gw=np.array(gw)
                gb=np.linalg.solve(gx.T@(gw[:,None]*gx)+np.diag([0,1e-3,0,0,0]),gx.T@(gw*gy))
                shift=np.repeat(np.r_[0,gb[2:]],3)
                residual_matrix=[]
                for i in tr:
                    if len(trajs[i])==4 and all(len(v)==3 for v in trajs[i]):
                        residual_matrix.append(np.concatenate(trajs[i])-gb[0]-gb[1]*df.loc[i,'centroid']-shift)
                rr=np.array(residual_matrix);cov=rr.T@rr/len(rr)
                # Average over all within-round candidate permutations: exchangeable labels.
                sym=np.zeros((12,12))
                for j in range(4):
                    for l in range(4):
                        block=cov[3*j:3*j+3,3*l:3*l+3]
                        if j==l:
                            diag=np.trace(block)/3;off=(block.sum()-np.trace(block))/6
                            sym[3*j:3*j+3,3*j:3*j+3]=off
                            for z in range(3):sym[3*j+z,3*j+z]=diag
                        else:sym[3*j:3*j+3,3*l:3*l+3]=block.mean()
                cov=.8*sym+.2*np.diag(np.diag(sym)) # fixed, not tuned on held-out targets
                ev,vec=np.linalg.eigh(cov);assert ev.min()>-1e-10
                rng=np.random.default_rng(20261024)
                z=rng.normal(size=(8192,12));z=np.vstack([z,-z])
                draws=(z@(vec*np.sqrt(np.maximum(ev,0))).T+shift).max(1)
                gloc=gb[0]+gb[1]*df.loc[te,'centroid']
                bounded=np.clip(draws[:,None]+gloc.to_numpy()[None,:],-1,1)
                df.loc[te,'process']=bounded.mean(0)
                df.loc[te,'process_lo']=np.quantile(bounded,.1,axis=0)
                df.loc[te,'process_hi']=np.quantile(bounded,.9,axis=0)
                coeff.append(dict(gen=gen,arm=arm,fold=fold,intercept=beta[0],slope=beta[1],
                                  trajectory_gain=float(np.mean(residuals)),iid12_gain=emax(first,12),
                                  process_intercept=gb[0],process_slope=gb[1],
                                  round1=gb[2],round2=gb[3],round3=gb[4],
                                  covariance_targets=len(rr),process_gain=float(draws.mean()),
                                  covariance_min_eigenvalue=float(ev.min())))
    df.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(coeff).to_csv(out/'coefficients.csv',index=False)
    summaries=[]
    for keys,h in list(df.groupby('arm'))+[(('ALL',g),h) for g,h in df.groupby('gen')]+[('ALL',df)]:
        r=dict(group=str(keys),n=len(h),observed=h.landing.median(),
               coverage80=((h.landing>=h.process_lo)&(h.landing<=h.process_hi)).mean())
        for name in ('process','trajectory','iid12','iid3','one','kappa','anchoring'):
            r[name+'_median']=h[name].median();r[name+'_mae']=abs(h[name]-h.landing).mean()
        summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    # Paired uncertainty clustered by k across arms/generators; folds are fixed, not refitted.
    rng=np.random.default_rng(20261024);ci=[]
    for arm,h in list(df.groupby('arm'))+[('ALL',df)]:
        for other in ('kappa','anchoring'):
            delta=abs(h.process-h.landing)-abs(h[other]-h.landing)
            blocks=pd.DataFrame({'k':h.k,'delta':delta}).groupby('k').delta.agg(['sum','count'])
            ix=rng.integers(0,len(blocks),(2000,len(blocks)))
            boot=blocks['sum'].to_numpy()[ix].sum(1)/blocks['count'].to_numpy()[ix].sum(1)
            ci.append(dict(arm=arm,versus=other,delta_mae=delta.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
    pd.DataFrame(ci).to_csv(out/'paired_uncertainty.csv',index=False)
    print(summary.round(4).to_string(index=False));print(pd.DataFrame(ci).round(4).to_string(index=False))
    print('candidate counts',df.groupby('gen').candidates.agg(['min','max']).to_dict())

if __name__=='__main__':main()
