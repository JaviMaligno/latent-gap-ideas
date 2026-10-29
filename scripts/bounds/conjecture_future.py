"""Corpus-only Round 3 predictions; --measure is a separately committed experiment.

--predict NEVER computes a corpus maximum or selects a corpus-scored direction.
The quadrature uses only mean/covariance and latitude-bin moments. --retrospective
joins the already recorded 48 results; it does not remeasure them. No model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conjecture_battery import DOMAINS, SPACES, corpus, predict, subsample
from geometry import DATA, expected_max, unit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'prereg'
CACHE = DATA / 'runs' / 'bounds3'
VERSION = 'latitude-mixture-v1'
BINS = 32
POWER = 15
REPLICATES = 2
SEED = 202610243
SHIFT = 2.5
KEYS = ['domain', 'space', 'N', 'K', 'M']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def load_future(domain, space):
    base = DATA / 'prepared' / domain
    meta = pd.read_parquet(base / 'corpus.parquet', columns=['split', 'adult'])
    keep = (meta['split'].eq('future') & ~meta['adult']).to_numpy()
    raw = np.load(base / f'emb_{space}.npy', mmap_mode='r')
    x = unit(np.array(raw[keep], dtype=np.float32)).astype(np.float32)
    if len(x) < 20000:
        raise ValueError(f'{domain}: only {len(x)} eligible future rows; need 20,000')
    return x


def future_subsample(x, n, domain):
    if n is None:
        return x
    rng = np.random.default_rng([SEED, n, DOMAINS.index(domain)])
    return x[rng.choice(len(x), n, replace=False)]


def statistics(x):
    """Population-denominator moments only. No direction scores or maxima."""
    n, d = x.shape
    mu = x.mean(axis=0, dtype=np.float64)
    centered = x - mu
    cov = centered.T @ centered / n
    r = np.linalg.norm(mu)
    if r < 1e-10:
        raise ValueError('Latitude-mixture law requires nonzero mean')
    u = mu / r
    lat = centered @ u
    var_l = float(u @ cov @ u)
    if var_l < 1e-14:
        raise ValueError('Latitude-mixture law requires nonzero latitude variance')
    v = cov @ u / var_l
    # Residual of linear regression of X on its mean-axis latitude.
    residual = centered - lat[:, None] * v
    residual_energy = np.einsum('ij,ij->i', residual, residual)
    mean_energy = residual_energy.mean()
    groups = np.array_split(np.argsort(lat, kind='stable'), BINS)
    means = np.array([lat[g].mean() for g in groups])
    variances = np.array([lat[g].var() for g in groups])
    scales = np.array([residual_energy[g].mean()/mean_energy for g in groups])
    weights = np.array([len(g)/n for g in groups])
    eig, basis = np.linalg.eigh(cov)
    eig = np.maximum(eig, 0)
    return dict(n=n, d=d, mu=mu, r=r, eig=eig, mean_eig=basis.T@mu,
                regression_eig=basis.T@v, var_l=var_l, bin_mean=means,
                bin_var=variances, bin_scale=scales, bin_weight=weights,
                corpus_sha256=digest(x))


def quadrature(st, rep):
    """Importance Sobol quadrature of sphere integrals, never applied to corpus rows.

    G~N(-SHIFT*u,I), T=G/||G||; likelihood exp(SHIFT*G.u+SHIFT²/2).
    Eigen-coordinates preserve the complete alignment of mu and covariance.
    """
    d = int(st['d'])
    u = st['mean_eig']/st['r']
    sob = qmc.Sobol(d, scramble=True, seed=SEED+rep)
    a, h, variance, weight = [], [], [], []
    for _ in range(2**POWER//2048):
        g = ndtri(sob.random(2048)) - SHIFT*u
        norm_g = np.linalg.norm(g, axis=1)
        a.append(g@st['mean_eig']/norm_g)
        h.append(g@st['regression_eig']/norm_g)
        variance.append((g*g)@st['eig']/norm_g**2)
        weight.append(np.exp(SHIFT*(g@u)+SHIFT**2/2))
    return tuple(np.concatenate(y) for y in (a,h,variance,weight))


def evaluate(st, selections):
    records = []
    for rep in range(REPLICATES):
        a,h,var,weight = quadrature(st,rep)
        residual_var = np.maximum(var - st['var_l']*h*h, 0)
        means = a[:,None] + h[:,None]*st['bin_mean']
        sds = np.sqrt(np.maximum(residual_var[:,None]*st['bin_scale'] +
                                 h[:,None]**2*st['bin_var'], 1e-20))
        n = int(st['n'])
        def cdf(c, kind):
            if kind == 'mixture':
                sf = ndtr((means-c)/sds) @ st['bin_weight']
                prob = np.exp(n*np.log1p(-np.minimum(sf,1-1e-16)))
            elif kind == 'gaussian':
                prob = np.exp(n*log_ndtr((c-a)/np.sqrt(var)))
            elif kind == 'constant':
                prob = np.exp(n*log_ndtr((c-a)/np.sqrt(st['eig'].sum()/st['d'])))
            else:
                raise ValueError(kind)
            return float(np.mean(weight*prob))
        for k,m in selections:
            p=k/(2*m)
            values={kind:brentq(lambda c:cdf(c,kind)-p,-1,1,xtol=2e-8)
                    for kind in ('mixture','gaussian','constant')}
            # Model-only diagnostics: conditional first moments below median threshold.
            sf=ndtr((means-values['mixture'])/sds)@st['bin_weight']
            low_weight=weight*np.exp(n*np.log1p(-np.minimum(sf,1-1e-16)))
            values.update(K=k,M=m,rep=rep,
                low_variance_ratio=float(np.sum(low_weight*var)/low_weight.sum() /
                                         (st['eig'].sum()/st['d'])),
                low_mean=float(np.sum(low_weight*a)/low_weight.sum()))
            records.append(values)
    result=[]
    for k,m in selections:
        rows=[r for r in records if r['K']==k and r['M']==m]
        row={'K':k,'M':m}
        for key in ('mixture','gaussian','constant','low_variance_ratio','low_mean'):
            row[key]=float(np.mean([r[key] for r in rows]))
        row['quadrature_half_range']=abs(rows[0]['mixture']-rows[1]['mixture'])/2
        row['corrected']=row.pop('mixture')
        result.append(row)
    return result


def compute(x, selections, label):
    CACHE.mkdir(parents=True,exist_ok=True)
    fingerprint=digest(x)
    path=CACHE/f'{VERSION}-{label}-{fingerprint[:12]}.npz'
    if path.exists():
        with np.load(path) as f:
            st=dict(f)
    else:
        st=statistics(x)
        np.savez(path,**st)
    results=evaluate(st,selections)
    for row in results:
        row.update(N=len(x),d=x.shape[1],r=float(st['r']),
                   old=predict(x,row['K'],row['M']),
                   effective_rank=float(st['eig'].sum()**2/np.square(st['eig']).sum()),
                   corpus_sha256=fingerprint,model=VERSION)
    return results


def retrospect():
    known=pd.concat([pd.read_csv(OUT/'conjecture_battery_results.csv'),
                     pd.read_csv(OUT/'conjecture_battery_results_patents.csv')],ignore_index=True)
    rows=[]
    for dom in DOMAINS:
        for space in SPACES:
            x=corpus(dom,space)
            part=known[(known.domain==dom)&(known.space==space)]
            for n,group in part.groupby('N',sort=False):
                started=time.monotonic()
                sample=subsample(x,int(n),dom,space)
                selected=[(int(r.K),int(r.M)) for r in group.itertuples()]
                for row in compute(sample,selected,f'past-{dom}-{space}-{n}'):
                    row.update(domain=dom,space=space)
                    rows.append(row)
                print(f'past {dom} {space} N={n}: {time.monotonic()-started:.1f}s',flush=True)
                pd.DataFrame(rows).merge(known[KEYS+['observed']],on=KEYS).to_csv(
                    OUT/'conjecture_round3_retrospective.csv',index=False)
    df=pd.DataFrame(rows).merge(known[KEYS+['observed']],on=KEYS)
    for name in ('old','corrected','gaussian','constant'):
        df[f'{name}_error']=df.observed-df[name]
    df.to_csv(OUT/'conjecture_round3_retrospective.csv',index=False)
    print(df.groupby(['domain','space'])[['old_error','corrected_error']].mean())


def freeze_predictions():
    dest=OUT/'conjecture_future_predictions.csv'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite frozen predictions: {dest}')
    rows=[]
    for dom in DOMAINS:
        for space in SPACES:
            x=load_future(dom,space)
            for size in (None,20000):
                started=time.monotonic()
                sample=future_subsample(x,size,dom)
                for row in compute(sample,[(50,20000),(500,20000)],f'future-{dom}-{space}-{size}'):
                    row.update(domain=dom,space=space,size='full' if size is None else str(size))
                    rows.append(row)
                print(f'future statistics only {dom} {space} {size}: {time.monotonic()-started:.1f}s',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(dest,index=False,float_format='%.10g')
    manifest={'model':VERSION,'seed':SEED,'bins':BINS,'sobol_power':POWER,
              'replicates':REPLICATES,'importance_shift':SHIFT,
              'prediction_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'success':'Across all 24 rows: median absolute corrected error < old; every corrected absolute error <= 0.005.',
              'secondary':'Report domain/space signed biases, count within .005, maximum error; no row exclusions.',
              'future_ceilings_measured':False}
    (OUT/'conjecture_future_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(df[['domain','space','N','K','old','corrected']].to_string(index=False))


def measure_future():
    # Only the experimenter invokes this branch AFTER committing the predictions.
    for name in ('docs/prereg/conjecture_future_predictions.csv',
                 'docs/prereg/conjecture_future_manifest.json',
                 'scripts/bounds/conjecture_future.py'):
        committed=subprocess.run(['git','show',f'HEAD:{name}'],cwd=ROOT,capture_output=True,check=True).stdout
        if committed != (ROOT/name).read_bytes():
            raise RuntimeError(f'Commit frozen file before measurement: {name}')
    path=OUT/'conjecture_future_predictions.csv'
    manifest=json.loads((OUT/'conjecture_future_manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['prediction_sha256']:
        raise RuntimeError('Predictions do not match manifest')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=manifest['script_sha256']:
        raise RuntimeError('Script does not match manifest')
    results_path=OUT/'conjecture_future_results.csv'
    if results_path.exists():
        raise FileExistsError('Refusing to overwrite future results')
    df=pd.read_csv(path)
    rows=[]
    for (dom,space,size),group in df.groupby(['domain','space','size'],sort=False):
        x=future_subsample(load_future(dom,space),None if str(size)=='full' else int(size),dom)
        if any(group.corpus_sha256 != digest(x)):
            raise RuntimeError('Future corpus fingerprint changed')
        n,d=x.shape
        # One shared pool per corpus-size/space, so the two selections are nested.
        rng=np.random.default_rng([SEED+1,DOMAINS.index(dom),SPACES.index(space),
                                   0 if str(size)=='full' else int(size)])
        values=[]
        for start in range(0,20000,256):
            targets=unit(rng.standard_normal((min(256,20000-start),d))).astype(np.float32)
            values.append((x@targets.T).max(axis=0))
        ceilings=np.sort(np.concatenate(values))
        for source in group.to_dict('records'):
            source['observed']=float(np.median(ceilings[:int(source['K'])]))
            rows.append(source)
        print(f'MEASURED {dom} {space} {size}',flush=True)
    result=pd.DataFrame(rows)
    for name in ('old','corrected'):
        result[f'{name}_error']=result.observed-result[name]
    result.to_csv(results_path,index=False)
    old=result.old_error.abs(); new=result.corrected_error.abs()
    summary={'n':len(result),'old_median_abs_error':float(old.median()),
             'corrected_median_abs_error':float(new.median()),
             'corrected_max_abs_error':float(new.max()),'corrected_within_005':int((new<=.005).sum()),
             'success':bool(new.median()<old.median() and (new<=.005).all())}
    (OUT/'conjecture_future_verdict.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--predict',action='store_true')
    mode.add_argument('--measure',action='store_true')
    mode.add_argument('--retrospective',action='store_true')
    args=ap.parse_args()
    if args.predict:
        freeze_predictions()
    elif args.measure:
        measure_future()
    else:
        retrospect()

if __name__=='__main__':
    main()
