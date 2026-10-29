"""Reproduce docs/bounds.md; no network/model calls. Run: uv run python scripts/bounds/analyse.py.

All cross-validation folds hold out target indices across ALL generators/arms.
Outputs go to runs/steam/bounds under the external data root, never overwrite input runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import betainc
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import DATA, load_run, unit, expected_max

# Freeze the reviewed experiment's scope; geometry.py can acquire new study arms/runs.
DECODED_ARMS = ('REAL', 'INTERP', 'NOISE', 'CROWD', 'FAR')
RUNS = {'pilot':'luna-gw', 'grid-gpt-6-luna':'luna', 'grid-gpt-6-1-sol':'sol61',
        'grid-claude-opus-5-5':'opus', 'grid-claude-sonnet-5-5':'sonnet', 'grid-claude-haiku-5-5':'haiku'}


def cap_bound(b, a):
    b = np.asarray(b)
    return np.where(b >= a, 1., a*b + np.sqrt(np.maximum(0, (1-a*a)*(1-b*b))))


def quadratic_bound(t, mu, eig, radius):
    """Rigorous dual upper certificate (up to floating point); x constrained to unit BALL.
    Coordinates are in covariance eigenbasis; radius is squared Mahalanobis radius.
    Any nonnegative lambda, eta gives a bound; optimizer success is unnecessary for validity.
    """
    def dual(z):
        lam, eta = np.exp(z)
        return (t@mu + lam*radius + eta*(1-mu@mu)
                + np.sum((t-2*eta*mu)**2/(lam/eig+eta))/4)
    fit = minimize(dual, np.log([.001, .3]), method="L-BFGS-B", bounds=[(-25, 10)]*2)
    return min(1., float(dual(fit.x)))


def spherical_map(t, mu, eig, beta):
    """Mode on the sphere of exp(-q(x)/2 + beta*t.x), in eigen-coordinates.
    Positive-definite Lagrangian Hessian certifies the global mode, not textual reachability.
    """
    b = mu+beta*eig*t
    low = (-1/eig.max())*(1-1e-12)
    def secular(lam):
        return np.sum((b/(1+lam*eig))**2)-1
    high = max(1., low+1)
    while secular(high)>0:
        high *= 2
    lam = brentq(secular,low,high,xtol=1e-11)
    x = b/(1+lam*eig)
    assert abs(x@x-1)<1e-7 and np.min(1/eig+lam)>0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DATA)
    args = ap.parse_args()
    out = args.data/'runs/steam/bounds'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261024)
    root = args.data/'prepared/steam'
    meta = pd.read_parquet(root/'corpus.parquet', columns=['adult', 'split'])
    P = unit(np.load(root/'emb_qwen3e06.npy')[(~meta.adult & (meta.split == 'past')).to_numpy()].astype(float))
    n, d = P.shape
    mu = P.mean(0)
    u = unit(mu)
    Sig = P.T@P/n - np.outer(mu, mu)
    eig, V = np.linalg.eigh(Sig)
    assert eig.min() > 0
    muV = mu@V
    lat = P@u
    # Block the transformation to limit temporary memory.
    q = np.concatenate([np.sum(((p-mu)@V)**2/eig, axis=1) for p in np.array_split(P, 16)])
    a = float(lat.min())
    a01 = float(np.quantile(lat, .01))
    rmax, r99 = float(q.max()), float(np.quantile(q, .99))
    tg = np.load(args.data/'runs/steam/pilot/targets.npz', allow_pickle=True)
    info, arrays = [], {}
    for arm in DECODED_ARMS:
        T = unit(tg[arm].astype(float))
        S = P@T.T
        if f'{arm}_hidden' in tg.files:
            for k, h in enumerate(tg[f'{arm}_hidden']):
                S[np.asarray(h, dtype=int), k] = np.nan
        ceiling = np.nanmax(S, axis=0)
        m, sd = np.nanmean(S, axis=0), np.nanstd(S, axis=0)
        visible_n = np.sum(np.isfinite(S), axis=0)
        # Surrogate expectation above top-8 quantile, NOT at the maximum.
        z8 = norm.pdf(norm.isf(8/visible_n))/(8/visible_n)
        global_sd = np.sqrt(np.sum(T*(T@Sig),axis=1))
        G = mu + ((z8/global_sd)[:, None])*(T@Sig)
        Emax = np.array([expected_max(int(v)) for v in visible_n])
        for k, t in enumerate(T):
            mode = spherical_map(t@V,muV,eig,z8[k]/global_sd[k])
            info.append(dict(arm=arm, k=k, ceiling=ceiling[k], cap=cap_bound(t@u,a).item(),
                             cap99=cap_bound(t@u,a01).item(),
                             ellipsoid=quadratic_bound(t@V,muV,eig,rmax),
                             ellipsoid99=quadratic_bound(t@V,muV,eig,r99),
                             gaussian=m[k]+sd[k]*Emax[k], gaussian_centroid=G[k]@t,
                             gaussian_unit=unit(G[k])@t, spherical_map=mode@(t@V),
                             target_lat=t@u, sigma=sd[k]))
        arrays[arm] = (T, G)
    target = pd.DataFrame(info)
    target.to_csv(out/'targets.csv', index=False)
    frames, es, cs, gs = [], [], [], []
    for run, gen in RUNS.items():
        df = load_run(args.data/'runs/steam'/run, DECODED_ARMS)
        E = np.stack(df.e).astype(float)
        T = np.stack([arrays[a][0][k] for a,k in zip(df.arm,df.k)])
        G = np.stack([arrays[a][1][k] for a,k in zip(df.arm,df.k)])
        N = np.stack([P[nn[:8]] for nn in df.nn])
        C = unit(N.mean(1))
        loo = np.stack([unit(N.sum(1)-N[:,j]) for j in range(8)], axis=1)
        df['gen'] = gen
        df['landing'] = np.sum(E*T,axis=1)
        df['centroid_score'] = np.sum(C*T,axis=1)
        df['mean_score'] = T@mu
        df['anchor'] = np.sum(E*C,axis=1)
        df['anchor_term'] = df.anchor*df.centroid_score
        df['anchor_remainder'] = df.landing-df.anchor_term
        st = T@Sig
        # Does Sigma t explain anything after removing the shown centroid and global mean?
        cm = C@mu
        det = mu@mu-cm**2
        ec = np.sum(E*C,axis=1); em = E@mu
        sc = np.sum(st*C,axis=1); sm = st@mu
        er = E-((ec*(mu@mu)-em*cm)/det)[:,None]*C-((em-ec*cm)/det)[:,None]*mu
        sr = st-((sc*(mu@mu)-sm*cm)/det)[:,None]*C-((sm-sc*cm)/det)[:,None]*mu
        df['sigma_partial_cos'] = np.sum(unit(er)*unit(sr),axis=1)
        df['sigma_cos'] = np.sum(unit(E-mu)*unit(st),axis=1)
        df['neighbour_loo'] = np.median(np.sum(N*loo,axis=2),axis=1)
        df['latitude'] = E@u
        df['mahalanobis_sq'] = np.sum(((E-mu)@V)**2/eig,axis=1)
        df['gaussian_direction_cos'] = np.sum(unit(N.mean(1)-mu)*unit(G-mu),axis=1)
        df['gaussian_centroid_error'] = np.linalg.norm(N.mean(1)-G,axis=1)
        df['gaussian_centroid_norm'] = np.linalg.norm(G,axis=1)
        df['neighbour_centroid_norm'] = np.linalg.norm(N.mean(1),axis=1)
        es.extend(E); cs.extend(C); gs.extend(unit(G))
        frames.append(df.drop(columns=['nn','e']))
    df = pd.concat(frames,ignore_index=True).merge(target,on=['arm','k'],validate='many_to_one')
    E,C,G = map(np.stack,(es,cs,gs))
    # Merge above preserves input order (assert using the defining dot products).
    T = np.stack([arrays[a][0][k] for a,k in zip(df.arm,df.k)])
    assert np.max(abs(np.sum(E*T,axis=1)-df.landing)) < 1e-10
    coeff = []
    # Fit full vectors, not landing scores: e ~ alpha C + beta mu.
    # Group k to avoid reusing a target across generators. Five fixed modulo folds.
    for gen in df.gen.unique():
        for fold in range(5):
            train = (df.gen==gen)&(df.k%5!=fold)
            test = (df.gen==gen)&(df.k%5==fold)
            for name, X in [('neighbour', C), ('gaussian', G)]:
                xx = X[train]
                gram = np.array([[np.sum(xx*xx), np.sum(xx@mu)],
                                 [np.sum(xx@mu), train.sum()*(mu@mu)]])
                rhs = np.array([np.sum(xx*E[train]), np.sum(E[train]@mu)])
                alpha,beta = np.linalg.solve(gram,rhs)
                predvec = alpha*X[test]+beta*mu
                df.loc[test,name+'_prediction'] = np.sum(predvec*T[test],axis=1)
                coeff.append(dict(gen=gen,fold=fold,model=name,alpha=alpha,beta=beta))
            realtrain = train & (df.arm=='REAL')
            kap = np.median(df.loc[realtrain,'landing']/df.loc[realtrain,'ceiling'])
            df.loc[test,'kappa_prediction'] = kap*df.loc[test,'ceiling']
            for arm in DECODED_ARMS:
                tr = train & (df.arm==arm)
                te = test & (df.arm==arm)
                # Predict held-out anchoring from training vectors; no landing outcomes in fit.
                anchor = df.loc[tr,'anchor'].mean()
                df.loc[te,'anchor_prediction'] = anchor*df.loc[te,'centroid_score']
    df.to_csv(out/'items.csv',index=False)
    pd.DataFrame(coeff).to_csv(out/'coefficients.csv',index=False)
    stats = dict(n=n,d=d,mean_norm=float(np.linalg.norm(mu)), latitude_min=a,latitude_q01=a01,
                 mahalanobis_max=rmax,mahalanobis_q99=r99,mean_mahalanobis=float(q.mean()),
                 eigenvalue_min=float(eig.min()),trace=float(eig.sum()))
    # Exponential-tilt derivative at zero is exactly Sigma*t for ANY finite distribution.
    t = arrays['FAR'][0][0]
    scores = P@t
    def tilted_mean(beta):
        w = np.exp(beta*(scores-scores.max()))
        return (w@P)/w.sum()
    h = 1e-3
    numerical_derivative = (tilted_mean(h)-tilted_mean(-h))/(2*h)
    tilt_error = np.linalg.norm(numerical_derivative-Sig@t)/np.linalg.norm(Sig@t)
    assert tilt_error < 1e-5
    stats['tilt_derivative_relative_error'] = float(tilt_error)
    print('SHAPE', json.dumps(stats))
    print('\nTARGET medians\n',target.groupby('arm').median(numeric_only=True).round(4).to_string())
    summary = []
    for arm,h in df.groupby('arm'):
        r = dict(arm=arm,n=len(h),landing=h.landing.median(),anchor=h.anchor.median(),
                 neighbour_loo=h.neighbour_loo.median(),centroid_score=h.centroid_score.median(),
                 gaussian_direction_cos=h.gaussian_direction_cos.median(),
                 gaussian_centroid_error=h.gaussian_centroid_error.median(),
                 gaussian_centroid_norm=h.gaussian_centroid_norm.median(),
                 neighbour_centroid_norm=h.neighbour_centroid_norm.median(),
                 anchor_term=h.anchor_term.median(),anchor_remainder=h.anchor_remainder.median(),
                 anchor_remainder_mae=h.anchor_remainder.abs().mean(),
                 sigma_cos=h.sigma_cos.median(),sigma_partial_cos=h.sigma_partial_cos.median(),
                 cap99_violation=(h.latitude<a01).mean(),ellipsoid99_violation=(h.mahalanobis_sq>r99).mean(),
                 cap_violation=(h.latitude<a).mean(),ellipsoid_violation=(h.mahalanobis_sq>rmax).mean())
        for name in ('neighbour','gaussian','kappa','anchor'):
            r[name+'_median']=h[name+'_prediction'].median()
            r[name+'_mae']=abs(h[name+'_prediction']-h.landing).mean()
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary.to_csv(out/'summary.csv',index=False)
    print('\nDECODER summary\n',summary.round(4).to_string(index=False))
    print('\nCOEFFICIENTS\n',pd.DataFrame(coeff).groupby('model')[['alpha','beta']].agg(['min','median','max']).round(4))
    # Exact spherical coverage lower bound for any dictionary, independent of its distribution.
    cover = []
    for c in (.05,.1,.2,.3,.5):
        p = .5*betainc((d-1)/2,.5,1-c*c)
        cover.append(dict(c=c,cap_fraction=p,necessary_n=1/p,max_covered_fraction=min(1,n*p)))
    print('\nCOVERAGE\n',pd.DataFrame(cover).to_string(index=False))
    pd.DataFrame(cover).to_csv(out/'coverage.csv',index=False)
    # Numerical checks of cap support construction and ellipsoid certificates.
    for a_test in (a,a01,.64):
        T = unit(rng.normal(size=(100,d)))
        b = T@u
        X = a_test*u+np.sqrt(1-a_test*a_test)*unit(T-b[:,None]*u)
        assert np.max(abs(np.linalg.norm(X,axis=1)-1))<1e-10
        assert np.max(abs(np.sum(X*T,axis=1)-cap_bound(b,a_test)))<1e-10
    assert np.all(df.loc[df.mahalanobis_sq<=rmax,'landing'] <= df.loc[df.mahalanobis_sq<=rmax,'ellipsoid']+1e-7)
    assert np.all(df.loc[df.latitude>=a,'landing'] <= df.loc[df.latitude>=a,'cap']+1e-7)
    (out/'shape.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('\nAll numerical certificate checks passed. Outputs:',out)


if __name__ == '__main__':
    main()
