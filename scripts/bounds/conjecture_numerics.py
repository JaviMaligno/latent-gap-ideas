"""Refine frozen sphere integrals using cached corpus moments only; no new forecast.

Reads no embeddings and cannot measure any corpus ceiling. Frozen CSV unchanged.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
import conjecture_future as model


def main():
    # At 2^16 points a 30-bit Sobol grid can include zero. Map endpoints to
    # half a grid cell before the inverse normal. The frozen 2^15 runs had no
    # endpoints; this numerical safeguard changes no frozen prediction.
    original_ndtri=model.ndtri
    model.ndtri=lambda u: original_ndtri(np.clip(u,2.0**-31,1-2.0**-31))
    df=pd.read_csv(model.OUT/'conjecture_future_predictions.csv')
    rows=[]
    for source in df[(df['size']=='full') & (df.K==50)].to_dict('records'):
        path=model.CACHE/(f"{model.VERSION}-future-{source['domain']}-{source['space']}-None-"
                          f"{source['corpus_sha256'][:12]}.npz")
        with np.load(path) as f:
            st=dict(f)
        model.POWER=16
        refined=model.evaluate(st,[(50,20000)])[0]['corrected']
        a,h,var,weight=model.quadrature(st,0)
        rv=np.maximum(var-st['var_l']*h*h,0)
        means=a[:,None]+h[:,None]*st['bin_mean']
        sd=np.sqrt(np.maximum(rv[:,None]*st['bin_scale']+h[:,None]**2*st['bin_var'],1e-20))
        sf=ndtr((means-source['corrected'])/sd)@st['bin_weight']
        posterior=weight*np.exp(int(st['n'])*np.log1p(-np.minimum(sf,1-1e-16)))
        posterior/=posterior.sum()
        ac=a-posterior@a; vc=var-posterior@var
        corr=(posterior@(ac*vc))/np.sqrt((posterior@(ac*ac))*(posterior@(vc*vc)))
        row=dict(domain=source['domain'],space=source['space'],N=source['N'],K=50,M=20000,
                 frozen=source['corrected'],refined=refined,difference=refined-source['corrected'],
                 model_low_latitude_variance_correlation=corr)
        rows.append(row)
        print(row,flush=True)
        pd.DataFrame(rows).to_csv(model.OUT/'conjecture_round3_numerics.csv',index=False)

if __name__=='__main__':
    main()
