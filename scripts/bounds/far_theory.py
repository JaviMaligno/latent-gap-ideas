"""Evaluate proved model bounds only. Does NOT sample data or run falsification battery."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, beta, norm
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry import DATA, expected_max


def coordinate_isf(p,d):
    v=t.isf(p,d-1)
    return v/np.sqrt(d-1+v*v)


def certificate(n,d,r,m=20000,k=50,delta=.05):
    # Three failure budgets: residual maxima, latitudes, ranks.
    dw=da=dq=delta/3
    lo=coordinate_isf(-np.expm1(np.log(dw/(2*m))/n),d-1)
    hi=coordinate_isf(dw/(2*m*n),d-1)
    amax=coordinate_isf(da/(2*m),d)
    assert lo>=0
    # Simultaneously cover both middle ranks of the selected K, including even K.
    ranks=sorted(set([(k+1)//2,k//2+1]))
    alo=[];ahi=[]
    for rank in ranks:
        pl=beta.ppf(dq/(2*len(ranks)),rank,m+1-rank)
        ph=beta.isf(dq/(2*len(ranks)),rank,m+1-rank)
        alo.append(-coordinate_isf(pl,d));ahi.append(-coordinate_isf(ph,d))
    s=np.sqrt(1-r*r)
    lower=r*np.mean(alo)+s*np.sqrt(1-amax*amax)*lo
    upper=r*np.mean(ahi)+s*hi
    return dict(n=n,d=d,r=r,m=m,k=k,coverage=1-delta,w_lower=lo,w_upper=hi,
                max_latitude=amax,lower=lower,upper=upper,
                conjecture=(expected_max(n)*s+r*norm.ppf(k/(2*m)))/np.sqrt(d))


def main():
    result=certificate(68222,1024,.64)
    counter=dict(n=68222,d=1024,r=0,conjecture=expected_max(68222)/32,
                 antipodal_population_selected_median=coordinate_isf((1-50/40000)/2,1024))
    out=DATA/'runs/steam/bounds2';out.mkdir(parents=True,exist_ok=True)
    (out/'theory.json').write_text(json.dumps(dict(latitude_model=result,counterexample=counter),indent=2))
    print(json.dumps(dict(latitude_model=result,counterexample=counter),indent=2))

if __name__=='__main__':main()
