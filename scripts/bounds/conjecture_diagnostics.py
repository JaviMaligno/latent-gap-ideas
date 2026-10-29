"""Read-only past-corpus mechanism check on ALREADY SAVED Qwen FAR directions.

No target selection and no maxima, old or future. These are the pilot FAR targets,
not the battery's separate seeded pools. Quantile diagnostics do not change the law.
"""
import numpy as np
import pandas as pd
from conjecture_future import CACHE, OUT, corpus
from geometry import DATA


def main():
    rows=[]
    for domain in ('steam','arxiv'):
        x=corpus(domain,'qwen3e06')
        mu=x.mean(axis=0,dtype=np.float64)
        t=np.load(DATA/'runs'/domain/'pilot'/'targets.npz')['FAR']
        scores=x@t.T
        mean=mu@t.T
        var=scores.var(axis=0,dtype=np.float64)
        average_var=(np.mean(np.einsum('ij,ij->i',x,x))-mu@mu)/x.shape[1]
        standardized=(scores-mean)/np.sqrt(var)
        q99,q999=np.quantile(standardized,[.99,.999],axis=0)
        rows.append(dict(domain=domain,space='qwen3e06',N=len(x),targets=len(t),
                         variance_ratio_median=float(np.median(var/average_var)),
                         variance_ratio_mean=float(np.mean(var/average_var)),
                         latitude_mean=float(np.mean(mean)),
                         latitude_variance_correlation=float(np.corrcoef(mean,var)[0,1]),
                         standardized_q99_median=float(np.median(q99)),
                         standardized_q999_median=float(np.median(q999))))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'conjecture_round3_saved_target_diagnostics.csv',index=False)
    print(df.to_string(index=False))

if __name__=='__main__':
    main()
