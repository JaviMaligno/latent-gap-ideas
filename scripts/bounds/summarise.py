"""Extra tables from analyse.py outputs and the existing geometry logs; no new embeddings.
Run: uv run python scripts/bounds/summarise.py
"""
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry import DATA, expected_max

out = DATA/'runs/steam/bounds'
df = pd.read_csv(out/'items.csv')
stats = json.loads((out/'shape.json').read_text())
rows = []
for arm,h in df.groupby('arm'):
    r = {'arm':arm}
    for name in ('gaussian_unit','spherical_map','gaussian'):
        r[name+'_mae_landing'] = abs(h[name]-h.landing).mean()
    rows.append(r)
print('SURROGATE_ERRORS\n',pd.DataFrame(rows).round(4).to_string(index=False))
pd.DataFrame(rows).to_csv(out/'surrogate_errors.csv',index=False)

# A retrospective, parameter-free approximation: the selected targets are the lower
# K/M tail of m_t + constant residual extreme. Not a theorem about FAR selection.
M,K = 20000,50
def selection(r,d,n):
    return (np.sqrt(1-r*r)*expected_max(n)+r*norm.ppf(K/(2*M)))/np.sqrt(d)
rows = [dict(space='Qwen full corpus',n=stats['n'],r=stats['mean_norm'],
             observed=df[df.arm=='FAR'].drop_duplicates('k').ceiling.median())]
for filename,space in [('geometry.log','Qwen half B'),('geometry_bgem3.log','bge-m3 half B')]:
    log = (DATA/'runs/steam'/filename).read_text()
    shape = json.loads(re.search(r'G1 corpus shape[^:]*: (\{.*\})',log)[1])
    n = int(re.search(r'n_B = (\d+)',log)[1])
    line = next(s for s in log.splitlines() if s.strip().startswith('FAR (most isolated'))
    observed = float(line.split(': ',1)[1].split(' | ')[0].split()[-1])
    rows.append(dict(space=space,n=n,r=shape['norm_of_mean'],observed=observed))
for r in rows:
    r['predicted'] = selection(r['r'],1024,r['n'])
    r['abs_error'] = abs(r['predicted']-r['observed'])
    r['predicted_d_for_0.3'] = max(0,selection(r['r'],1,r['n']))**2/.3**2
pd.DataFrame(rows).to_csv(out/'selection.csv',index=False)
print('\nSELECTION CONJECTURE\n',pd.DataFrame(rows).round(5).to_string(index=False))

# Descriptive uncertainty: target-cluster bootstrap of cross-fitted errors (generators
# are paired within each target). This does not refit the model in each bootstrap.
rng = np.random.default_rng(20261024)
far = df[df.arm=='FAR'].copy()
far['advantage'] = abs(far.kappa_prediction-far.landing)-abs(far.anchor_prediction-far.landing)
err = far.groupby('k').advantage.mean().to_numpy()
boot = np.mean(rng.choice(err,size=(5000,len(err))),axis=1)
print('\nFAR MAE advantage anchor over kappa, target bootstrap:',
      err.mean(), np.quantile(boot,[.025,.975]))
print('FAR coefficients implied by decomposition:', {
    'anchor_over_centroid_norm': np.median(far.anchor/far.neighbour_centroid_norm),
    'top8_mean_over_ceiling': np.median(far.centroid_score*far.neighbour_centroid_norm/far.ceiling),
    'anchor_term_over_ceiling': np.median(far.anchor_term/far.ceiling)})
print('FAR raw and partial Sigma alignment by generator\n',
      far.groupby('gen')[['sigma_cos','sigma_partial_cos']].median().round(4))
