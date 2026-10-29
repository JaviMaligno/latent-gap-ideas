import json, numpy as np, pandas as pd
from pathlib import Path
D=Path.home()/"Documents/experiment-data/latent-gap-ideas"
meta=pd.read_parquet(D/"prepared/steam/corpus.parquet",columns=["name","split","adult"])
meta=meta[~meta.adult].reset_index(drop=True); pm=meta[meta.split=="past"].reset_index(drop=True)
t=np.load(D/"runs/steam/pilot/targets.npz",allow_pickle=True)
items=[json.loads(l) for l in open(D/"runs/steam/pilot/items.jsonl")]
by={}
for r in items:
    if r.get("best"): by.setdefault(r["arm"],[]).append(r)
def src(arm,k):
    if arm=="REAL": return f"juego oculto: {pm.name[t['REAL_idx'][k]]}"
    if arm in("INTERP","PAIR"): a,b=t[f"{arm}_pairs"][k]; return f"pareja: {pm.name[a]} + {pm.name[b]}"
    if arm in("NOISE","CROWD"): return f"base: {pm.name[int(t[arm+'_base'][k])]}, nivel {t[arm+'_level'][k]}"
    return "punto aleatorio lejano"
for arm,rs in by.items():
    print(f"\n===== {arm} (n={len(rs)})")
    if rs[0]["best"]["sim"] is not None:
        rs=sorted(rs,key=lambda r:-r["best"]["sim"]); pick=[rs[0],rs[len(rs)//2],rs[-1]]
    else: pick=rs[:3]
    for r in pick:
        b=r["best"]; it=b["item"]; nn=", ".join(pm.name[i] for i in r.get("neighbours",[])[:3])
        sim=f"{b['sim']:.3f}" if b["sim"] is not None else "-"
        print(f"\n-- k={r['k']} sim={sim} | {src(arm,r['k'])}\n   vecinos mostrados: {nn}\n   >> {it['name']} [{', '.join(it.get('genres',[])[:3])}]\n   {it["about"]}")
