"""Independent read-only verification of frozen Round 8 results (no model imports)."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
import csv
import hashlib
import json
import resource
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/bounds8'
DATA = Path.home() / 'Documents/experiment-data/latent-gap-ideas'
result = json.loads((OUT / 'results.json').read_text())
audit = [json.loads(line) for line in (OUT / 'text_audit.jsonl').open()]
known = {(r['domain'], r['sha256']): r for r in audit if r['embedded'] and any(r[g][0] is not None for g in ('original', 'opus'))}
vec = {}
for key in known:
    for cache in ('fluent-search', 'support-search'):
        p = DATA / 'runs' / cache / 'verified_cache/6ae3845eab14497b' / (key[1] + '.npy')
        if p.exists():
            v = np.asarray(np.load(p, mmap_mode='r'), dtype=np.float64)
            vec[key] = v / np.linalg.norm(v)
            break
for name in ('cover_embeddings.csv', 'cover_r6_embeddings.csv'):
    with (ROOT / 'docs/prereg' / name).open() as f:
        for r in csv.DictReader(f):
            for d in ('steam', 'arxiv'):
                key = (d, r['text_sha256'])
                if key in known and key not in vec:
                    v = np.array(json.loads(r['vector']), dtype=np.float64)
                    vec[key] = v / np.linalg.norm(v)
assert len(vec) == len(known)
checks = []
for r in result['regularity']:
    gate, domain = r['gate'], r['domain']
    kk = [k for k, v in known.items() if (domain == 'pooled' or k[0] == domain) and v[gate][0] is not None]
    p = [k for k in kk if known[k][gate][0] == 1]
    n = [k for k in kk if known[k][gate][0] == 0]
    b = np.stack([vec[k] for k in n])
    minimum = float('inf')
    for start in range(0, len(p), 64):
        a = np.stack([vec[k] for k in p[start:start + 64]])
        minimum = min(minimum, float(cdist(a, b, 'euclidean').min()))
    assert abs(minimum - r['d_min']) < 1e-12
    pair = [r['positive_pair'], r['negative_pair']]
    for q in pair:
        assert hashlib.sha256(q['text'].encode()).hexdigest() == q['sha256']
    assert abs(np.linalg.norm(vec[domain if domain != 'pooled' else pair[0]['domain'], pair[0]['sha256']] - vec[pair[1]['domain'], pair[1]['sha256']]) - minimum) < 1e-12
    checks.append({'test': 'independent scipy cdist all opposite pairs', 'gate': gate, 'domain': domain, 'd_min': minimum})
na = json.loads((OUT / 'nll_audit.json').read_text())
for r in result['nll']:
    q = [x for x in na if x['run'] == r['run'] and x['domain'] == r['domain'] and 'excluded' not in x]
    vals = sorted({x['nll'] for x in q})
    thresholds = [vals[0] - 1] + vals
    best = min((sum((x['nll'] <= h) != bool(x['label']) for x in q), h) for h in thresholds)
    p = [x['nll'] for x in q if x['label']]
    n = [x['nll'] for x in q if not x['label']]
    auc = sum((a < b) + .5 * (a == b) for a in p for b in n) / (len(p) * len(n))
    assert best[0] == r['best_errors'] and auc == r['auc_negative_nll']
    assert max(p) == r['max_accepted'][1] and min(n) == r['min_rejected'][1]
    checks.append({'test': 'independent exhaustive threshold and pair-count AUC', 'run': r['run'], 'domain': r['domain'], 'best_errors': best[0], 'auc': auc})
# Input immutability, including every local cached vector consumed.
manifest = json.loads((OUT / 'input_manifest.json').read_text())
for r in manifest:
    h = hashlib.sha256()
    with Path(r['path']).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == r['sha256'], r['path']
verification = {'utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
                'unchanged_input_files': len(manifest), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with (OUT / 'verification.json').open('x') as f:
    json.dump(verification, f, indent=2)
    f.write('\n')
print(json.dumps(verification, indent=2))
