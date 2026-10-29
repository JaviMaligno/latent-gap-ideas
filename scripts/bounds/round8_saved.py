"""Frozen, offline Round 8 analysis. Never imports an encoder or a model client.

Run `--self-test`, then freeze this file's SHA256 in the document and freeze.json,
then `--run`. Inputs are opened read-only; outputs use exclusive creation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / 'Documents/experiment-data/latent-gap-ideas'
OUT = REPO / 'runs/bounds8'
SIG = '6ae3845eab14497b'
JUDGES = {'claude:claude-haiku-5-5': 'haiku', 'codex:gpt-6-luna': 'luna',
          'claude:claude-opus-5-5': 'opus'}
HEADERS = {'steam': 'A video game page on the Steam store:\n',
           'arxiv': 'arXiv computer science paper. The title is followed by the abstract, a paragraph that summarises the problem, the approach and the contribution.\n'}
INPUTS = {}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def remember(p):
    p = Path(p)
    if str(p) not in INPUTS:
        INPUTS[str(p)] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}
    return p


def rows(p):
    with remember(p).open(newline='') as f:
        yield from csv.DictReader(f)


def jsonlines(p):
    with remember(p).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def readjson(p):
    return json.loads(remember(p).read_text())


def writejson(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def valid_score(v):
    return type(v) is int and 1 <= v <= 10


def decision(votes, strict=False):
    names = ['haiku', 'luna'] + (['opus'] if strict else [])
    if strict and not votes.get('opus'):
        return None, 'missing_opus'
    if any(len({v >= 7 for v in votes.get(k, [])}) > 1 for k in names):
        return None, 'conflicting_ratings'
    if any(any(v < 7 for v in votes.get(k, [])) for k in names):
        return 0, 'settled'
    if all(votes.get(k) and min(votes[k]) >= 7 for k in names):
        return 1, 'settled'
    return None, 'unresolved'


class Inventory:
    def __init__(self):
        self.texts = {}
        self.by_source = defaultdict(set)
        self.votes = defaultdict(lambda: defaultdict(list))
        self.rating_sources = defaultdict(list)
        self.vectors = {}
        self.vector_sources = defaultdict(list)
        self.eligible = defaultdict(set)
        self.nlls = defaultdict(list)
        self.nll_labels = {}
        self.exclusions = {}

    def add(self, domain, text, source, expected=None, eligible=None):
        h = sha_bytes(text.encode())
        assert expected is None or expected == h, (source, h, expected)
        key = (domain, h)
        if key in self.texts:
            assert self.texts[key] == text
        self.texts[key] = text
        self.by_source[source].add(key)
        if eligible is not None:
            self.eligible[key].add(bool(eligible))
        return key

    def rate(self, key, judge, score, source):
        if judge in JUDGES:
            judge = JUDGES[judge]
        if judge in ('haiku', 'luna', 'opus') and valid_score(score):
            self.votes[key][judge].append(score)
            self.rating_sources[key].append({'source': source, 'judge': judge, 'score': score})

    def vector(self, key, v, source):
        v = np.asarray(v, dtype=np.float64)
        assert v.shape == (1024,) and np.isfinite(v).all()
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1) < 1e-5, (source, norm)
        v = v / norm
        if key in self.vectors:
            assert np.max(np.abs(v - self.vectors[key])) < 1e-12, ('embedding conflict', key, source)
        else:
            self.vectors[key] = v
        self.vector_sources[key].append(source)

    def cached_judgments(self, p, keys):
        index = defaultdict(list)
        for key in keys:
            index[key[1]].append(key)
        for j in jsonlines(p):
            for key in index.get(j['sha256'], []):
                self.rate(key, j['judge'], j.get('coherence'), str(p))

    def load(self):
        sr = DATA / 'runs/seed-rewrite'
        for run in sorted(p for p in sr.iterdir() if p.is_dir()):
            source = 'seed-rewrite/' + run.name
            winners = {}
            for p in sorted(run.glob('*/*/*.json')):
                r = readjson(p)
                for c in r.get('seeds', []) + r.get('candidates', []):
                    self.add(r['domain'], c['text'], source, c.get('sha256'),
                             c.get('valid', False) and not c.get('gibberish', True))
                g = r.get('gate', {})
                if g.get('coh_text'):
                    key = self.add(r['domain'], g['coh_text'], source)
                    for judge, field in [('haiku', 'j1'), ('luna', 'j2')]:
                        self.rate(key, judge, g.get(field), str(p) + ':gate')
                    winners[f"{r['domain']}/{p.parent.name}/{r['arm']}-{r['k']}"] = key
            if (run / 'judgements.jsonl').exists():
                self.cached_judgments(run / 'judgements.jsonl', self.by_source[source])
            if (run / 'opus_check.jsonl').exists():
                for j in jsonlines(run / 'opus_check.jsonl'):
                    self.rate(winners[j['key']], 'opus', j.get('coherence'), str(run / 'opus_check.jsonl'))

        fs = DATA / 'runs/fluent-search'
        for run in sorted(p for p in fs.iterdir() if (p / 'judgements.jsonl').exists()):
            source = 'fluent-search/' + run.name
            for p in sorted((run / 'kaggle-out').glob('fluent_gpu*.jsonl')):
                for r in jsonlines(p):
                    states = [{'text': r['generated']}] + r['pool'] + ([r['final']] if r.get('final') else [])
                    for c in states:
                        key = self.add(r['domain'], c['text'], source)
                        if run.name in ('full', 'steam4b') and c.get('nll') is not None:
                            expected = 'Qwen/Qwen3-4B-Base' if run.name == 'steam4b' else 'Qwen/Qwen3-1.7B-Base'
                            assert r['lm'] == expected
                            self.nlls[(run.name, key)].append((float(c['nll']), str(p)))
            index = {key[1]: key for key in self.by_source[source]}
            for c in rows(run / 'candidates.csv'):
                key = index[c['sha256']]
                assert key[0] == c['domain']
                self.eligible[key].add(c['valid'] == 'True' and c['gibberish'] == 'False')
                if run.name in ('full', 'steam4b') and c['nll']:
                    self.nlls[(run.name, key)].append((float(c['nll']), str(run / 'candidates.csv')))
            self.cached_judgments(run / 'judgements.jsonl', self.by_source[source])
            # 8.2 is run-local: no ratings imported from other experiments.
            if run.name in ('full', 'steam4b'):
                local = defaultdict(lambda: defaultdict(list))
                for j in jsonlines(run / 'judgements.jsonl'):
                    if j['sha256'] in index and j['judge'] in JUDGES and valid_score(j.get('coherence')):
                        local[index[j['sha256']]][JUDGES[j['judge']]].append(j['coherence'])
                for key in self.by_source[source]:
                    self.nll_labels[(run.name, key)] = decision(local[key])

        qpath, jpath = REPO / 'runs/cover/judge_queue.jsonl', REPO / 'runs/cover/judge_results.jsonl'
        # The DATA copies are expected exact copies, not independent observations.
        for p in (qpath, jpath):
            mirror = DATA / 'runs/cover' / p.name
            assert digest(remember(p)) == digest(remember(mirror)), ('cover mirror mismatch', p)
        queue = {}
        for r in jsonlines(qpath):
            key = self.add(r['domain'], r['text'], 'cover/queue', r['text_sha256'], r['valid'] and not r['gibberish_flag'])
            queue[r['id']] = key
        for j in jsonlines(jpath):
            key = queue[j['id']]
            assert j['text_sha256'] == key[1]
            for judge in ('haiku', 'luna', 'opus'):
                self.rate(key, judge, j.get(judge), str(jpath))

        base = DATA / 'runs/coherent_calibration'
        for r in jsonlines(base / 'texts.jsonl'):
            self.add(r['domain'], r['text'], 'coherent_calibration', r['sha256'], not r['bounds'] and not r['gib_rule'])
        self.cached_judgments(base / 'judgements.jsonl', self.by_source['coherent_calibration'])

        # Same exact-text rubric: retain only the required judge identities.
        for p in sorted((DATA / 'runs').glob('*/coherent_search/*/items.jsonl')):
            source = str(p.parent.relative_to(DATA / 'runs'))
            for r in jsonlines(p):
                cfg = r.get('config', {})
                for c in r.get('candidates', []):
                    if not c.get('text'):
                        continue
                    key = self.add(r.get('domain', p.parents[2].name), c['text'], source, c.get('sha256'),
                                   c.get('valid', False) and not c.get('gibberish', True))
                    for field, spec in [('coherence', cfg.get('judge', r.get('judge'))),
                                        ('coherence2', cfg.get('judge2', r.get('judge2')))]:
                        if spec in JUDGES:
                            self.rate(key, spec, c.get(field), str(p))

        # Only same-fingerprint local vectors. No prepared/GPU/BGE substitution.
        by_hash = defaultdict(list)
        for key in self.texts:
            by_hash[key[1]].append(key)
        for root in ('fluent-search', 'support-search'):
            base = DATA / 'runs' / root / 'verified_cache' / SIG
            for h, keys in by_hash.items():
                p = base / (h + '.npy')
                if p.exists():
                    v = np.load(remember(p), mmap_mode='r', allow_pickle=False)
                    for key in keys:
                        self.vector(key, v, str(p))
        for name in ('cover_embeddings.csv', 'cover_r6_embeddings.csv'):
            for r in rows(REPO / 'docs/prereg' / name):
                assert r['encoder'] == SIG
                for key in by_hash.get(r['text_sha256'], []):
                    self.vector(key, json.loads(r['vector']), name + ':' + r['text_sha256'])

        # Pitch-rubric ratings cannot label the original rendered descriptions.
        # Inline embeddings may still fill an exact-text join to genuine gate ratings.
        for p in sorted((DATA / 'runs').glob('*/*/items.jsonl')):
            if not (p.parent.name == 'pilot' or p.parent.name.startswith('grid-')):
                continue
            count, embedded = 0, 0
            for r in jsonlines(p):
                count += 1
                embedded += bool((r.get('best') or {}).get('emb'))
                for c in r.get('history', []) + [r.get('best') or {}]:
                    if c.get('text') and c.get('emb'):
                        for key in by_hash.get(sha_bytes(c['text'].encode()), []):
                            self.vector(key, c['emb'], str(p))
            self.exclusions[str(p.parent.relative_to(DATA / 'runs'))] = {
                'items': count, 'best_embedding_present': embedded,
                'reason': 'normalized-pitch rubric; no exact-text Haiku/luna original-gate labels'}

    def evidence(self, key):
        return {'domain': key[0], 'sha256': key[1], 'text': self.texts[key],
                'ratings': {k: sorted(set(v)) for k, v in self.votes[key].items()},
                'rating_sources': self.rating_sources[key],
                'embedding_sources': self.vector_sources[key],
                'eligibility_flags': sorted(self.eligible[key])}


def nearest(a, b):
    """Float64 Euclidean distances; bounded block scratch (<4 MB)."""
    if not len(b):
        return np.full(len(a), np.inf), np.full(len(a), -1)
    out, idx = np.full(len(a), np.inf), np.full(len(a), -1)
    bn = np.einsum('ij,ij->i', b, b)
    for start in range(0, len(a), 64):
        block = a[start:start + 64]
        d2 = np.einsum('ij,ij->i', block, block)[:, None] + bn - 2 * (block @ b.T)
        j = np.argmin(d2, axis=1)
        # Re-evaluate each nearest candidate by subtraction, avoiding cancellation.
        out[start:start + len(block)] = np.linalg.norm(block - b[j], axis=1)
        idx[start:start + len(block)] = j
    return out, idx


def threshold_stats(items):
    # one (sha, NLL, binary label) per exact text; low NLL predicts positive.
    items = sorted(items, key=lambda r: (r[1], r[0]))
    p = sum(r[2] for r in items)
    n = len(items) - p
    if not p or not n:
        return {'n': len(items), 'positive': p, 'negative': n, 'status': 'one_class'}
    accepted = [r for r in items if r[2]]
    rejected = [r for r in items if not r[2]]
    hi = max(accepted, key=lambda r: (r[1], r[0]))
    lo = min(rejected, key=lambda r: (r[1], r[0]))
    err, best, best_h, fp, fn = p, p, None, 0, p
    auc_wins, neg_above, i = 0.0, n, 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j][1] == items[i][1]:
            j += 1
        pos = sum(x[2] for x in items[i:j])
        neg = j - i - pos
        neg_above -= neg
        auc_wins += pos * (neg_above + .5 * neg)
        err += neg - pos
        if err < best:  # earliest (smallest threshold) wins a tie
            best, best_h = err, items[i][1]
            fp, fn = sum(1 - x[2] for x in items[:j]), sum(x[2] for x in items[j:])
        i = j
    return {'n': len(items), 'positive': p, 'negative': n, 'max_accepted': hi,
            'min_rejected': lo, 'separable': hi[1] < lo[1],
            'best_errors': best, 'best_h': best_h, 'best_h_null_means': 'below all observations',
            'false_accepts': fp, 'false_rejects': fn, 'auc_negative_nll': auc_wins / (p * n)}


def analyse(inv):
    summary = {'sources': {}, 'regularity': [], 'nll': [], 'exclusions': inv.exclusions}
    labels = {gate: {key: decision(inv.votes[key], gate == 'opus') for key in inv.texts}
              for gate in ('original', 'opus')}
    with (OUT / 'text_audit.jsonl').open('x') as f:
        for key in sorted(inv.texts):
            r = {'domain': key[0], 'sha256': key[1], 'text': inv.texts[key],
                 'original': labels['original'][key], 'opus': labels['opus'][key],
                 'ratings': {k: sorted(set(v)) for k, v in inv.votes[key].items()},
                 'embedded': key in inv.vectors, 'eligible': sorted(inv.eligible[key])}
            f.write(json.dumps(r) + '\n')
    for source, keys in sorted(inv.by_source.items()):
        counts = Counter()
        for key in keys:
            for gate in ('original', 'opus'):
                lab, status = labels[gate][key]
                counts[f'{gate}_{status}'] += 1
                if lab is not None:
                    counts[f'{gate}_embedded' if key in inv.vectors else f'{gate}_missing_embedding'] += 1
                    counts[f'{gate}_positive' if lab else f'{gate}_negative'] += 1
        summary['sources'][source] = {'distinct_texts': len(keys), **counts}
    keys = sorted(inv.vectors)
    vectors = np.stack([inv.vectors[k] for k in keys])
    ix = {key: i for i, key in enumerate(keys)}
    # Exact normalized-vector collisions are checked by full byte identity.
    buckets = defaultdict(list)
    for key in keys:
        buckets[inv.vectors[key].tobytes()].append(key)
    summary['opposite_label_collisions'] = []
    for bucket in buckets.values():
        for gate in ('original', 'opus'):
            known = [k for k in bucket if labels[gate][k][0] is not None]
            if {labels[gate][k][0] for k in known} == {0, 1}:
                summary['opposite_label_collisions'].append({'gate': gate, 'texts': [inv.evidence(k) for k in known]})
    refs = {(r['domain'], r['arm'], int(r['k'])): r for r in rows(DATA / 'runs/fluent-search/refs.csv')}
    targets = readjson(DATA / 'runs/fluent-search/inputs/fluent_inputs.json')
    far_rows = []
    for gate in ('original', 'opus'):
        for domain in ('steam', 'arxiv', 'pooled'):
            kk = [k for k in keys if (domain == 'pooled' or k[0] == domain) and labels[gate][k][0] is not None]
            pos = [k for k in kk if labels[gate][k][0] == 1]
            neg = [k for k in kk if labels[gate][k][0] == 0]
            result = {'gate': gate, 'domain': domain, 'positive': len(pos), 'negative': len(neg)}
            if not pos or not neg:
                result['status'] = 'one_class'
                summary['regularity'].append(result)
                continue
            a, b = vectors[[ix[k] for k in pos]], vectors[[ix[k] for k in neg]]
            distances, nearest_idx = nearest(a, b)
            i = int(np.argmin(distances))
            dmin = float(distances[i])
            result.update(d_min=dmin, L_min=1 / dmin if dmin else None,
                          positive_pair=inv.evidence(pos[i]), negative_pair=inv.evidence(neg[int(nearest_idx[i])]))
            # A concrete check that surviving finite constants need not be vacuous:
            # count same-label edges at r=d_min and r=d_min/2, excluding self.
            edges = {}
            for factor in (1, 2):
                count = 0
                for group in (a, b):
                    for start in range(0, len(group), 64):
                        block = group[start:start + 64]
                        d2 = np.maximum(0, np.sum(block * block, axis=1)[:, None] + np.sum(group * group, axis=1) - 2 * block @ group.T)
                        for j in range(len(block)):
                            count += int(np.count_nonzero(d2[j, start + j + 1:] < (dmin / factor) ** 2))
                edges[str(factor)] = count
            result['same_label_edges_at_L_factor'] = edges
            summary['regularity'].append(result)
            if domain == 'pooled':
                continue
            eligible = [k for k in keys if k[0] == domain and inv.eligible[k] == {True}]
            unknown = [k for k in eligible if labels[gate][k][0] is None]
            u = vectors[[ix[k] for k in unknown]] if unknown else np.empty((0, 1024))
            # Direct negative-anchor propagation: valid for an ASSUMED global L;
            # cannot certify that assumption or coverage outside this finite cache.
            nd, _ = nearest(u, b)
            pd, _ = nearest(u, a)
            for target in targets:
                if target['domain'] != domain or target['arm'] != 'FAR' or not 1 <= int(target['k']) <= 20:
                    continue
                ref = refs[(domain, 'FAR', int(target['k']))]
                t = np.array(target['target'], dtype=np.float64)
                t /= np.linalg.norm(t)
                ceiling = max(float(ref['ceiling']), float(ref['heldout_ceiling']))
                for margin in (.03, .05):
                    bound = ceiling + margin
                    danger = u @ t > bound
                    for factor in (1, 2):
                        radius = dmin / factor
                        far_rows.append({'gate': gate, 'domain': domain, 'k': target['k'], 'margin': margin,
                                         'L_factor': factor, 'L': factor / dmin if dmin else None,
                                         'b': bound, 'known_positive_exceeds_b': bool(np.any(a @ t > bound)),
                                         'cached_unknown_dangerous': int(danger.sum()),
                                         'within_negative_ball': int(np.count_nonzero(danger & (nd < radius))),
                                         'within_positive_ball': int(np.count_nonzero(danger & (pd < radius))),
                                         'within_both_balls_all_unknown': int(np.count_nonzero((nd < radius) & (pd < radius)))})
    writejson('far_diagnostics.json', far_rows)
    nll_audit = []
    for run, domain in [('full', 'steam'), ('full', 'arxiv'), ('steam4b', 'steam')]:
        counts = Counter()
        used = []
        keys_for_run = sorted(inv.by_source['fluent-search/' + run])
        for key in keys_for_run:
            if key[0] != domain:
                continue
            lab, status = inv.nll_labels[(run, key)]
            vals = sorted({v for v, _ in inv.nlls.get((run, key), []) if math.isfinite(v)})
            counts['distinct_texts'] += 1
            counts[status] += 1
            if lab is None:
                continue
            if not vals:
                counts['settled_missing_nll'] += 1
                continue
            # Exact-text LM function: numerical duplicates tolerated to 1e-4 nats.
            # Larger discrepancies excluded, never choose one using its label.
            if vals[-1] - vals[0] > 1e-4:
                counts['settled_nll_conflict'] += 1
                nll_audit.append({'run': run, **inv.evidence(key), 'label': lab, 'nll_values': vals, 'excluded': 'nll_conflict'})
                continue
            value = vals[0]  # fixed numerical representative, independent of labels
            used.append((key[1], value, lab))
            nll_audit.append({'run': run, **inv.evidence(key), 'label': lab, 'nll_values': vals, 'nll': value})
        summary['nll'].append({'run': run, 'domain': domain, 'header': HEADERS[domain],
                               'counts': dict(counts), **threshold_stats(used)})
    writejson('nll_audit.json', nll_audit)
    writejson('input_manifest.json', sorted(INPUTS.values(), key=lambda r: r['path']))
    summary['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary['completed_utc'] = datetime.now(timezone.utc).isoformat()
    writejson('results.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k in ('nll', 'peak_rss_bytes', 'completed_utc')}, indent=2))
    print('Regularity:', [(r['gate'], r['domain'], r.get('d_min'), r.get('L_min')) for r in summary['regularity']])


def self_test():
    assert decision({'haiku': [6]})[0] == 0
    assert decision({'haiku': [7]})[0] is None
    assert decision({'haiku': [7], 'luna': [7]})[0] == 1
    assert decision({'luna': [6]})[0] == 0
    assert decision({'haiku': [6, 8], 'luna': [8]})[1] == 'conflicting_ratings'
    assert decision({'haiku': [8], 'luna': [8]}, True)[0] is None
    # Exhaustively verify ties, errors and AUC against pairwise/direct definitions.
    import itertools
    for n in range(2, 7):
        for labs in itertools.product((0, 1), repeat=n):
            if len(set(labs)) < 2:
                continue
            vals = [i // 2 for i in range(n)]
            items = [(str(i), vals[i], labs[i]) for i in range(n)]
            r = threshold_stats(items)
            brute = [(sum((v <= h) != y for _, v, y in items), h) for h in [-1] + sorted(set(vals))]
            assert r['best_errors'] == min(brute)[0]
            pairs = [(a, b) for _, a, y in items if y for _, b, z in items if not z]
            assert r['auc_negative_nll'] == sum((a < b) + .5 * (a == b) for a, b in pairs) / len(pairs)
    a = np.array([[1., 0], [0, 1]])
    d, ix = nearest(a, np.array([[1., 0], [-1, 0]]))
    assert np.allclose(d, [0, np.sqrt(2)]) and ix.tolist() == [0, 0]
    print('PASS: synthetic gate, tied-threshold, AUC and distance checks')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if args.run:
        freeze = json.loads((OUT / 'freeze-v2.json').read_text())
        assert freeze['code_sha256'] == digest(Path(__file__))
        doc = (REPO / 'docs/bounds7_impossibility.md').read_text()
        assert freeze['code_sha256'] in doc and freeze['utc'] in doc
        assert not (OUT / 'results.json').exists(), 'Never overwrite an outcome'
        inv = Inventory()
        inv.load()
        analyse(inv)
