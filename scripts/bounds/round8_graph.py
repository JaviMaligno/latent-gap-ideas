"""Frozen completion of Round 8's FAR component/query diagnostic; saved data only."""
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import resource

import round8_saved as base
import numpy as np

OUT = base.OUT


class Forest:
    def __init__(self, n):
        self.p = list(range(n))
        self.size = [1] * n
        self.tree = defaultdict(list)

    def root(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j, distance):
        a, b = self.root(i), self.root(j)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.p[b] = a
        self.size[a] += self.size[b]
        self.tree[i].append((j, distance))
        self.tree[j].append((i, distance))

    def path(self, a, b, keys):
        q, seen = deque([a]), {a: None}
        while q:
            i = q.popleft()
            if i == b:
                break
            for j, d in self.tree[i]:
                if j not in seen:
                    seen[j] = (i, d)
                    q.append(j)
        out = []
        i = b
        while seen[i] is not None:
            j, d = seen[i]
            out.append({'from': keys[j][1], 'to': keys[i][1], 'distance': d})
            i = j
        return out[::-1]


def run():
    freeze = json.loads((OUT / 'graph-freeze.json').read_text())
    assert freeze['code_sha256'] == base.digest(Path(__file__))
    assert freeze['base_code_sha256'] == base.digest(Path(base.__file__))
    assert freeze['code_sha256'] in (base.REPO / 'docs/bounds7_impossibility.md').read_text()
    for r in json.loads((OUT / 'input_manifest.json').read_text()):
        assert base.digest(Path(r['path'])) == r['sha256'], ('input changed', r['path'])
    inv = base.Inventory()
    inv.load()
    results = json.loads((OUT / 'results.json').read_text())
    refs = {(r['domain'], int(r['k'])): r for r in base.rows(base.DATA / 'runs/fluent-search/refs.csv') if r['arm'] == 'FAR'}
    targets = [r for r in base.readjson(base.DATA / 'runs/fluent-search/inputs/fluent_inputs.json')
               if r['arm'] == 'FAR' and 1 <= int(r['k']) <= 20]
    output = []
    for domain in ('steam', 'arxiv'):
        keys = sorted(k for k in inv.vectors if k[0] == domain and inv.eligible[k] == {True})
        x = np.stack([inv.vectors[k] for k in keys])
        graphs = {}
        radii = {}
        labels = {}
        for gate in ('original', 'opus'):
            labels[gate] = [base.decision(inv.votes[k], gate == 'opus')[0] for k in keys]
            r = next(a['d_min'] for a in results['regularity'] if a['gate'] == gate and a['domain'] == domain)
            for factor in (1, 2):
                graphs[gate, factor] = Forest(len(keys))
                radii[gate, factor] = r / factor
        max_radius = max(radii.values())
        norms = np.einsum('ij,ij->i', x, x)
        for start in range(0, len(x), 64):
            block = x[start:start + 64]
            d2 = norms[start:start + len(block), None] + norms[None] - 2 * block @ x.T
            for j in range(len(block)):
                i = start + j
                # Slightly broaden only the prefilter; all edges are re-evaluated.
                near = np.flatnonzero(d2[j, i + 1:] <= max_radius ** 2 + 1e-12) + i + 1
                for k in near:
                    d = float(np.linalg.norm(x[i] - x[k]))
                    for cfg, radius in radii.items():
                        if d < radius:
                            graphs[cfg].union(i, int(k), d)
        for (gate, factor), graph in graphs.items():
            lab = labels[gate]
            comps = defaultdict(list)
            for i in range(len(keys)):
                comps[graph.root(i)].append(i)
            comp_labs = {root: {lab[i] for i in members if lab[i] is not None} for root, members in comps.items()}
            conflicts = []
            for root, members in comps.items():
                if comp_labs[root] == {0, 1}:
                    a = next(i for i in members if lab[i] == 1)
                    b = next(i for i in members if lab[i] == 0)
                    conflicts.append({'positive': inv.evidence(keys[a]), 'negative': inv.evidence(keys[b]),
                                      'path': graph.path(a, b, keys)})
            out = {'domain': domain, 'gate': gate, 'L_factor': factor, 'radius': radii[gate, factor],
                   'n_vertices': len(keys), 'n_components': len(comps), 'mixed_components': len(conflicts),
                   'conflict_paths': conflicts, 'targets': []}
            for t in targets:
                if t['domain'] != domain:
                    continue
                v = np.array(t['target'], dtype=np.float64)
                v /= np.linalg.norm(v)
                scores = x @ v
                ref = refs[domain, int(t['k'])]
                for margin in (.03, .05):
                    bound = max(float(ref['ceiling']), float(ref['heldout_ceiling'])) + margin
                    danger = np.flatnonzero(scores > bound)
                    roots = {graph.root(int(i)) for i in danger}
                    unknown_danger = [int(i) for i in danger if lab[i] is None]
                    unknown_comps = sum(not comp_labs[root] for root in roots)
                    neg_known = sum(comp_labs[graph.root(i)] == {0} for i in unknown_danger)
                    positive_components = sum(1 in comp_labs[root] for root in roots)
                    out['targets'].append({'k': t['k'], 'margin': margin,
                                           'M_cached': len(unknown_danger), 'unknown_dangerous_components': unknown_comps,
                                           'negative_implied_unknown_texts': neg_known,
                                           'dangerous_positive_components': positive_components,
                                           'known_positive_exceeds': any(lab[int(i)] == 1 for i in danger),
                                           'conditional_query_saving': len(unknown_danger) - unknown_comps})
            output.append(out)
    base.writejson('graph_results.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                          'graphs': output})
    for r in output:
        print(r['domain'], r['gate'], r['L_factor'], 'vertices', r['n_vertices'], 'components', r['n_components'],
              'mixed', r['mixed_components'], 'saving', sum(t['conditional_query_saving'] for t in r['targets']))


if __name__ == '__main__':
    run()
