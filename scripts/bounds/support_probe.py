"""Support diagnostics, without LLM calls or new targets.

Round 4 (read-only inputs, JSON to stdout):
  OMP_NUM_THREADS=4 uv run python scripts/bounds/support_probe.py --audit
  OMP_NUM_THREADS=4 uv run python scripts/bounds/support_probe.py --audit --local-edits 8
  OMP_NUM_THREADS=4 uv run python scripts/bounds/support_probe.py --audit --cone-steps 32

The cache-only audit takes about a minute; 32 cone steps on all 40 FAR targets
also took about a minute (most converged earlier). Eight corpus anchors per domain, each
with six deterministic edits, need at most 112 ONNX forwards: measured throughput
on the 16 GB Intel Mac is about 1 s / 200 tokens, plus 14 s model loading. Allow
3-8 minutes. A conservative online estimate refuses work beyond --minutes (28 by
default); the deadline is checked before every forward. No network, judges, or LM.
Local slopes/ranks/residuals are sample diagnostics, NEVER global certificates.

Legacy invocation (unchanged output contract): no flags prepares GPU inputs;
--probe also runs the old k=0 probe and writes in runs/support-search/.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from latent_gap.embedder import MODEL_DIR, LocalQwen

DATA = Path.home()/'Documents/experiment-data/latent-gap-ideas'


def legacy(args):
    out=DATA/'runs/support-search';out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for domain in ('steam','arxiv'):
        root=DATA/'prepared'/domain
        meta=pd.read_parquet(root/'corpus.parquet')
        keep=(~meta.adult & (meta.split=='past')).to_numpy()
        texts=meta.loc[keep,'text'].tolist()
        p=np.load(root/'emb_qwen3e06.npy')[keep].astype(np.float32)
        p/=np.linalg.norm(p,axis=1,keepdims=True)
        tg=np.load(DATA/f'runs/{domain}/pilot/targets.npz')
        for arm in ('FAR','GAP'):
            for k,t in enumerate(tg[arm]):
                t=t/np.linalg.norm(t); scores=p@t
                ix=np.argsort(scores)[-3:][::-1]
                jobs.append(dict(domain=domain,arm=arm,k=k,target=t.tolist(),
                                 seeds=[texts[i] for i in ix],corpus_best=float(scores[ix[0]])))
        print(domain,'prepared existing FAR/GAP targets',flush=True)
    # Reuse existing Steam winners as additional witnesses; never read arXiv decodes.
    lookup={(j['domain'],j['arm'],j['k']):j for j in jobs}
    runs=('pilot','grid-gpt-6-luna','grid-gpt-6-1-sol','grid-claude-opus-5-5',
          'grid-claude-sonnet-5-5','grid-claude-haiku-5-5')
    for run in runs:
        for line in (DATA/f'runs/steam/{run}/items.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['arm'] in ('FAR','GAP') and r.get('best') and r['best'].get('text'):
                lookup['steam',r['arm'],r['k']].setdefault('decoder_seeds',[]).append(r['best']['text'])
    (out/'support_inputs.json').write_text(json.dumps(jobs))
    (out/'dataset-metadata.json').write_text(json.dumps(dict(title='latent-gap-support-inputs',
        id='javieraguilarmartin/latent-gap-support-inputs',licenses=[{'name':'CC0-1.0'}]),indent=2))
    if not args.probe:return
    model=LocalQwen(threads=4)
    # Fixed small vocabulary/length probe: these are unrestricted witnesses, not a semantic benchmark.
    probes=['', '.', 'game', 'paper', 'the '*16, '0 '*16, 'xyz '*16,
            'A puzzle game in which players rotate mirrors to guide light to a door.',
            'We study graph coloring and compare a greedy algorithm with exact search.']
    embeddings=model.embed(probes)
    results=[]
    for j in jobs:
        if j['k']!=0:continue
        t=np.array(j['target']); seed=j['seeds'][0]
        # Shortening can remove the domain syntax: deliberately keep any/coherent separate.
        variants=[seed,' '.join(seed.split()[:24]),' '.join(seed.split()[-24:])]
        e=model.embed(variants); alltexts=probes+variants
        scores=np.vstack([embeddings,e])@t;best=int(np.argmax(scores))
        r={k:j[k] for k in ('domain','arm','k','corpus_best')}
        r.update(probe_best=float(scores[best]),witness=alltexts[best],
                 exact_seed=float(scores[-3]),token_count=len(model._ids(alltexts[best])),
                 coherence='unadjudicated', scores=[float(s) for s in scores])
        results.append(r);print(json.dumps(r),flush=True)
    (out/'local_probe.json').write_text(json.dumps(results,indent=2))

def unit(a):
    a = np.asarray(a, dtype=np.float64)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or zero embedding")
    return a / norms


def fingerprint():
    # Exactly the existing fluent-search / seed-rewrite cache contract.
    files = [(str(p.relative_to(MODEL_DIR)), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(MODEL_DIR.rglob('*')) if p.is_file()]
    return hashlib.sha256((inspect.getsource(LocalQwen) + str(files)).encode()).hexdigest()[:16]


def cap_bound(latitude, target_latitude):
    b = np.clip(target_latitude, -1., 1.)
    return float(1. if b >= latitude else latitude * b +
                 np.sqrt(max(0., 1. - latitude**2) * max(0., 1. - b**2)))


def cone_certificate(p, t, axis, latitude, hidden, steps, deadline):
    """Column-generation NNLS; a polar dual supplies an upper bound at ANY iterate.

    A near-optimal primal mixture is not claimed reachable as text. The bound's
    validity does not depend on optimizer convergence. Float64, with a 1e-9
    feasibility buffer: checked numerically, not interval arithmetic.
    """
    from scipy.optimize import nnls
    active, weights = [], np.empty(0)
    projection = np.zeros_like(t)
    for iteration in range(steps + 1):
        if time.monotonic() > deadline:
            raise TimeoutError('Cone-certificate deadline exceeded')
        residual = t - projection
        corr = p @ residual
        corr[hidden] = -np.inf
        violation = float(corr.max())
        polar = residual - ((max(0., violation) + 1e-9) / latitude) * axis
        upper = float(np.linalg.norm(t - polar))
        pn = np.linalg.norm(projection)
        lower = float(projection @ t / pn) if pn else -1.
        if iteration == steps or upper - lower < 1e-5 or violation < 1e-8:
            break
        idx = int(corr.argmax())
        if idx in active:
            break
        active.append(idx)
        weights, _ = nnls(p[active].T, t, maxiter=10*len(active))
        projection = weights @ p[active]
    check = p @ polar
    check[hidden] = -np.inf
    if check.max() > 1e-8:
        raise AssertionError('Cone polar certificate failed feasibility check')
    return dict(upper=min(1., upper), virtual_mixture_lower=lower,
                duality_gap=upper-lower, iterations=iteration,
                polar_max_inner=float(check.max()), polar=polar.tolist(),
                active_corpus_positions=active, weights=weights.tolist())


def self_check():
    # An exactly soluble cone with one negative target coordinate. A certificate
    # must bound the true continuum, not just the two dictionary atoms.
    p = np.eye(2)
    t = unit(np.array([-.6, .8]))
    axis = unit(p.mean(0))
    cert = cone_certificate(p, t, axis, float((p@axis).min()), np.array([], dtype=int),
                            8, time.monotonic()+30)
    assert .8 <= cert['upper'] < .8000001
    assert abs(cert['virtual_mixture_lower']-.8) < 1e-12
    # Here normalization genuinely raises the ceiling: a two-atom cone contains t.
    t = axis
    cert = cone_certificate(p, t, axis, float((p@axis).min()), np.array([], dtype=int),
                            8, time.monotonic()+30)
    assert abs(cert['upper']-1.) < 1e-12
    assert abs(cert['virtual_mixture_lower']-1.) < 1e-12
    assert (p@t).max() < .71
    print('PASS: orthant support, primal/dual certificate, normalization amplification')


def token_distance(a, b):
    """Levenshtein metric on canonical token sequences (not a derivative)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]


def edit_texts(domain, text):
    """Four training edits, two held-out edit types; same fields, no coherence claim."""
    delimiter = '. About: ' if domain == 'steam' else '. Abstract: '
    if delimiter not in text:
        return []
    prefix, body = text.split(delimiter, 1)
    sentences = re.split(r'(?<=[.!?])\s+', body)
    if len(sentences) < 3 or not 70 <= len(body.split()) <= 170:
        return []
    bodies = [
        ('drop_first', ' '.join(sentences[1:])),
        ('drop_last', ' '.join(sentences[:-1])),
        ('swap_first_two', ' '.join([sentences[1], sentences[0], *sentences[2:]])),
        ('lower_body', body.lower()),
        ('drop_middle', ' '.join(s for i, s in enumerate(sentences) if i != len(sentences)//2)),
        ('sentence_newlines', '\n'.join(sentences)),
    ]
    return [(name, prefix + delimiter + b) for name, b in bodies]


def local_probes(domain, texts, model, count, deadline, rng):
    from latent_gap.coherent import bounds_violation
    from latent_gap.domains import DOMAINS

    def valid(text):
        # Same render round trip and length caps; no gibberish/judge assertion.
        if domain == 'steam':
            if not text.startswith('Genres: ') or '. Tags: ' not in text or '. About: ' not in text:
                return False
            g, rest = text[8:].split('. Tags: ', 1)
            tags, body = rest.split('. About: ', 1)
            item = dict(name='', genres=g.split(','), tags=tags.split(','), about=body)
        else:
            if not text.startswith('Title: ') or '. Abstract: ' not in text:
                return False
            title, body = text[7:].split('. Abstract: ', 1)
            item = dict(title=title, abstract=body)
        return (DOMAINS[domain].render(item) == text and
                not bounds_violation(domain, item, len(model.tok.encode(text).ids)))

    selected = []
    for idx in rng.permutation(len(texts)):
        edits = edit_texts(domain, texts[idx])
        if edits and valid(texts[idx]) and all(valid(s) for _, s in edits):
            selected.append((int(idx), edits))
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError('Not enough eligible, in-format source texts')
    all_deltas, rows, durations = [], [], []
    for idx, edits in selected:
        vectors = []
        for s in [texts[idx], *[s for _, s in edits]]:
            remaining = (count - len(rows)) * 7 - len(vectors)
            seconds_each = max(durations, default=2.)
            if time.monotonic() + 1.5 * seconds_each * remaining > deadline:
                raise TimeoutError('Conservative estimate exceeds local-probe deadline; reduce --local-edits')
            start = time.monotonic()
            vectors.append(unit(model.embed_one(s)))
            durations.append(time.monotonic() - start)
        a = vectors[0]
        delta = np.stack(vectors[1:]) - a
        tangent = delta - (delta @ a)[:, None] * a
        _, sv, vt = np.linalg.svd(tangent[:4], full_matrices=False)
        energy = sv**2
        rank = int(np.searchsorted(np.cumsum(energy), .95 * energy.sum()) + 1) if energy.sum() else 0
        basis = vt[:rank]
        residual = tangent[4:] - (tangent[4:] @ basis.T) @ basis
        ids = model._ids(texts[idx])
        distances = [token_distance(ids, model._ids(s)) for _, s in edits]
        slopes = [float(np.linalg.norm(d) / n) for d, n in zip(delta, distances) if n]
        rows.append(dict(corpus_pos=idx, text_sha256=hashlib.sha256(texts[idx].encode()).hexdigest(),
                         train_rank95=rank, train_pr=float(energy.sum()**2 / (energy @ energy)) if energy.sum() else 0.,
                         train_radius=float(np.linalg.norm(delta[:4], axis=1).max()),
                         test_radius=float(np.linalg.norm(delta[4:], axis=1).max()),
                         heldout_tangent_residual=float(np.linalg.norm(residual, axis=1).max()),
                         secant_lipschitz_lower=max(slopes, default=0.),
                         token_edit_distances=distances))
        all_deltas.extend(delta[:4])
        print(f'{domain}: local anchor {len(rows)}/{count}', file=sys.stderr, flush=True)
    return np.stack(all_deltas), rows, sum(durations)


def summarise(frame):
    result = []
    for (domain, writer, arm), g in frame.groupby(['domain', 'writer', 'arm'], sort=True):
        measured = g[g.h_coh.notna()]
        row = dict(domain=domain, writer=writer, arm=arm, n=len(g), measured=len(measured))
        for col in ['ceiling', 'ceiling_star', 'h_coh', 'cap', 'mix_zero', 'cone_ratio', 'cone_optimized',
                    'cone_distance_lower', 'optimized_cone_distance_lower', 'mix_local', 'cover_local',
                    'directional_local', 'nearest_distance', 'mixture_norm', 'mixture_residual',
                    'anchor', 'anchor_remainder']:
            if col in g:
                row[col + '_median'] = float(g[col].median()) if g[col].notna().any() else None
        for name, bound in [('corpus', g.ceiling), ('star', g.ceiling_star),
                            ('star_plus_03', g.ceiling_star + .03), ('cap', g.cap),
                            ('mix_zero', g.mix_zero), ('cone_ratio', g.cone_ratio)]:
            row[name + '_violations'] = int((g.h_coh > bound + 1e-6).sum())
        for name in ['cover_local', 'directional_local', 'mix_local', 'cone_optimized']:
            if name in g and g[name].notna().any():
                row[name + '_violations'] = int((g.h_coh > g[name] + 1e-6).sum())
        row['star_plus_05_reached'] = int((g.h_coh >= g.ceiling_star + .05).sum())
        row['required_corpus_slack'] = float((measured.h_coh - measured.ceiling).max()) if len(measured) else None
        row['required_star_slack'] = float((measured.h_coh - measured.ceiling_star).max()) if len(measured) else None
        if 'cone_optimized' in measured and measured.cone_optimized.notna().any():
            row['required_cone_slack'] = float((measured.h_coh - measured.cone_optimized).max())
            row['required_cone_distance_slack'] = float(measured.optimized_cone_distance_lower.max())
        if 'cover_local' in g:
            row['cover_hypothesis_violations'] = int((g.nearest_distance > g.sample_radius + 1e-6).sum())
        result.append(row)
    return result


def audit(args):
    start = time.monotonic()
    data = args.data
    fs = data / 'runs/fluent-search'
    sr = data / 'runs/seed-rewrite/test'
    sig = fingerprint()
    cache = fs / 'verified_cache' / sig
    if not cache.is_dir():
        raise FileNotFoundError(f'Encoder-fingerprinted cache missing: {cache}')
    refs = {(r['domain'], r['arm'], r['k']): r for r in
            map(json.loads, (fs / 'refs.jsonl').read_text().splitlines())}
    targets = {(r['domain'], r['arm'], r['k']): unit(r['target']) for r in
               json.loads((fs / 'inputs/fluent_inputs.json').read_text())}
    files = sorted(sr.glob('*/*/*.json'))
    records = [json.loads(p.read_text()) for p in files]
    if len(records) != 200:
        raise ValueError(f'Expected the 200 rev-3 test records, found {len(records)}')
    opus = {r['key']: r.get('coherence') for r in
            map(json.loads, (sr / 'opus_check.jsonl').read_text().splitlines())}
    manifest_files = [fs/'refs.jsonl', fs/'inputs/fluent_inputs.json', sr/'opus_check.jsonl', *files]
    manifest = {str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_files}
    output = dict(status='EMPIRICAL LAW: diagnostics, not a support certificate', encoder_fingerprint=sig,
                  seed=args.seed, local_edits_per_domain=args.local_edits, input_sha256=manifest,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cone_steps=args.cone_steps, domains={}, rows=[], cone_certificates=[])
    model = LocalQwen(threads=4) if args.local_edits else None
    for domain in ['steam', 'arxiv']:
        root = data / 'prepared' / domain
        meta = pd.read_parquet(root / 'corpus.parquet', columns=['text', 'split', 'adult'])
        keep = (~meta.adult & (meta.split == 'past')).to_numpy()
        # One domain at a time, < 1 GB arrays. Never read the future corpus rows.
        p = unit(np.load(root/'emb_qwen3e06.npy', mmap_mode='r')[keep])
        texts = meta.loc[keep, 'text'].tolist()
        mean = p.mean(0)
        axis = unit(mean)
        latitudes = p @ axis
        latitude = float(latitudes.min())
        if latitude <= 0:
            raise ValueError('The reported normalized-mixture certificate requires positive latitude')
        deltas = None
        info = dict(n=len(p), d=p.shape[1], mean_norm=float(np.linalg.norm(mean)), min_latitude=latitude)
        digest = hashlib.sha256()
        for offset in range(0, len(p), 1024):
            digest.update(p[offset:offset+1024].tobytes())
        info['normalized_past_vectors_sha256'] = digest.hexdigest()
        info['past_texts_sha256'] = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        if model is not None:
            deltas, details, seconds = local_probes(domain, texts, model, args.local_edits,
                                                    start + args.minutes*60, np.random.default_rng(args.seed))
            info.update(local_probes=details, onnx_seconds=seconds,
                        sample_radius=float(np.linalg.norm(deltas, axis=1).max()))
        output['domains'][domain] = info
        by_target, polars = {}, {}
        for rec in records:
            if rec['domain'] != domain:
                continue
            key = (domain, rec['arm'], rec['k'])
            t = targets[key]
            ref = refs[key]
            if key not in by_target:
                scores = p @ t
                hidden = np.asarray(ref['hidden_pos'], dtype=int)
                scores[hidden] = -np.inf
                c = float(scores.max())
                if abs(c - ref['ceiling']) > 3e-4:
                    raise ValueError(f'Masking / normalization mismatch for {key}')
                bound = dict(ceiling=c, ceiling_star=max(c, ref['heldout_ceiling']),
                             cap=cap_bound(latitude, t @ axis), mix_zero=min(1., c/latitude))
                beta = max(0., float((scores/latitudes).max()))
                bound['cone_beta'] = beta
                bound['cone_ratio'] = min(1., beta)
                if args.cone_steps and rec['arm'] == 'FAR':
                    cert = cone_certificate(p, t, axis, latitude, hidden, args.cone_steps,
                                            start + args.minutes*60)
                    bound['cone_optimized'] = min(bound['cone_ratio'], cert['upper'])
                    polars[key] = np.asarray(cert['polar'])
                    output['cone_certificates'].append(dict(domain=domain, arm=rec['arm'], k=rec['k'], **cert))
                if deltas is not None:
                    bound.update(cover_local=min(1., c + info['sample_radius']),
                                 sample_radius=info['sample_radius'],
                                 mix_local=min(1., c/latitude + info['sample_radius']),
                                 directional_local=min(1., c + max(0., float((deltas @ t).max()))))
                by_target[key] = bound
            writer = rec['config']['writer'].split(':')[1]
            row = dict(domain=domain, writer=writer, arm=rec['arm'], k=rec['k'], **by_target[key])
            gate = rec.get('gate', {})
            row['h_coh'] = gate.get('h_coh')
            row['opus_accepts_winner'] = (opus.get(f'{domain}/{writer}/{rec["arm"]}-{rec["k"]}') or 0) >= 7
            if row['h_coh'] is not None:
                if min(gate['j1'], gate['j2']) < 7:
                    raise ValueError('Recorded witness did not pass both judges')
                sha = hashlib.sha256(gate['coh_text'].encode()).hexdigest()
                e = unit(np.load(cache / f'{sha}.npy'))
                score = float(e @ t)
                if abs(score - row['h_coh']) > 3e-5:
                    raise ValueError('Cached embedding does not reproduce witness score')
                row['h_coh'] = score
                ratio_polar = t - by_target[key]['cone_beta'] * axis
                # A separating hyperplane gives a lower bound on distance to the
                # entire corpus cone, stronger than merely refuting a specific m8.
                row['cone_distance_lower'] = max(0., float(e @ ratio_polar)) / max(1e-15, np.linalg.norm(ratio_polar))
                if key in polars:
                    z = polars[key]
                    row['optimized_cone_distance_lower'] = max(0., float(e @ z)) / max(1e-15, np.linalg.norm(z))
                sims = p @ e
                sims[np.asarray(ref['hidden_pos'], dtype=int)] = -np.inf
                nn = np.argpartition(sims, -8)[-8:]
                m = p[nn].mean(0)
                centre = unit(m)
                anch = float(e @ centre)
                row.update(text_sha256=sha, nearest_distance=float(np.sqrt(max(0., 2 - 2*sims.max()))),
                           mixture_norm=float(np.linalg.norm(m)), mixture_residual=float(np.linalg.norm(e-centre)),
                           anchor=anch, anchor_remainder=float((e-anch*centre)@t))
                # A per-witness algebraic check, not fitting a universal slack.
                if score > min(1., by_target[key]['ceiling']/np.linalg.norm(m)+row['mixture_residual'])+1e-8:
                    raise AssertionError('Normalized-mixture inequality failed')
            output['rows'].append(row)
        print(f'{domain}: corpus and recorded witnesses audited', file=sys.stderr, flush=True)
        del p
    frame = pd.DataFrame(output['rows'])
    combined = []
    for _, g in frame.groupby(['domain', 'arm', 'k']):
        row = g.loc[g.h_coh.idxmax()].to_dict() if g.h_coh.notna().any() else g.iloc[0].to_dict()
        row['writer'] = 'max over writers'
        combined.append(row)
    all_rows = pd.concat([frame, pd.DataFrame(combined)], ignore_index=True)
    output['summary'] = summarise(all_rows)
    # Sensitivity only on already rated winners: never infer unseen Opus judgments.
    sensitivity = frame.copy()
    sensitivity.loc[~sensitivity.opus_accepts_winner, 'h_coh'] = np.nan
    comb = []
    for _, g in sensitivity[sensitivity.arm != 'REAL'].groupby(['domain', 'arm', 'k']):
        row = g.loc[g.h_coh.idxmax()].to_dict() if g.h_coh.notna().any() else g.iloc[0].to_dict()
        row['writer'] = 'max with Opus on recorded winners'
        comb.append(row)
    output['opus_sensitivity'] = summarise(pd.DataFrame(comb))
    output['seconds'] = time.monotonic() - start
    # JSON null, never nonstandard NaN; preserve missing coherent scores.
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    print(json.dumps(clean(output), indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--probe', action='store_true', help='Legacy k=0 witness probe')
    modes.add_argument('--audit', action='store_true', help='Round-4 diagnostics, read-only; JSON to stdout')
    modes.add_argument('--self-check', action='store_true', help='Check exact soluble cone examples; no data or ONNX')
    ap.add_argument('--local-edits', type=int, default=0, help='Real-text anchors per domain, each with six edits')
    ap.add_argument('--cone-steps', type=int, default=0, help='NNLS column steps per FAR target; 0 skips optimization')
    ap.add_argument('--minutes', type=float, default=28.)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--data', type=Path, default=DATA)
    args = ap.parse_args()
    if not 0 < args.minutes <= 30 or args.local_edits < 0 or args.cone_steps < 0:
        ap.error('Require 0 < minutes <= 30 and local-edits, cone-steps >= 0')
    if (args.local_edits or args.cone_steps) and not args.audit:
        ap.error('--local-edits / --cone-steps require --audit')
    if args.self_check:
        self_check()
    elif args.audit:
        audit(args)
    else:
        legacy(args)


if __name__ == '__main__':
    main()
