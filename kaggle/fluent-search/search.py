"""Fluency-constrained discrete search towards embedding targets (docs/prereg/fluent_search.md). Kaggle, GPU T4 x2.

Objective J(s) = cos(enc(s), t) - lam * meanNLL_LM(s): enc = Qwen3-Embedding-0.6B (document embedding, explicit EOS,
last-token pooling, as in kaggle/support-search), LM = Qwen3-0.6B-Base (same tokenizer), NLL in nats per text token
given a fixed one-line domain header. No corpus text is ever used: the search starts from the empty domain scaffold
(Steam "Genres: ... Tags: ... About: ...", arXiv "Title: .... Abstract: ..."), exactly the format of DOMAINS[d].render.

Phase 1, guided generation (AutoDAN-style): spans are written left to right; at each step the candidates are the LM's
top-k next tokens plus the encoder-gradient top-k, every candidate is scored exactly (encoder with KV cache), the best
J is appended; a span closes when closing it scores higher than any extension (after its minimum length).
Phase 2, GCG-style refinement of the whole text: replacements (gradient top-k and LM top-k at the position),
deletions and LM insertions at random editable positions; each proposal is decoded, re-parsed into the domain format,
re-tokenised and scored exactly with both models (encoder on one GPU, LM on the other); the best is accepted if it
raises J. Two worker processes (encoder on GPU i, LM on GPU 1-i) split the targets. Progress is printed per block.
Outputs (append-only): /kaggle/working/fluent_gpu{i}.jsonl (one record per target x lambda, with the pool of accepted
states), nll_reference.jsonl. GPU scores are proposals only: local ONNX re-embedding is authoritative.
"""
import copy
import glob
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ENC = 'Qwen/Qwen3-Embedding-0.6B'
LM = 'Qwen/Qwen3-1.7B-Base'  # smoke 1-6 used Qwen3-0.6B-Base: judges rejected its prose (see prereg)
EOS = 151643
SEED = 20261026
# Local CPU test only (FLUENT_CPU=1): inputs/outputs elsewhere, float32 on CPU, one worker. Kaggle ignores these.
CPU = os.environ.get('FLUENT_CPU') == '1'
IN_DIR = os.environ.get('FLUENT_IN', '/kaggle/input')
OUT_DIR = Path(os.environ.get('FLUENT_OUT', '/kaggle/working'))

# ---- run configuration (the only block edited between the smoke and the full run; recorded in every output row)
RUN = dict(
    name='tune-seeds-arxiv',
    lm='Qwen/Qwen3-1.7B-Base',              # arXiv tuning seeds: same LM as the rev-1 arXiv test seeds
    lambdas=[0.0, 0.05],
    ks={'REAL': list(range(11, 21))},  # seed_rewrite tuning targets (not the test set)
    domains=['arxiv'],
    gen_k_lm=32, gen_k_grad=32,             # phase 1 candidates per step
    grad_lm_top=100,                        # lam > 0: gradient proposals restricted to the LM's top-M tokens
    close_top=5,                            # a list/title span closes only if the scaffold token is in the LM top-5
    ref_seconds=35, ref_positions=6,        # phase 2 budget per target x lambda, positions per step
    ref_k_grad=6, ref_k_lm=6, ref_k_ins=2,  # phase 2 proposals per position (+1 deletion)
    stop_minutes=60, kill_minutes=70,       # workers stop starting jobs / parent watchdog (Kaggle limit 12 h)
)

if os.environ.get('FLUENT_RUN'):
    RUN.update(json.loads(os.environ['FLUENT_RUN']))
LM = RUN.get('lm', LM)

# LM context only (never embedded, never scored): a fixed content-free header, written before any run and independent
# of every target. No example text (an invented exemplar, smoke 4-5, became a topic attractor) and, for Steam, no field
# names (smoke 6-7: the LM echoed them into the lists), no corpus text.
HEADER = {
    'steam': 'A video game page on the Steam store:\n',
    'arxiv': ('arXiv computer science paper. The title is followed by the abstract, a paragraph that summarises the '
              'problem, the approach and the contribution.\n'),
}
# (scaffold piece before the span, min tokens, max tokens, kind); spans start with a space-prefixed token
SPANS = {'steam': [('Genres:', 1, 10, 'short'), ('. Tags:', 3, 40, 'short'), ('. About:', 40, 110, 'long')],
         'arxiv': [('Title:', 4, 30, 'short'), ('. Abstract:', 60, 130, 'long')]}
LIMITS = {'steam': {'genres': 7, 'tags': 20, 'about': 225}, 'arxiv': {'title': 18, 'abstract': 197}}
MAX_TOKENS = 511


def parse(domain, text):
    """Rendered text -> fields, or None if it is not exactly in the domain's rendered format (render(parse) == text)
    or breaks the corpus-p99 bounds of latent_gap.coherent.BOUNDS."""
    lim = LIMITS[domain]
    if domain == 'steam':
        if not text.startswith('Genres: ') or '. Tags: ' not in text:
            return None
        g, rest = text[len('Genres: '):].split('. Tags: ', 1)
        if '. About: ' not in rest:
            return None
        tg, about = rest.split('. About: ', 1)
        item = {'genres': g.split(','), 'tags': tg.split(','), 'about': about}
        if f"Genres: {','.join(item['genres'])}. Tags: {','.join(item['tags'])}. About: {about[:1200]}" != text:
            return None
        if (len(item['genres']) > lim['genres'] or len(item['tags']) > lim['tags']
                or len(about[:1200].split()) > lim['about'] or not g.strip() or not tg.strip() or not about.strip()):
            return None
        return item
    if not text.startswith('Title: ') or '. Abstract: ' not in text:
        return None
    title, ab = text[len('Title: '):].split('. Abstract: ', 1)
    if f'Title: {title}. Abstract: {ab[:1200]}' != text:
        return None
    if len(title.split()) > lim['title'] or len(ab[:1200].split()) > lim['abstract'] or not title.strip() or not ab.strip():
        return None
    return {'title': title, 'abstract': ab}


ITEM_MAX = {('steam', 0): 4, ('steam', 1): 12}  # proposal caps, inside the corpus p99 bounds (7 / 20)
ITEM_WORDS = 4  # words per genre / tag item (corpus tags are 1-3 words)
# rev 2 (4B smoke): items are plain labels: letters, digits, apostrophes, '+', single spaces and in-word hyphens,
# no 'and' / 'or' chaining, at most 30 characters
ITEM_CHARS = 30
ITEM_RE = re.compile(r"[a-z0-9'+]+(?:[ -][a-z0-9'+]+)*")
BAD_ITEM = re.compile(r"\b(?:and|or)\b")


def span_ok(domain, si, kind, span_text, closing):
    """Proposal constraints (not validity): in list spans every completed item is non-empty and unique and the item
    count stays within the corpus p99; the last (prose) span may only close at a sentence end."""
    if kind == 'short' and (domain, si) in ITEM_MAX:
        items = [x.strip().lower() for x in span_text.split(',')]
        done = items if closing else items[:-1]
        if any(not x for x in done) or len(set(done)) < len(done) or len(items) > ITEM_MAX[domain, si]:
            return False
        if any(len(x.split()) > ITEM_WORDS or len(x) > ITEM_CHARS or not ITEM_RE.fullmatch(x) or BAD_ITEM.search(x)
               for x in items if x):
            return False
        seen = set()  # every item adds a word not used by earlier items (no recombined keyword stuffing)
        for x in done:
            w = set(x.replace('-', ' ').split())
            if w <= seen or len(w & seen) > 1:  # rev 2: at most one word shared with earlier items
                return False
            seen |= w
    if domain == 'arxiv' and si == 0 and len(span_text.split()) > LIMITS['arxiv']['title']:
        return False
    if kind == 'long' and len(span_text.split()) > max(LIMITS[domain].values()):
        return False
    if kind == 'long' and closing:
        return span_text.rstrip()[-1:] in ('.', '!', '?')
    return True


def struct_ok(domain, text, need_end):
    item = parse(domain, text)
    if item is None:
        return False
    if domain == 'steam':
        if not (span_ok(domain, 0, 'short', ','.join(item['genres']), True)
                and span_ok(domain, 1, 'short', ','.join(item['tags']), True)):
            return False
        body = item['about']
    else:
        body = item['abstract']
    return not need_end or body.rstrip()[-1:] in ('.', '!', '?')


def worker(wid, jobs, deadline, nll_ref):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    de, dl = ('cpu', 'cpu') if CPU else (f'cuda:{wid}', f'cuda:{1 - wid}')
    dt = torch.float32 if CPU else torch.float16
    rng = random.Random(SEED + wid)
    torch.manual_seed(SEED + wid)
    print(f'W{wid} loading encoder on {de}, LM on {dl}', flush=True)
    tok = AutoTokenizer.from_pretrained(ENC)
    enc = AutoModel.from_pretrained(ENC, dtype=dt, attn_implementation='sdpa').to(de).eval()
    lm = AutoModelForCausalLM.from_pretrained(LM, dtype=dt, attn_implementation='sdpa').to(dl).eval()
    for p in list(enc.parameters()) + list(lm.parameters()):
        p.requires_grad_(False)
    W = enc.get_input_embeddings().weight
    V = W.shape[0]  # encoder vocabulary (151669); the LM's logits are padded to 151936 and sliced to V
    vocab = tok.batch_decode([[i] for i in range(min(V, len(tok)))]) + [''] * max(0, V - len(tok))
    special = set(tok.all_special_ids)
    ascii_ok = torch.tensor([i not in special and len(s.strip()) > 0 and all(32 <= ord(c) < 127 for c in s)
                             for i, s in enumerate(vocab)])
    masks = {'long': ascii_ok, 'short': ascii_ok & torch.tensor([not any(c in s for c in '.:;()') for s in vocab])}
    space = torch.tensor([s.startswith(' ') and len(s.strip()) > 0 for s in vocab])
    masks_d = {(kind, first): (m & space if first else m).to(de) for kind, m in masks.items() for first in (0, 1)}
    masks_l = {key: m.to(dl) for key, m in masks_d.items()}
    word_d, word_l = space.to(de), space.to(dl)
    hdr = {d: tok.encode(h, add_special_tokens=False) for d, h in HEADER.items()}
    # token 4-grams of the header: never reproduced in the text
    banned4 = {}
    for d, h in HEADER.items():
        ex = tok.encode(h, add_special_tokens=False)
        banned4[d] = {tuple(ex[i:i + 4]) for i in range(len(ex) - 3)}

    def n_rep3(a):
        tri = [tuple(a[i:i + 3]) for i in range(len(a) - 2)]
        return len(tri) - len(set(tri))

    def copies(domain, a):
        return any(tuple(a[i:i + 4]) in banned4[domain] for i in range(len(a) - 3))
    print(f'W{wid} ready: {len(jobs)} jobs, vocabulary ascii {int(ascii_ok.sum())}/{V}', flush=True)

    def ids_of(text):
        a = tok.encode(text, add_special_tokens=False)
        return a[:MAX_TOKENS]

    # ---------------- exact scoring, both GPUs (launch both, then synchronise)
    def enc_launch(seqs, bs=32):
        outs = []
        with torch.no_grad():
            for b in range(0, len(seqs), bs):
                chunk = [s + [EOS] for s in seqs[b:b + bs]]
                n = max(map(len, chunk))
                x = torch.full((len(chunk), n), EOS, device=de, dtype=torch.long)
                m = torch.zeros_like(x)
                for r, a in enumerate(chunk):
                    x[r, :len(a)] = torch.tensor(a, device=de)
                    m[r, :len(a)] = 1
                h = enc(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
                e = F.normalize(h[torch.arange(len(chunk), device=de), m.sum(1) - 1].float(), dim=-1)
                outs.append(e)
        return torch.cat(outs)

    def lm_launch(domain, seqs, bs=4):
        """Mean NLL (nats) of the text tokens given the header; logits only at text positions (memory)."""
        outs = []
        H = hdr[domain]
        h0 = len(H) - 1
        with torch.no_grad():
            for b in range(0, len(seqs), bs):
                chunk = [H + s for s in seqs[b:b + bs]]
                n = max(map(len, chunk))
                x = torch.full((len(chunk), n), EOS, device=dl, dtype=torch.long)
                m = torch.zeros_like(x)
                for r, a in enumerate(chunk):
                    x[r, :len(a)] = torch.tensor(a, device=dl)
                    m[r, :len(a)] = 1
                hs = lm.model(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[:, h0:-1]
                logits = lm.lm_head(hs).float()
                nll = F.cross_entropy(logits.transpose(1, 2), x[:, h0 + 1:], reduction='none')
                keep = m[:, h0 + 1:]
                outs.append((nll * keep).sum(1) / keep.sum(1))
                del hs, logits, nll
        return torch.cat(outs)

    def score(domain, seqs, t):
        e = enc_launch(seqs)
        nl = lm_launch(domain, seqs)
        return (e @ t).cpu().tolist(), nl.cpu().tolist()

    # ---------------- phase 1: guided left-to-right generation
    def generate(domain, t, lam):
        spans = SPANS[domain]
        ids = []                       # text tokens so far (shared tokenizer)
        sum_nll, n_txt = 0.0, 0
        evals = 0
        # encoder cache over the text prefix; LM cache over header + text prefix
        with torch.no_grad():
            lo = lm(input_ids=torch.tensor([hdr[domain]], device=dl), use_cache=True)
        lm_cache, lm_logits = lo.past_key_values, lo.logits[0, -1, :V].float()
        enc_cache = None

        def push(new):  # append tokens to both caches
            nonlocal enc_cache, lm_cache, lm_logits, sum_nll, n_txt
            with torch.no_grad():
                for x in new:
                    lp = torch.log_softmax(lm_logits, -1)
                    sum_nll += float(-lp[x]); n_txt += 1
                    o = lm(input_ids=torch.tensor([[x]], device=dl), past_key_values=lm_cache, use_cache=True)
                    lm_cache, lm_logits = o.past_key_values, o.logits[0, -1, :V].float()
                ex = torch.tensor([new], device=de)
                o = enc(input_ids=ex, past_key_values=enc_cache, use_cache=True)
                enc_cache = o.past_key_values
            ids.extend(new)

        def closing_score(piece_ids):  # J of prefix + piece (or of the prefix alone if piece_ids is empty)
            seq = ids + piece_ids
            c, nl = score(domain, [seq], t)
            return c[0] - lam * nl[0], c[0], nl[0]

        for si, (piece, lo_n, hi_n, kind) in enumerate(spans):
            push(tok.encode(piece, add_special_tokens=False))
            n = 0
            span_start = len(ids)
            nxt = tok.encode(spans[si + 1][0], add_special_tokens=False) if si + 1 < len(spans) else []
            while True:
                lp = torch.log_softmax(lm_logits, -1)
                mask_l = masks_l[kind, int(n == 0)].clone()
                mask_e = masks_d[kind, int(n == 0)].clone()
                # no repeated token 3-gram
                if len(ids) >= 2:
                    a, b = ids[-2], ids[-1]
                    bad = [ids[i + 2] for i in range(len(ids) - 2) if ids[i] == a and ids[i + 1] == b]
                    if len(ids) >= 3:
                        bad += [g[3] for g in banned4[domain] if g[:3] == tuple(ids[-3:])]
                    if bad:
                        mask_l[bad] = False; mask_e[bad] = False
                cand = set(torch.topk(lp.masked_fill(~mask_l, -1e9), RUN['gen_k_lm']).indices.tolist())
                # encoder-gradient proposals: gradient at an appended placeholder (the LM's top-1 token)
                dummy = int(torch.argmax(lp.masked_fill(~mask_l, -1e9)))
                seq = torch.tensor([ids + [dummy, EOS]], device=de)
                emb = enc.get_input_embeddings()(seq).detach().requires_grad_(True)
                h = enc(inputs_embeds=emb, use_cache=False).last_hidden_state
                (F.normalize(h[0, -1].float(), dim=-1) @ t).backward()
                g = emb.grad[0, -2].float()
                if lam > 0:  # lam > 0: gradient proposals only among the LM's top-M next tokens (AutoDAN-style)
                    keep = torch.zeros_like(mask_e)
                    keep[torch.topk(lp.masked_fill(~mask_l, -1e9), RUN['grad_lm_top']).indices.to(de)] = True
                    mask_e &= keep
                gains = (W.float() @ g).masked_fill(~mask_e, -1e9)
                cand |= set(torch.topk(gains, RUN['gen_k_grad']).indices.tolist())
                del emb, h, g, gains
                sp = ids[span_start:]
                cand = [c for c in sorted(cand)
                        if span_ok(domain, si, kind, tok.decode(sp + [c], skip_special_tokens=True), False)]
                hard_max = hi_n + 15  # past hi_n the span closes at the first point where closing is allowed
                can_close = n >= lo_n and span_ok(domain, si, kind, tok.decode(sp, skip_special_tokens=True), True)
                if can_close and nxt and n < hard_max and (domain, si) not in ITEM_MAX:  # titles: close only where the LM would
                    can_close = nxt[0] in torch.topk(lp, RUN['close_top']).indices.tolist()
                if not cand or n >= hard_max:
                    break
                B = len(cand)
                with torch.no_grad():
                    pc = copy.deepcopy(enc_cache)
                    pc.batch_repeat_interleave(B)
                    x = torch.tensor([[c, EOS] for c in cand], device=de)
                    m = torch.ones((B, len(ids) + 2), device=de, dtype=torch.long)
                    h = enc(input_ids=x, attention_mask=m, past_key_values=pc, use_cache=True).last_hidden_state
                    cos = (F.normalize(h[:, -1].float(), dim=-1) @ t).cpu()
                    del pc, h
                nll_c = (sum_nll - lp[cand].cpu()) / (n_txt + 1)
                J = cos - lam * nll_c
                evals += B
                best = int(torch.argmax(J))
                if can_close:
                    close = closing_score(nxt)
                    evals += 1
                    if n >= hi_n or close[0] >= float(J[best]):
                        break
                push([cand[best]])
                n += 1
        text = tok.decode(ids, skip_special_tokens=True)
        return text, evals

    # ---------------- phase 2: refinement of the whole text
    def span_ranges(domain, text):
        """Character ranges of the free spans in a rendered text."""
        if domain == 'steam':
            a = len('Genres:')
            b = text.index('. Tags: ')
            c = text.index('. About: ', b)
            return [(a, b), (b + len('. Tags:'), c), (c + len('. About:'), len(text))], ['short', 'short', 'long']
        a = len('Title:')
        b = text.index('. Abstract: ')
        return [(a, b), (b + len('. Abstract:'), len(text))], ['short', 'long']

    def refine(domain, t, lam, text, budget_s, pool, log):
        start = time.monotonic()
        ids = ids_of(text)
        c, nl = score(domain, [ids], t)
        cur = dict(text=text, cos=c[0], nll=nl[0], J=c[0] - lam * nl[0], step=-1)
        pool.append(cur)
        evals, step, accepted = 1, 0, 0
        need_end = struct_ok(domain, text, True)  # keep a sentence-final ending once there is one
        while time.monotonic() - start < budget_s:
            enc_ = tok(cur['text'], add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc_['input_ids'][:MAX_TOKENS], enc_['offset_mapping'][:MAX_TOKENS]
            ranges, kinds = span_ranges(domain, cur['text'])
            edit = []   # (position, kind, first-of-span)
            for p, (s, e) in enumerate(offs):
                # whole-word edits only: token p starts a word and the next token starts a new word or is punctuation
                nxt_s = vocab[ids[p + 1]] if p + 1 < len(ids) else ' '
                if not vocab[ids[p]].startswith(' ') or (nxt_s[:1].isalnum()):
                    continue
                for (a, b), kd in zip(ranges, kinds):
                    if a <= s and e <= b:
                        edit.append((p, kd, int(s == a)))
            if not edit:
                break
            # encoder gradient (GPU de) and LM next-token distribution (GPU dl) on the current text
            seq = torch.tensor([ids + [EOS]], device=de)
            emb = enc.get_input_embeddings()(seq).detach().requires_grad_(True)
            h = enc(inputs_embeds=emb, use_cache=False).last_hidden_state
            (F.normalize(h[0, -1].float(), dim=-1) @ t).backward()
            grad = emb.grad[0].float()
            with torch.no_grad():
                H = hdr[domain]
                lpos = torch.log_softmax(lm(input_ids=torch.tensor([H + ids], device=dl)).logits[0, :, :V].float(), -1)
            props = []
            for p, kd, first in rng.sample(edit, min(RUN['ref_positions'], len(edit))):
                me, ml = masks_d[kd, first] & word_d, masks_l[kd, first] & word_l
                lmrow = lpos[len(H) + p - 1].masked_fill(~ml, -1e9)
                if lam > 0:
                    keep = torch.zeros_like(me)
                    keep[torch.topk(lmrow, RUN['grad_lm_top']).indices.to(de)] = True
                    me = me & keep
                gains = ((W.float() @ grad[p]) - float(W[ids[p]].float() @ grad[p])).masked_fill(~me, -1e9)
                reps = set(torch.topk(gains, RUN['ref_k_grad']).indices.tolist())
                reps |= set(torch.topk(lmrow, RUN['ref_k_lm']).indices.tolist())
                reps.discard(ids[p])
                for r in reps:
                    props.append(ids[:p] + [r] + ids[p + 1:])
                if not first:
                    props.append(ids[:p] + ids[p + 1:])
                for r in torch.topk(lmrow, RUN['ref_k_ins']).indices.tolist():
                    props.append(ids[:p] + [r] + ids[p:])
            del emb, h, grad, lpos
            texts, seqs, seen = [], [], {cur['text']}
            rep0 = n_rep3(ids)
            for a in props:
                s = tok.decode(a, skip_special_tokens=True)
                if s in seen or not struct_ok(domain, s, need_end):
                    continue
                seen.add(s)
                q = ids_of(s)
                if len(q) >= MAX_TOKENS or copies(domain, q) or n_rep3(q) > rep0:
                    continue
                texts.append(s); seqs.append(q)
            if not seqs:
                step += 1
                continue
            cs, ns = score(domain, seqs, t)
            evals += len(seqs)
            Js = [c_ - lam * n_ for c_, n_ in zip(cs, ns)]
            b = max(range(len(Js)), key=Js.__getitem__)
            if Js[b] > cur['J']:
                cur = dict(text=texts[b], cos=cs[b], nll=ns[b], J=Js[b], step=step)
                pool.append(cur); accepted += 1
            # also keep the highest-cos proposal of the step (the gate walks the pool by cos)
            bc = max(range(len(cs)), key=cs.__getitem__)
            if bc != b and cs[bc] > cur['cos']:
                pool.append(dict(text=texts[bc], cos=cs[bc], nll=ns[bc], J=Js[bc], step=step, side=True))
            step += 1
            if step % 10 == 0:
                log(f'refine step {step}: J={cur["J"]:.4f} cos={cur["cos"]:.4f} nll={cur["nll"]:.3f} '
                    f'accepted={accepted} evals={evals}')
        return cur, evals, step, accepted

    # ---------------- NLL reference (corpus fluency), worker 0 only
    if wid == 0 and nll_ref:
        rows = []
        for d in RUN['domains']:
            texts = [r['text'] for r in nll_ref if r['domain'] == d]
            nl = lm_launch(d, [ids_of(s) for s in texts]).cpu().tolist()
            rows += [dict(domain=d, text=s, nll=v) for s, v in zip(texts, nl)]
            print(f'W0 NLL reference {d}: median {sorted(nl)[len(nl) // 2]:.3f} over {len(nl)} corpus texts', flush=True)
        (OUT_DIR / 'nll_reference.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    dest = OUT_DIR / f'fluent_gpu{wid}.jsonl'
    done_jobs = 0
    began = time.monotonic()
    for j in jobs:
        for lam in RUN['lambdas']:
            if time.monotonic() > deadline:
                print(f'W{wid} deadline: stopping before {j["domain"]}/{j["arm"]}/{j["k"]} lam={lam}', flush=True)
                return
            tag = f'W{wid} {j["domain"]}/{j["arm"]}/{j["k"]} lam={lam}'
            log = lambda m: print(f'{tag} {m}', flush=True)
            t0 = time.monotonic()
            t = F.normalize(torch.tensor(j['target'], device=de, dtype=torch.float32), dim=0)
            text, gen_evals = generate(j['domain'], t, lam)
            gen_s = time.monotonic() - t0
            log(f'generated in {gen_s:.1f}s ({gen_evals} evals): {text[:160]!r}')
            pool = []
            if parse(j['domain'], text) is None:
                log('WARNING generated text is not in the domain format; refinement skipped')
                cur, ref_evals, steps, acc = None, 0, 0, 0
            else:
                cur, ref_evals, steps, acc = refine(j['domain'], t, lam, text, RUN['ref_seconds'], pool, log)
            row = dict(domain=j['domain'], arm=j['arm'], k=j['k'], lam=lam, generated=text,
                       final=cur, pool=pool, gen_evals=gen_evals, ref_evals=ref_evals, ref_steps=steps,
                       ref_accepted=acc, gen_seconds=round(gen_s, 1),
                       seconds=round(time.monotonic() - t0, 1), run=RUN, enc=ENC, lm=LM, worker=wid)
            with dest.open('a') as f:
                f.write(json.dumps(row) + '\n')
            fin = f'cos={cur["cos"]:.4f} nll={cur["nll"]:.3f}' if cur else 'none'
            log(f'saved: {fin} pool={len(pool)} steps={steps} ({row["seconds"]}s)')
        done_jobs += 1
        el = time.monotonic() - began
        print(f'W{wid} block {done_jobs}/{len(jobs)} targets done, {el / 60:.1f} min, '
              f'ETA {el / done_jobs * (len(jobs) - done_jobs) / 60:.1f} min', flush=True)


def main():
    began = time.monotonic()
    if not CPU:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1', 'accelerate'],
                       check=True, timeout=300)
    import torch
    if not CPU and torch.cuda.device_count() < 2:
        raise RuntimeError('Select Kaggle GPU T4 x2')
    found = glob.glob(f'{IN_DIR}/**/fluent_inputs.json', recursive=True)
    if len(found) != 1:
        raise RuntimeError(f'Expected one fluent_inputs.json; got {found}')
    jobs = [j for j in json.loads(Path(found[0]).read_text())
            if j['domain'] in RUN['domains'] and j['k'] in RUN['ks'].get(j['arm'], [])]
    ref = glob.glob(f'{IN_DIR}/**/nll_reference.json', recursive=True)
    nll_ref = json.loads(Path(ref[0]).read_text()) if ref else []
    jobs.sort(key=lambda j: (j['k'], j['domain'], j['arm']))  # a timeout still covers every stratum
    print(f'{len(jobs)} targets x {len(RUN["lambdas"])} lambdas; run {RUN["name"]}', flush=True)
    deadline = began + RUN['stop_minutes'] * 60
    if CPU:  # local check: in-process, one worker
        worker(0, jobs, deadline, nll_ref)
        return
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(i, jobs[i::2], deadline, nll_ref)) for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(max(1, began + RUN['kill_minutes'] * 60 - time.monotonic()))
    for p in ps:
        if p.is_alive():
            p.terminate(); p.join()
            print('Watchdog stopped a worker; partial outputs saved', flush=True)
        elif p.exitcode:
            raise RuntimeError(f'worker exited {p.exitcode}')
    print('Finished. Re-embed every saved text with the local ONNX encoder before reporting.', flush=True)


if __name__ == '__main__':
    main()
