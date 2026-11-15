"""R4 stage 3 of docs/prereg/review_far_decoding.md: invert REAL, GAP and FAR targets rebuilt in the gtr-t5-base
space with vec2text's pretrained gtr-base corrector, one domain per T4 (CUDA_VISIBLE_DEVICES per worker).

Budgets: C0 num_steps=0 (inverter hypothesis only), C1 num_steps=20 sequence_beam_width=1, C2 num_steps=50
sequence_beam_width=4. Vectors passed to the corrector = unit target x median raw norm of the corpus (`scale`).
Round-trip check (budget "RT"): the 50 REAL items' own raw embeddings at C1, BLEU (sacrebleu sentence BLEU) and
exact-token F1 (multiset overlap of T5 token ids) against their 32-token text. Every inverted text, the 32-token
truncations of the calibration items and of the REAL items are re-embedded with the same gtr pipeline as
kaggle/review-gtr-embed (raw mean-pooled vectors, scored locally by scripts/review_far_inversion.py score).
Progress: one line per batch of 25 targets per budget; outputs rewritten after every batch; OOM halves the batch.
run_info.json {"smoke": true} = 25 REAL targets only (all budgets + round trip), the registered smoke run.
"""
import glob
import json
import os
import subprocess
import sys
import time

OUT = "/kaggle/working"
DOMS = ("steam", "arxiv")

WORKER = r'''
import json, os, sys, time
import numpy as np, torch
dom, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.load(open(f"{inp}/run_info.json"))
import vec2text
from transformers import AutoModel, AutoTokenizer
import sacrebleu
t0 = time.time()
def log(m):
    print(f"[{dom} {time.time() - t0:6.0f}s] {m}", flush=True)
z = np.load(f"{inp}/targets_{dom}.npz")
tx = json.load(open(f"{inp}/texts_{dom}.json"))
scale = float(z["scale"])
corrector = vec2text.load_pretrained_corrector("gtr-base")
tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda").eval()
log(f"loaded corrector and gtr encoder; scale {scale:.4f}; smoke {info['smoke']}")

def embed(texts):
    vs = []
    for i in range(0, len(texts), 128):
        x = tok(texts[i:i + 128], return_tensors="pt", max_length=32, truncation=True, padding="max_length").to("cuda")
        with torch.no_grad():
            h = enc(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
        m = x["attention_mask"]
        vs.append(((h * m[..., None]).sum(1) / m.sum(1)[:, None]).float().cpu().numpy())
    return np.concatenate(vs) if vs else np.zeros((0, 768), np.float32)

def trunc32(text):
    ids = tok(text, max_length=32, truncation=True)["input_ids"]
    return tok.decode(ids, skip_special_tokens=True)

C0_PATH = []
def invert(E, steps, beam):
    x = torch.tensor(E, dtype=torch.float32, device="cuda")
    if steps == 0:
        try:
            r = vec2text.invert_embeddings(embeddings=x, corrector=corrector, num_steps=0, sequence_beam_width=0)
            if not C0_PATH:
                C0_PATH.append("num_steps=0")
            return r
        except Exception as e:  # recorded: the hypothesis-only path of the same function
            if not C0_PATH:
                C0_PATH.append(f"num_steps=None after num_steps=0 failed: {type(e).__name__}: {str(e)[:200]}")
                log(C0_PATH[0])
            return vec2text.invert_embeddings(embeddings=x, corrector=corrector, num_steps=None)
    return vec2text.invert_embeddings(embeddings=x, corrector=corrector, num_steps=steps, sequence_beam_width=beam)

def run(E, steps, beam):
    try:
        return invert(E, steps, beam)
    except torch.OutOfMemoryError:
        oom = True
    torch.cuda.empty_cache()
    if len(E) == 1:
        raise RuntimeError("OOM on a single target")
    h = len(E) // 2
    log(f"OOM on {len(E)} targets, splitting")
    return run(E[:h], steps, beam) + run(E[h:], steps, beam)

def token_f1(a, b):
    ta, tb = tok(a)["input_ids"], tok(b)["input_ids"]
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)

rows = []
def save():
    with open(f"{out}/inversions_{dom}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

budgets = {k: tuple(v) for k, v in info["budgets"].items()}
arms = ("REAL",) if info["smoke"] else ("REAL", "GAP", "FAR")
n = info["n_smoke"] if info["smoke"] else len(z["REAL"])
BATCH = 25
for arm in arms:
    T = z[arm][:n].astype(np.float32)
    for bname, (steps, beam) in budgets.items():
        tb = time.time()
        for s in range(0, n, BATCH):
            texts = run(T[s:s + BATCH] * scale, steps, beam)
            embs = embed(texts)
            for j, (t, e) in enumerate(zip(texts, embs)):
                rows.append({"domain": dom, "arm": arm, "k": s + j, "budget": bname, "text": t,
                             "emb": [round(float(v), 6) for v in e]})
            save()
            log(f"{arm} {bname} targets {min(s + BATCH, n)}/{n} ({time.time() - tb:.0f}s this budget)")
# round trip: REAL items' own raw embeddings at C1
refs = [trunc32(r["text"]) for r in tx["real"]][:n]
steps, beam = budgets["C1"]
for s in range(0, n, BATCH):
    texts = run(z["REAL_raw"][s:s + BATCH].astype(np.float32), steps, beam)
    embs = embed(texts)
    for j, (t, e) in enumerate(zip(texts, embs)):
        ref = refs[s + j]
        rows.append({"domain": dom, "arm": "REAL", "k": s + j, "budget": "RT", "text": t, "ref_text": ref,
                     "bleu": float(sacrebleu.sentence_bleu(t, [ref]).score), "token_f1": token_f1(t, ref),
                     "emb": [round(float(v), 6) for v in e]})
    save()
    log(f"round trip {min(s + BATCH, n)}/{n}")
# 32-token truncations for the gate calibration, re-embedded
tr = [{"kind": "calib", "row": r["row"], "text": trunc32(r["text"])} for r in tx["calib"]]
tr += [{"kind": "real", "k": r["k"], "row": r["row"], "text": trunc32(r["text"])} for r in tx["real"]]
for r, e in zip(tr, embed([r["text"] for r in tr])):
    r["emb"] = [round(float(v), 6) for v in e]
with open(f"{out}/truncations_{dom}.jsonl", "w") as fh:
    for r in tr:
        fh.write(json.dumps(r) + "\n")
json.dump({"C0_path": C0_PATH, "n_rows": len(rows), "seconds": time.time() - t0}, open(f"{out}/meta_{dom}.json", "w"))
log(f"done: {len(rows)} inversions, {len(tr)} truncations")
'''


def sh(cmd: str) -> int:
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


def vec2text_ok() -> bool:
    return sh(f'{sys.executable} -c "import vec2text, sacrebleu; vec2text.load_pretrained_corrector(\'gtr-base\'); print(\'vec2text ok\')"') == 0


def main() -> None:
    t0 = time.time()
    sh(f"{sys.executable} -m pip install -q vec2text sacrebleu")
    if not vec2text_ok():  # vec2text pins an older transformers API; retry with a pinned stack
        # Kaggle's image is Python 3.13: transformers 4.44.2 (tokenizers<0.20) has no wheel there. Try stacks that
        # install on 3.13 and predate the meta-device from_pretrained change that breaks vec2text's nested loads.
        for stack in ('"transformers==4.48.3" "accelerate==1.3.0"', '"transformers==4.46.3" "accelerate==1.1.1"',
                      '"transformers==4.51.3" "accelerate==1.6.0"'):
            sh(f"{sys.executable} -m pip install -q {stack}")
            if vec2text_ok():
                break
        else:
            raise SystemExit("vec2text cannot load the gtr-base corrector")
    import torch
    ngpu = max(1, torch.cuda.device_count())
    inp = os.path.dirname(glob.glob("/kaggle/input/**/run_info.json", recursive=True)[0])
    print(f"gpus {ngpu}; inputs {inp}; run_info {open(f'{inp}/run_info.json').read().strip()}", flush=True)
    wpath = f"{OUT}/worker.py"
    open(wpath, "w").write(WORKER)
    procs = []
    for i, dom in enumerate(DOMS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % ngpu)}
        procs.append(subprocess.Popen([sys.executable, wpath, dom, inp, OUT], env=env))
        if ngpu == 1:
            procs[-1].wait()
    codes = [p.wait() for p in procs]
    os.remove(wpath)
    print(f"exit codes {codes}; {(time.time() - t0) / 60:.1f} min", flush=True)
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
