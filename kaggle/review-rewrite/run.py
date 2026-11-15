"""OLMo-7B-0424-Instruct rewrites for OLMO-C (docs/prereg/review_tangent_confounds.md, Part B, Section 11):
RW-F (consensus continuations + seeded sample of non-continuation follow-ups, prompt R), NA (context paraphrases,
prompt PA) and NB (nearby later papers, prompt R); prompts are built locally by scripts/review_rewrite_generate.py
(kaggle-olmo-inputs) and read from items.json. Model, sampling, parser and embedders as kaggle/olmo-direction/run.py
(temperature 1.0, top_p 1.0, max_new_tokens 450 as the OLMo ideas; JSON-or-prose parser); one retry on a parse
failure, then the item is dropped (counted). The LLM is split over both T4s (device_map="auto"); a batch that runs out
of GPU memory is halved. Progress is printed in blocks (no tqdm). Outputs are rewritten after each pass:
rewrites.jsonl (one row per item; empty title/abstract = dropped), raw_generations.jsonl, rw_{qwen3e06,bgem3}.npy
(rows = items with an abstract, in rewrites.jsonl order), run_info.json.
"""
import glob, json, os, re, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.0"], check=True)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/kaggle/working"
INP = os.path.dirname(glob.glob("/kaggle/input/**/items.json", recursive=True)[0])
MODEL = "allenai/OLMo-7B-0424-Instruct-hf"
ITEMS = json.load(open(f"{INP}/items.json"))
T = json.load(open(f"{INP}/templates.json"))
INFO = json.load(open(f"{INP}/run_info.json"))
SYSTEM, SEED, ATTEMPTS = T["system"], INFO["seed"], INFO["attempts"]
ABSTRACT_CHARS = 1200
MAX_CTX, MAX_NEW, BATCH = 2048, 450, 12
json.dump(INFO, open(f"{OUT}/run_info.json", "w"))
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)


log(f"model {MODEL} items {len(ITEMS)} gpus {torch.cuda.device_count()} seed {SEED}")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
llm.eval()
torch.manual_seed(SEED)


def item_text(title, abstract):
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def parse(raw):  # = kaggle/olmo-direction/run.py parse(raw, "title")
    cands = [c for c in (_parse_json(raw), _parse_prose(raw)) if c]
    return max(cands, key=lambda c: len(c["abstract"])) if cands else None


def _parse_json(raw):
    for m in re.finditer(r"\{[^{}]*\}", raw, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        d = {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
        if d.get("title") and d.get("abstract"):
            return {"title": str(d["title"]).strip(), "abstract": " ".join(str(d["abstract"]).split())}
    return None


def _parse_prose(raw):
    t = raw.replace("**", "")
    m = re.search(r"title\s*[:\-]\s*(.+?)\s*\n+\s*abstract[^:\n]*:\s*(.+)", t, re.S | re.I)
    if m:
        ab = re.split(r"\n\s*(?:json|\{)", m.group(2), maxsplit=1, flags=re.I)[0]
        title, ab = m.group(1).strip().strip('"'), " ".join(ab.split()).strip('"')
        if title and len(ab) > 40:
            return {"title": title, "abstract": ab}
    return None


RAW = []


def generate(prompts, label):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    ids = [list(tok(c, add_special_tokens=False)["input_ids"]) for c in chats]
    ids = [x[-(MAX_CTX - MAX_NEW):] for x in ids]
    out = [""] * len(prompts)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    nb = -(-len(order) // BATCH)
    t1 = time.time()

    def attempt(idx):
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            g = llm.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=MAX_NEW,
                             pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, batch["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)

    def run(idx):  # a batch that does not fit in GPU memory is split in halves
        oom = False
        try:
            attempt(idx)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            oom = True
        torch.cuda.empty_cache()
        if oom:
            log(f"  {label}: OOM on a batch of {len(idx)}, splitting")
            h = len(idx) // 2
            run(idx[:h])
            run(idx[h:])

    for bi, b in enumerate(range(0, len(order), BATCH)):
        run(order[b:b + BATCH])
        if (bi + 1) % 10 == 0 or bi + 1 == nb:
            el = time.time() - t1
            log(f"  {label}: batch {bi + 1}/{nb} ({el:.0f}s, ~{el / (bi + 1) * (nb - bi - 1):.0f}s left in this pass)")
    return out


def save(res, att):
    with open(f"{OUT}/rewrites.jsonl", "w") as fh:
        for it, r, a in zip(ITEMS, res, att):
            rec = {k: v for k, v in it.items() if k != "prompt"} | (r or {"title": "", "abstract": ""}) | {"attempts": a}
            fh.write(json.dumps(rec) + "\n")
    with open(f"{OUT}/raw_generations.jsonl", "w") as fh:
        for r in RAW:
            fh.write(json.dumps(r) + "\n")


res, att = [None] * len(ITEMS), [0] * len(ITEMS)
todo = list(range(len(ITEMS)))
for a in range(ATTEMPTS):
    if not todo:
        break
    log(f"pass {a}: {len(todo)} generations")
    outs = generate([SYSTEM + "\n\n" + ITEMS[i]["prompt"] for i in todo], f"pass {a}")
    nxt = []
    for i, o in zip(todo, outs):
        RAW.append({"id": ITEMS[i]["id"], "attempt": a, "raw": o})
        att[i] = a + 1
        p = parse(o)
        if p:
            res[i] = p
        else:
            nxt.append(i)
    todo = nxt
    save(res, att)
    log(f"pass {a} done: parsed {sum(r is not None for r in res)}/{len(res)}; saved")
log(f"dropped after {ATTEMPTS} attempts: {len(todo)} " + json.dumps({s: sum(ITEMS[i]['set'] == s for i in todo) for s in ('RW-F', 'NA', 'NB')}))

# ---- embeddings in both spaces (settings of kaggle/olmo-direction/run.py)
texts = [item_text(r["title"], r["abstract"]) for r in res if r]
del llm
torch.cuda.empty_cache()
for name, model in (("qwen3e06", "Qwen/Qwen3-Embedding-0.6B"), ("bgem3", "BAAI/bge-m3")):
    emb = SentenceTransformer(model, device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
    emb.max_seq_length = 512
    parts = []
    for b in range(0, len(texts), 512):
        parts.append(emb.encode(texts[b:b + 512], batch_size=64, normalize_embeddings=True, convert_to_numpy=True))
        log(f"  {name}: embedded {min(b + 512, len(texts))}/{len(texts)}")
    np.save(f"{OUT}/rw_{name}.npy", np.concatenate(parts).astype(np.float16))
    del emb
    torch.cuda.empty_cache()
log("all done")
