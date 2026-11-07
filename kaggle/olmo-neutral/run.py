"""OLMo NEUTRAL revision arm (docs/prereg/strengthen_olmo_neutral.md).
Generation settings, prompt functions, parser and generate() are copied verbatim from kaggle/llama-continuation/run.py
(the only change: generate() prints progress after every batch). Round 0 = OLMo's GEO round-0 idea (geo_r0.jsonl,
from llama-continuation-v1/iter_llama.jsonl); rounds 1-2 revise with the content-free NEUTRAL feedback used for gpt-4o."""
N_ROUNDS = 3
BAND = (0.574, 0.661)
ABSTRACT_CHARS = 1200
NEUTRAL_FB = "Revise your proposal: write an improved version."  # scripts/iterative_critique.py NEUTRAL_FB

def prompt(arm: str, ctx_block: str, extra: list[str], previous: list[str]) -> str:
    p = ("Here is a paper:\n\n" + ctx_block + "\n\nPropose the paper that should come next, building on this one: "
         "an original follow-up a research group could write. Give its title and an arXiv-style abstract of "
         "120-180 words.")
    if arm == "CTX+NB":
        p += "\n\nThe closest existing papers to it are:\n\n" + "\n\n".join(extra)
    if arm == "CTX+MOM":
        p += "\n\nThe most recent papers in the same area, showing where it is moving, are:\n\n" + "\n\n".join(extra)
    if arm == "CTX+MEM" and previous:
        p += "\n\nYou already proposed these; propose something different from all of them:\n" + "\n".join(f"- {t}" for t in previous)
    return p


def geo_feedback(s: float) -> str:
    head = f"Your proposal's similarity to the original paper is {s:.2f} (embedding cosine). Real follow-up papers typically sit between {BAND[0]:.2f} and {BAND[1]:.2f}. "
    if s > BAND[1]:
        return head + "You are too close: build on the paper but take a bigger step - a new question, method or setting, not a variation of the same one."
    if s < BAND[0]:
        return head + "You are too far: keep a clearer link to the original paper."
    return head + "You are within the range: keep the step, sharpen the contribution."


def revise_prompt(ctx_block: str, prev: dict, feedback: str) -> str:
    return (prompt("CTX", ctx_block, [], []) + f"\n\nYour previous proposal was:\nTitle: {prev['title']}\nAbstract: {prev['abstract']}\n\n"
            f"Feedback: {feedback}\n\nWrite a revised proposal: a full new title and abstract.")



# ---- kernel body
import glob, json, os, re, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.0"], check=True)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

t0 = time.time()
OUT = "/kaggle/working"
INP = os.path.dirname(glob.glob("/kaggle/input/**/contexts.json", recursive=True)[0])
GEO0 = glob.glob("/kaggle/input/**/geo_r0.jsonl", recursive=True)[0]
MODEL = "allenai/OLMo-7B-0424-Instruct-hf"
CTX = json.load(open(f"{INP}/contexts.json"))
T = json.load(open(f"{INP}/templates.json"))
SYSTEM = T["system"]
SEED = 20261105
G0 = {r["ctx"]: r for r in map(json.loads, open(GEO0))}
assert sorted(G0) == list(range(len(CTX))) == list(range(150)), "geo_r0 must cover the 150 contexts"
print("model", MODEL, "contexts", len(CTX), "gpus", torch.cuda.device_count(), "geo_r0", len(G0), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
llm.eval()
torch.manual_seed(SEED)
MAX_CTX, MAX_NEW, BATCH = 2048, 450, 12
print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)


def item_text(title, abstract):
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def parse(raw, key):
    """OLMo-0424 ignores the JSON instruction and writes 'Title: ... Abstract: ...' (diagnostic kernel, 28/30 such).
    Candidates: any JSON object with the keys, and the prose layout; the longest abstract wins (an appended JSON
    block is usually a shortened copy of the prose)."""
    if key == "title":
        cands = [c for c in (_parse_json(raw, key), _parse_prose(raw)) if c]
        return max(cands, key=lambda c: len(c["abstract"])) if cands else None
    return _parse_json(raw, key) or _parse_prose_review(raw)


def _parse_json(raw, key):
    for m in re.finditer(r"\{[^{}]*\}", raw, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        d = {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else {}
        if key == "title" and d.get("title") and d.get("abstract"):
            return {"title": str(d["title"]).strip(), "abstract": " ".join(str(d["abstract"]).split())}
        if key == "review" and d.get("review"):
            return {"review": str(d["review"]).strip()}
    return None


def _parse_prose(raw):
    t = raw.replace("**", "")
    if True:
        m = re.search(r"title\s*[:\-]\s*(.+?)\s*\n+\s*abstract[^:\n]*:\s*(.+)", t, re.S | re.I)
        if m:
            ab = re.split(r"\n\s*(?:json|\{)", m.group(2), maxsplit=1, flags=re.I)[0]
            title, ab = m.group(1).strip().strip('"'), " ".join(ab.split()).strip('"')
            if title and len(ab) > 40:
                return {"title": title, "abstract": ab}
        return None


def _parse_prose_review(raw):
    t = raw.replace("**", "")
    m = re.search(r"review\s*[:\-]\s*(.+)", t, re.S | re.I)
    txt = " ".join((m.group(1) if m else t).split())
    return {"review": txt} if len(txt) > 20 else None


RAW = []  # every raw generation, saved at the end


def chat_batch(texts, key, system):
    """One vLLM pass per attempt; failed parses are retried up to twice, as in the gateway runs."""
    res = [None] * len(texts)
    todo = list(range(len(texts)))
    for _ in range(3):
        if not todo:
            break
        # the system text is prepended to the user turn: the OLMo chat template has no system role
        outs = generate([system + "\n\n" + texts[i] for i in todo])
        nxt = []
        RAW.extend({"key": key, "raw": o} for o in outs)
        for i, o in zip(todo, outs):
            p = parse(o, key)
            if p:
                res[i] = p
            else:
                nxt.append(i)
        todo = nxt
    blank = {"title": "", "abstract": ""} if key == "title" else {"review": ""}
    print(f"  parsed {sum(r is not None for r in res)}/{len(res)} ({key})", flush=True)
    return [r or blank for r in res]


def generate(prompts):
    """Sampled generation (temperature 1, as in the gateway runs), batched by prompt length."""
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    ids = [list(tok(c, add_special_tokens=False)["input_ids"]) for c in chats]
    ids = [x[-(MAX_CTX - MAX_NEW):] for x in ids]
    out = [""] * len(prompts)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            g = llm.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=MAX_NEW,
                             pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, batch["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)
        del g, batch
        torch.cuda.empty_cache()
        print(f"    gen batch {b // BATCH + 1}/{-(-len(order) // BATCH)} ({time.time()-t0:.0f}s)", flush=True)
    return out




def dump(rows):
    with open(f"{OUT}/iter_olmo_neutral.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    with open(f"{OUT}/raw_generations_neutral.jsonl", "w") as fh:
        for x in RAW:
            fh.write(json.dumps(x) + "\n")


# ---- NEUTRAL arm: round 0 copied from GEO, rounds 1-2 revised with the content-free feedback
iters, state = [], {}
for i in range(len(CTX)):
    g = G0[i]
    rec = {"ctx": i, "arm": "NEUTRAL", "k": 0, "round": 0, "title": g["title"], "abstract": g["abstract"],
           "feedback_in": "", "r0_from": "GEO"}
    iters.append(rec)
    state[i] = {"rec": rec, "fb": NEUTRAL_FB if rec["abstract"] else ""}
for r in range(1, N_ROUNDS):
    texts = [revise_prompt(CTX[i]["ctx_block"], state[i]["rec"], state[i]["fb"]) for i in range(len(CTX))]
    outs = chat_batch(texts, "title", SYSTEM)
    for i, o in enumerate(outs):
        rec = {"ctx": i, "arm": "NEUTRAL", "k": 0, "round": r, **o, "feedback_in": state[i]["fb"]}
        iters.append(rec)
        state[i] = {"rec": rec, "fb": NEUTRAL_FB if o["abstract"] else ""}  # empty record -> empty feedback, as the critique run
    dump(iters)
    print(f"neutral round {r} done ({time.time()-t0:.0f}s)", flush=True)

# ---- embeddings (critique run's settings): contexts and every NEUTRAL row with an abstract, file order
del llm
torch.cuda.empty_cache()
rows = [x for x in iters if x["abstract"]]
texts = [item_text(x["title"], x["abstract"]) for x in rows]
ctx_texts = [c["ctx_text"] for c in CTX]
for space, name in (("qwen3e06", "Qwen/Qwen3-Embedding-0.6B"), ("bgem3", "BAAI/bge-m3")):
    m = SentenceTransformer(name, device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    np.save(f"{OUT}/neutral_emb_{space}.npy", m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16))
    np.save(f"{OUT}/ctx_emb_{space}.npy", m.encode(ctx_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16))
    print(f"embedded {space}: {len(texts)} neutral rows + {len(ctx_texts)} contexts ({time.time()-t0:.0f}s)", flush=True)
    del m
    torch.cuda.empty_cache()
print(f"all done ({time.time()-t0:.0f}s)", flush=True)
