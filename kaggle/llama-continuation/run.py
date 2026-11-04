"""OLMo-7B-0424-Instruct (AI2; Dolma pretraining data, cutoff Feb/Mar 2023; Llama 3.1 access requests were disabled on Kaggle) as second generator for continue_from_context Part B and iterative_critique.
Generated from scripts/ by kaggle/llama-continuation/build; prompt functions copied verbatim with inspect."""
N_IDEAS = 5
N_IDEAS_ITER = 1  # OLMo replication: one idea per context and arm (time limit), gpt-4o used 3
N_ROUNDS = 3
BAND = (0.574, 0.661)
ABSTRACT_CHARS = 1200

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


# ---- kernel body (generation protocol mirrors scripts/continue_from_context.py and scripts/iterative_critique.py)
import glob, json, os, re, subprocess, sys, time
# vLLM was dropped: on Kaggle's T4 image it upgrades torch to CUDA 13 and then fails at engine start-up.
# Plain transformers generation keeps the preinstalled torch.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.0"], check=True)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/kaggle/working"
INP = os.path.dirname(glob.glob("/kaggle/input/**/contexts.json", recursive=True)[0])
MODEL = "allenai/OLMo-7B-0424-Instruct-hf"
CTX = json.load(open(f"{INP}/contexts.json"))
T = json.load(open(f"{INP}/templates.json"))
SYSTEM, REVIEW = T["system"], T["review"]
SEED = 20261030
print("model", MODEL, "contexts", len(CTX), "gpus", torch.cuda.device_count(), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
llm.eval()
torch.manual_seed(SEED)
MAX_CTX, MAX_NEW, BATCH = 2048, 450, 12  # v6 ran out of memory on GPU 0 at batch 24
emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
emb.max_seq_length = 512


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
    return out


def enc(texts):
    return emb.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


t0 = time.time()
Cv = enc([c["ctx_text"] for c in CTX])
Fv = [enc(c["fups"]) for c in CTX]
print(f"embedded contexts and follow-ups ({time.time()-t0:.0f}s)", flush=True)

# ---- Part B arms (reused from the v6 run when its output is in the inputs: CTX took 2.3 h on two T4s)
partB = []
_done = f"{INP}/partB_done_v8.jsonl"  # v6 Part B (16 % parsed with the strict JSON parser) is not reused
if os.path.exists(_done):
    partB = [json.loads(l) for l in open(_done)]
    print(f"partB reused: {len(partB)} ideas", flush=True)
for arm in (() if partB else ("CTX",)):  # CTX+NB / CTX+MOM prompts (~2,650 tokens) exceed OLMo-0424's 2,048-token window
    jobs = [(i, k) for i in range(len(CTX)) for k in range(N_IDEAS)]
    texts = [prompt(arm, CTX[i]["ctx_block"], CTX[i]["nb"] if arm == "CTX+NB" else CTX[i]["mom"] if arm == "CTX+MOM" else [], []) for i, k in jobs]
    for (i, k), r in zip(jobs, chat_batch(texts, "title", SYSTEM)):
        partB.append({"ctx": i, "arm": arm, "k": k, **r})
    print(f"partB {arm} done ({time.time()-t0:.0f}s)", flush=True)
prev = {i: [] for i in range(len(CTX))}
for k in (range(0) if any(r["arm"] == "CTX+MEM" for r in partB) else range(N_IDEAS)):
    texts = [prompt("CTX+MEM", CTX[i]["ctx_block"], [], prev[i]) for i in range(len(CTX))]
    for i, r in enumerate(chat_batch(texts, "title", SYSTEM)):
        partB.append({"ctx": i, "arm": "CTX+MEM", "k": k, **r})
        prev[i].append(r["title"])
    print(f"partB MEM k={k} ({time.time()-t0:.0f}s)", flush=True)
with open(f"{OUT}/ideas_llama.jsonl", "w") as fh:
    for r in partB:
        fh.write(json.dumps(r) + "\n")

# ---- iterative critique
iters, state = [], {}
for r in range(N_ROUNDS):
    jobs = [(i, a, k) for i in range(len(CTX)) for a in ("GEO", "LLM", "ORACLE") for k in range(N_IDEAS_ITER)]
    texts = []
    for i, a, k in jobs:
        cb = CTX[i]["ctx_block"]
        texts.append(prompt("CTX", cb, [], []) if r == 0 else revise_prompt(cb, state[(i, a, k)]["rec"], state[(i, a, k)]["fb"]))
    outs = chat_batch(texts, "title", SYSTEM)
    vec = enc([item_text(o["title"], o["abstract"]) if o["abstract"] else "" for o in outs])
    for (i, a, k), o, v in zip(jobs, outs, vec):
        step = float(v @ Cv[i]) if o["abstract"] else float("nan")
        fit = float((Fv[i] @ v).max()) if o["abstract"] else float("nan")
        rec = {"ctx": i, "arm": a, "k": k, "round": r, **o, "feedback_in": state.get((i, a, k), {}).get("fb", ""), "step": step, "fit": fit}
        iters.append(rec)
        state[(i, a, k)] = {"rec": rec}
    if r < N_ROUNDS - 1:
        rev_jobs = [(i, a, k) for (i, a, k) in jobs if a == "LLM" and state[(i, a, k)]["rec"]["abstract"]]
        revs = chat_batch([REVIEW.format(ctx=CTX[i]["ctx_block"], title=state[(i, a, k)]["rec"]["title"], abstract=state[(i, a, k)]["rec"]["abstract"]) for i, a, k in rev_jobs], "review", "Answer with JSON.")
        rv = {j: x["review"] for j, x in zip(rev_jobs, revs)}
        for (i, a, k) in jobs:
            rec = state[(i, a, k)]["rec"]
            if not rec["abstract"]:
                state[(i, a, k)]["fb"] = ""
            elif a == "GEO":
                state[(i, a, k)]["fb"] = geo_feedback(rec["step"])
            elif a == "ORACLE":
                state[(i, a, k)]["fb"] = (f"Your proposal's similarity to the closest paper actually published as a follow-up is "
                                          f"{rec['fit']:.2f} (embedding cosine). Try to get closer.")
            else:
                state[(i, a, k)]["fb"] = rv.get((i, a, k), "")
    print(f"iter round {r} done ({time.time()-t0:.0f}s)", flush=True)
with open(f"{OUT}/iter_llama.jsonl", "w") as fh:
    for r in iters:
        fh.write(json.dumps(r) + "\n")

with open(f"{OUT}/raw_generations.jsonl", "w") as fh:
    for r in RAW:
        fh.write(json.dumps(r) + "\n")

# ---- embeddings of every output and of contexts/follow-ups, in both spaces (same settings as the corpus)
texts = ([c["ctx_text"] for c in CTX] + [t for c in CTX for t in c["fups"]]
         + [item_text(r["title"], r["abstract"]) for r in partB if r["abstract"]]
         + [item_text(r["title"], r["abstract"]) for r in iters if r["abstract"]])
np.save(f"{OUT}/emb_qwen3e06.npy", enc(texts).astype(np.float16))
del llm
torch.cuda.empty_cache()
bge = SentenceTransformer("BAAI/bge-m3", device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
bge.max_seq_length = 512
np.save(f"{OUT}/emb_bgem3.npy", bge.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16))
print(f"all done ({time.time()-t0:.0f}s), {len(texts)} texts embedded", flush=True)
