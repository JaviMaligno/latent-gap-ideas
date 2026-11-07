"""OLMo-7B-0424-Instruct critique arms + NEUTRAL on the eligible contexts of one wave/part
(docs/prereg/strengthen_olmo_direction.md, stage 3). Derived from kaggle/llama-continuation/run.py: same model,
sampling, parser, prompts and embedders; no Part B; 3 ideas per context and arm; arms GEO, LLM (reviewer = OLMo),
ORACLE, NEUTRAL; NEUTRAL's round 0 is GEO's round 0 (r0_from GEO). Inputs built by scripts/olmo_direction_kaggle.py,
which checks that the prompt functions below are verbatim copies of scripts/continue_from_context.py and
scripts/iterative_critique.py. Progress is printed in blocks; outputs are rewritten after every round."""
N_IDEAS_ITER = 3
N_ROUNDS = 3
BAND = (0.574, 0.661)
ABSTRACT_CHARS = 1200
NEUTRAL_FB = "Revise your proposal: write an improved version."  # docs/prereg/neutral_revision.md
GEN_ARMS = ("GEO", "LLM", "ORACLE", "NEUTRAL")


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

OUT = "/kaggle/working"
INP = os.path.dirname(glob.glob("/kaggle/input/**/contexts.json", recursive=True)[0])
MODEL = "allenai/OLMo-7B-0424-Instruct-hf"
CTX = json.load(open(f"{INP}/contexts.json"))
T = json.load(open(f"{INP}/templates.json"))
INFO = json.load(open(f"{INP}/run_info.json"))
SYSTEM, REVIEW = T["system"], T["review"]
assert tuple(T["band"]) == BAND
SEED = INFO["seed"]
json.dump(INFO, open(f"{OUT}/run_info.json", "w"))
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)


log(f"model {MODEL} wave {INFO['wave']} part {INFO['part']} contexts {len(CTX)} gpus {torch.cuda.device_count()} seed {SEED}")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
llm.eval()
torch.manual_seed(SEED)
MAX_CTX, MAX_NEW, BATCH = 2048, 450, 12
emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
emb.max_seq_length = 512


def item_text(title, abstract):
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def parse(raw, key):
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


RAW = []


def chat_batch(texts, key, system, label):
    res = [None] * len(texts)
    todo = list(range(len(texts)))
    for attempt in range(3):
        if not todo:
            break
        outs = generate([system + "\n\n" + texts[i] for i in todo], f"{label} try{attempt}")
        nxt = []
        RAW.extend({"key": key, "label": label, "raw": o} for o in outs)
        for i, o in zip(todo, outs):
            p = parse(o, key)
            if p:
                res[i] = p
            else:
                nxt.append(i)
        todo = nxt
    blank = {"title": "", "abstract": ""} if key == "title" else {"review": ""}
    log(f"  {label}: parsed {sum(r is not None for r in res)}/{len(res)} ({key})")
    return [r or blank for r in res]


def generate(prompts, label):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    ids = [list(tok(c, add_special_tokens=False)["input_ids"]) for c in chats]
    ids = [x[-(MAX_CTX - MAX_NEW):] for x in ids]
    out = [""] * len(prompts)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    nb = -(-len(order) // BATCH)
    t1 = time.time()
    for bi, b in enumerate(range(0, len(order), BATCH)):
        idx = order[b:b + BATCH]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            g = llm.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=MAX_NEW,
                             pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, batch["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)
        del g, batch
        torch.cuda.empty_cache()
        if (bi + 1) % 10 == 0 or bi + 1 == nb:
            el = time.time() - t1
            log(f"  {label}: batch {bi + 1}/{nb} ({el:.0f}s, ~{el / (bi + 1) * (nb - bi - 1):.0f}s left in this pass)")
    return out


def enc(texts):
    return emb.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


Cv = enc([c["ctx_text"] for c in CTX])
Fv = [enc(c["fups"]) for c in CTX]
log(f"embedded {len(CTX)} contexts and {sum(len(c['fups']) for c in CTX)} follow-ups")


def save_iters(iters):
    with open(f"{OUT}/iter_olmo.jsonl", "w") as fh:
        for r in iters:
            fh.write(json.dumps(r) + "\n")
    with open(f"{OUT}/raw_generations.jsonl", "w") as fh:
        for r in RAW:
            fh.write(json.dumps(r) + "\n")


iters, state = [], {}
for r in range(N_ROUNDS):
    arms = ("GEO", "LLM", "ORACLE") if r == 0 else GEN_ARMS
    jobs = [(i, a, k) for i in range(len(CTX)) for a in arms for k in range(N_IDEAS_ITER)]
    texts = []
    for i, a, k in jobs:
        cb = CTX[i]["ctx_block"]
        texts.append(prompt("CTX", cb, [], []) if r == 0 else revise_prompt(cb, state[(i, a, k)]["rec"], state[(i, a, k)]["fb"]))
    log(f"round {r}: {len(jobs)} generations")
    outs = chat_batch(texts, "title", SYSTEM, f"round {r}")
    vec = enc([item_text(o["title"], o["abstract"]) if o["abstract"] else "" for o in outs])
    for (i, a, k), o, v in zip(jobs, outs, vec):
        step = float(v @ Cv[i]) if o["abstract"] else float("nan")
        fit = float((Fv[i] @ v).max()) if o["abstract"] else float("nan")
        rec = {"ctx": i, "pos": CTX[i]["pos"], "arm": a, "k": k, "round": r, **o,
               "feedback_in": state.get((i, a, k), {}).get("fb", ""), "step": step, "fit": fit}
        iters.append(rec)
        state[(i, a, k)] = {"rec": rec}
    if r == 0:  # NEUTRAL starts from GEO's round-0 ideas (neutral_revision.md)
        for i in range(len(CTX)):
            for k in range(N_IDEAS_ITER):
                rec = {**state[(i, "GEO", k)]["rec"], "arm": "NEUTRAL", "r0_from": "GEO"}
                iters.append(rec)
                state[(i, "NEUTRAL", k)] = {"rec": rec}
    if r < N_ROUNDS - 1:
        cur = [(i, a, k) for i in range(len(CTX)) for a in GEN_ARMS for k in range(N_IDEAS_ITER)]
        rev_jobs = [(i, a, k) for (i, a, k) in cur if a == "LLM" and state[(i, a, k)]["rec"]["abstract"]]
        log(f"round {r}: {len(rev_jobs)} reviews")
        revs = chat_batch([REVIEW.format(ctx=CTX[i]["ctx_block"], title=state[(i, a, k)]["rec"]["title"],
                                         abstract=state[(i, a, k)]["rec"]["abstract"]) for i, a, k in rev_jobs],
                          "review", "Answer with JSON.", f"round {r} reviews")
        rv = {j: x["review"] for j, x in zip(rev_jobs, revs)}
        for (i, a, k) in cur:
            rec = state[(i, a, k)]["rec"]
            if not rec["abstract"]:
                state[(i, a, k)]["fb"] = ""
            elif a == "GEO":
                state[(i, a, k)]["fb"] = geo_feedback(rec["step"])
            elif a == "ORACLE":
                state[(i, a, k)]["fb"] = (f"Your proposal's similarity to the closest paper actually published as a follow-up is "
                                          f"{rec['fit']:.2f} (embedding cosine). Try to get closer.")
            elif a == "NEUTRAL":
                state[(i, a, k)]["fb"] = NEUTRAL_FB
            else:
                state[(i, a, k)]["fb"] = rv.get((i, a, k), "")
    save_iters(iters)
    log(f"round {r} done and saved ({len(iters)} rows)")

# ---- embeddings in both spaces (same settings as the corpus): contexts, follow-ups, ideas with an abstract
ideas = [item_text(r["title"], r["abstract"]) for r in iters if r["abstract"]]
fups = [t for c in CTX for t in c["fups"]]
np.save(f"{OUT}/fup_owner.npy", np.array([i for i, c in enumerate(CTX) for _ in c["fups"]]))
for name, model in (("qwen3e06", None), ("bgem3", "BAAI/bge-m3")):
    if model:
        del emb
        if "llm" in globals():
            del llm
        torch.cuda.empty_cache()
        emb = SentenceTransformer(model, device="cuda:0", model_kwargs={"torch_dtype": torch.float16})
        emb.max_seq_length = 512
    for kind, texts in (("ctx", [c["ctx_text"] for c in CTX]), ("fup", fups), ("iter", ideas)):
        np.save(f"{OUT}/{kind}_{name}.npy", enc(texts).astype(np.float16))
    log(f"embedded {len(CTX)} contexts, {len(fups)} follow-ups, {len(ideas)} ideas in {name}")
log("all done")
