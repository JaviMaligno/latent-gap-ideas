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


def item_text(title, abstract):
    return f"Title: {title}. Abstract: {abstract[:ABSTRACT_CHARS]}"


def parse(raw, key):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if key == "title":
        return {"title": d.get("title", ""), "abstract": d.get("abstract", "")} if d.get("title") and d.get("abstract") else None
    return {"review": d["review"]} if d.get("review") else None


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
        for i, o in zip(todo, outs):
            p = parse(o, key)
            if p:
                res[i] = p
            else:
                nxt.append(i)
        todo = nxt
    blank = {"title": "", "abstract": ""} if key == "title" else {"review": ""}
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



# ---- diagnostic: raw outputs for 20 CTX prompts and 10 revise prompts, to see how the JSON breaks
t0 = time.time()
texts = [prompt("CTX", CTX[i]["ctx_block"], [], []) for i in range(20)]
fake = {"title": "A placeholder follow-up", "abstract": "We extend the method to a new setting and evaluate it."}
texts += [revise_prompt(CTX[i]["ctx_block"], fake, geo_feedback(0.75)) for i in range(10)]
raw = generate([SYSTEM + "\n\n" + t for t in texts])
rows = [{"i": n, "prompt_tokens": len(tok(SYSTEM + "\n\n" + t)["input_ids"]), "raw": r, "parsed": parse(r, "title") is not None,
         "new_tokens": len(tok(r)["input_ids"])} for n, (t, r) in enumerate(zip(texts, raw))]
json.dump(rows, open(f"{OUT}/raw_outputs.json", "w"), indent=1)
print("parsed", sum(r["parsed"] for r in rows), "of", len(rows), f"({time.time()-t0:.0f}s)", flush=True)
