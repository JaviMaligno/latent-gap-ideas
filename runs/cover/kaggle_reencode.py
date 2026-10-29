"""Specified Round 6 census job. NOT executed locally or submitted to Kaggle.

Dataset layout: reencode_job.json, kaggle_reencode.py, model/{frozen files},
prepared/{steam,arxiv}/{corpus.parquet,emb_qwen3e06.npy}.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(p, value):
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--domain",choices=["steam","arxiv"],required=True)
    ap.add_argument("--limit",type=int,default=200)
    ap.add_argument("--minutes",type=float,default=35)
    args = ap.parse_args()
    if not 0<args.minutes<=40 or args.limit<1:
        ap.error("Use positive limit and <=40 minutes")
    start = time.monotonic()
    deadline = start+args.minutes*60-20
    manifest = json.loads((args.input/"reencode_job.json").read_text())
    for pkg,version in manifest["packages"].items():
        if importlib.metadata.version(pkg)!=version:
            raise ValueError(f"Version mismatch: {pkg}; install pinned {version}")
    if digest(Path(__file__)) != manifest["runner_sha256"]:
        raise ValueError("Runner differs from specified job")
    for rel, expected in manifest["input_sha256"].items():
        if time.monotonic()>deadline:
            raise TimeoutError("Input verification exceeded bounded invocation")
        if digest(args.input/rel) != expected:
            raise ValueError(f"Input changed: {rel}")
    import numpy as np
    import pandas as pd
    import onnxruntime as ort
    from tokenizers import Tokenizer
    d = args.domain
    texts = pd.read_parquet(args.input/f"prepared/{d}/corpus.parquet")["text"].tolist()
    first_text = {}
    for i, text in enumerate(texts):
        first_text.setdefault(text_hash(text), i)
    prepared = np.load(args.input/f"prepared/{d}/emb_qwen3e06.npy",mmap_mode="r")
    if prepared.shape!=(len(texts),1024):
        raise ValueError("Corpus row alignment/shape mismatch")
    out = args.output/d
    out.mkdir(parents=True,exist_ok=True)
    job_hash = digest(args.input/"reencode_job.json")
    identity = out/"identity.json"
    ident = dict(job_sha256=job_hash,n=len(texts),platform=os.uname().sysname,
                 machine=os.uname().machine,provider="CPUExecutionProvider")
    if identity.exists() and json.loads(identity.read_text())!=ident:
        raise ValueError("Output belongs to another job/platform")
    if not identity.exists():
        write_json(identity,ident)
    todo = [i for i in range(len(texts)) if not (out/f"{i:07d}.npz").exists()][:args.limit]
    if todo:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(args.input/"model/onnx/model.onnx"),opts,
                                    providers=["CPUExecutionProvider"])
        kv = [i.name for i in sess.get_inputs() if i.name.startswith("past_key_values")]
        tok = Tokenizer.from_file(str(args.input/"model/tokenizer.json"))
        for i in todo:
            if time.monotonic()>deadline:
                break
            ids = tok.encode(texts[i]).ids
            if not ids or ids[-1]!=151643:
                ids += [151643]
            if len(ids)>512:
                ids = ids[:511]+[151643]
            arr = np.asarray([ids],dtype=np.int64)
            n = len(ids)
            feed = dict(input_ids=arr, attention_mask=np.ones((1,n),dtype=np.int64),
                        position_ids=np.arange(n,dtype=np.int64)[None])
            empty = np.zeros((1,8,0,128),dtype=np.float32)
            feed.update({key:empty for key in kv})
            first = first_text[text_hash(texts[i])]
            if first < i:
                with np.load(out/f"{first:07d}.npz",allow_pickle=False) as previous:
                    v = previous["vector"].copy()
            else:
                h = sess.run(["last_hidden_state"],feed)[0][0,-1]
                v = (h/np.linalg.norm(h)).astype(np.float32)
            ref = np.asarray(prepared[i],dtype=np.float64)
            ref /= np.linalg.norm(ref)
            eu = v.astype(np.float64); eu /= np.linalg.norm(eu)
            metadata = dict(row=i,text_sha256=text_hash(texts[i]),tokens_sha256=text_hash(json.dumps(ids)),
                vector_sha256=hashlib.sha256(v.tobytes()).hexdigest(),norm=float(np.linalg.norm(v)),
                prepared_discrepancy=float(np.linalg.norm(eu-ref)),job_sha256=job_hash)
            if not np.all(np.isfinite(v)) or abs(metadata["norm"]-1)>1e-5:
                raise ValueError("Invalid encoding")
            p = out/f"{i:07d}.npz"
            tmp = out/f"{i:07d}.tmp"
            with tmp.open("wb") as f:
                np.savez(f,vector=v,metadata=json.dumps(metadata))
            tmp.replace(p)
            print(d,i,"seconds",round(time.monotonic()-start,1),flush=True)
    missing = sum(not (out/f"{i:07d}.npz").exists() for i in range(len(texts)))
    if missing:
        print("Incomplete; rows remaining:",missing,flush=True)
        return
    # Resumable census audit, in contiguous index order. Never declare completion
    # from filenames alone. Bind checkpoints to per-row content and text identity.
    audit_path = out/"audit_progress.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else dict(next_row=0,max_discrepancy=0.,chain="")
    for i in range(audit["next_row"],len(texts)):
        if time.monotonic()>deadline:
            write_json(audit_path,audit)
            print("Audit resume row",i,flush=True)
            return
        p = out/f"{i:07d}.npz"
        with np.load(p,allow_pickle=False) as z:
            v, m = z["vector"],json.loads(str(z["metadata"]))
        if (m["row"]!=i or m["text_sha256"]!=text_hash(texts[i]) or m["job_sha256"]!=job_hash
                or v.shape!=(1024,) or v.dtype!=np.float32 or not np.all(np.isfinite(v))
                or hashlib.sha256(v.tobytes()).hexdigest()!=m["vector_sha256"]):
            raise ValueError(f"Bad checkpoint {i}")
        first = first_text[text_hash(texts[i])]
        if first < i:
            with np.load(out/f"{first:07d}.npz",allow_pickle=False) as previous:
                if not np.array_equal(v,previous["vector"]):
                    raise ValueError("Duplicate text has inconsistent saved encoding")
        audit.update(next_row=i+1,max_discrepancy=max(audit["max_discrepancy"],m["prepared_discrepancy"]),
                     chain=text_hash(audit["chain"]+digest(p)))
        if (i+1)%200==0:
            write_json(audit_path,audit)
    write_json(audit_path,audit)
    write_json(out/"COMPLETE.json",dict(**ident,**audit,scope="all rows in original order; audited saved outputs"))
    print("COMPLETE",d,len(texts),audit["max_discrepancy"],flush=True)


if __name__ == "__main__":
    main()
