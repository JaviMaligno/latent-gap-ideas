"""Round 5 C2. Offline, single process; all outputs are docs/prereg/cover_*.csv.

Order: --plan; --embed --limit 8 (benchmark); --embed --limit 240 (repeat);
--fit; --freeze; --measure; --check. No network or language-model calls.
Embedding rows are append-only checkpoints. Fit/freeze/results refuse overwrite.
See docs/bounds6_cover.md for the distinction between support and calibration.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import hashlib
import inspect
import json
import re
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from tokenizers import Tokenizer

from latent_gap.embedder import LocalQwen, MODEL_DIR
from support_probe import cone_certificate, unit

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
OUT = ROOT / "docs/prereg"
FS, SR = DATA / "runs/fluent-search", DATA / "runs/seed-rewrite"
DOMAINS = ("steam", "arxiv")
WRITERS = ("claude-sonnet-5-5", "gpt-6-luna")
OPS = ("F", "S", "T", "FS", "FT", "ST")
SEED = 202610285
DELTA = .05
COUNTS = {"train": 16, "cal": 64, "check": 32}
LN2, LN10 = np.log(2.), np.log(10.)
DEADLINE = float("inf")
STOP = set("abstract title about genres tags which their there these those between through without however because therefore different proposed presents based using results method methods approach problem paper study".split())


def check_deadline():
    if time.monotonic() > DEADLINE:
        raise TimeoutError("Stage deadline exceeded; no inference should exceed 40 minutes")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            check_deadline()
            h.update(b)
    return h.hexdigest()


def textsha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def path(name):
    return OUT / f"cover_{name}.csv"


def save(name, rows):
    p = path(name)
    if p.exists():
        raise FileExistsError(p)
    pd.DataFrame(rows).to_csv(p, index=False, float_format="%.17g")
    return p


def load(name):
    return pd.read_csv(path(name), float_precision="round_trip", keep_default_na=False)


def fingerprint():
    files = [(str(p.relative_to(MODEL_DIR)), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(MODEL_DIR.rglob("*")) if p.is_file()]
    return textsha(inspect.getsource(LocalQwen) + str(files))[:16]


def corpus(domain, vectors=True, future=False):
    base = DATA / "prepared" / domain
    meta = pd.read_parquet(base / "corpus.parquet", columns=["text", "split", "adult"])
    keep = (~meta.adult & (meta.split == ("future" if future else "past"))).to_numpy()
    texts = meta.loc[keep, "text"].tolist()
    if not vectors:
        return texts
    x = np.array(np.load(base / "emb_qwen3e06.npy", mmap_mode="r")[keep], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return texts, x


def refs_jobs():
    refs = {(r["domain"], r["arm"], r["k"]): r for r in
            map(json.loads, (FS / "refs.jsonl").read_text().splitlines())}
    jobs = [j for j in json.loads((FS / "inputs/fluent_inputs.json").read_text()) if j["k"] > 0]
    return refs, jobs


def parts(domain, text):
    sep = ". About: " if domain == "steam" else ". Abstract: "
    if sep not in text:
        return None
    header, body = text.split(sep, 1)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return header, body, sentences


def term(sentence):
    words = {w for w in re.findall(r"\b[A-Za-z][A-Za-z-]{5,}\b", sentence)
             if w.lower() not in STOP}
    return sorted(words, key=lambda w: (-len(w), w))[0] if words else None


def valid(domain, text, tok):
    p = parts(domain, text)
    if p is None:
        return False
    header, body, sentences = p
    if len(body) > 1200 or not 70 <= len(body.split()) <= 170 or len(sentences) < 3:
        return False
    if len(tok.encode(text).ids) > 511 or " ".join(sentences) != body:
        return False
    if domain == "steam":
        if not header.startswith("Genres: ") or ". Tags: " not in header:
            return False
        g, t = header[8:].split(". Tags: ", 1)
        return len(g.split(",")) <= 7 and len(t.split(",")) <= 20
    return header.startswith("Title: ") and len(header[7:].split()) <= 18


def compose(domain, source, donor, op):
    """F: whole header. S: middle sentence. T: <=2 middle-sentence terms.

    Order is F,S,T; first and last source sentences always survive verbatim.
    No truncation, invented vocabulary, or recursive reuse of generated texts.
    """
    h, _, ss = parts(domain, source)
    dh, _, ds = parts(domain, donor)
    ss = list(ss)
    middle = len(ss) // 2
    if "F" in op:
        h = dh
    if "S" in op:
        ss[middle] = ds[len(ds) // 2]
    if "T" in op:
        old = term(ss[middle])
        # With S, use the source as the substitution donor to avoid a no-op.
        replacement = term(parts(domain, source)[2][middle] if "S" in op else ds[len(ds) // 2])
        if old is None or replacement is None or old == replacement:
            return None
        ss[middle] = re.sub(r"\b" + re.escape(old) + r"\b", replacement, ss[middle], count=2)
    sep = ". About: " if domain == "steam" else ". Abstract: "
    text = h + sep + " ".join(ss)
    return text if text != source else None


def eligible(domain, texts, tok, hidden):
    return [i for i, s in enumerate(texts) if i not in hidden and valid(domain, s, tok)
            and term(parts(domain, s)[2][len(parts(domain, s)[2]) // 2])]


def plan():
    if path("manifest").exists():
        raise RuntimeError("Already frozen")
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    refs, _ = refs_jobs()
    rows, banks = [], []
    for di, domain in enumerate(DOMAINS):
        texts = corpus(domain, vectors=False)
        # A single target-independent bank for each domain: remove ALL hidden REAL sources.
        hidden = {i for (d, a, _), r in refs.items() if d == domain and a == "REAL" for i in r["hidden_pos"]}
        bank = eligible(domain, texts, tok, hidden)
        banks.extend(dict(domain=domain, position=i, text_sha256=textsha(texts[i])) for i in bank)
        for oi, op in enumerate((*OPS, "FST")):
            phases = COUNTS if op in OPS else {"depth3": 32}
            for pi, (phase, n) in enumerate(phases.items()):
                # Separate deterministic RNG streams represent independent iid pair draws.
                rng = np.random.default_rng(np.random.SeedSequence([SEED, di, oi, pi]))
                count = 0
                while count < n:
                    a, b = map(int, rng.choice(bank, 2, replace=True))
                    if a == b:
                        continue
                    s = compose(domain, texts[a], texts[b], op)
                    if s is None or not valid(domain, s, tok):
                        continue
                    rows.append(dict(domain=domain, op=op, phase=phase, draw=count,
                                     source=a, donor=b, text=s, text_sha256=textsha(s),
                                     tokens=len(tok.encode(s).ids)))
                    count += 1
        # Two independent original texts for ONNX-versus-prepared parity checks.
        for j, i in enumerate(np.random.default_rng(SEED + di).choice(bank, 2, replace=False)):
            rows.append(dict(domain=domain, op="original", phase="parity", draw=j,
                             source=int(i), donor=int(i), text=texts[i], text_sha256=textsha(texts[i]),
                             tokens=len(tok.encode(texts[i]).ids)))
        print(domain, "eligible", len(bank), "rows", sum(r["domain"] == domain for r in rows), flush=True)
    save("bank", banks)
    # Interleave domain/operation/phase, so a short benchmark is representative.
    rows.sort(key=lambda r: (r["draw"], r["phase"], r["domain"], r["op"]))
    save("plan", rows)
    save("protocol", [dict(key=k, value=v, utc=now()) for k, v in {
        "seed": SEED, "encoder": fingerprint(), "threads": 4,
        "counts": json.dumps(COUNTS), "delta_family": DELTA,
        "law": "For each D/op, iid uniform ordered distinct eligible past pairs conditioned on valid, changed output; streams independent across train/cal/check. FST is outside L.",
        "language": "F,S,T,FS,FT,ST on all eligible ordered pairs. No identity cell, no recursion. All hidden REAL source positions excluded globally. Original rows are parity diagnostics only.",
        "model": "q=unit((1-lambda)*p_source+lambda*p_donor); prepared vectors are fixed unit reference anchors, not assumed equal to local encodings. lambda in 0,.1,..,1 minimizes train mean squared residual, separately D/op; cal maximum residual.",
        "rewrite_extension": "m8(x)=unit(mean of 8 nearest visible corpus rows). epsilon=max(grammar calibration maxima, all measured tune-v2 REAL11..20 m8 residuals). Tested, not assumed to follow from grammar.",
        "tail": "q95 empirical body plus scaled Beta(a,b) exceedances ending at 1; MLE on past only; future rows untouched until measure; REAL masks per refs.",
        "tail_criterion": "Each domain/arm: median future observed/predicted exceedance ratios at fixed past-model q99,q999 in [.5,2]; absolute median maximum bias <=.03 for REAL/GAP, <=.01 for FAR/RANDOM; 95% maximum interval coverage >=.80.",
        "search_criterion": "Each D/writer/arm: gate coverage>=.9, absolute paired median prediction error<=.03, and 95% plug-in maximum interval coverage>=.8. REAL-only fit of median log10 N. No refitting on test.",
        "disclosure": "The test results and round-4 summaries are already known; out-of-set reuse, not prospective validation. Only tuning records enter fitting. No new coherence judgments.",
    }.items()])


def embeddings():
    if not path("embeddings").exists():
        return {}
    out = {}
    with path("embeddings").open(newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("vector"):
                raise ValueError("Incomplete checkpoint row; inspect before resuming")
            if r["encoder"] != fingerprint():
                raise ValueError("Encoder changed")
            v = np.asarray(json.loads(r["vector"]), dtype=np.float64)
            assert v.shape == (1024,) and abs(np.linalg.norm(v)-1) < 1e-5
            out[r["text_sha256"]] = unit(v)
    return out


def embed(limit, minutes):
    if path("manifest").exists():
        raise RuntimeError("Already frozen")
    df = load("plan").drop_duplicates("text_sha256")
    old = embeddings()
    todo = [r for r in df.itertuples() if r.text_sha256 not in old][:limit]
    if not todo:
        print("All requested embeddings already checkpointed")
        return
    timings = pd.read_csv(path("embeddings")).seconds.tolist() if old else []
    estimate = 20 + len(todo) * (2 * max(timings) if timings else 5.)
    print(f"Preflight {len(todo)} forwards, conservative estimate {estimate/60:.1f} min; hard stop {minutes} min", flush=True)
    if estimate > minutes * 60:
        raise RuntimeError("Reduce --limit to keep conservative estimate inside deadline")
    start = time.monotonic()
    model = LocalQwen(threads=4)
    loaded = time.monotonic() - start
    sig = fingerprint()
    exists = path("embeddings").exists()
    with path("embeddings").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text_sha256", "encoder", "tokens", "seconds", "vector"])
        if not exists:
            writer.writeheader()
        for i, r in enumerate(todo):
            if time.monotonic()-start > minutes*60-15:
                print("Deadline: clean resumable stop", flush=True)
                break
            before = time.monotonic()
            v = model.embed_one(r.text)
            sec = time.monotonic()-before
            writer.writerow(dict(text_sha256=r.text_sha256, encoder=sig, tokens=r.tokens,
                                 seconds=sec, vector=json.dumps(v.tolist(), separators=(",", ":"))))
            f.flush()
            if (i+1) % 20 == 0 or i == 0:
                print(f"Embedded {i+1}/{len(todo)}; last={sec:.2f}s; elapsed={time.monotonic()-start:.1f}s", flush=True)
    print(f"ONNX load={loaded:.2f}s; chunk total={time.monotonic()-start:.2f}s", flush=True)


def cached(text):
    p = FS / "verified_cache" / fingerprint() / f"{textsha(text)}.npy"
    return unit(np.load(p)), p


def m8_residual(x, e, hidden):
    sims = x @ e
    sims[np.asarray(hidden, dtype=int)] = -np.inf
    ix = np.argpartition(sims, -8)[-8:]
    q = unit(x[ix].mean(0, dtype=np.float64))
    return float(np.linalg.norm(e-q)), float(np.sqrt(max(0., 2-2*sims.max()))), ix.tolist()


def pair_anchor(x, source, donor, lam):
    return unit((1-lam)*unit(x[source])+lam*unit(x[donor]))


def tail_fit(scores):
    u = float(np.quantile(scores, .95))
    z = (scores[scores > u] - u) / (1-u)
    if not (len(z) and np.all((z > 0) & (z < 1))):
        raise ValueError("Invalid bounded tail data")
    a, b, _, _ = beta.fit(z, floc=0, fscale=1)
    return dict(u=u, a=float(a), b=float(b), mass=len(z)/len(scores),
                body_x=np.quantile(scores[scores <= u], np.linspace(0, 1, 101)).tolist())


def logsf(h, p):
    if h >= 1:
        return -np.inf
    if h <= -1:
        return 0.
    if h >= p["u"]:
        return float(np.log(p["mass"]) + beta.logsf((h-p["u"])/(1-p["u"]), p["a"], p["b"]))
    f = (1-p["mass"]) * np.interp(h, p["body_x"], np.linspace(0, 1, 101))
    return float(np.log1p(-f))


def logq(h, p):
    s = logsf(h, p)
    if s == 0:
        return np.inf
    return s if s < -35 else float(np.log(-np.log(-np.expm1(s))))


def inverse(h, p):
    return float((np.log(LN2)-logq(h, p))/LN10)


def max_quantile(logn, p, probability=.5):
    goal = np.log(-np.log(probability))-logn*LN10
    # Exact conversion between max quantile and one-draw survival.
    sf = -np.expm1(-np.exp(goal)) if goal > -35 else np.exp(goal)
    if sf >= p["mass"]:
        return float(np.interp((1-sf)/(1-p["mass"]), np.linspace(0, 1, 101), p["body_x"]))
    z = beta.isf(sf/p["mass"], p["a"], p["b"])
    return float(p["u"]+(1-p["u"])*z)


def single_quantile(probability, p):
    return max_quantile(0., p, probability)


def fit():
    if path("manifest").exists():
        raise RuntimeError("Already frozen")
    plan_df, bank_df, em = load("plan"), load("bank"), embeddings()
    assert set(plan_df.text_sha256) <= set(em)
    refs, jobs = refs_jobs()
    calrows, models, tunings, sources, tails, fits = [], [], [], [], [], []
    sourcepaths = [Path(__file__), ROOT/"src/latent_gap/embedder.py", ROOT/"scripts/bounds/support_probe.py",
                   FS/"refs.jsonl", FS/"inputs/fluent_inputs.json"]
    for domain in DOMAINS:
        check_deadline()
        _, x = corpus(domain)
        sourcepaths.extend([DATA/"prepared"/domain/"corpus.parquet", DATA/"prepared"/domain/"emb_qwen3e06.npy"])
        dplan = plan_df[plan_df.domain == domain]
        epsilons = []
        calvectors, tunevectors = [], []
        for op in OPS:
            g = dplan[(dplan.op == op) & dplan.phase.isin(["train", "cal"])]
            train = g[g.phase == "train"]
            grid = np.linspace(0, 1, 11)
            costs = []
            for lam in grid:
                costs.append(np.mean([np.linalg.norm(em[r.text_sha256] - pair_anchor(x,r.source,r.donor,lam))**2
                                      for r in train.itertuples()]))
            lam = float(grid[int(np.argmin(costs))])
            vals = []
            for r in g.itertuples():
                q = pair_anchor(x,r.source,r.donor,lam)
                residual = float(np.linalg.norm(em[r.text_sha256]-q))
                calrows.append(dict(domain=domain, op=op, phase=r.phase, draw=r.draw,
                                    residual=residual, source=r.source, donor=r.donor))
                if r.phase == "cal":
                    vals.append(residual)
                    calvectors.append(em[r.text_sha256]-q)
            eps = max(vals)
            epsilons.append(eps)
            models.append(dict(domain=domain, op=op, lam=lam, epsilon=eps, ncal=len(vals),
                               train_mse=min(costs), epsilon_train=max(r["residual"] for r in calrows
                               if r["domain"] == domain and r["op"] == op and r["phase"] == "train"),
                               marginal_failure=1/(len(vals)+1),
                               family_pac_failure=1-(DELTA/(len(DOMAINS)*len(OPS)))**(1/len(vals))))
        for r in dplan[dplan.phase == "parity"].itertuples():
            error = float(np.linalg.norm(em[r.text_sha256]-unit(x[r.source])))
            sources.append(dict(kind="encoder_parity", domain=domain, key=r.source, value=error))
            print(domain, "prepared/local anchor discrepancy", error, flush=True)
        for k in range(11, 21):
            t = next(unit(j["target"]) for j in jobs if (j["domain"],j["arm"],j["k"]) == (domain,"REAL",k))
            for writer in WRITERS:
                rp = SR/"tune-v2"/domain/writer/f"REAL-{k}.json"
                sourcepaths.append(rp)
                rec = json.loads(rp.read_text())
                assert rec["done"] and rec["config"]["prompt"] == "v2"
                assert (rec["domain"],rec["arm"],rec["k"]) == (domain,"REAL",k)
                gate = rec["gate"]
                row = dict(domain=domain, writer=writer, k=k, h_coh=gate["h_coh"], residual=None)
                if gate["h_coh"] is not None:
                    e, cp = cached(gate["coh_text"])
                    sourcepaths.append(cp)
                    assert abs(e @ t - gate["h_coh"]) < 3e-5
                    row["residual"], _, ix = m8_residual(x, e, refs[domain,"REAL",k]["hidden_pos"])
                    tunevectors.append(e-unit(x[ix].mean(0,dtype=np.float64)))
                tunings.append(row)
        ext = max([*epsilons, *[r["residual"] for r in tunings if r["domain"] == domain and r["residual"] is not None]])
        models.append(dict(domain=domain, op="extension", lam=None, epsilon=ext, ncal=None,
                           train_mse=None, epsilon_train=None, marginal_failure=None, family_pac_failure=None))
        selected = [j for j in jobs if j["domain"] == domain]
        rng = np.random.default_rng(SEED + 90 + DOMAINS.index(domain))
        selected += [dict(domain=domain, arm="RANDOM", k=k+1, target=unit(rng.normal(size=x.shape[1])).tolist())
                     for k in range(24)]
        ts = np.stack([unit(j["target"]) for j in selected])
        directional_cal = np.maximum(0., (np.stack(calvectors) @ ts.T).max(axis=0))
        directional_ext = np.maximum(directional_cal, (np.stack(tunevectors) @ ts.T).max(axis=0))
        scores = x @ ts.T
        for i, j in enumerate(selected):
            check_deadline()
            key = domain, j["arm"], j["k"]
            hidden = refs[key]["hidden_pos"] if key in refs else []
            s = np.delete(scores[:, i], hidden)
            if key in refs:
                assert abs(float(s.max())-refs[key]["ceiling"]) < 3e-4
            par = tail_fit(s)
            row = dict(domain=domain, arm=j["arm"], k=j["k"], parameters=json.dumps(par),
                       target=json.dumps(ts[i].tolist()), n_past=len(s), ceiling=float(s.max()),
                       directional_grammar=float(directional_cal[i]), directional_extension=float(directional_ext[i]),
                       past_max_predicted=max_quantile(np.log10(len(s)), par),
                       q99=single_quantile(.99, par), q999=single_quantile(.999, par))
            tails.append(row)
            if j["arm"] == "REAL" and j["k"] >= 11:
                for r in tunings:
                    if (r["domain"], r["k"]) == (domain,j["k"]) and r["h_coh"] is not None:
                        r["log10_neff"] = inverse(r["h_coh"], par)
        for writer in WRITERS:
            vals = [r["log10_neff"] for r in tunings if r["domain"] == domain and r["writer"] == writer and r["h_coh"] is not None]
            fits.append(dict(domain=domain, writer=writer, log10_neff=float(np.median(vals)), n=len(vals)))
        del scores, x
        print(domain, "fit complete; grammar max epsilon", max(epsilons), "extension", ext, flush=True)
    save("calibration", calrows)
    save("models", models)
    save("tuning", tunings)
    save("tails", tails)
    save("fits", fits)
    sources.extend(dict(kind="file", domain="", key=str(p), value=sha(p)) for p in sorted(set(sourcepaths)))
    save("sources", sources)


def freeze():
    models, tails, fits, bank_df = load("models"), load("tails"), load("fits"), load("bank")
    refs, _ = refs_jobs()
    predictions, certs = [], []
    for domain in DOMAINS:
        _, x = corpus(domain)
        # Universal grammar relaxation: cone of all allowed eligible sources.
        bank = bank_df.loc[bank_df.domain == domain, "position"].to_numpy(dtype=int)
        p = unit(x[bank])
        axis = unit(p.mean(0))
        latitude = float((p @ axis).min())
        assert latitude > 0
        dm = models[models.domain == domain]
        eps = max(float(v) for v in dm.loc[dm.op.isin(OPS), "epsilon"])
        ext = float(dm.loc[dm.op == "extension", "epsilon"].iloc[0])
        # Extension uses full visible-corpus cone, not just the grammar bank.
        xf = unit(x)
        axf = unit(xf.mean(0))
        latf = float((xf @ axf).min())
        for r in tails[tails.domain == domain].itertuples():
            check_deadline()
            if r.arm == "RANDOM" or (r.arm == "REAL" and r.k > 10):
                continue
            t, par = np.array(json.loads(r.target)), json.loads(r.parameters)
            hidden = np.asarray(refs[domain,r.arm,r.k]["hidden_pos"], dtype=int)
            if r.arm == "FAR":
                c = cone_certificate(p, t, axis, latitude, np.array([],dtype=int), 48, min(DEADLINE,time.monotonic()+180))
                cf = cone_certificate(xf, t, axf, latf, hidden, 48, min(DEADLINE,time.monotonic()+180))
                for label, cc in (("grammar",c),("extension",cf)):
                    certs.append(dict(domain=domain, arm=r.arm, k=r.k, cone=label,
                                      **{k:json.dumps(v) if isinstance(v,list) else v for k,v in cc.items()}))
                cone, cone_ext = c["upper"], cf["upper"]
            else:
                cone = min(1., max(0., float(((p@t)/(p@axis)).max())))
                ratio = (xf@t)/(xf@axf)
                ratio[hidden] = -np.inf
                cone_ext = min(1., max(0., float(ratio.max())))
            for writer in WRITERS:
                n = float(fits.loc[(fits.domain == domain)&(fits.writer == writer),"log10_neff"].iloc[0])
                predictions.append(dict(domain=domain, arm=r.arm, k=r.k, writer=writer,
                    grammar_cone=cone, grammar_epsilon=eps, grammar_bound=min(1.,cone+eps),
                    extension_cone=cone_ext, extension_epsilon=ext, extension_bound=min(1.,cone_ext+ext),
                    directional_grammar_residual=r.directional_grammar,
                    directional_extension_residual=r.directional_extension,
                    directional_grammar_bound=min(1.,cone+r.directional_grammar),
                    directional_extension_bound=min(1.,cone_ext+r.directional_extension),
                    ceiling=r.ceiling, ceiling_star=max(r.ceiling,refs[domain,r.arm,r.k]["heldout_ceiling"]),
                    tail_prediction=max_quantile(n,par), tail_lo=max_quantile(n,par,.025),
                    tail_hi=max_quantile(n,par,.975), log10_neff=n,
                    escape_probability=float(-np.expm1(-np.exp(np.clip(n*LN10+logq(max(r.ceiling,refs[domain,r.arm,r.k]["heldout_ceiling"])+.05,par),-745,700))))))
        del x, xf, p
        print(domain, "predictions complete", flush=True)
    save("predictions", predictions)
    save("cones", certs)
    names = ("plan", "bank", "protocol", "embeddings", "calibration", "models", "tuning", "tails", "fits", "sources", "predictions", "cones")
    files = [Path(__file__), *[path(n) for n in names]]
    save("manifest", [dict(path=str(p), sha256=sha(p), frozen_utc=now()) for p in files])
    print("FROZEN", now(), sha(path("predictions")), flush=True)


def verify_freeze():
    for r in load("manifest").itertuples():
        if sha(r.path) != r.sha256:
            raise ValueError(f"Frozen file changed: {r.path}")
    for r in load("sources").itertuples():
        if r.kind == "file" and sha(r.key) != r.value:
            raise ValueError(f"Source changed: {r.key}")


def membership(domain, text, texts, bank, pair_index, tok):
    """Exact membership; unchanged boundary sentences prune the finite search."""
    pp = parts(domain, text)
    if pp is None or not valid(domain,text,tok):
        return False
    candidates = pair_index.get((pp[2][0], pp[2][-1]), [])
    for i in candidates:
        for j in bank:
            check_deadline()
            if i == j:
                continue
            if any(compose(domain, texts[i], texts[j], op) == text for op in OPS):
                return True
    return False


def measure():
    verify_freeze()
    start = time.monotonic()
    pred, models, tails, plan_df, bank_df = (load(n) for n in ("predictions","models","tails","plan","bank"))
    refs, _ = refs_jobs()
    em = embeddings()
    tok = Tokenizer.from_file(str(MODEL_DIR/"tokenizer.json"))
    opus_path = SR/"test/opus_check.jsonl"
    opus = {r["key"]: r.get("coherence") for r in map(json.loads, opus_path.read_text().splitlines())}
    results, checks, futures, auditpaths = [], [], [], [opus_path]
    for domain in DOMAINS:
        texts, x = corpus(domain)
        bank = bank_df.loc[bank_df.domain == domain,"position"].to_numpy(dtype=int)
        pair_index = {}
        for i in bank:
            ss = parts(domain,texts[i])[2]
            pair_index.setdefault((ss[0],ss[-1]),[]).append(i)
        dm = models[models.domain == domain]
        for r in plan_df[(plan_df.domain == domain)&plan_df.phase.isin(["check","depth3"])].itertuples():
            # Depth-3 extrapolation uses the frozen FS predictor/epsilon.
            mr = dm[dm.op == ("FS" if r.op == "FST" else r.op)].iloc[0]
            lam, eps = float(mr.lam), float(mr.epsilon)
            q = pair_anchor(x,r.source,r.donor,lam)
            residual = float(np.linalg.norm(em[r.text_sha256]-q))
            checks.append(dict(domain=domain, op=r.op, phase=r.phase, draw=r.draw, residual=residual,
                               epsilon=eps, violation=residual>eps+1e-6, source=r.source, donor=r.donor,
                               lam=lam, text_sha256=r.text_sha256))
        for r in pred[pred.domain == domain].itertuples():
            check_deadline()
            rp = SR/"test"/domain/r.writer/f"{r.arm}-{r.k}.json"
            auditpaths.append(rp)
            rec = json.loads(rp.read_text())
            assert rec["done"] and rec["config"]["prompt"] == "v2"
            assert (rec["domain"],rec["arm"],rec["k"]) == (domain,r.arm,r.k)
            assert rec["config"]["writer"].split(":")[-1] == r.writer
            gate = rec["gate"]
            row = r._asdict(); row.pop("Index")
            row.update(h_coh=gate["h_coh"], in_language=None, m8_residual=None, nearest_distance=None,
                       opus_rating=opus.get(f"{domain}/{r.writer}/{r.arm}-{r.k}"),
                       grammar_score_violation=None, extension_score_violation=None, extension_residual_violation=None,
                       directional_grammar_violation=None, directional_extension_violation=None,
                       directional_residual=None, directional_residual_violation=None)
            if gate["h_coh"] is not None:
                assert min(gate["j1"],gate["j2"]) >= 7
                e, cp = cached(gate["coh_text"])
                auditpaths.append(cp)
                tr = tails[(tails.domain==domain)&(tails.arm==r.arm)&(tails.k==r.k)].iloc[0]
                t = np.array(json.loads(tr.target))
                assert abs(e @ t-gate["h_coh"]) < 3e-5
                row["h_coh"] = float(e@t)
                row["in_language"] = membership(domain,gate["coh_text"],texts,bank,pair_index,tok)
                row["text_sha256"] = textsha(gate["coh_text"])
                row["m8_residual"], row["nearest_distance"], ix = m8_residual(x,e,refs[domain,r.arm,r.k]["hidden_pos"])
                row["m8_positions"] = json.dumps(ix)
                row["grammar_score_violation"] = row["h_coh"]>r.grammar_bound+1e-6
                row["extension_score_violation"] = row["h_coh"]>r.extension_bound+1e-6
                row["extension_residual_violation"] = row["m8_residual"]>r.extension_epsilon+1e-6
                row["directional_residual"] = float((e-unit(x[ix].mean(0,dtype=np.float64)))@t)
                row["directional_residual_violation"] = row["directional_residual"]>r.directional_extension_residual+1e-6
                row["directional_grammar_violation"] = row["h_coh"]>r.directional_grammar_bound+1e-6
                row["directional_extension_violation"] = row["h_coh"]>r.directional_extension_bound+1e-6
            results.append(row)
        del x
        _, future = corpus(domain, future=True)
        dt = tails[tails.domain == domain]
        ts = np.stack([json.loads(r.target) for r in dt.itertuples()])
        sc = future @ ts.T
        for i,r in enumerate(dt.itertuples()):
            par = json.loads(r.parameters)
            s, n = sc[:,i], len(future)
            predicted = max_quantile(np.log10(n),par)
            lo,hi = max_quantile(np.log10(n),par,.025),max_quantile(np.log10(n),par,.975)
            futures.append(dict(domain=domain,arm=r.arm,k=r.k,n=n,
                observed_max=float(s.max()),predicted_max=predicted,max_lo=lo,max_hi=hi,
                max_covered=lo<=s.max()<=hi,
                exceed_q99=int((s>r.q99).sum()),exceed_q999=int((s>r.q999).sum()),
                ratio_q99=float((s>r.q99).mean()/.01),ratio_q999=float((s>r.q999).mean()/.001)))
        del future,sc
        print(domain, "test and future checks complete", flush=True)
    save("results",results)
    save("checks",checks)
    save("future",futures)
    # No imputations: missing winners remain absent, filtered winners are not reranked.
    summary, combined = [], []
    result_df = pd.DataFrame(results)
    for gate_name in ("original","opus"):
        df = result_df.copy()
        if gate_name == "opus":
            df.loc[~(pd.to_numeric(df.opus_rating,errors="coerce")>=7),"h_coh"] = np.nan
        for (d,a),g in df.groupby(["domain","arm"]):
            winners = []
            for _,gg in g.groupby("k"):
                mm = gg[gg.h_coh.notna()]
                if len(mm):
                    winners.append(mm.loc[mm.h_coh.idxmax()])
            m = pd.DataFrame(winners)
            if not len(m):
                continue
            combined.append(dict(gate=gate_name,domain=d,arm=a,n=g.k.nunique(),measured=len(m),
                h_median=m.h_coh.median(),in_language=int(m.in_language.sum()),
                grammar_bound_median=g.drop_duplicates("k").grammar_bound.median(),
                extension_bound_median=g.drop_duplicates("k").extension_bound.median(),
                directional_grammar_bound_median=g.drop_duplicates("k").directional_grammar_bound.median(),
                directional_extension_bound_median=g.drop_duplicates("k").directional_extension_bound.median(),
                grammar_score_violations=int(m.grammar_score_violation.sum()),
                extension_score_violations=int(m.extension_score_violation.sum()),
                extension_residual_violations=int(m.extension_residual_violation.sum()),
                directional_grammar_violations=int(m.directional_grammar_violation.sum()),
                directional_extension_violations=int(m.directional_extension_violation.sum()),
                directional_residual_violations=int(m.directional_residual_violation.sum()),
                observed_escapes=int((m.h_coh>=m.ceiling_star+.05).sum()),
                predicted_escape_lower=g.groupby("k").escape_probability.max().mean(),
                predicted_escape_upper=g.groupby("k").escape_probability.sum().clip(upper=1).mean()))
        for (d,w,a),g in df.groupby(["domain","writer","arm"]):
            m = g[g.h_coh.notna()]
            err = float(m.h_coh.median()-m.tail_prediction.median()) if len(m) else np.nan
            pi = float(((m.h_coh>=m.tail_lo)&(m.h_coh<=m.tail_hi)).mean()) if len(m) else np.nan
            summary.append(dict(gate=gate_name,domain=d,writer=w,arm=a,n=len(g),measured=len(m),
                in_language=int(m.in_language.sum()),grammar_bound_median=g.grammar_bound.median(),
                extension_bound_median=g.extension_bound.median(),h_median=m.h_coh.median(),
                grammar_score_violations=int(m.grammar_score_violation.sum()),
                extension_score_violations=int(m.extension_score_violation.sum()),
                extension_residual_violations=int(m.extension_residual_violation.sum()),
                directional_grammar_bound_median=g.directional_grammar_bound.median(),
                directional_extension_bound_median=g.directional_extension_bound.median(),
                directional_grammar_violations=int(m.directional_grammar_violation.sum()),
                directional_extension_violations=int(m.directional_extension_violation.sum()),
                directional_residual_violations=int(m.directional_residual_violation.sum()),
                residual_max=m.m8_residual.max(), tail_predicted_median=m.tail_prediction.median(),
                tail_median_error=err,tail_pi_coverage=pi,
                search_pass=len(m)/len(g)>=.9 and abs(err)<=.03 and pi>=.8,
                observed_excess_median=(m.h_coh-m.ceiling).median(),
                predicted_excess_median=(m.tail_prediction-m.ceiling).median(),
                observed_escapes=int((m.h_coh>=m.ceiling_star+.05).sum()),
                predicted_escape_fraction=g.escape_probability.mean()))
    save("summary",summary)
    save("combined",combined)
    fsumm = []
    for (d,a),g in pd.DataFrame(futures).groupby(["domain","arm"]):
        # REAL11..20 are tuning directions, report separately below.
        groups = [("all",g)] if a != "REAL" else [("test",g[g.k<=10]),("tuning",g[g.k>10])]
        for subset,gg in groups:
            bias = float(np.median(gg.predicted_max-gg.observed_max))
            r99,r999 = gg.ratio_q99.median(),gg.ratio_q999.median()
            coverage = gg.max_covered.mean()
            fsumm.append(dict(domain=d,arm=a,subset=subset,n=len(gg),ratio_q99=r99,ratio_q999=r999,
                observed_max=gg.observed_max.median(),predicted_max=gg.predicted_max.median(),
                paired_median_bias=bias,max_coverage=coverage,
                passed=.5<=r99<=2 and .5<=r999<=2 and abs(bias)<=(.01 if a in ("FAR","RANDOM") else .03) and coverage>=.8))
    save("future_summary",fsumm)
    save("audit",[dict(path=str(p),sha256=sha(p),measured_utc=now(),seconds=time.monotonic()-start)
                  for p in sorted(set(auditpaths))])
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(fsumm).to_string(index=False))


def check():
    # Mathematical identities at moderate/extreme, physically bounded quantiles.
    p = dict(u=.2,a=.8,b=20.,mass=.05,body_x=np.linspace(-1,.2,101).tolist())
    for n in (2,5,10,20,40):
        h = max_quantile(n,p)
        assert -1 <= h < 1 and abs(inverse(h,p)-n)<1e-6
    assert logsf(1.,p) == -np.inf and logsf(-1.,p)==0
    assert max_quantile(10,p,.025)<max_quantile(10,p,.5)<max_quantile(10,p,.975)
    # Geometry: exact two-ray mixtures and the bounded cell residual inequality.
    a,b = np.array([1.,0.]),np.array([0.,1.])
    q=unit(a+b); e=unit([.3,1.]);t=unit([-.4,1.])
    assert e@t <= q@t+np.linalg.norm(e-q)+1e-12
    if path("manifest").exists():
        verify_freeze()
    if path("results").exists():
        rr=load("results")
        assert len(rr)==200 and len(load("future"))==168
        assert len(load("checks"))==448
        assert np.all((rr.grammar_bound.astype(float)<=1)&(rr.extension_bound.astype(float)<=1))
    print("PASS: bounded-tail inverse/quantile, endpoints, intervals, cell inequality, artifacts and freeze")


def runtime_row(mode, seconds):
    """Mac ru_maxrss is bytes. Append-only operational log, outside model freeze."""
    exists = path("runtime").exists()
    with path("runtime").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "utc", "seconds", "peak_rss_gib"])
        if not exists:
            writer.writeheader()
        writer.writerow(dict(mode=mode, utc=now(), seconds=seconds,
                             peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    for flag in ("plan","embed","fit","freeze","measure","check"):
        group.add_argument("--"+flag,action="store_true")
    ap.add_argument("--limit",type=int,default=240)
    ap.add_argument("--minutes",type=float,default=35.)
    args=ap.parse_args()
    if not 0<args.minutes<=40 or args.limit<=0:
        ap.error("Use positive limit, 0<minutes<=40")
    start = time.monotonic()
    DEADLINE = start + args.minutes*60
    if args.embed:
        embed(args.limit,args.minutes)
        runtime_row("embed", time.monotonic()-start)
    else:
        for flag in ("plan","fit","freeze","measure","check"):
            if getattr(args,flag):
                globals()[flag]()
                runtime_row(flag, time.monotonic()-start)
