"""Round 4A: corpus-equivalent size. Offline, three deliberately separate phases.

  uv run --no-sync --no-cache python scripts/bounds/neff.py --fit
  uv run --no-sync --no-cache python scripts/bounds/neff.py --predict
  uv run --no-sync --no-cache python scripts/bounds/neff.py --measure

Only --measure opens seed-rewrite/test outcomes. No encoder, judge, network, or
LLM imports. All writes are docs/prereg/neff_*.csv. See docs/bounds4_neff.md.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / "Documents/experiment-data/latent-gap-ideas"
FS = DATA / "runs/fluent-search"
SR = DATA / "runs/seed-rewrite"
OUT = ROOT / "docs/prereg"
DOMAINS = ("steam", "arxiv")
WRITERS = ("claude-sonnet-5-5", "gpt-6-luna")
MODELS = ("mixture32", "gpd95")
SEED = 202610284
B = 10000
PERM = 19999
LN2, LN10 = np.log(2.0), np.log(10.0)
CRITERION = ("For each domain/writer, all three arms must have coverage>=0.90, "
             "abs(observed median - paired predicted median)<=0.03, and no "
             "tuning-vs-arm median-log10N permutation rejection at Holm 0.05 "
             "over the 12 domain/writer/arm comparisons, separately per tail/scope. "
             "Domain-only survives only if both writers survive. Bootstrap "
             "intervals are descriptive 95%, conditional on corpus and gate. "
             "An observed score beyond a fitted finite tail endpoint fails support.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def save(name, rows):
    path = OUT / f"neff_{name}.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")
    return path


def load(name):
    return pd.read_csv(OUT / f"neff_{name}.csv", float_precision="round_trip")


def moment_model(x, raw_second, hidden):
    """Exact visible-row moments, avoiding N x d float64 centred/residual arrays."""
    n, d = x.shape
    keep = np.ones(n, bool)
    keep[hidden] = False
    count = int(keep.sum())
    excluded = x[hidden].astype(np.float64)
    mu = (x.sum(0, dtype=np.float64) - excluded.sum(0)) / count
    cov = (raw_second - excluded.T @ excluded) / count - np.outer(mu, mu)
    r = np.linalg.norm(mu)
    if r <= 1e-10:
        raise ValueError("Zero mean: latitude mixture undefined")
    u = mu / r
    tau = u @ cov @ u
    if tau <= 1e-14:
        raise ValueError("Zero latitude variance")
    v = cov @ u / tau
    # One small matrix multiplication supplies all row-wise quantities.
    proj = x @ np.stack([u, v, mu], axis=1)
    lat = proj[:, 0] - r
    energy = (np.einsum("ij,ij->i", x, x, dtype=np.float64) - 2*proj[:, 2] + r*r
              - 2*lat*(proj[:, 1] - mu @ v) + lat*lat*(v @ v))
    groups = np.array_split(np.flatnonzero(keep)[np.argsort(lat[keep], kind="stable")], 32)
    weights = np.array([len(g)/count for g in groups])
    means = np.array([lat[g].mean() for g in groups])
    variances = np.array([lat[g].var() for g in groups])
    scales = np.array([energy[g].mean()/energy[keep].mean() for g in groups])
    assert abs(weights @ means) < 1e-10
    assert abs(weights @ (variances+means**2)-tau) < 1e-10
    assert abs(weights @ scales-1) < 1e-10
    return dict(n=count, mu=mu, cov=cov, r=r, u=u, tau=tau, v=v,
                weights=weights, means=means, variances=variances, scales=scales)


def directional(st, t):
    a = float(st["mu"] @ t)
    h = float(st["v"] @ t)
    var = float(t @ st["cov"] @ t)
    resvar = max(var-st["tau"]*h*h, 0)
    means = a + h*st["means"]
    variances = resvar*st["scales"] + h*h*st["variances"]
    w = st["weights"]
    assert abs(w @ means-a) < 1e-10
    assert abs(w @ (variances+(means-a)**2)-var) < 1e-10
    return dict(means=means.tolist(), sds=np.sqrt(variances).tolist(), weights=w.tolist()), {
        "N_visible": st["n"], "mean_score": a, "directional_variance": var,
        "mean_axis_latitude": float(st["u"] @ t), "regression_projection": h,
        "mean_norm": st["r"], "latitude_variance": st["tau"]}


def logsf(c, model, par):
    c = np.asarray(c)
    if model == "mixture32":
        z = (c[..., None]-np.asarray(par["means"]))/np.asarray(par["sds"])
        return logsumexp(np.log(par["weights"])+log_ndtr(-z), axis=-1)
    u, xi, scale = par["threshold"], par["shape"], par["scale"]
    tail = np.log(par["tail_mass"]) + genpareto.logsf(np.maximum(c-u, 0), xi, scale=scale)
    # Continuous body only for observations below the pre-fixed 95% threshold.
    body_lcdf = (np.log1p(-par["tail_mass"]) + log_ndtr((c-par["mean"])/par["sd"])
                 - log_ndtr((u-par["mean"])/par["sd"]))
    body = np.log(-np.expm1(np.minimum(body_lcdf, -1e-16)))
    return np.where(c >= u, tail, body)


def log_q(c, model, par):
    """log[-log F(c)], stable even where 1-F is far below machine epsilon."""
    ls = np.minimum(logsf(c, model, par), -1e-16)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.log(-np.log1p(-np.exp(np.maximum(ls, -36))))
    return np.where(ls < -35, ls, exact)


def inverse(c, model, par):
    return (np.log(LN2)-log_q(c, model, par))/LN10


def quantile(log10n, model, par, p=.5):
    if not np.isfinite(log10n):
        if model == "gpd95" and par["shape"] < 0:
            return par["threshold"]-par["scale"]/par["shape"]
        return np.inf
    goal = np.log(-np.log(p))-log10n*LN10
    if model == "gpd95":
        # For an extreme tail, solve survival directly, without 1-F cancellation.
        sf = -np.expm1(-np.exp(goal)) if goal > -35 else np.exp(goal)
        if sf <= par["tail_mass"]:
            return float(par["threshold"] + genpareto.isf(sf/par["tail_mass"],
                         par["shape"], scale=par["scale"]))
    lo, hi = -2.0, 2.0
    while log_q(lo, model, par) < goal:
        lo *= 2
    while log_q(hi, model, par) > goal:
        hi *= 2
    return brentq(lambda c: float(log_q(c, model, par))-goal, lo, hi, xtol=1e-12)


def boot_ci(values, seed=SEED):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.median(values[rng.integers(len(values), size=(B, len(values)))], axis=1)
    return tuple(np.quantile(med, [.025, .975], method="inverted_cdf"))


def outcome(domain, writer, arm, k, split):
    path = SR / split / domain / writer / f"{arm}-{k}.json"
    rec = json.loads(path.read_text())
    assert (rec["domain"], rec["arm"], rec["k"]) == (domain, arm, k)
    assert rec["done"] and rec["config"]["prompt"] == "v2"
    assert rec["config"]["S"] == 2 and rec["config"]["R"] == 3
    assert rec["config"]["writer"].split(":")[-1] == writer
    h = rec["gate"]["h_coh"]
    return np.nan if h is None else float(h), path


def fit():
    if (OUT / "neff_predictions.csv").exists():
        raise RuntimeError("Predictions already frozen; fitting is closed")
    for name in ("models", "tuning", "fits", "sources"):
        if (OUT/f"neff_{name}.csv").exists():
            raise FileExistsError(name)
    paths = [FS/"inputs/fluent_inputs.json", FS/"refs.jsonl", FS/"refs.csv"]
    jobs = json.loads(paths[0].read_text())
    refs = pd.read_json(paths[1], lines=True)
    ref = {(r.domain, r.arm, r.k): r for r in refs.itertuples()}
    models, tuning = [], []
    for domain in DOMAINS:
        base = DATA/"prepared"/domain
        paths.extend([base/"corpus.parquet", base/"emb_qwen3e06.npy"])
        meta = pd.read_parquet(base/"corpus.parquet", columns=["split", "adult"])
        keep = ((meta.split == "past") & ~meta.adult).to_numpy()
        raw = np.load(base/"emb_qwen3e06.npy", mmap_mode="r")
        x = np.array(raw[keep], dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        fingerprint = array_sha(x)
        second = np.zeros((x.shape[1], x.shape[1]))
        for start in range(0, len(x), 4096):
            block = x[start:start+4096].astype(np.float64)
            second += block.T @ block
        st_full = moment_model(x, second, [])
        selected = [j for j in jobs if j["domain"] == domain and j["k"] > 0]
        ts = np.array([j["target"] for j in selected], dtype=np.float64)
        ts /= np.linalg.norm(ts, axis=1, keepdims=True)
        scores = x @ ts.T
        for i, (job, t) in enumerate(zip(selected, ts)):
            arm, k = job["arm"], int(job["k"])
            rr = ref[domain, arm, k]
            hidden = list(rr.hidden_pos)
            st = moment_model(x, second, hidden) if hidden else st_full
            mix, fields = directional(st, t)
            s = np.delete(scores[:, i], hidden)
            u = float(np.quantile(s, .95))
            excess = s[s > u]-u
            xi, loc, scale = genpareto.fit(excess, floc=0)
            gpd = dict(threshold=u, shape=float(xi), scale=float(scale), tail_mass=len(excess)/len(s),
                       mean=float(s.mean()), sd=float(s.std()))
            for model, par in zip(MODELS, (mix, gpd)):
                row = dict(domain=domain, arm=arm, k=k, model=model, parameters=json.dumps(par),
                           corpus_sha256=fingerprint, target_sha256=array_sha(t), **fields,
                           ceiling=float(rr.ceiling), ceiling_star=max(rr.ceiling, rr.heldout_ceiling),
                           corpus_q99=float(rr.corpus_q99))
                row["corpus_max_predicted"] = quantile(np.log10(len(s)), model, par)
                # Descriptive corpus tail check, not independent tail validation.
                for p in (.99, .999):
                    c = float(np.quantile(s, p))
                    row[f"sf_ratio_at_q{str(p)[2:]}"] = float(np.exp(logsf(c, model, par))/(1-p))
                models.append(row)
                if arm == "REAL" and 11 <= k <= 20:
                    for writer in WRITERS:
                        h, path = outcome(domain, writer, arm, k, "tune-v2")
                        paths.append(path)
                        tuning.append(dict(domain=domain, writer=writer, arm=arm, k=k, model=model,
                                           h_coh=h, log10_neff=float(inverse(h, model, par)),
                                           outcome_sha256=sha(path)))
        del x, scores, raw
        print(f"Corpus moments and tuning only: {domain}", flush=True)
    td = pd.DataFrame(tuning)
    fits = []
    for (domain, model), group in td.groupby(["domain", "model"]):
        for writer in (*WRITERS, "pooled"):
            g = group if writer == "pooled" else group[group.writer == writer]
            # Domain-only fit clusters the two writers on the same target.
            vals = g.groupby("k").log10_neff.median().dropna().to_numpy()
            lo, hi = boot_ci(vals)
            fits.append(dict(domain=domain, model=model, writer=writer,
                             scope="domain" if writer == "pooled" else "domain_writer",
                             n_targets=len(vals), tuning_coverage=g.h_coh.notna().mean(),
                             log10_neff=float(np.median(vals)),
                             log10_neff_lo=lo, log10_neff_hi=hi,
                             unsupported_tuning=int(np.isposinf(g.log10_neff).sum())))
    save("models", models)
    save("tuning", tuning)
    save("fits", fits)
    save("sources", [dict(path=str(p), sha256=sha(p)) for p in sorted(set(paths))])
    print(pd.DataFrame(fits).to_string(index=False))


def predict():
    for name in ("predictions", "manifest"):
        if (OUT/f"neff_{name}.csv").exists():
            raise FileExistsError(name)
    models, fits = load("models"), load("fits")
    rows = []
    for r in models.itertuples():
        if r.arm == "REAL" and r.k > 10:
            continue
        par = json.loads(r.parameters)
        for writer in WRITERS:
            for scope in ("domain_writer", "domain"):
                w = writer if scope == "domain_writer" else "pooled"
                f = fits[(fits.domain == r.domain) & (fits.model == r.model) & (fits.writer == w)].iloc[0]
                row = dict(domain=r.domain, writer=writer, arm=r.arm, k=r.k, model=r.model, scope=scope,
                           log10_neff=f.log10_neff, log10_neff_lo=f.log10_neff_lo,
                           log10_neff_hi=f.log10_neff_hi, ceiling=r.ceiling, ceiling_star=r.ceiling_star)
                for key, n, p in (("predicted", f.log10_neff, .5),
                                  ("fit_lo", f.log10_neff_lo, .5), ("fit_hi", f.log10_neff_hi, .5),
                                  ("max_pi_lo", f.log10_neff, .025), ("max_pi_hi", f.log10_neff, .975)):
                    row[key] = quantile(n, r.model, par, p)
                lq = log_q(r.ceiling_star+.05, r.model, par)
                rate = np.exp(np.clip(f.log10_neff*LN10+lq, -745, 700))
                row["escape_probability"] = float(-np.expm1(-rate))
                row["median_escape"] = row["predicted"] >= r.ceiling_star+.05
                rows.append(row)
    dest = save("predictions", rows)
    now = datetime.now(timezone.utc).isoformat()
    paths = [Path(__file__), dest] + [OUT/f"neff_{n}.csv" for n in ("models", "tuning", "fits", "sources")]
    manifest = [dict(kind="file", key=str(p.relative_to(ROOT)), value=sha(p), frozen_utc=now) for p in paths]
    manifest += [dict(kind="protocol", key=k, value=v, frozen_utc=now) for k, v in {
        "criterion": CRITERION, "seed": str(SEED), "bootstrap": str(B), "permutations": str(PERM),
        "disclosure": "Experimenter already measured test. Requested prereg summaries were read. "
                      "No seed-rewrite/test per-target outcomes opened before this freeze. Out-of-set, not prospective.",
        "alternative": "gpd95 threshold fixed at corpus q95, MLE shape/scale on excesses; same REAL-only size fit",
        "gpd_rescue": "Rescue only if all 12 cells survive and pooled target MAE improves >=20% over mixture32",
        "iid_diagnostic": "Secondary held-out maximum PIT KS calibrated by 10000 null simulations including tuning fit; "
                          "Holm over 12 domain/writer/arm cells, domain_writer only; 9 training "
                          "maxima for arxiv/Sonnet (REAL-17 missing), 10 otherwise",
    }.items()]
    save("manifest", manifest)
    print(f"FROZEN {len(rows)} predictions at {now}; sha256={sha(dest)}")


def permutation_p(a, b, seed):
    a, b = np.asarray(a), np.asarray(b)
    if not np.all(np.isfinite(np.r_[a, b])):
        return np.nan
    obs = abs(np.median(a)-np.median(b))
    pool = np.r_[a, b]
    rng = np.random.default_rng(seed)
    hit = 0
    for start in range(0, PERM, 1000):
        inds = np.argsort(rng.random((min(1000, PERM-start), len(pool))), axis=1)
        perm = pool[inds]
        stats = abs(np.median(perm[:, :len(a)], axis=1)-np.median(perm[:, len(a):], axis=1))
        hit += int((stats >= obs-1e-12).sum())
    return (hit+1)/(PERM+1)


def holm(p):
    p = np.asarray(p)
    order = np.flatnonzero(np.isfinite(p))[np.argsort(p[np.isfinite(p)])]
    result = np.full(len(p), np.nan)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(order)))))
    return result


def ks_uniform(u):
    u = np.sort(u, axis=-1)
    n = u.shape[-1]
    return np.maximum(np.max(np.arange(1, n+1)/n-u, axis=-1),
                      np.max(u-np.arange(n)/n, axis=-1))


def iid_pit_p(log_q_values, log10n, n_train, seed):
    if not np.isfinite(log10n):
        return 0.0
    u = np.exp(-np.exp(np.clip(log_q_values+log10n*LN10, -745, 700)))
    stat = ks_uniform(u)
    rng = np.random.default_rng(seed)
    # Scale equivariance permits true N=1; inverse-median estimator is unchanged.
    e_train = rng.exponential(size=(B, n_train))
    n_fit = np.exp(np.median(np.log(LN2)-np.log(e_train), axis=1))
    e_test = rng.exponential(size=(B, len(u)))
    null = ks_uniform(np.exp(-e_test*n_fit[:, None]))
    return float((1+(null >= stat).sum())/(B+1))


def measure():
    for name in ("results", "summary", "escape", "audit"):
        if (OUT/f"neff_{name}.csv").exists():
            raise FileExistsError(name)
    manifest = load("manifest")
    for r in manifest[manifest.kind == "file"].itertuples():
        if sha(ROOT/r.key) != r.value:
            raise RuntimeError(f"Frozen artifact changed: {r.key}")
    for r in load("sources").itertuples():
        if sha(r.path) != r.sha256:
            raise RuntimeError(f"Source changed: {r.path}")
    models, tuning, predictions = load("models"), load("tuning"), load("predictions")
    pars = {(r.domain, r.arm, r.k, r.model): json.loads(r.parameters) for r in models.itertuples()}
    observed, hashes = {}, []
    for d in DOMAINS:
        for w in WRITERS:
            for arm, count in (("REAL", 10), ("GAP", 20), ("FAR", 20)):
                for k in range(1, count+1):
                    h, path = outcome(d, w, arm, k, "test")
                    observed[d, w, arm, k] = h
                    hashes.append(dict(path=str(path), sha256=sha(path)))
    rows = []
    for r in predictions.to_dict("records"):
        h = observed[r["domain"], r["writer"], r["arm"], r["k"]]
        par = pars[r["domain"], r["arm"], r["k"], r["model"]]
        r.update(observed=h, observed_log10_neff=float(inverse(h, r["model"], par)),
                 error=h-r["predicted"], observed_logq=float(log_q(h, r["model"], par)))
        rows.append(r)
    results = pd.DataFrame(rows)
    summaries = []
    keys = ["model", "scope", "domain", "writer", "arm"]
    for index, (key, group) in enumerate(results.groupby(keys)):
        model, scope, domain, writer, arm = key
        g = group[group.observed.notna()]
        vals = g.observed_log10_neff.to_numpy()
        # Pooling writers changes null dispersion: test arm drift within writer.
        tr = tuning[(tuning.model == model) & (tuning.domain == domain) & (tuning.writer == writer)]
        train = tr.groupby("k").log10_neff.median().dropna().to_numpy()
        lo, hi = boot_ci(vals, SEED+index)
        bias_lo, bias_hi = boot_ci(g.error.to_numpy(), SEED+index)
        row = dict(zip(keys, key))
        row.update(n=len(group), measured=len(g), coverage=len(g)/len(group),
                   predicted_median=group.predicted.median(), paired_predicted_median=g.predicted.median(),
                   observed_median=g.observed.median(), median_error=g.observed.median()-g.predicted.median(),
                   paired_error_median=g.error.median(), paired_error_lo=bias_lo, paired_error_hi=bias_hi,
                   mae=g.error.abs().mean(), log10_neff=g.log10_neff.iloc[0],
                   log10_neff_lo=g.log10_neff_lo.iloc[0], log10_neff_hi=g.log10_neff_hi.iloc[0],
                   observed_log10_neff=float(np.median(vals)), observed_log10_neff_lo=lo,
                   observed_log10_neff_hi=hi, unsupported_observed=int(np.isposinf(vals).sum()),
                   permutation_p=permutation_p(train, vals, SEED+index),
                   iid_pit_p=iid_pit_p(g.observed_logq.to_numpy(), g.log10_neff.iloc[0], len(train), SEED+index)
                   if scope == "domain_writer" else np.nan,
                   pi_coverage=((g.observed >= g.max_pi_lo) & (g.observed <= g.max_pi_hi)).mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    for _, inds in summary.groupby(["model", "scope"]).groups.items():
        summary.loc[inds, "permutation_holm"] = holm(summary.loc[inds, "permutation_p"])
        if summary.loc[inds, "scope"].iloc[0] == "domain_writer":
            summary.loc[inds, "iid_pit_holm"] = holm(summary.loc[inds, "iid_pit_p"])
    summary["survives"] = ((summary.coverage >= .9) & (summary.median_error.abs() <= .03)
                            & (summary.permutation_holm > .05) & (summary.unsupported_observed == 0))
    # Independent-judge sensitivity: the original accepted witness only, no new gate walk.
    opus_path = SR/"test/opus_check.jsonl"
    opus = {r["key"]: r.get("coherence") for r in
            (json.loads(line) for line in opus_path.read_text().splitlines() if line.strip())}
    hashes.append(dict(path=str(opus_path), sha256=sha(opus_path)))
    escape = []
    for (model, scope, domain), group in results[results.arm == "FAR"].groupby(["model", "scope", "domain"]):
        for writer in (*WRITERS, "max"):
            g = group if writer == "max" else group[group.writer == writer]
            p = g.pivot(index="k", columns="writer", values="escape_probability")
            obs = g.pivot(index="k", columns="writer", values="observed")
            threshold = g.groupby("k").ceiling_star.first()+.05
            accepted = obs.copy()
            for k in accepted.index:
                for w in accepted.columns:
                    score = opus.get(f"{domain}/{w}/FAR-{k}")
                    if score is None or score < 7:
                        accepted.loc[k, w] = np.nan
            escape.append(dict(model=model, scope=scope, domain=domain, writer=writer, n=len(obs),
                expected_share_lo=p.max(axis=1).mean(), expected_share_hi=p.sum(axis=1).clip(upper=1).mean(),
                expected_share_independent=(1-(1-p).prod(axis=1)).mean(),
                median_prediction_share=(g.groupby("k").predicted.max() >= threshold).mean(),
                observed_share=(obs.max(axis=1) >= threshold).mean(),
                opus_observed_share=(accepted.max(axis=1) >= threshold).mean(),
                opus_coverage=accepted.max(axis=1).notna().mean()))
    save("results", results)
    save("summary", summary)
    save("escape", escape)
    save("audit", hashes+[dict(path="measurement_utc", sha256=datetime.now(timezone.utc).isoformat())])
    cols = keys+["predicted_median", "observed_median", "observed_log10_neff", "permutation_holm", "survives"]
    print(summary[cols].to_string(index=False))
    print(pd.DataFrame(escape).to_string(index=False))


def checks():
    par = dict(means=[.2], sds=[.1], weights=[1.0])
    for n in (0, 2, 6, 30, 100):
        for p in (.025, .5, .975):
            h = quantile(n, "mixture32", par, p)
            assert abs(float(log_q(h, "mixture32", par))-(np.log(-np.log(p))-n*LN10)) < 1e-8
        assert abs(float(inverse(quantile(n, "mixture32", par), "mixture32", par))-n) < 1e-8
    gpd = dict(threshold=.3, shape=-.1, scale=.05, tail_mass=.05, mean=0., sd=.2)
    for n in (2, 6, 12):
        assert abs(float(inverse(quantile(n, "gpd95", gpd), "gpd95", gpd))-n) < 1e-7
    assert np.isposinf(inverse(.81, "gpd95", gpd))
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(1000, 8))+.5
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    st = moment_model(x, x.T@x, [2, 5])
    visible = np.delete(x, [2, 5], axis=0)
    assert np.allclose(st["cov"], np.cov(visible.T, bias=True), atol=1e-12)
    directional(st, np.ones(8)/np.sqrt(8))
    print("PASS: inversion/quantile round trips (through N=1e100), GPD endpoint, masked moments, mixture moments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("fit", "predict", "measure", "check"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.check:
        checks()
    elif args.fit:
        fit()
    elif args.predict:
        predict()
    else:
        measure()
