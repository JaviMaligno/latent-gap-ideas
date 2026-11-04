"""M1 of docs/prereg/modern_and_oracle_history.md: one-shot CTX (and CTX+MEM) step gap for gpt-5.4-mini on the 2025
contexts. Embeds the Part B ideas locally (Qwen3, same encoder as the corpus). Output: key "M1" in oracle_history.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import continue_from_context as M  # noqa: E402
from latent_gap.arxiv import item_text  # noqa: E402
from latent_gap.embedder import LocalQwen  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/prereg/oracle_history.json"
z = np.load(M.RUN / "emb.npz"); C, F, owner = z["C"], z["F"], z["owner"]
rows = [r for r in map(json.loads, (M.RUN / "ideas.jsonl").read_text().splitlines()) if r["abstract"]]
cache = M.RUN / "partB_emb.npy"
V = np.load(cache) if cache.exists() else LocalQwen(threads=int(os.environ.get("OMP_NUM_THREADS", "6"))).embed([item_text(r["title"], r["abstract"]) for r in rows])
np.save(cache, V)
res = {}
for arm in ("CTX", "CTX+MEM"):
    gap, div = [], []
    for i in range(len(C)):
        idx = [n for n, r in enumerate(rows) if r["ctx"] == i and r["arm"] == arm]
        X = V[idx]
        gap.append(float(np.median(X @ C[i]) - np.median(F[owner == i] @ C[i])))
        S = X @ X.T; div.append(float(S[np.triu_indices(len(X), 1)].mean()))
    g = np.array(gap)
    res[arm] = {"mean_step_gap": float(g.mean()), "p": float(stats.wilcoxon(g, alternative="greater").pvalue),
                "followup_step_median": float(np.median([np.median(F[owner == i] @ C[i]) for i in range(len(C))])),
                "mean_div": float(np.mean(div))}
allres = json.loads(OUT.read_text()); allres.setdefault("continue-modern", {})["M1"] = res
OUT.write_text(json.dumps(allres, indent=1)); print(json.dumps(res, indent=1))
