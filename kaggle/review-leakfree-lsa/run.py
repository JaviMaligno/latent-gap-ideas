"""review_leakfree_encoder (docs/prereg/review_leakfree_encoder.md), LSA space: TF-IDF + SVD fitted on past text only,
then Section 4 (where_future_lands.analyse) and the gap-score validity (review_hole_validity, raw, K = 256) for Steam,
arXiv and patents. CPU kernel: no model, no GPU needed. Progress is printed in plain lines (no tqdm).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

W = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/review_leakfree_encoder.py", recursive=True)[0])
os.makedirs(f"{W}/repo/scripts", exist_ok=True)
os.makedirs(f"{W}/repo/docs/prereg", exist_ok=True)
for f in ("where_future_lands.py", "review_hole_validity.py", "revision_future_cells.py", "review_leakfree_encoder.py"):
    shutil.copy(f"{src}/{f}", f"{W}/repo/scripts/{f}")
shutil.copy(f"{src}/where_future_lands_result.json", f"{W}/repo/docs/prereg/")
for d in ("steam", "arxiv", "patents"):
    os.makedirs(f"{W}/data/prepared/{d}", exist_ok=True)
    if not os.path.exists(f"{W}/data/prepared/{d}/corpus.parquet"):
        os.symlink(f"{src}/{d}_corpus.parquet", f"{W}/data/prepared/{d}/corpus.parquet")
env = dict(os.environ, LF_DATA=f"{W}/data", LF_OUT=f"{W}/out", LF_WORKERS=str(os.cpu_count() or 4),
           OMP_NUM_THREADS=str(os.cpu_count() or 4), PYTHONUNBUFFERED="1")
print(f"cpus {os.cpu_count()}", flush=True)
t0 = time.time()
for args in (["lsa"], ["analyse", "lsa256", "steam", "arxiv", "patents"]):
    print(time.strftime("%H:%M:%S"), "running", args, flush=True)
    subprocess.run([sys.executable, f"{W}/repo/scripts/review_leakfree_encoder.py", *args], env=env, check=True)
    print(time.strftime("%H:%M:%S"), "finished", args, f"{(time.time() - t0) / 60:.1f} min", flush=True)
