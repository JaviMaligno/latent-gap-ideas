"""Compile the repo's Lean project (lean/, embedded below at push time) against Mathlib on Kaggle, printing progress.

CPU kernel: Lean does not use GPUs, so the repo's two-T4 rule for GPU kernels does not apply here.
"""
import base64
import os
import subprocess
import sys
import time

FILES = __FILES__
ROOT = "/tmp/proj"  # outside /kaggle/working: the output stays the log, not 2 GB of .lake
THEOREMS = ["inner_split", "landing_le", "ladder_norm_sq", "ladder_inner", "ladder_cos", "sum_norm_sub_mean_sq",
            "sum_directional_variance", "expected_max_le", "expected_max_le_of_bounds"] + [
    f"LatentGap.Bounds.{t}" for t in ("cap_dual", "unit_distance", "anchor_decomposition", "normalized_mean_score",
                                      "quadratic_young", "ellipsoid_ball_dual", "covering_measure_le")]
T0 = time.time()


def run(cmd: str) -> int:
    print(f"[{time.time() - T0:6.0f}s] $ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "PATH": os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"]})
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    return p.wait()


for path, data in FILES.items():
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "wb").write(base64.b64decode(data))
print(f"{len(FILES)} project files written", flush=True)
subprocess.run("curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y "
               "--default-toolchain none", shell=True, check=True)
run("lake --version")
run("lake exe cache get 2>&1 | grep -vE '^(Downloading|Decompressing|Unpacking)' | tail -20")
code = run("lake build 2>&1")
open(os.path.join(ROOT, "Axioms.lean"), "w").write("import LatentGap\n" + "\n".join(f"#print axioms {t}" for t in THEOREMS) + "\n")
run("lake env lean Axioms.lean 2>&1")
print(f"[{time.time() - T0:6.0f}s] LAKE BUILD EXIT {code}", flush=True)
sys.exit(0)
