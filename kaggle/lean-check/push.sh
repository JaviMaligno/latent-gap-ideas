#!/usr/bin/env bash
# Embed lean/ (sources and lake config, not .lake) into check.py and push the CPU kernel.
set -eu
cd "$(dirname "$0")"
python3 - <<'PY'
import base64, json, pathlib
root = pathlib.Path("../../lean")
files = {str(p.relative_to(root)): base64.b64encode(p.read_bytes()).decode()
         for p in root.rglob("*") if p.is_file() and ".lake" not in p.parts and p.suffix in (".lean", ".toml", ".json", "")
         and p.name != "README.md"}
pathlib.Path("check.py").write_text(pathlib.Path("template.py").read_text().replace("__FILES__", json.dumps(files)))
print(sorted(files))
PY
kaggle kernels push -p .
