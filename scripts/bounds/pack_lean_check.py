"""Refresh the existing Kaggle CPU Lean checker from current repo sources (no push)."""
import ast
import base64
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
CHECK=ROOT/'kaggle/lean-check/check.py'
s=CHECK.read_text()
assignment=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
old=ast.literal_eval(assignment.value)
files={name:base64.b64encode((ROOT/'lean'/name).read_bytes()).decode() for name in old}
s,n=re.subn(r'^FILES = .*$',lambda _: 'FILES = '+json.dumps(files),s,count=1,flags=re.M)
assert n==1
s=re.sub(r'# bounds2 axioms\n.*?# end bounds2 axioms\n','',s,flags=re.S)
extra=['unit_support_le_one','antipodal_ceiling','two_point_latitude','threshold_count_sandwich']
block='# bounds2 axioms\nTHEOREMS += '+repr(['LatentGap.Bounds.'+x for x in extra])+'\n# end bounds2 axioms\n'
s=s.replace('T0 = time.time()',block+'T0 = time.time()')
CHECK.write_text(s)
print('Updated CPU checker sources and four axiom checks; no kernel pushed.')
