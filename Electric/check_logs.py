import subprocess, json
p = subprocess.run(['kaggle','kernels','logs','chinnaphatkhun/ev-s6e9-push-v2'],capture_output=True, text=True)
raw = p.stdout + p.stderr
# logs come as comma-separated json records; try to parse each {..}
import re
recs = re.findall(r'\{[^{}]*?"data"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
def unesc(s):
    return s.encode().decode('unicode_escape', errors='ignore')[-300:]
keys = ['INPUT ROOT', 'CANDS', 'USING', 'Error', 'error', 'Traceback', 'FileNotFound', 'ModuleNot', 'ImportError', 'Memory', 'Killed', 'hela']
for r in recs:
    t = unesc(r)
    if any(k in t for k in keys):
        print(t)
print('---- total records:', len(recs))
