import subprocess, re, sys
slug = 'chinnaphatkhun/' + sys.argv[1]
p = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
print('STATUS:', (p.stdout + p.stderr).strip()[-200:])
p = subprocess.run(['kaggle', 'kernels', 'logs', slug], capture_output=True, text=True)
raw = p.stdout + p.stderr
recs = re.findall(r'\{[^{}]*?"data"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
def unesc(s):
    return s.encode().decode('unicode_escape', errors='ignore')
keys = ['>>> train', '<<<', 'BLEND', 'Error', 'error', 'Traceback', 'FileNotFound', 'ModuleNot', 'ImportError', 'Memory', 'Killed', 'prior models', 'matrices', 'batch:']
shown = 0
for r in recs:
    t = unesc(r).strip()
    if any(k in t for k in keys):
        print(t[-400:])
        shown += 1
print('---- matched lines:', shown, '/', len(recs))
