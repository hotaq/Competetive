ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
lib = open(ROOT + "/ensemble_farm/farm_lib.py").read()
body = open(ROOT + "/farm_tabm/train.py").read()
start = body.index('HERE = os.path.dirname')
end = body.index('tr = pd.read_csv(TRAIN_P)')
body = body[:start] + body[end:]
assert 'farm_lib' not in body, [l for l in body.splitlines() if 'farm_lib' in l]
assert 'HERE' not in body
out = lib + "\n\n# ================= TABM BODY =================\n\n" + body
open(ROOT + "/farm_tabm/train.py", "w").write(out)
print("baked KB:", len(out) // 1024)
import ast
ast.parse(out)
print("SYNTAX OK")
