ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
lib = open(ROOT + "/ensemble_farm/farm_lib.py").read()
body = open(ROOT + "/farm_tt/train.py").read()
cut = 'HERE = os.path.dirname(os.path.abspath(__file__))'
assert cut in body
body = body[:body.index('import sys\n')] + body[body.index('tr = pd.read_csv(TRAIN_P)'):]
assert 'farm_lib' not in body, [l for l in body.splitlines() if 'farm_lib' in l]
out = lib + "\n\n# ================= TT BODY =================\n\n" + body
open(ROOT + "/farm_tt/train.py", "w").write(out)
print("baked KB:", len(out) // 1024)
import ast
ast.parse(out)
print("SYNTAX OK")
