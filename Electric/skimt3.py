import json
nb = json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest2/ev-adoption-tokens-xgboost-tinytokentransformer.ipynb"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    low = src.lower()
    if c.get("cell_type") == "code" and any(k in low for k in ["class tiny", "nn.module", "def train_", "def make_", "dataloader", "auc", "oof"]):
        print("===== code", i, "len", len(src), "=====")
        print(src[:3000])
        print()
