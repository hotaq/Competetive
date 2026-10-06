import json
nb = json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest2/s6e9-94-6-transformer-and-gbdt-ensemble.ipynb"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    low = src.lower()
    if c.get("cell_type") == "markdown":
        print("--- md", i, "---")
        print(src[:1000])
        print()
    elif any(k in low for k in ["transformer", "torch", "nn.module", "attention", "auc", "score", "blend", "weight"]):
        print("--- code", i, "len", len(src), "---")
        print(src[:2500])
        print()
