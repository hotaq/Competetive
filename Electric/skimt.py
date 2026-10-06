import json
nb = json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest2/s6e9-94-6-transformer-and-gbdt-ensemble.ipynb"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown":
        print("--- md", i, "---")
        print(src[:1500])
        print()
    else:
        print("--- code", i, "len", len(src), "---")
        print(src[:800])
        print()
