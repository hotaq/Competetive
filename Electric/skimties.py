import json
nb = json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest3/s6e9-does-breaking-ties-help.ipynb"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown":
        print("--- md", i, "---")
        print(src[:2000])
        print()
    elif len(src) < 1200:
        print("--- code", i, "---")
        print(src[:1200])
        print()
    else:
        print("--- code", i, "len", len(src), "(skip)")
