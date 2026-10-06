import json
nb = json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest3/0-94650-why-breaking-ties-wins-s6e9-guide.ipynb"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown":
        print("--- md", i, "---")
        print(src[:2200])
        print()
