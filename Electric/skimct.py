import json
nb=json.load(open("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest/s6e9-ctboost-not-catboost-astra-baseline.ipynb"))
for i,c in enumerate(nb["cells"]):
    if c.get("cell_type")=="code":
        src="".join(c.get("source",[]))
        print(f"===== code {i} len={len(src)} =====")
        print(src[:4500])
        print()
