import json, glob
for f in sorted(glob.glob("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/klatest/*.ipynb")):
    nb=json.load(open(f))
    print("="*80)
    print(f.split("/")[-1], "cells=", len(nb["cells"]))
    for i,c in enumerate(nb["cells"]):
        src="".join(c.get("source",[]))
        if c.get("cell_type")=="markdown":
            print(f"--- md {i} ---")
            print(src[:2000])
            print()
