"""E042 = E040 (public 0.943 stack, unmodified) + our E041 refined-label student as a 5th Raptor arm.

The student is a 224 px CoAtNet trained on knee-raptor-corpus (44 slices, 336 px, span 0.06-0.94),
the same geometry as the native384-v8 arm except the image size. It joins the Raptor probability
blend at w=0.20, so its share is 0.20/1.20 = 16.7% of the Raptor family (predeclared, not tuned).
It runs on cuda:1 after v10 and v8; GPU0 still carries the shared MaxSpan pair.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "e040_public_0943_fork/e040-public-0943-fork.ipynb"
STUDENT_W = 0.20

nb = json.loads(SRC.read_text())
cell = nb["cells"][22]
s = "".join(cell["source"])


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('''     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]''', f'''     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20}},
    {{"name": "e041-refined-student", "file": "raptor_ft_e041_refined.pt",
     "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224,
     "img": 336, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": {STUDENT_W}}},
]''')
sub("""            run_single(3, torch.device('cuda:1'))
""", """            run_single(3, torch.device('cuda:1'))
            run_single(4, torch.device('cuda:1'))
""")
cell["source"] = s

# The asset finder only walks an explicit root list; register the student dataset there.
c1 = "".join(nb["cells"][1]["source"])
old = "ASSET_ROOTS = ["
assert c1.count(old) == 1
nb["cells"][1]["source"] = c1.replace(old, old + "'/kaggle/input/datasets/chinnaphatkhun/rsna-knee-e042-student', "
                                      "'/kaggle/input/rsna-knee-e042-student', ")
(HERE / "e042-stack-plus-student.ipynb").write_text(json.dumps(nb, indent=1))

meta = json.loads((HERE.parent / "e040_public_0943_fork/kernel-metadata.json").read_text())
meta.update(id="chinnaphatkhun/rsna-knee-e042-stack-plus-student", title="RSNA Knee E042 Stack Plus Student",
            code_file="e042-stack-plus-student.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + ["chinnaphatkhun/rsna-knee-e042-student"]
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote e042-stack-plus-student.ipynb + kernel-metadata.json")
