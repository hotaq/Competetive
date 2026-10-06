"""E040 smoke test: our notebook is the upstream 0.943 notebook, unmodified, with the same inputs."""
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
UPSTREAM_NB = HERE / "upstream/mattiaangeli-speedy-raptors-latest-20260925.ipynb"
UPSTREAM_SHA256 = "2fc5a4206d76c13a2ef79a3943a787f13713f873443b535b8471eb3141dc9489"
SAME_KEYS = ("dataset_sources", "kernel_sources", "model_sources", "competition_sources",
             "docker_image", "machine_shape", "enable_gpu", "enable_internet")


def code(nb_path):
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


assert hashlib.sha256(UPSTREAM_NB.read_bytes()).hexdigest() == UPSTREAM_SHA256, "upstream snapshot changed"

meta = json.loads((HERE / "kernel-metadata.json").read_text())
ours = code(HERE / meta["code_file"])
assert ours == code(UPSTREAM_NB), "notebook code drifted from upstream"
for i, cell in enumerate(ours):
    ast.parse(cell, filename=f"cell{i}")

up = json.loads((HERE / "upstream/kernel-metadata.json").read_text())
for k in SAME_KEYS:
    assert meta[k] == up[k], f"metadata {k} differs from upstream"
assert meta["id"].startswith("chinnaphatkhun/") and meta["is_private"] is True

# Every /kaggle/input/<slug> the code reads must be mounted by a declared source.
mounted = {s.split("/")[1] for s in meta["dataset_sources"] + meta["kernel_sources"]}
mounted |= {meta["model_sources"][0].split("/")[1], *meta["competition_sources"],
            "datasets", "notebooks", "models", "competitions"}
used = set(re.findall(r"/kaggle/input/([A-Za-z0-9_.-]+)", "\n".join(ours)))
assert used <= mounted, f"unmounted inputs: {used - mounted}"

print(f"OK: {len(ours)} code cells identical to upstream and compile; {len(used)} input roots mounted")
