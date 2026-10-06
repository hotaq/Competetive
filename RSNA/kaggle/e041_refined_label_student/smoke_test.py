"""Pre-push check: assets match their manifest, the trainer is the generated one, the launcher parses."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
man = json.loads((A / "e041_label_manifest.json").read_text())
for name, digest in man["sha256"].items():
    assert hashlib.sha256((A / name).read_bytes()).hexdigest() == digest, f"{name} drifted from manifest"
assert man["refined_rows"] == 4349 and abs(man["gold58_mix_auc"] - 0.9282) < 5e-4

before = (A / "train_e041.py").read_bytes()
subprocess.run([sys.executable, str(HERE / "make_trainer.py")], check=True, capture_output=True)
assert (A / "train_e041.py").read_bytes() == before, "assets/train_e041.py is stale; rerun make_trainer.py"

launcher = (HERE / "e041-launcher.py").read_text()
ast.parse(launcher)
meta = json.loads((HERE / "kernel-metadata.json").read_text())
assert meta["code_file"] == "e041-launcher.py" and meta["is_private"] and meta["enable_internet"]
assert "chinnaphatkhun/rsna-knee-e041-assets" in meta["dataset_sources"]
for arm in ("labels_e041_refined.parquet", "labels_e041_control.parquet"):
    assert arm in launcher and (A / arm).is_file()
mode = launcher.split('MODE = "')[1].split('"')[0]
print(f"OK: assets match manifest, trainer current, launcher MODE={mode}")
