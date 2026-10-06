"""E041: train the refined-label and control-label students side by side, one arm per T4.

Same trainer, seed and recipe on both GPUs; only the --labels parquet differs, so the pair is
a clean A/B of the label lever. Both judge on Gold58 at k_eval 42, like the v4 anchor (0.9174).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = "full"  # "smoke": 300 studies x 1 epoch, throughput + fp16 check; "full": all 4,349 x 16 epochs
EXTRA = {"smoke": ["--limit", "300", "--epochs", "1"], "full": ["--epochs", "16"]}[MODE]
ARMS = [("e041_refined", "labels_e041_refined.parquet"), ("e041_control", "labels_e041_control.parquet")]
WORK = Path("/kaggle/working")
ASSETS = next(p for p in (Path("/kaggle/input/rsna-knee-e041-assets"),
                          Path("/kaggle/input/datasets/chinnaphatkhun/rsna-knee-e041-assets"))
              if (p / "train_e041.py").is_file())
print(f"E041 {MODE} | assets {ASSETS}", flush=True)

procs = []
for gpu, (tag, labels) in enumerate(ARMS):
    cmd = [sys.executable, "-u", str(ASSETS / "train_e041.py"), "--labels", str(ASSETS / labels),
           "--tag", tag, "--bs", "8", "--max_hours", "10", *EXTRA]
    log = open(WORK / f"{tag}.log", "w")
    procs.append((tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})))

t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(120)
    for tag, _ in procs:
        tail = (WORK / f"{tag}.log").read_text().strip().splitlines()[-1:] or [""]
        print(f"[{(time.time() - t0) / 60:6.1f} min] {tag}: {tail[0][:200]}", flush=True)

for tag, p in procs:
    print(f"\n===== {tag} exit {p.returncode}\n" + (WORK / f"{tag}.log").read_text()[-4000:], flush=True)
sys.exit(max(p.returncode for _, p in procs))
