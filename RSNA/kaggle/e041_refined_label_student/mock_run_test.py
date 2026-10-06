"""CPU wiring test for assets/train_e041.py: fake 12-study corpus, random-init resnet18, 1 epoch.

Also checks that the real 224 arch builds with 1024 features and grad checkpointing.
Run with a python that has torch + timm:  python mock_run_test.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

HERE = Path(__file__).resolve().parent
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

work = Path(tempfile.mkdtemp())
run = work / "run"
run.mkdir()
shutil.copy(HERE / "assets/train_e041.py", run / "train_e041.py")
rng = np.random.default_rng(0)
ids = np.array([f"1.2.{i}" for i in range(12)])
np.save(run / "all_vols.npy", rng.integers(0, 255, (12, 8, 64, 64), dtype=np.uint8))
np.save(run / "all_masks.npy", np.ones((12, 8), dtype=np.uint8))
np.save(run / "all_ids.npy", ids)
gold = pd.DataFrame(rng.integers(0, 2, (4, 12)), columns=LAB)
gold.loc[0], gold.loc[1] = 0, 1  # both classes present for every target
gold.insert(0, "StudyInstanceUID", ids[:4])
train = pd.concat([gold, pd.DataFrame({"StudyInstanceUID": ids[4:]})], ignore_index=True)
train.to_csv(work / "train.csv", index=False)  # trainer looks in dirname(HERE)
labels = pd.DataFrame(rng.uniform(0.02, 0.98, (12, 12)), columns=LAB)
labels.insert(0, "StudyInstanceUID", ids)
labels.to_parquet(work / "labels.parquet", index=False)

cmd = [sys.executable, str(run / "train_e041.py"), "--labels", str(work / "labels.parquet"),
       "--ckpt", "none", "--arch", "resnet18", "--res", "64", "--epochs", "2", "--bs", "2",
       "--k", "3", "--k_eval", "3", "--workers", "0", "--tag", "mock", "--max_hours", "0"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=run)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "trainer crashed"
assert "train 8 | gold-val 4" in r.stdout, "gold studies must be excluded from training"
assert "[guard]" in r.stdout and "ep1 " not in r.stdout, "max_hours guard must stop after ep0"
ck = torch.load(run / "raptor_ft_mock.pt", map_location="cpu", weights_only=False)
assert {"model", "arch", "res", "lab"} <= set(ck) and ck["res"] == 64 and ck["lab"] == LAB

bb = timm.create_model("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", pretrained=False, num_classes=0, global_pool="avg")
bb.set_grad_checkpointing(True)
bb.train()
f = bb(torch.rand(2, 3, 224, 224))
assert f.shape == (2, 1024) and bb.num_features == 1024
f.sum().backward()
shutil.rmtree(work)
print("OK: trainer wiring, gold exclusion, time guard, checkpoint format, 224 arch forward/backward")
