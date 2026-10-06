# E040 — unmodified fork of the public 0.943 stack

Goal: reset the anchor from E026 (0.927) to the public 0.943 floor before trying any lever on top.

- Upstream: `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original`, latest version pulled 2026-09-25
  (last run 2026-09-22). The author reports 0.943 for v34 (`scriptVersionId=351665011`). The CLI cannot pull
  by version (403), so we cannot prove that latest == v34. The recipe matches v34 as described by
  `jiweiliu/rsna-knee-fast-2xt4-inference` v9: four CoAt readers at 0.25 each, rank of the probability mean,
  CoAt/Raptor 0.40/0.60, and Raptor maxspan/rev/dense/native at 0.60/0.10/0.10/0.20.
  **The LB score of this fork is the test.**
- Snapshot: `upstream/` (notebook sha256 `2fc5a420…c9489`). Our notebook is byte-identical in code.
- Inputs: 14 datasets, 2 kernel outputs and DINOv2-small, plus the docker image pinned to upstream's digest.
  Dataset versions are not pinnable from the CLI. The notebook's own artifact checks pin checkpoint hashes and
  stop the run if an asset drifts.
- Licences: CC0, except RadImageNet-derived heads (CC-BY-NC-SA-4.0) and prvsiyan (`other`).
  Rules §6 allow public no-cost external models, and §5.a.4 exempts incompatible-licence models from the
  winner licence.
- Runtime: upstream says it runs "comfortably in less than 9 hrs" on 2×T4. That is the hidden-rerun limit.

```bash
python3 kaggle/e040_public_0943_fork/smoke_test.py
kaggle kernels push -p kaggle/e040_public_0943_fork
kaggle competitions submit -c rsna-knee-abnormality-detection -k chinnaphatkhun/rsna-knee-e040-public-0943-fork -v <N> -f submission.csv -m "E040 ..."
```
