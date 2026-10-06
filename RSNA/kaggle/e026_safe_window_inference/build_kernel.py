#!/usr/bin/env python3
"""Build E026 from the exact E020 inference source with one windowing change."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
OUTPUT = HERE / "e026-single-coatnet-safe-windows.py"


SAFE_EVAL_CENTERS = '''def _eval_centers(mask, D, k):
    """Resample centers only where the full 2.5D window stays in one series slot.

    ``build_study`` assigns every acquisition to one fixed contiguous slot and never
    reuses a SeriesInstanceUID.  Requiring all three mask values and keeping the
    center inside one slot is therefore equivalent to E025's same-SeriesInstanceUID
    predicate, without changing pixels, weights, normalization, or model code.
    """
    safe = []
    cursor = 0
    for _, _, slot_depth in SLOTS:
        lo = cursor
        hi = min(cursor + int(slot_depth), D)
        safe.extend(
            center
            for center in range(lo + 1, hi - 1)
            if bool(mask[center - 1:center + 2].all())
        )
        cursor += int(slot_depth)
        if cursor >= D:
            break

    # Preserve E020's no-study-dropped contract for a pathological all-missing
    # study.  Normal studies use only the series-safe branch above.
    if not safe:
        valid = np.where(mask > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, D))
        lo, hi = int(valid.min()), int(valid.max())
        safe = [
            center
            for center in range(lo + 1, hi)
            if center - 1 >= lo and center + 1 <= hi
        ]
        if not safe:
            safe = [max(1, min((lo + hi) // 2, D - 2))]

    indexes = np.linspace(0, len(safe) - 1, k).round().astype(int)
    return [safe[index] for index in indexes]
'''


def build() -> None:
    source = SOURCE.read_text()
    pattern = re.compile(
        r"def _eval_centers\(mask, D, k\):\n.*?(?=\ndef eval_windows\(vol, mask, k, res, norm=NORM\):)",
        re.DOTALL,
    )
    transformed, replacements = pattern.subn(SAFE_EVAL_CENTERS.rstrip() + "\n", source)
    if replacements != 1:
        raise RuntimeError(f"expected one _eval_centers replacement, got {replacements}")

    transformed = transformed.replace(
        "Knee MRI: twelve findings from a single model",
        "Knee MRI: twelve findings with series-safe 2.5D windows",
        1,
    )
    transformed = transformed.replace(
        "Scoring uses 42 windows per study.",
        "Scoring uses 42 windows per study, resampled only from windows that stay inside one acquisition series.",
        1,
    )
    OUTPUT.write_text(transformed)

    metadata = {
        "id": "chinnaphatkhun/rsna-knee-e026-safe-window-inference",
        "title": "RSNA Knee E026 Safe Window Inference",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["dreaddevelopment/raptor-knee-widedense"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    build()
