#!/usr/bin/env python3
"""Build the Kaggle-native RSNA knee preprocessing visual audit notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "e023-rsna-knee-preprocess-visual-audit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        r"""
# E023 — RSNA knee MRI preprocessing visual audit

This notebook runs entirely on Kaggle and audits the exact image path used by the
single-checkpoint CoAtNet submission that scored **0.924 public LB**:

`raw DICOM → geometric ordering → 6–94% slice sampling → series-wise 2–98% windowing
→ image-centred 140 mm crop → 336 px stack → 3-slice 2.5D window → 384 px + ImageNet norm`

It is diagnostic only: it does **not** train a model and does **not** create a competition
submission. The output includes visual panels and CSV/JSON evidence for deciding the next
preprocessing experiment.

The high-value questions are:

1. Does the public 44-slice fine-tuning corpus match the 64-slice E020 inference stack?
2. Do any 3-slice RGB windows cross a series/plane boundary instead of using physically
   neighbouring slices?
3. How often are required series slots missing or ambiguous?
4. Does using only `PixelSpacing[0]` distort a nominally square 140 mm crop?
5. How much information is lost by resizing 336 → 384 instead of cropping directly to 384?
"""
    ),
    code(
        r"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import torch
import torch.nn.functional as F
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

IMG = 336
MODEL_RES = 384
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.06, 0.94
K_EVAL = 42
SLOTS = [
    {"slot": "sagittal_fluid", "plane": "Sagittal", "fluid": 1, "count": 18},
    {"slot": "sagittal_other", "plane": "Sagittal", "fluid": 0, "count": 14},
    {"slot": "coronal_fluid", "plane": "Coronal", "fluid": 1, "count": 12},
    {"slot": "coronal_other", "plane": "Coronal", "fluid": 0, "count": 8},
    {"slot": "axial", "plane": "Axial", "fluid": -1, "count": 12},
]
MAXS = sum(s["count"] for s in SLOTS)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Optional: paste one or more StudyInstanceUIDs here. The notebook otherwise chooses
# deterministic representative studies (gold/complete, many-series, and missing-slot).
MANUAL_STUDY_IDS: list[str] = []

print(f"Exact E020 layout: {MAXS} slices at {IMG}px; CoAtNet input {MODEL_RES}px; K_EVAL={K_EVAL}")
"""
    ),
    md("## 1. Discover the Kaggle competition and attached public corpora"),
    code(
        r"""
def find_competition_root() -> Path:
    preferred = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for root in preferred:
        if (root / "train_series.csv").is_file():
            return root
    # Bounded discovery: only inspect immediate Kaggle input children.
    for root in Path("/kaggle/input").iterdir():
        if root.is_dir() and (root / "train_series.csv").is_file():
            return root
    raise FileNotFoundError("Could not find the RSNA Knee competition mount")


ROOT = find_competition_root()
TRAIN_DCM = ROOT / "train_series"
if not TRAIN_DCM.is_dir():
    TRAIN_DCM = ROOT / "train_images"

series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train_path = ROOT / "train.csv"
train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str}) if train_path.is_file() else pd.DataFrame()
if len(train):
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

corpus_candidates = list(Path("/kaggle/input").glob("*"))
corpus_candidates += list(Path("/kaggle/input").glob("datasets/*/*"))
corpus_dirs = sorted(
    {p.resolve() for p in corpus_candidates if p.is_dir() and "corpus" in p.name.lower()},
    key=lambda p: str(p),
)
inventory = pd.DataFrame(
    [
        {"item": "competition root", "value": str(ROOT)},
        {"item": "DICOM root", "value": str(TRAIN_DCM)},
        {"item": "series rows", "value": len(series)},
        {"item": "unique studies", "value": series["StudyInstanceUID"].nunique()},
        {"item": "structured-label rows", "value": len(train)},
        {"item": "attached corpus dirs", "value": ", ".join(p.name for p in corpus_dirs)},
    ]
)
display(inventory)
inventory.to_csv(WORK / "input_inventory.csv", index=False)
display(series.head())
"""
    ),
    md("## 2. Reproduce E020 series-slot selection and measure coverage"),
    code(
        r"""
def fluid_value(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def choose_slot(rows: list[dict], spec: dict, used: set[str]):
    candidates = [
        r for r in rows
        if str(r.get("Anatomical_Plane", "")).lower() == spec["plane"].lower()
        and str(r["SeriesInstanceUID"]) not in used
    ]
    preferred = candidates
    if spec["fluid"] in (0, 1):
        exact = [r for r in candidates if fluid_value(r.get("Fluid_Sensitive", 0)) == spec["fluid"]]
        if exact:
            preferred = exact
    return (preferred[0] if preferred else None), len(candidates), len(preferred)


def select_slots(frame: pd.DataFrame):
    rows = frame.to_dict("records")
    used: set[str] = set()
    selected = []
    for spec in SLOTS:
        row, n_plane, n_preferred = choose_slot(rows, spec, used)
        if row is not None:
            used.add(str(row["SeriesInstanceUID"]))
        selected.append({"spec": spec, "row": row, "plane_candidates": n_plane, "preferred_candidates": n_preferred})
    return selected


coverage_rows = []
selection_cache = {}
for uid, frame in series.groupby("StudyInstanceUID", sort=False):
    picked = select_slots(frame)
    selection_cache[str(uid)] = picked
    row = {
        "StudyInstanceUID": str(uid),
        "n_series": len(frame),
        "slots_present": sum(x["row"] is not None for x in picked),
        "ambiguous_slots": sum(x["preferred_candidates"] > 1 for x in picked),
    }
    for x in picked:
        row[x["spec"]["slot"]] = int(x["row"] is not None)
    coverage_rows.append(row)

coverage = pd.DataFrame(coverage_rows)
slot_summary = pd.DataFrame(
    {
        "slot": [s["slot"] for s in SLOTS],
        "present_rate": [coverage[s["slot"]].mean() for s in SLOTS],
        "missing_studies": [int((coverage[s["slot"]] == 0).sum()) for s in SLOTS],
    }
)
display(slot_summary.style.format({"present_rate": "{:.2%}"}))
print("Studies with all five slots:", int((coverage["slots_present"] == 5).sum()), "/", len(coverage))
print("Studies with at least one ambiguous slot:", int((coverage["ambiguous_slots"] > 0).sum()))
coverage.to_csv(WORK / "study_slot_coverage.csv", index=False)
slot_summary.to_csv(WORK / "slot_coverage_summary.csv", index=False)


def stable_key(uid: str):
    return hashlib.sha1(uid.encode()).hexdigest()


known = set(coverage["StudyInstanceUID"])
chosen = [str(x) for x in MANUAL_STUDY_IDS if str(x) in known]
gold_ids = set(train["StudyInstanceUID"].astype(str)) if len(train) and "StudyInstanceUID" in train else set()
complete = coverage[coverage["slots_present"] == 5].copy()
gold_complete = complete[complete["StudyInstanceUID"].isin(gold_ids)].copy()
if len(gold_complete):
    chosen.append(sorted(gold_complete["StudyInstanceUID"], key=stable_key)[0])
if len(complete):
    many = complete.sort_values(["n_series", "ambiguous_slots"], ascending=False).iloc[0]["StudyInstanceUID"]
    chosen.append(str(many))
missing = coverage[(coverage["slots_present"] < 5) & (coverage["slots_present"] >= 3)].copy()
if len(missing):
    chosen.append(str(missing.sort_values(["slots_present", "n_series"], ascending=False).iloc[0]["StudyInstanceUID"]))
if len(chosen) < 3:
    chosen.extend(sorted(complete["StudyInstanceUID"], key=stable_key)[:3])
selected_uids = list(dict.fromkeys(chosen))[:3]

print("Selected studies:", selected_uids)
display(coverage[coverage["StudyInstanceUID"].isin(selected_uids)])
"""
    ),
    md("## 3. Exact raw-DICOM preprocessing, plus provenance for every output slice"),
    code(
        r"""
def dicom_records(series_dir: Path):
    records = []
    spacing_pairs = []
    for file in sorted(series_dir.glob("*.dcm")):
        try:
            header = pydicom.dcmread(str(file), stop_before_pixels=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.asarray(iop[:3], dtype=float)
                c = np.asarray(iop[3:], dtype=float)
                normal = np.cross(r, c)
                position = float(np.dot(np.asarray(ipp, dtype=float), normal))
            else:
                position = float(getattr(header, "InstanceNumber", 0) or 0)
            spacing = getattr(header, "PixelSpacing", [0.5, 0.5])
            ps_row = float(spacing[0])
            ps_col = float(spacing[1]) if len(spacing) > 1 else ps_row
            spacing_pairs.append((ps_row, ps_col))
            records.append(
                {
                    "position": position,
                    "path": file,
                    "ps_row": ps_row,
                    "ps_col": ps_col,
                    "rows": int(getattr(header, "Rows", 0) or 0),
                    "cols": int(getattr(header, "Columns", 0) or 0),
                }
            )
        except Exception:
            continue
    records.sort(key=lambda x: x["position"])
    if spacing_pairs:
        med = np.median(np.asarray(spacing_pairs), axis=0)
        fallback = (float(med[0]), float(med[1]))
    else:
        fallback = (0.5, 0.5)
    return records, fallback


def read_pixels(path: Path) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    arr = apply_modality_lut(dcm.pixel_array, dcm).astype(np.float32)
    if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def exact_crop_bounds(shape, ps_scalar: float):
    # E020 behaviour: PixelSpacing[0] controls both axes; bounds clamp to the image.
    h, w = shape
    half = max(2, int(round(CROP_MM / max(ps_scalar, 1e-3) / 2.0)))
    cy, cx = h // 2, w // 2
    return max(0, cy - half), min(h, cy + half), max(0, cx - half), min(w, cx + half)


def crop_resize_exact(arr01: np.ndarray, ps_scalar: float, out_size=IMG):
    y0, y1, x0, x1 = exact_crop_bounds(arr01.shape, ps_scalar)
    crop = arr01[y0:y1, x0:x1]
    if crop.size == 0:
        crop = arr01
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized, (y0, y1, x0, x1)


def crop_resize_axis_aware(arr01: np.ndarray, ps_row: float, ps_col: float, out_size=IMG):
    # Audit candidate only: independent row/column spacing for a true 140x140 mm box.
    h, w = arr01.shape
    half_y = max(2, int(round(CROP_MM / max(ps_row, 1e-3) / 2.0)))
    half_x = max(2, int(round(CROP_MM / max(ps_col, 1e-3) / 2.0)))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - half_y), min(h, cy + half_y)
    x0, x1 = max(0, cx - half_x), min(w, cx + half_x)
    crop = arr01[y0:y1, x0:x1]
    if crop.size == 0:
        crop = arr01
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA), (y0, y1, x0, x1)


def build_study(uid: str):
    volume = np.zeros((MAXS, IMG, IMG), dtype=np.uint8)
    mask = np.zeros(MAXS, dtype=np.uint8)
    provenance = []
    examples = []
    selected_rows = []
    out_index = 0
    for slot_index, picked in enumerate(selection_cache[uid]):
        spec, row = picked["spec"], picked["row"]
        if row is None:
            for _ in range(spec["count"]):
                provenance.append({"output_index": out_index, "slot_index": slot_index, "slot": spec["slot"], "series_uid": None, "raw_index": None})
                out_index += 1
            selected_rows.append({"StudyInstanceUID": uid, "slot": spec["slot"], "selected": False})
            continue

        series_uid = str(row["SeriesInstanceUID"])
        records, fallback_spacing = dicom_records(TRAIN_DCM / uid / series_uid)
        selected_rows.append(
            {
                "StudyInstanceUID": uid,
                "slot": spec["slot"],
                "selected": True,
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": row.get("Anatomical_Plane"),
                "Fluid_Sensitive": row.get("Fluid_Sensitive"),
                "SeriesDescription": row.get("SeriesDescription", ""),
                "raw_slice_count": len(records),
                "plane_candidates": picked["plane_candidates"],
                "preferred_candidates": picked["preferred_candidates"],
            }
        )
        if not records:
            for _ in range(spec["count"]):
                provenance.append({"output_index": out_index, "slot_index": slot_index, "slot": spec["slot"], "series_uid": series_uid, "raw_index": None})
                out_index += 1
            continue

        n = len(records)
        lo = int(n * SPAN_LO)
        hi = max(lo, int(n * SPAN_HI) - 1)
        picks = np.linspace(lo, hi, spec["count"]).round().astype(int) if n > 1 else np.zeros(spec["count"], dtype=int)
        arrays = []
        meta = []
        for raw_index in picks:
            rec = records[min(int(raw_index), n - 1)]
            try:
                arr = read_pixels(rec["path"])
            except Exception:
                arr = None
            arrays.append(arr)
            meta.append((int(raw_index), rec))

        valid_arrays = [a for a in arrays if a is not None]
        if valid_arrays:
            all_values = np.concatenate([a.reshape(-1) for a in valid_arrays])
            p02, p98 = np.percentile(all_values, [2.0, 98.0])
        else:
            p02, p98 = 0.0, 1.0

        for local_index, (arr, (raw_index, rec)) in enumerate(zip(arrays, meta)):
            if out_index >= MAXS:
                break
            ps_row = rec.get("ps_row", fallback_spacing[0]) or fallback_spacing[0]
            ps_col = rec.get("ps_col", fallback_spacing[1]) or fallback_spacing[1]
            if arr is not None:
                norm = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0, 1)
                exact, bounds = crop_resize_exact(norm, ps_row, IMG)
                volume[out_index] = np.round(exact * 255).astype(np.uint8)
                mask[out_index] = 1
                if local_index == len(arrays) // 2:
                    axis_aware, axis_bounds = crop_resize_axis_aware(norm, ps_row, ps_col, IMG)
                    examples.append(
                        {
                            "slot": spec["slot"], "series_uid": series_uid, "raw": arr,
                            "norm": norm, "exact": exact, "axis_aware": axis_aware,
                            "bounds": bounds, "axis_bounds": axis_bounds,
                            "ps_row": ps_row, "ps_col": ps_col, "raw_index": raw_index,
                            "raw_count": n, "p02": float(p02), "p98": float(p98),
                        }
                    )
                y0, y1, x0, x1 = bounds
                realized_row_mm = (y1 - y0) * ps_row
                realized_col_mm = (x1 - x0) * ps_col
            else:
                realized_row_mm = realized_col_mm = np.nan
            provenance.append(
                {
                    "output_index": out_index, "slot_index": slot_index, "slot": spec["slot"],
                    "series_uid": series_uid, "raw_index": raw_index, "raw_count": n,
                    "ps_row": ps_row, "ps_col": ps_col,
                    "realized_row_mm": realized_row_mm, "realized_col_mm": realized_col_mm,
                }
            )
            out_index += 1
    return {
        "volume": volume,
        "mask": mask,
        "provenance": pd.DataFrame(provenance),
        "examples": examples,
        "selection": pd.DataFrame(selected_rows),
    }


built = {}
for uid in selected_uids:
    print("Building", uid)
    built[uid] = build_study(uid)

selection_audit = pd.concat([x["selection"] for x in built.values()], ignore_index=True)
provenance_audit = pd.concat(
    [x["provenance"].assign(StudyInstanceUID=uid) for uid, x in built.items()], ignore_index=True
)
selection_audit.to_csv(WORK / "selected_series_audit.csv", index=False)
provenance_audit.to_csv(WORK / "slice_provenance.csv", index=False)
display(selection_audit)
"""
    ),
    md("## 4. Raw series view with the exact 140 mm image-centred crop"),
    code(
        r"""
for uid, item in built.items():
    examples = item["examples"]
    if not examples:
        continue
    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 8), squeeze=False)
    for col, ex in enumerate(examples):
        ax = axes[0, col]
        ax.imshow(ex["norm"], cmap="gray")
        y0, y1, x0, x1 = ex["bounds"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2, label="exact E020"))
        ay0, ay1, ax0, ax1 = ex["axis_bounds"]
        ax.add_patch(Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, fill=False, edgecolor="orange", linewidth=1.5, linestyle="--", label="axis-aware"))
        ax.set_title(f"{ex['slot']}\nraw {ex['raw_index']+1}/{ex['raw_count']} | spacing {ex['ps_row']:.3f}×{ex['ps_col']:.3f} mm")
        ax.axis("off")
        if col == 0:
            ax.legend(loc="lower right", fontsize=7)
        axes[1, col].imshow(ex["exact"], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Exact model crop → 336")
        axes[1, col].axis("off")
    fig.suptitle(f"Study {uid}: raw selected series and exact crop", fontsize=14)
    fig.tight_layout()
    path = WORK / f"raw_crop_{uid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
"""
    ),
    md("## 5. The complete 64-slice stack and its five slot boundaries"),
    code(
        r"""
slot_colors = ["tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:red"]
for uid, item in built.items():
    volume = item["volume"]
    prov = item["provenance"]
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for index, ax in enumerate(axes.flat):
        ax.imshow(volume[index], cmap="gray", vmin=0, vmax=255)
        record = prov.iloc[index]
        color = slot_colors[int(record["slot_index"])]
        ax.set_title(f"{index:02d} {record['slot'][:5]}", fontsize=7, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Study {uid}: exact E020 64-slice stack (colour = source slot)", fontsize=15)
    fig.tight_layout()
    fig.savefig(WORK / f"stack64_{uid}.png", dpi=140, bbox_inches="tight")
    plt.show()
"""
    ),
    md("## 6. Audit 2.5D windows: physically adjacent vs crossing a series boundary"),
    code(
        r"""
def e020_eval_centers(mask: np.ndarray, depth: int, k: int):
    # Verbatim behaviour: mask chooses only min/max; internal holes and boundaries are ignored.
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    centers = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not centers:
        centers = [max(1, min((lo + hi) // 2, depth - 2))]
    positions = np.linspace(0, len(centers) - 1, k).round().astype(int)
    return [centers[i] for i in positions]


boundary_rows = []
for uid, item in built.items():
    prov = item["provenance"]
    centers = e020_eval_centers(item["mask"], MAXS, K_EVAL)
    classes = []
    for center in centers:
        tri = prov.iloc[[center - 1, center, center + 1]]
        series_ids = [x for x in tri["series_uid"].tolist() if pd.notna(x)]
        slots = tri["slot_index"].tolist()
        crosses = len(set(series_ids)) > 1 or len(set(slots)) > 1
        contains_zero = bool((item["mask"][[center - 1, center, center + 1]] == 0).any())
        classes.append((center, crosses, contains_zero))
        boundary_rows.append(
            {
                "StudyInstanceUID": uid, "center": center, "crosses_series_or_slot": crosses,
                "contains_zero_slice": contains_zero,
                "left_slot": tri.iloc[0]["slot"], "center_slot": tri.iloc[1]["slot"], "right_slot": tri.iloc[2]["slot"],
            }
        )

    normal = next((x for x in classes if not x[1] and not x[2]), None)
    boundary = next((x for x in classes if x[1]), None)
    zero = next((x for x in classes if x[2]), None)
    examples = [("physically adjacent", normal), ("cross-series/slot", boundary), ("contains missing zero", zero)]
    examples = [(name, x) for name, x in examples if x is not None]
    if examples:
        fig, axes = plt.subplots(2, len(examples), figsize=(5 * len(examples), 8), squeeze=False)
        for col, (name, info) in enumerate(examples):
            center = info[0]
            tri = item["volume"][[center - 1, center, center + 1]].astype(np.float32) / 255.0
            rgb = np.moveaxis(tri, 0, -1)
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"{name}\ncenter={center}, RGB = slices {center-1}/{center}/{center+1}")
            axes[0, col].axis("off")
            axes[1, col].plot(tri.reshape(3, -1).mean(1), marker="o", label="mean")
            axes[1, col].plot(tri.reshape(3, -1).std(1), marker="s", label="std")
            axes[1, col].set_xticks([0, 1, 2], ["R/left", "G/centre", "B/right"])
            axes[1, col].set_ylim(0, 1)
            axes[1, col].legend()
            axes[1, col].grid(alpha=0.25)
        fig.suptitle(f"Study {uid}: actual E020 2.5D window classes", fontsize=14)
        fig.tight_layout()
        fig.savefig(WORK / f"window_boundary_{uid}.png", dpi=150, bbox_inches="tight")
        plt.show()

boundary_audit = pd.DataFrame(boundary_rows)
boundary_summary = boundary_audit.groupby("StudyInstanceUID")[["crosses_series_or_slot", "contains_zero_slice"]].agg(["sum", "mean"])
display(boundary_summary)
boundary_audit.to_csv(WORK / "window_boundary_audit.csv", index=False)
"""
    ),
    md("## 7. Compare the public 44-slice fine-tuning corpus with rebuilt E020 64-slice inputs"),
    code(
        r"""
def corpus_file(name: str):
    for directory in corpus_dirs:
        direct = directory / name
        if direct.is_file():
            return direct
        # Dataset mounts may include a version subdirectory. Recursion is bounded to
        # the two small corpus metadata trees, never the competition DICOM tree.
        hits = list(directory.rglob(name))
        if hits:
            return hits[0]
    return None


part_specs = [
    (corpus_file("all_ids.npy"), corpus_file("all_vols.npy"), corpus_file("all_masks.npy")),
    (corpus_file("extra_ids.npy"), corpus_file("extra_vols.npy"), corpus_file("extra_masks.npy")),
]
corpus_parts = []
for ids_path, vols_path, masks_path in part_specs:
    if ids_path and vols_path and masks_path:
        ids = [str(x) for x in np.load(ids_path, allow_pickle=True).tolist()]
        corpus_parts.append(
            {
                "ids": ids, "lookup": {uid: i for i, uid in enumerate(ids)},
                "vols": np.load(vols_path, mmap_mode="r"),
                "masks": np.load(masks_path, mmap_mode="r"),
                "vols_path": str(vols_path),
            }
        )


def public_corpus_row(uid: str):
    for part in corpus_parts:
        if uid in part["lookup"]:
            row = part["lookup"][uid]
            return np.asarray(part["vols"][row]), np.asarray(part["masks"][row]), part["vols_path"]
    return None


corpus_shapes = [tuple(p["vols"].shape) for p in corpus_parts]
print("Attached public corpus shapes:", corpus_shapes)
comparison_rows = []
for uid, item in built.items():
    public = public_corpus_row(uid)
    if public is None:
        print("UID absent from attached corpus:", uid)
        continue
    public_vol, public_mask, source = public
    rebuilt = item["volume"]
    comparison_rows.append(
        {
            "StudyInstanceUID": uid, "public_depth": public_vol.shape[0], "rebuilt_depth": rebuilt.shape[0],
            "public_valid": int((public_mask > 0).sum()), "rebuilt_valid": int(item["mask"].sum()),
            "public_mean": float(public_vol.mean()), "rebuilt_mean": float(rebuilt.mean()),
            "public_std": float(public_vol.std()), "rebuilt_std": float(rebuilt.std()), "source": source,
        }
    )
    fig, axes = plt.subplots(2, 12, figsize=(22, 5))
    for row_index, (label, vol, mask) in enumerate(
        [("public fine-tune corpus", public_vol, public_mask), ("rebuilt E020 inference", rebuilt, item["mask"])]
    ):
        valid = np.where(mask > 0)[0]
        picks = np.linspace(valid.min(), valid.max(), 12).round().astype(int) if len(valid) else np.arange(12)
        for col, index in enumerate(picks):
            axes[row_index, col].imshow(vol[int(index)], cmap="gray", vmin=0, vmax=255)
            axes[row_index, col].set_title(f"{label}\nindex {int(index)}", fontsize=7)
            axes[row_index, col].axis("off")
    fig.suptitle(f"Study {uid}: 44-slice public training input vs 64-slice E020 inference input", fontsize=14)
    fig.tight_layout()
    fig.savefig(WORK / f"corpus44_vs_stack64_{uid}.png", dpi=150, bbox_inches="tight")
    plt.show()

corpus_comparison = pd.DataFrame(comparison_rows)
display(corpus_comparison)
corpus_comparison.to_csv(WORK / "corpus44_vs_stack64.csv", index=False)
"""
    ),
    md("## 8. Quantify crop-spacing assumptions and the 336 → 384 second resize"),
    code(
        r"""
# Bounded header-only sample: enough to reveal spacing/crop issues without scanning the full DICOM tree.
header_rows = []
sample_series = series.assign(_key=series["SeriesInstanceUID"].map(stable_key)).sort_values("_key").head(256)
for _, row in sample_series.iterrows():
    uid, sid = str(row["StudyInstanceUID"]), str(row["SeriesInstanceUID"])
    files = sorted((TRAIN_DCM / uid / sid).glob("*.dcm"))
    if not files:
        continue
    try:
        h = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True)
        spacing = getattr(h, "PixelSpacing", [0.5, 0.5])
        ps_row = float(spacing[0]); ps_col = float(spacing[1] if len(spacing) > 1 else spacing[0])
        rows = int(getattr(h, "Rows", 0)); cols = int(getattr(h, "Columns", 0))
        y0, y1, x0, x1 = exact_crop_bounds((rows, cols), ps_row)
        header_rows.append(
            {
                "StudyInstanceUID": uid, "SeriesInstanceUID": sid,
                "plane": row.get("Anatomical_Plane"), "ps_row": ps_row, "ps_col": ps_col,
                "spacing_ratio": max(ps_row, ps_col) / max(min(ps_row, ps_col), 1e-6),
                "rows": rows, "cols": cols,
                "exact_row_fov_mm": (y1 - y0) * ps_row,
                "exact_col_fov_mm": (x1 - x0) * ps_col,
                "row_crop_truncated": (y1 - y0) * ps_row < CROP_MM * 0.98,
                "col_crop_truncated": (x1 - x0) * ps_col < CROP_MM * 0.98,
            }
        )
    except Exception:
        pass

spacing_audit = pd.DataFrame(header_rows)
spacing_audit.to_csv(WORK / "spacing_crop_audit.csv", index=False)
if len(spacing_audit):
    display(spacing_audit[["ps_row", "ps_col", "spacing_ratio", "exact_row_fov_mm", "exact_col_fov_mm"]].describe(percentiles=[.01, .05, .5, .95, .99]))

resize_rows = []
for uid, item in built.items():
    for ex in item["examples"]:
        direct384, _ = crop_resize_exact(ex["norm"], ex["ps_row"], MODEL_RES)
        direct384 = np.asarray(direct384, dtype=np.float32)
        two_stage384 = cv2.resize(ex["exact"], (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_LINEAR)
        two_stage384 = np.asarray(two_stage384, dtype=np.float32)
        diff = np.abs(direct384 - two_stage384)
        direct_edge = cv2.Laplacian(direct384, cv2.CV_32F).var()
        staged_edge = cv2.Laplacian(two_stage384, cv2.CV_32F).var()
        resize_rows.append(
            {
                "StudyInstanceUID": uid, "slot": ex["slot"],
                "mean_abs_difference": float(diff.mean()), "p99_abs_difference": float(np.percentile(diff, 99)),
                "direct384_edge_variance": float(direct_edge), "two_stage_edge_variance": float(staged_edge),
                "edge_ratio_two_stage_over_direct": float(staged_edge / (direct_edge + 1e-8)),
            }
        )

resize_audit = pd.DataFrame(resize_rows)
resize_audit.to_csv(WORK / "resize336_to384_audit.csv", index=False)
display(resize_audit)

first_uid = selected_uids[0]
if built[first_uid]["examples"]:
    ex = built[first_uid]["examples"][0]
    direct384, _ = crop_resize_exact(ex["norm"], ex["ps_row"], MODEL_RES)
    direct384 = np.asarray(direct384, dtype=np.float32)
    staged384 = cv2.resize(ex["exact"], (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_LINEAR)
    staged384 = np.asarray(staged384, dtype=np.float32)
    axis_aware = ex["axis_aware"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    panels = [
        (ex["norm"], "Series 2–98% normalized"),
        (ex["exact"], "Exact crop → 336"),
        (staged384, "Exact path 336 → 384"),
        (direct384, "Audit candidate: direct → 384"),
        (np.abs(direct384 - staged384), "Absolute resize difference"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image, cmap="magma" if "difference" in title.lower() else "gray")
        ax.set_title(title); ax.axis("off")
    fig.suptitle(f"Study {first_uid}, {ex['slot']}: exact and candidate resize paths")
    fig.tight_layout()
    fig.savefig(WORK / "resize_path_comparison.png", dpi=160, bbox_inches="tight")
    plt.show()
"""
    ),
    md("## 9. Automated gap report and next experiment"),
    code(
        r"""
public_depths = sorted({shape[1] for shape in corpus_shapes}) if corpus_shapes else []
cross_rate = float(boundary_audit["crosses_series_or_slot"].mean()) if len(boundary_audit) else np.nan
zero_rate = float(boundary_audit["contains_zero_slice"].mean()) if len(boundary_audit) else np.nan
missing_rate = float((coverage["slots_present"] < 5).mean())
ambiguity_rate = float((coverage["ambiguous_slots"] > 0).mean())
anisotropic_rate = float((spacing_audit["spacing_ratio"] > 1.01).mean()) if len(spacing_audit) else np.nan
truncated_rate = float((spacing_audit[["row_crop_truncated", "col_crop_truncated"]].any(axis=1)).mean()) if len(spacing_audit) else np.nan
edge_ratio = float(resize_audit["edge_ratio_two_stage_over_direct"].median()) if len(resize_audit) else np.nan

gaps = [
    {
        "severity": "HIGH" if public_depths and public_depths != [64] else "PASS",
        "check": "Warm fine-tune stack depth matches E020 inference",
        "evidence": f"attached corpus depths={public_depths}; E020 inference depth=64",
        "impact": "E022 adapts the checkpoint on a different slice allocation and context distribution.",
        "next_experiment": "Rebuild a 64-slice training corpus inside Kaggle with the exact E020 slot/sampling code before full warm fine-tuning.",
    },
    {
        "severity": "HIGH" if np.isfinite(cross_rate) and cross_rate > 0 else "PASS",
        "check": "Every 2.5D RGB window uses one physical series",
        "evidence": f"{cross_rate:.1%} of sampled E020 windows cross a series/slot boundary" if np.isfinite(cross_rate) else "not measured",
        "impact": "RGB channels can represent different planes or pulse sequences rather than neighbouring anatomy.",
        "next_experiment": "Carry a series-id vector and sample centres only when c-1,c,c+1 share the same SeriesInstanceUID; retrain/fine-tune with the same rule.",
    },
    {
        "severity": "HIGH" if np.isfinite(zero_rate) and zero_rate > 0.02 else ("MEDIUM" if np.isfinite(zero_rate) and zero_rate > 0 else "PASS"),
        "check": "Evaluation windows exclude missing zero-filled slots",
        "evidence": f"{zero_rate:.1%} of sampled windows contain at least one zero slice" if np.isfinite(zero_rate) else "not measured",
        "impact": "The current centre generator uses only min/max valid indices and can traverse internal mask holes.",
        "next_experiment": "Require all three mask entries to be valid when constructing window centres.",
    },
    {
        "severity": "MEDIUM" if missing_rate > 0.02 else "LOW",
        "check": "Five requested series slots exist",
        "evidence": f"{missing_rate:.1%} of studies miss at least one slot",
        "impact": "Missing sequences change both context count and window semantics.",
        "next_experiment": "Use explicit missing-slot tokens/masks in the pooling head and prevent windows spanning gaps.",
    },
    {
        "severity": "MEDIUM" if ambiguity_rate > 0.10 else "LOW",
        "check": "First matching series is unambiguous",
        "evidence": f"{ambiguity_rate:.1%} of studies have more than one preferred candidate for a slot",
        "impact": "CSV row order, rather than resolution/coverage/sequence quality, can choose the model input.",
        "next_experiment": "Rank matching series using slice count, spacing, coverage and description instead of taking the first row.",
    },
    {
        "severity": "MEDIUM" if np.isfinite(anisotropic_rate) and anisotropic_rate > 0.05 else "LOW",
        "check": "PixelSpacing[0] is valid for both crop axes",
        "evidence": f"{anisotropic_rate:.1%} of sampled series have >1% in-plane spacing anisotropy" if np.isfinite(anisotropic_rate) else "not measured",
        "impact": "A nominal 140 mm square can cover different physical widths and heights.",
        "next_experiment": "Use PixelSpacing[0] and [1] independently, then retrain or ablate before changing scored inference.",
    },
    {
        "severity": "MEDIUM" if np.isfinite(truncated_rate) and truncated_rate > 0.05 else "LOW",
        "check": "The requested 140 mm crop fits inside the acquired image",
        "evidence": f"{truncated_rate:.1%} of sampled series produce a clamped field of view" if np.isfinite(truncated_rate) else "not measured",
        "impact": "Clamping silently changes anatomical scale across studies.",
        "next_experiment": "Pad to a true physical 140 mm field of view before resize and expose a padding mask.",
    },
    {
        "severity": "LOW",
        "check": "336 → 384 double interpolation",
        "evidence": f"median edge-variance ratio two-stage/direct={edge_ratio:.3f}" if np.isfinite(edge_ratio) else "not measured",
        "impact": "The second resize can slightly smooth small tears and cartilage edges, but changing it alone breaks train/inference parity.",
        "next_experiment": "Only test direct 384 crops in a retrained boundary-aware branch.",
    },
]

gap_report = pd.DataFrame(gaps)
severity_order = pd.CategoricalDtype(["HIGH", "MEDIUM", "LOW", "PASS"], ordered=True)
gap_report["severity"] = gap_report["severity"].astype(severity_order)
gap_report = gap_report.sort_values("severity").reset_index(drop=True)
display(gap_report)
gap_report.to_csv(WORK / "preprocessing_gap_report.csv", index=False)

summary = {
    "experiment": "E023 RSNA Knee preprocessing visual audit",
    "selected_studies": selected_uids,
    "exact_e020": {"stack_depth": 64, "crop_mm": 140.0, "stored_size": 336, "model_size": 384, "k_eval": 42},
    "observations": {
        "public_corpus_depths": public_depths,
        "cross_series_window_rate": cross_rate,
        "zero_slice_window_rate": zero_rate,
        "missing_slot_study_rate": missing_rate,
        "ambiguous_series_study_rate": ambiguity_rate,
        "spacing_anisotropy_rate": anisotropic_rate,
        "crop_truncation_rate": truncated_rate,
        "median_two_stage_edge_ratio": edge_ratio,
    },
    "recommended_next_branch": "rebuild exact 64-slice corpus with series-aware valid window centres; then warm-start CoAtNet",
    "submission_created": False,
}
(WORK / "preprocessing_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

print("\nDECISION")
print("Do not spend a submission from this notebook.")
print("Use the visual and CSV evidence to choose between:")
print("  A) exact 64-slice corpus + series-aware windows (recommended), or")
print("  B) keep E020 unchanged if the audit does not reproduce a high-severity gap.")
print("\nArtifacts written to", WORK)
for path in sorted(WORK.glob("*audit*")):
    print(" -", path.name)
"""
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {
            "accelerator": "none",
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
)
nbf.write(nb, OUT)
print(OUT)
