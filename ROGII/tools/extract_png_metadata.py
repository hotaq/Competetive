"""Extract original well/typewell identifiers from the provided training PNG titles."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image


TITLE_PATTERN = re.compile(
    r"Well:\s*(Well\d+)\s*\|\s*Typewell:\s*(Typewell\d+)\s*\|\s*Azimuth\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)


def extract_one(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        title = image.crop(
            (
                int(width * 0.15),
                0,
                int(width * 0.90),
                min(220, int(height * 0.10)),
            )
        )
        text = pytesseract.image_to_string(title, config="--psm 7").strip()

    match = TITLE_PATTERN.search(text)
    if match is None:
        return {
            "id": path.stem,
            "well_name": None,
            "typewell_name": None,
            "azimuth": None,
            "ocr_text": text,
        }
    well_name, typewell_name, azimuth = match.groups()
    return {
        "id": path.stem,
        "well_name": well_name,
        "typewell_name": typewell_name,
        "azimuth": float(azimuth),
        "ocr_text": text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/train_png_metadata.csv"),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.train_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found under {args.train_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(extract_one, paths))

    frame = pd.DataFrame(rows).sort_values("id")
    failures = frame["typewell_name"].isna()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        {
            "output": str(args.output),
            "rows": len(frame),
            "parsed": int((~failures).sum()),
            "failed": int(failures.sum()),
        }
    )
    if failures.any():
        print(frame.loc[failures, ["id", "ocr_text"]].to_string(index=False))
        raise RuntimeError("Some PNG titles could not be parsed")


if __name__ == "__main__":
    main()
