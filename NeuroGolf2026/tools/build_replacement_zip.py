#!/usr/bin/env python3
"""Build a NeuroGolf submission by replacing selected ONNX members."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_replacement(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("replacement must be member=path")
    member, path = text.split("=", 1)
    member = Path(member).name
    if not member.endswith(".onnx"):
        raise argparse.ArgumentTypeError(f"replacement member must be .onnx: {member}")
    return member, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--zip-out", type=Path, required=True)
    parser.add_argument("--replace", action="append", type=parse_replacement, required=True)
    args = parser.parse_args()

    replacements = dict(args.replace)
    raw_replacements: dict[str, bytes] = {}
    for member, path in replacements.items():
        raw_replacements[member] = path.read_bytes()

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(args.base_zip) as src, zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            if member in raw_replacements:
                dst.writestr(member, raw_replacements[member])
                seen.add(member)
            else:
                dst.writestr(member, src.read(name))
        for member, raw in raw_replacements.items():
            if member not in seen:
                dst.writestr(member, raw)

    print(f"base: {args.base_zip}")
    print(f"output: {args.zip_out}")
    print("replaced:", ", ".join(sorted(raw_replacements)))


if __name__ == "__main__":
    main()
