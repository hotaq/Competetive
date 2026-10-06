"""Build a self-contained Kaggle tarball for live-response PPO v8."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submissions" / "live_response_ppo_v8" / "main.py"
CHECKPOINT = ROOT / "rl" / "checkpoints" / "live_response_ppo_v8.npz"
DEFAULT_NATIVE = ROOT / "bindings" / "python" / "kaggriculture_rs" / "_native.abi3.so"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_format(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if prefix == b"\x7fELF":
        return "elf"
    if prefix in {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"}:
        return "mach-o"
    return "unknown"


def _install_native(source: Path, destination: Path) -> None:
    if source.suffix != ".whl":
        shutil.copy2(source, destination)
        return
    with zipfile.ZipFile(source) as wheel:
        candidates = [
            name
            for name in wheel.namelist()
            if name.endswith("/_native.abi3.so") or name == "_native.abi3.so"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"wheel must contain exactly one _native.abi3.so; got {candidates}"
            )
        destination.write_bytes(wheel.read(candidates[0]))


def build(native: Path, output_dir: Path, target: str) -> Path:
    expected_format = "elf" if target == "kaggle" else "mach-o"
    staging = output_dir / f"live_response_ppo_v8_{target}"
    staging.mkdir(parents=True, exist_ok=True)
    files = {
        "main.py": TEMPLATE,
        "model.npz": CHECKPOINT,
    }
    for name, source in files.items():
        shutil.copy2(source, staging / name)
    native_path = staging / "_native.abi3.so"
    _install_native(native, native_path)
    actual_format = _binary_format(native_path)
    if actual_format != expected_format:
        raise ValueError(
            f"{target} build requires {expected_format} native library; got {actual_format}"
        )
    files["_native.abi3.so"] = native

    manifest = {
        "agent": "live-response-contextual-ppo-v2",
        "checkpoint": "live_response_ppo_v8.npz",
        "promotion_status": "live-league-challenger",
        "target": target,
        "files": {
            name: {"sha256": _sha256(staging / name), "bytes": (staging / name).stat().st_size}
            for name in sorted(files)
        },
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    archive = output_dir / f"live_response_ppo_v8_{target}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in (*sorted(files), "manifest.json"):
            bundle.add(staging / name, arcname=name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "dist")
    parser.add_argument("--target", choices=("local", "kaggle"), default="local")
    args = parser.parse_args()
    archive = build(args.native.resolve(), args.output_dir.resolve(), args.target)
    print(archive)


if __name__ == "__main__":
    main()
