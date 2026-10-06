"""Build the V67 quantity-conserving front-run agent as one Kaggle file."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT / "submissions" / "meta_v67_quantity_conserving_front_run_standalone"
)

MODULES = (
    (
        "kaggriculture_rl.learning.tape_selector",
        ROOT / "src/kaggriculture_rl/learning/tape_selector.py",
    ),
    (
        "kaggriculture_rl.policy.primitive_legalizer",
        ROOT / "src/kaggriculture_rl/policy/primitive_legalizer.py",
    ),
    (
        "kaggriculture_rl.policy.unit_rules",
        ROOT / "src/kaggriculture_rl/policy/unit_rules.py",
    ),
    (
        "artifacts.top20_live_tapes80_20260826.tape_99566826_s0",
        ROOT / "artifacts/top20_live_tapes80_20260826/tape_99566826_s0.py",
    ),
    (
        "artifacts.top20_live_tapes80_20260826.tape_99596430_s1",
        ROOT / "artifacts/top20_live_tapes80_20260826/tape_99596430_s1.py",
    ),
    (
        "artifacts.top20_live_tapes80_20260826.tape_99600846_s0",
        ROOT / "artifacts/top20_live_tapes80_20260826/tape_99600846_s0.py",
    ),
    (
        "artifacts.top20_live_tapes80_20260826.tape_99612231_s0",
        ROOT / "artifacts/top20_live_tapes80_20260826/tape_99612231_s0.py",
    ),
    (
        "submissions.meta_v54_market_splice_state_gate_local.main",
        ROOT / "submissions/meta_v54_market_splice_state_gate_local/main.py",
    ),
    (
        "submissions.meta_v59_causal_route_selector_b120_local.main",
        ROOT / "submissions/meta_v59_causal_route_selector_b120_local/main.py",
    ),
    (
        "artifacts.v67_front_run_20260826.overlay",
        ROOT / "artifacts/v67_front_run_20260826/overlay.py",
    ),
    (
        "artifacts.v67_front_run_20260826.all_v16",
        ROOT / "artifacts/v67_front_run_20260826/all_v16.py",
    ),
)

REPORTS = {
    "submissions.meta_v54_market_splice_state_gate_local.main": (
        ROOT / "rl/evaluations/v54_market_splice_gate_old80_to_fresh95_20260826.json"
    ),
    "submissions.meta_v59_causal_route_selector_b120_local.main": (
        ROOT
        / "rl/evaluations/v59_causal_route_selector_b120_old80_to_fresh95_20260826.json"
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_assignment(source: str, name: str, replacement: str) -> str:
    """Replace one top-level assignment without changing the rest of the module."""
    tree = ast.parse(source)
    matches: list[ast.Assign | ast.AnnAssign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one assignment to {name}, found {len(matches)}")
    node = matches[0]
    if node.end_lineno is None:
        raise RuntimeError(f"assignment to {name} has no end position")
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return source[:start] + replacement + "\n" + source[end:]


def _module_source(name: str, path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    report_path = REPORTS.get(name)
    if report_path is not None:
        report_json = report_path.read_text(encoding="utf-8")
        source = _replace_assignment(
            source,
            "_REPORT",
            f"_REPORT = json.loads({report_json!r})",
        )
    compile(source, str(path), "exec")
    return source


def _standalone_source() -> str:
    embedded = [(name, _module_source(name, path)) for name, path in MODULES]
    runtime = '''\
from __future__ import annotations

import sys
import types


def _ensure_embedded_package(name):
    if not name:
        return None
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    parent_name, _, child_name = name.rpartition(".")
    parent = _ensure_embedded_package(parent_name)
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = []
    package.__file__ = "<embedded-package:" + name + ">"
    sys.modules[name] = package
    if parent is not None:
        setattr(parent, child_name, package)
    return package


def _load_embedded_module(name, source):
    parent_name, _, child_name = name.rpartition(".")
    parent = _ensure_embedded_package(parent_name)
    module = types.ModuleType(name)
    module.__package__ = parent_name
    module.__file__ = "<embedded-module:" + name + ">"
    sys.modules[name] = module
    if parent is not None:
        setattr(parent, child_name, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


_EMBEDDED_MODULES = __EMBEDDED_MODULES__
for _embedded_name, _embedded_source in _EMBEDDED_MODULES:
    _load_embedded_module(_embedded_name, _embedded_source)

agent = sys.modules[
    "artifacts.v67_front_run_20260826.all_v16"
].agent
'''
    source = runtime.replace("__EMBEDDED_MODULES__", repr(embedded))
    compile(source, "<v67-standalone>", "exec")
    return source


def _write_deterministic_archive(path: Path, main_bytes: bytes) -> None:
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file,
        tarfile.open(
            fileobj=gzip_file,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive,
    ):
        info = tarfile.TarInfo("main.py")
        info.size = len(main_bytes)
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        archive.addfile(info, io.BytesIO(main_bytes))


def build(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    archive_path = output_dir / "submission.tar.gz"
    main_bytes = _standalone_source().encode("utf-8")
    main_path.write_bytes(main_bytes)
    _write_deterministic_archive(archive_path, main_bytes)
    archive_bytes = archive_path.read_bytes()
    metadata: dict[str, object] = {
        "name": "meta_v67_quantity_conserving_front_run_standalone",
        "agent_file": str(main_path.relative_to(ROOT)),
        "archive": str(archive_path.relative_to(ROOT)),
        "main_sha256": _sha256(main_bytes),
        "archive_sha256": _sha256(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "source_agent": "artifacts/v67_front_run_20260826/all_v16.py",
        "selection_evidence": [
            "rl/evaluations/v67_all_v16_old80_20260826.json",
            "rl/evaluations/v67_all_v16_fresh95_20260826.json",
        ],
        "standalone_validation_evidence": [
            "rl/evaluations/v67_archive_raw_smoke_seat0_20260826.json",
            "rl/evaluations/v67_archive_raw_smoke_seat1_20260826.json",
            "rl/evaluations/v67_local_vs_standalone_99429769_s1_20260826.json",
            "rl/evaluations/v67_local_vs_standalone_99434330_s0_20260826.json",
            "rl/evaluations/v67_standalone_fresh95_exact_20260826.json",
        ],
        "status": "standalone_validated",
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
