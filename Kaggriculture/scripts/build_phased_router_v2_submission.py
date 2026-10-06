"""Build the standalone phased-router-v2 submission bundle.

Inlines the vendored boatlee V21 source plus the four per-seat schedule
traces into one self-contained main.py, following the _EMBEDDED_MODULES
pattern of the earlier standalone bundles. The router matches
PhasedRouterV2 defaults exactly: yusuke bundle on both seats.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_PATH = ROOT / "artifacts" / "recovered_boatlee_v21_source" / "main.py"
SCHEDULE_PATHS = {
    "dusta": {
        0: ROOT / "artifacts" / "dusta_schedule_s0.jsonl",
        1: ROOT / "artifacts" / "dusta_schedule_s1.jsonl",
    },
    "yusuke": {
        0: ROOT / "artifacts" / "yusuke_schedule_s0.jsonl",
        1: ROOT / "artifacts" / "yusuke_schedule_s1.jsonl",
    },
}
DEFAULT_OUTPUT_DIR = ROOT / "submissions" / "upload_v2_yusuke_20260904"
BUNDLE_NAME = "phased_router_v2_yusuke_standalone"

ROUTER_SOURCE = """import copy
from v2_schedules import SCHEDULES
import v21_vendor as _vendor

_DEFAULT_BUNDLE_BY_SEAT = {0: "yusuke", 1: "yusuke"}
_PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _seat_of(observation):
    return 1 if int(observation.get("player", 0) or 0) == 1 else 0


def agent(observation, configuration=None):
    seat = _seat_of(observation)
    if _DEFAULT_BUNDLE_BY_SEAT[seat] == "v21":
        return _vendor.agent(observation, configuration)
    trace = SCHEDULES[_DEFAULT_BUNDLE_BY_SEAT[seat]][seat]
    step = int(observation.get("step", 0) or 0)
    if step < 0 or step >= len(trace):
        return copy.deepcopy(_PASS_ACTION)
    return copy.deepcopy(trace[step])
"""

LOADER_SOURCE = """from __future__ import annotations

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
"""

def read_schedule(path):
    actions = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            actions.append(json.loads(line))
    if len(actions) not in (719, 720):
        raise ValueError("bad schedule length in " + str(path))
    return actions


def build_schedules_source():
    bundles = {}
    for name, seats in SCHEDULE_PATHS.items():
        bundles[name] = {seat: read_schedule(path) for seat, path in seats.items()}
    return "SCHEDULES = " + repr(bundles) + chr(10)


def build_main():
    vendor_source = VENDOR_PATH.read_text()
    schedules_source = build_schedules_source()
    modules = [
        ("v21_vendor", vendor_source),
        ("v2_schedules", schedules_source),
        ("v2_router", ROUTER_SOURCE),
    ]
    chr(10).join([
        "from __future__ import annotations",
        "",
        "import sys",
        "import types",
    ])
    lines = [LOADER_SOURCE.rstrip(chr(10)), "", "_EMBEDDED_MODULES = ["]
    for name, source in modules:
        lines.append("    (" + repr(name) + ", " + repr(source) + "),")
    lines.append("]")
    lines.append("")
    lines.append("for _embedded_name, _embedded_source in _EMBEDDED_MODULES:")
    lines.append("    _load_embedded_module(_embedded_name, _embedded_source)")
    lines.append("")
    lines.append("agent = sys.modules[" + repr("v2_router") + "].agent")
    lines.append("")
    return chr(10).join(lines)

def write_tarball(output_dir, main_bytes):
    tar_buffer = __import__("io").BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(main_bytes)
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        archive.addfile(info, __import__("io").BytesIO(main_bytes))
    raw = __import__("io").BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
        zipped.write(tar_buffer.getvalue())
    path = output_dir / "submission.tar.gz"
    path.write_bytes(raw.getvalue())
    return path


def verify_equivalence(main_path, checks):
    import runpy

    from kaggriculture_rs import Engine

    from kaggriculture_rl.policy.phased_router_v2 import PhasedRouterV2
    namespace = runpy.run_path(str(main_path))
    bundled = namespace["agent"]
    live = PhasedRouterV2(bundle_by_seat={0: "yusuke", 1: "yusuke"})
    checked = 0
    for seed in checks:
        live.reset()
        engine = Engine({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005})
        for _step in range(60):
            for seat in (0, 1):
                observation = engine.observation(seat)
                if bundled(observation, None) != live.agent(observation, None):
                    raise RuntimeError("divergence at seed " + str(seed))
                checked += 1
            left = live.agent(engine.observation(0), None)
            right = live.agent(engine.observation(1), None)
            engine.step([left, right])
    return checked


def build(output_dir, seeds=(208248505, 1219301430)):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_text = build_main()
    main_bytes = main_text.encode()
    main_path = output_dir / "main.py"
    main_path.write_text(main_text)
    checked = verify_equivalence(main_path, seeds)
    tarball = write_tarball(output_dir, main_bytes)
    metadata = {
        "name": BUNDLE_NAME,
        "agent_file": str(
            output_dir.relative_to(ROOT) / "main.py"
            if output_dir.is_relative_to(ROOT)
            else output_dir / "main.py"
        ),
        "archive": str(
            tarball.relative_to(ROOT)
            if tarball.is_relative_to(ROOT)
            else tarball
        ),
        "archive_bytes": len(tarball.read_bytes()),
        "archive_sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
        "main_bytes": len(main_bytes),
        "main_sha256": hashlib.sha256(main_bytes).hexdigest(),
        "vendor_sha256": hashlib.sha256(VENDOR_PATH.read_bytes()).hexdigest(),
        "default_bundles": {0: "yusuke", 1: "yusuke"},
        "calibration": "46 games per bundle: yusuke 24, v21 21, dusta 5",
        "equivalence_checks": checked,
        "status": "standalone_validated",
    }
    payload = json.dumps(metadata, indent=2, sort_keys=True) + chr(10)
    (output_dir / "candidate.json").write_text(payload)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
