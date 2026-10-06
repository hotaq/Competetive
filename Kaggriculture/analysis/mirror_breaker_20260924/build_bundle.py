"""Build the H54c bundle: C's main.py with 3 constants changed + C's LICENSE/NOTICE.
Usage: uv run python analysis/mirror_breaker_20260924/build_bundle.py"""

import hashlib
import io
import tarfile
from pathlib import Path

C_TAR = Path("artifacts/public_notebooks_20260924/outputs/kaggriculture-harvest-ledger/submission.tar.gz")
VARIANT = Path("analysis/mirror_breaker_20260924/variants_r2/H54c/main.py")
OUT = Path("submissions/mirror_h54c_standalone")
NOTE = ("# MODIFIED by team zero waste on 2026-09-24 (Apache-2.0 s4b): V9_RACE_DEFAULT/V9_RACE_MAX "
        "41/48 -> 54/54 and _CA_MARGIN -5.0 -> -7.0; nothing else changed.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_py = NOTE + VARIANT.read_text()
    (OUT / "main.py").write_text(main_py)
    with tarfile.open(C_TAR) as src:
        extra = {m.name: src.extractfile(m).read() for m in src.getmembers() if m.name in ("LICENSE.txt", "NOTICE.txt")}
    with tarfile.open(OUT / "submission.tar.gz", "w:gz") as tar:
        for name, data in [("main.py", main_py.encode()), *sorted(extra.items())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    print("tar sha256", hashlib.sha256((OUT / "submission.tar.gz").read_bytes()).hexdigest()[:16], "members", ["main.py", *sorted(extra)])


if __name__ == "__main__":
    main()
