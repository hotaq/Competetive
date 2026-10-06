"""Write single-constant variants of C (PREREGISTRATION.md). Usage: uv run python analysis/mirror_breaker_20260924/make_variants.py"""

import re
from pathlib import Path

SRC = Path("artifacts/public_notebooks_20260924/outputs/kaggriculture-harvest-ledger/main.py")
OUT = Path("analysis/mirror_breaker_20260924/variants")
EDITS = {
    "V9_RACE_DEFAULT": ["38", "42", "43", "44", "46", "48"],
    "V9_RACE_MARGIN": ["8", "16"], "V9_RACE_GAP": ["2", "4"], "V9_RACE_WINDOW": ["24", "36"],
    "V9_CARROT_RATIO": ["1.6", "2.0"], "CROP_MIN_PRICE": ["65", "75"],
    "_OR2_SLOT_MARGIN": ["15.0", "25.0"], "_SR_MARGIN": ["6", "10"], "_R37_HINGE_GAIN": ["6.0", "10.0"],
    "_HD2_RATIO": ["1.2", "1.4"], "_CS_RATIO": ["1.2", "1.4"], "_CA_MARGIN": ["-3.0", "-7.0"],
    "_CH_SHED": ["85", "95"],
}


def edit(lines: list[str], name: str, value: str) -> list[str]:
    idx = [i for i, line in enumerate(lines) if re.match(rf"{re.escape(name)}\s*=", line)]
    i = idx[-1]  # the last module-level binding is the one in force
    out = list(lines)
    out[i] = re.sub(rf"^({re.escape(name)}\s*=\s*)[-0-9.]+", rf"\g<1>{value}", lines[i])
    assert out[i] != lines[i], (name, value)
    return out


def main() -> None:
    lines = SRC.read_text().split("\n")
    for name, values in EDITS.items():
        for value in values:
            new = edit(lines, name, value)
            assert sum(a != b for a, b in zip(lines, new, strict=True)) == 1
            d = OUT / f"{name.strip('_').lower()}_{value}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "main.py").write_text("\n".join(new))
    print(len(list(OUT.glob("*/main.py"))), "variants")


if __name__ == "__main__":
    main()
