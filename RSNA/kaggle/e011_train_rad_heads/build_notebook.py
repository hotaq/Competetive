#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


INTRO = """# E011 — self-trained RadImageNet query heads

This no-submission training run uses the byte-pinned official RadImageNet
ResNet-50 checkpoint from the MIT-licensed upstream repository. It trains our
own two-seed × five-fold target-query heads on RSNA Knee training studies and
writes fold-safe OOF diagnostics before allowing an E012 inference blend.

The public non-commercial head package is deliberately **not attached**.
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    code = (here / "e011_train.py").read_text()
    compile(code, "e011_train.py", "exec")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "e011-intro",
                "metadata": {},
                "source": INTRO.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "e011-train",
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = here / "e011-train-rad-heads.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")))
    print(output)


if __name__ == "__main__":
    main()
