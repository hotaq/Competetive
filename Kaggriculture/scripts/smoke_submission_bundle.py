"""Load a submission bundle and plan one action from an observation on stdin."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("main", type=Path)
    args = parser.parse_args()
    namespace = runpy.run_path(str(args.main.resolve()))
    action = namespace["agent"](json.load(sys.stdin))
    if set(action) != {"farmer", "hands", "market"}:
        raise ValueError(f"invalid action keys: {sorted(action)}")
    print(json.dumps(action, separators=(",", ":")))


if __name__ == "__main__":
    main()
