"""Extract each splice configuration into a standard episode report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("input has no splice configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for configuration in configurations:
        suffix = Path(configuration["suffix_agent"]).stem
        branch_step = int(configuration["branch_step"])
        output = args.output_dir / f"{suffix}_b{branch_step}.json"
        report = {
            "schema_version": 1,
            "source_splice_report": str(args.input),
            "suffix_agent": configuration["suffix_agent"],
            "branch_step": branch_step,
            "splice_component": payload["splice_component"],
            "state_aware_repair": payload["state_aware_repair"],
            "summary": configuration["summary"],
            "episodes": configuration["episodes"],
        }
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outputs.append(str(output))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
