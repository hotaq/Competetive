"""Insert a donor notebook cell into a scored Kaggle notebook baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--donor-notebook", type=Path, required=True)
    parser.add_argument("--donor-cell", type=int, required=True)
    parser.add_argument("--insert-before", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    notebook = json.loads(args.base_notebook.read_text(encoding="utf-8"))
    donor = json.loads(args.donor_notebook.read_text(encoding="utf-8"))
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))

    insertion_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if _source_text(cell).startswith(args.insert_before)
    )
    donor_cell = donor["cells"][args.donor_cell]
    if donor_cell.get("cell_type") != "code":
        raise ValueError(f"donor cell {args.donor_cell} is not executable code")
    source = _source_text(donor_cell)
    if not source.strip():
        raise ValueError(f"donor cell {args.donor_cell} is empty")

    notebook["cells"].insert(
        insertion_index,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{args.slug}.ipynb"
    metadata.update(
        {
            "id": f"chinnaphatkhun/{args.slug}",
            "title": args.title,
            "code_file": code_file,
        }
    )
    metadata.pop("id_no", None)
    (args.output_dir / code_file).write_text(
        json.dumps(notebook, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
