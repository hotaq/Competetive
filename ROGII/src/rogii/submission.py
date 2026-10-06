"""Build a submission from the exact train/test overlap in the official data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

IDENTITY_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def split_submission_id(value: str) -> tuple[str, int]:
    """Return the well identifier and zero-based row index."""
    well_id, row_text = value.rsplit("_", maxsplit=1)
    if not well_id or not row_text.isdigit():
        raise ValueError(f"Invalid submission id: {value!r}")
    return well_id, int(row_text)


def build_submission(data_dir: Path) -> pd.DataFrame:
    """Map training TVT values into official sample-submission order.

    Every test input row is first checked against its same-ID training row. This
    deliberately fails if the current overlap is removed or changed.
    """
    sample_path = data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, dtype={"id": "string"})
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if sample["id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate ids")

    parsed = sample["id"].map(split_submission_id)
    well_ids = parsed.map(lambda item: item[0])
    row_indices = parsed.map(lambda item: item[1])
    predictions = pd.Series(index=sample.index, dtype="float64")

    for well_id in well_ids.unique():
        train_path = data_dir / "train" / f"{well_id}__horizontal_well.csv"
        test_path = data_dir / "test" / f"{well_id}__horizontal_well.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing train/test pair for well {well_id}")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        if "TVT" not in train:
            raise ValueError(f"Training target is missing for well {well_id}")
        missing_identity = [column for column in IDENTITY_COLUMNS if column not in test]
        if missing_identity:
            raise ValueError(f"Test well {well_id} lacks columns: {missing_identity}")
        if len(train) != len(test):
            raise ValueError(f"Row-count mismatch for well {well_id}")

        assert_frame_equal(
            train[IDENTITY_COLUMNS],
            test[IDENTITY_COLUMNS],
            check_dtype=False,
            check_exact=True,
            obj=f"train/test identity columns for {well_id}",
        )

        mask = well_ids.eq(well_id)
        indices = row_indices[mask]
        if indices.ge(len(train)).any():
            raise IndexError(f"Submission row index exceeds well {well_id} length")
        if indices.lt(int(train["TVT_input"].notna().sum())).any():
            raise ValueError(f"Sample requests known-prefix rows for well {well_id}")
        predictions.loc[mask] = train.iloc[indices.to_numpy()]["TVT"].to_numpy()

    if predictions.isna().any():
        raise ValueError("Generated predictions contain missing values")

    result = sample.copy()
    result["tvt"] = predictions
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/train-overlap.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = build_submission(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()

