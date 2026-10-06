from pathlib import Path

import pandas as pd
import pytest

from rogii.submission import build_submission, split_submission_id


def test_split_submission_id() -> None:
    assert split_submission_id("000d7d20_1442") == ("000d7d20", 1442)


def test_split_submission_id_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        split_submission_id("bad-id")


def test_official_data_maps_every_target_exactly() -> None:
    data_dir = Path("data/raw")
    if not (data_dir / "sample_submission.csv").is_file():
        pytest.skip("Official competition data is not present")

    submission = build_submission(data_dir)
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    assert submission["id"].tolist() == sample["id"].tolist()
    assert submission["tvt"].notna().all()
    assert len(submission) == len(sample)

