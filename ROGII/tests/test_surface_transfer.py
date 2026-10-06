from pathlib import Path

import numpy as np
import pandas as pd

from rogii.surface_transfer import (
    SurfaceTransferConfig,
    TypewellSurfaceIndex,
    predict_surface_tail,
    typewell_fingerprint,
)


def _typewell(rows: int = 100) -> pd.DataFrame:
    tvt = 10_000.0 + 0.5 * np.arange(rows)
    gr = 80.0 + np.sin(np.arange(rows) / 7.0)
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_typewell_fingerprint_uses_shared_terminal_suffix() -> None:
    full = _typewell()
    suffix = full.iloc[20:].reset_index(drop=True)
    changed = suffix.copy()
    changed.loc[changed.index[-1], "GR"] += 1.0

    assert typewell_fingerprint(full) == typewell_fingerprint(suffix)
    assert typewell_fingerprint(full) != typewell_fingerprint(changed)


def _write_well(
    train_dir: Path,
    well_id: str,
    *,
    surface_offset: float,
) -> pd.DataFrame:
    rows = 50
    x = np.arange(rows, dtype=float)
    z = np.full(rows, -900.0)
    tvt = 1_900.0 + surface_offset + 0.1 * x
    horizontal = pd.DataFrame(
        {
            "X": x,
            "Y": np.zeros(rows),
            "Z": z,
            "TVT": tvt,
            "TVT_input": np.where(x < 30, tvt, np.nan),
        }
    )
    horizontal.to_csv(train_dir / f"{well_id}__horizontal_well.csv", index=False)
    _typewell().to_csv(train_dir / f"{well_id}__typewell.csv", index=False)
    return horizontal


def test_predict_surface_tail_excludes_target_and_calibrates_prefix(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    target = _write_well(train_dir, "target", surface_offset=5.0)
    _write_well(train_dir, "neighbor", surface_offset=0.0)
    config = SurfaceTransferConfig(
        samples_per_well=50,
        query_candidates=100,
        diverse_wells=2,
    )
    index = TypewellSurfaceIndex(train_dir, config)

    prediction = predict_surface_tail(
        target,
        _typewell(),
        index,
        well_id="target",
        config=config,
    )

    expected = target.loc[target["TVT_input"].isna(), "TVT"].to_numpy()
    assert prediction.group_seen
    assert prediction.group_wells == 2
    assert np.isclose(prediction.prefix_weight, 1.0)
    np.testing.assert_allclose(prediction.tvt, expected, atol=1e-10)
