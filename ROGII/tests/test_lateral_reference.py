import numpy as np

from rogii.lateral_reference import (
    LateralReferenceBank,
    LateralReferenceConfig,
    concatenate_window_tables,
    window_signatures,
    window_table_from_arrays,
)


def _signal(rows: int = 512) -> np.ndarray:
    x = np.arange(rows, dtype=float)
    return 60.0 + 12.0 * np.sin(x / 13.0) + 4.0 * np.sin(x / 4.0)


def test_reverse_windows_recover_reference_tvt() -> None:
    config = LateralReferenceConfig(
        stride=4,
        window_offsets=(-24, -16, -8, 0, 8, 16, 24),
        tvt_scale=8.0,
        tvt_corridor=20.0,
        query_candidates=64,
        diverse_wells=1,
    )
    gr = _signal()
    tvt = 10_000.0 + 0.25 * np.arange(len(gr))
    table = window_table_from_arrays(
        gr,
        tvt,
        well_number=7,
        config=config,
    )
    bank = LateralReferenceBank({"group": table}, config)

    reversed_gr = gr[::-1]
    centers = np.array([120, 180, 240, 300], dtype=np.int64)
    signatures, _ = window_signatures(reversed_gr, centers, config)
    expected = tvt[len(gr) - 1 - centers]
    match = bank.query(
        "group",
        signatures,
        expected + 2.0,
        allowed_wells=np.array([7]),
        diverse_wells=1,
    )

    np.testing.assert_allclose(match.tvt, expected, atol=1.1)
    assert (match.orientation == -1).all()
    assert (match.score > 0.95).all()


def test_allowed_wells_are_enforced_before_consensus() -> None:
    config = LateralReferenceConfig(
        stride=4,
        window_offsets=(-16, -8, 0, 8, 16),
        tvt_scale=10.0,
        tvt_corridor=100.0,
        query_candidates=128,
        diverse_wells=2,
    )
    gr = _signal(320)
    first = window_table_from_arrays(
        gr,
        1_000.0 + 0.2 * np.arange(len(gr)),
        well_number=1,
        config=config,
    )
    second = window_table_from_arrays(
        gr,
        1_050.0 + 0.2 * np.arange(len(gr)),
        well_number=2,
        config=config,
    )
    bank = LateralReferenceBank(
        {"group": concatenate_window_tables([first, second])},
        config,
    )
    centers = np.array([100, 160, 220], dtype=np.int64)
    signatures, _ = window_signatures(gr, centers, config)
    expected = 1_050.0 + 0.2 * centers
    match = bank.query(
        "group",
        signatures,
        expected,
        allowed_wells=np.array([2]),
    )

    assert (match.support == 1).all()
    np.testing.assert_allclose(match.tvt, expected, atol=0.9)


def test_no_gr_query_retains_group_and_allowed_well_contract() -> None:
    config = LateralReferenceConfig(
        stride=4,
        window_offsets=(-16, -8, 0, 8, 16),
        tvt_corridor=30.0,
        query_candidates=64,
        diverse_wells=1,
    )
    gr = _signal(320)
    tvt = 2_000.0 + 0.1 * np.arange(len(gr))
    table = window_table_from_arrays(
        gr,
        tvt,
        well_number=3,
        config=config,
    )
    bank = LateralReferenceBank({"group": table}, config)
    centers = np.array([100, 160, 220], dtype=np.int64)
    signatures, _ = window_signatures(gr, centers, config)
    base = tvt[centers] + 0.3
    match = bank.query(
        "group",
        signatures,
        base,
        allowed_wells=np.array([3]),
        use_gr=False,
        diverse_wells=1,
    )

    assert (match.support == 1).all()
    np.testing.assert_allclose(match.tvt, base, atol=0.6)
