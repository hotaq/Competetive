from kaggriculture_rl.evaluation.live_profile_matrix import (
    evaluate_live_profile_matrix,
)
from kaggriculture_rl.policy.production_profile import ADAPTIVE_PROFILES


def test_live_profile_matrix_is_seat_balanced_and_baseline_is_zero() -> None:
    profiles = (ADAPTIVE_PROFILES[0], ADAPTIVE_PROFILES[3])

    result = evaluate_live_profile_matrix(
        seeds=(991,),
        candidates=profiles,
        opponents=profiles,
        baseline_name=ADAPTIVE_PROFILES[3].name,
    )

    assert len(result["games"]) == 8
    assert {game["candidate_seat"] for game in result["games"]} == {0, 1}
    baseline = next(
        row
        for row in result["rows"]
        if row["candidate"] == ADAPTIVE_PROFILES[3].name
    )
    assert baseline["paired_to_baseline"]["mean_margin"] == 0.0
    assert baseline["paired_to_baseline"]["worst_money"] == 0.0
