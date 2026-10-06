from kaggriculture_rl.policy.fixed_handoff import FixedHandoffPolicy


def test_fixed_handoff_keeps_both_policies_synchronized() -> None:
    calls = {"opening": [], "continuation": []}

    def opening(observation, configuration=None):
        del configuration
        calls["opening"].append(observation["step"])
        return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}

    def continuation(observation, configuration=None):
        del configuration
        calls["continuation"].append(observation["step"])
        return {"farmer": ["NORTH"], "hands": [], "market": []}

    policy = FixedHandoffPolicy(opening, continuation, boundary=1)

    assert policy({"step": 0})["market"] == [["HIRE"]]
    assert policy({"step": 1})["farmer"] == ["NORTH"]
    assert calls == {"opening": [0, 1], "continuation": [0, 1]}
