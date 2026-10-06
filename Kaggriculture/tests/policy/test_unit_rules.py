from __future__ import annotations

import copy

from kaggriculture_rl.policy.unit_rules import (
    AdjacentHighValueShedReturnRule,
    AdjacentShedReturnController,
    CapHarvRule,
    HourGatedCapHarvRule,
    LocalWorkFillRule,
    OrphanStructureRecoveryController,
    ValueGatedCapHarvRule,
)


def _observation(*, wool: int = 15, wool_price: int = 200) -> dict:
    farm = {
        "money": 1000,
        "farmer": [5, 4],
        "hands": [[4, 3]],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    return {
        "player": 0,
        "farms": [farm, {}],
        "private": {"inventories": [{}, {"WOOL": wool}]},
        "market": {"prices": {"WOOL": wool_price}},
    }


def _action(command: str = "EAST") -> dict:
    return {"farmer": ["PASS"], "hands": [[command]], "market": []}


def test_redirects_adjacent_loaded_hand_into_open_shed() -> None:
    result = AdjacentHighValueShedReturnRule().apply(_observation(), _action())

    assert result["hands"] == [["SOUTH"]]


def test_preserves_move_when_inventory_value_is_below_threshold() -> None:
    action = _action()

    result = AdjacentHighValueShedReturnRule().apply(_observation(wool=4, wool_price=200), action)

    assert result == action


def test_preserves_move_that_already_enters_shed() -> None:
    action = _action("SOUTH")

    result = AdjacentHighValueShedReturnRule().apply(_observation(), action)

    assert result == action


def test_does_not_redirect_into_occupied_shed_tile() -> None:
    observation = _observation()
    observation["farms"][0]["farmer"] = [4, 4]
    action = _action()

    result = AdjacentHighValueShedReturnRule().apply(observation, action)

    assert result == action


def test_controller_enters_drops_once_and_resumes() -> None:
    controller = AdjacentShedReturnController()
    entering = _observation()
    entering["step"] = 164

    first = controller.apply(entering, _action())

    at_shed = _observation()
    at_shed["step"] = 165
    at_shed["farms"][0]["hands"] = [[4, 4]]
    second = controller.apply(at_shed, _action("NORTH"))

    after_drop = _observation()
    after_drop["step"] = 166
    after_drop["private"]["inventories"][1] = {}
    third = controller.apply(after_drop, _action("NORTH"))

    assert first["hands"] == [["SOUTH"]]
    assert second["hands"] == [["DROP"]]
    assert third["hands"] == [["NORTH"]]

    later = _observation()
    later["step"] = 300
    assert controller.apply(later, _action()) == _action()


def test_controller_sells_projected_wool_on_drop_step() -> None:
    controller = AdjacentShedReturnController(
        minimum_inventory_value=2_500,
        liquidation_items=("WOOL",),
    )
    entering = _observation(wool=15, wool_price=220)
    entering["step"] = 164
    controller.apply(entering, _action())

    at_shed = _observation(wool=15, wool_price=198)
    at_shed["step"] = 165
    at_shed["farms"][0]["hands"] = [[4, 4]]
    base = _action("NORTH")
    base["market"] = [["SELL", "WOOL", 2], ["HIRE"]]

    result = controller.apply(at_shed, base)

    assert result["hands"] == [["DROP"]]
    assert result["market"] == [
        ["SELL", "WOOL", 13],
        ["SELL", "WOOL", 2],
        ["HIRE"],
    ]


def test_controller_does_not_sell_unselected_carried_items() -> None:
    controller = AdjacentShedReturnController(
        liquidation_items=("WOOL",),
    )
    entering = _observation()
    entering["private"]["inventories"][1]["FERTILIZER"] = 3
    entering["step"] = 164
    controller.apply(entering, _action())

    at_shed = copy.deepcopy(entering)
    at_shed["step"] = 165
    at_shed["farms"][0]["hands"] = [[4, 4]]

    result = controller.apply(at_shed, _action("NORTH"))

    assert result["market"] == [["SELL", "WOOL", 15]]


def test_controller_resets_intervention_budget_for_new_game() -> None:
    controller = AdjacentShedReturnController()
    observation = _observation()
    observation["step"] = 164
    controller.apply(observation, _action())

    new_game = _observation()
    new_game["step"] = 0

    assert controller.apply(new_game, _action())["hands"] == [["SOUTH"]]


def _orphan_observation(
    step: int,
    *,
    hand: list[int],
    inventory: dict | None = None,
    filled: bool = False,
    fed: bool = False,
    cared: bool = False,
) -> dict:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[2][2] = {
        "kind": "PASTURE",
        "animal": "COW" if filled else None,
        "fed_today": fed,
        "cared_today": cared,
    }
    farm = {
        "farmer": [5, 5],
        "hands": [hand],
        "tiles": tiles,
    }
    return {
        "player": 0,
        "step": step,
        "farms": [farm, {}],
        "private": {
            "inventories": [{}, inventory or {}],
            "shed": {"COW": 1, "WHEAT": 4},
        },
    }


def test_orphan_structure_controller_refills_pasture_after_day_boundary() -> None:
    controller = OrphanStructureRecoveryController()
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    assert controller.apply(
        _orphan_observation(193, hand=[4, 4]), action
    )["hands"] == [["PICKUP", "COW", 1]]
    assert controller.apply(
        _orphan_observation(194, hand=[4, 4], inventory={"COW": 1}), action
    )["hands"] == [["PICKUP", "WHEAT", 4]]
    assert controller.apply(
        _orphan_observation(
            195, hand=[4, 4], inventory={"COW": 1, "WHEAT": 4}
        ),
        action,
    )["hands"] == [["WEST"]]
    assert controller.apply(
        _orphan_observation(
            196, hand=[3, 4], inventory={"COW": 1, "WHEAT": 4}
        ),
        action,
    )["hands"] == [["WEST"]]
    assert controller.apply(
        _orphan_observation(
            197, hand=[2, 4], inventory={"COW": 1, "WHEAT": 4}
        ),
        action,
    )["hands"] == [["NORTH"]]
    assert controller.apply(
        _orphan_observation(
            198, hand=[2, 3], inventory={"COW": 1, "WHEAT": 4}
        ),
        action,
    )["hands"] == [["NORTH"]]
    assert controller.apply(
        _orphan_observation(
            199, hand=[2, 2], inventory={"COW": 1, "WHEAT": 4}
        ),
        action,
    )["hands"] == [["PLACE", "COW", 1]]
    assert controller.apply(
        _orphan_observation(
            200, hand=[2, 2], inventory={"WHEAT": 4}, filled=True
        ),
        action,
    )["hands"] == [["FEED"]]
    assert controller.apply(
        _orphan_observation(
            201,
            hand=[2, 2],
            inventory={"WHEAT": 3},
            filled=True,
            fed=True,
        ),
        action,
    )["hands"] == [["CARE"]]


def test_orphan_structure_controller_ignores_same_day_planned_structure() -> None:
    controller = OrphanStructureRecoveryController()
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    result = controller.apply(_orphan_observation(169, hand=[4, 4]), action)

    assert result == action


def test_local_work_fill_feeds_animal_instead_of_idle_pass() -> None:
    observation = _observation()
    observation["farms"][0]["farmer"] = [2, 2]
    observation["farms"][0]["tiles"][2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": False,
        "cared_today": False,
    }
    observation["private"]["inventories"][0] = {"WHEAT": 1}
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    result = LocalWorkFillRule().apply(observation, action)

    assert result["farmer"] == ["FEED"]


def test_local_work_fill_cares_when_idle_unit_has_no_feed() -> None:
    observation = _observation()
    observation["farms"][0]["farmer"] = [2, 2]
    observation["farms"][0]["tiles"][2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": False,
        "cared_today": False,
    }
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    result = LocalWorkFillRule().apply(observation, action)

    assert result["farmer"] == ["CARE"]


def test_local_work_fill_preserves_non_pass_route_command() -> None:
    observation = _observation()
    observation["farms"][0]["farmer"] = [2, 2]
    observation["farms"][0]["tiles"][2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 1,
        "fed_today": False,
        "cared_today": False,
    }
    action = {"farmer": ["EAST"], "hands": [["PASS"]], "market": []}

    result = LocalWorkFillRule().apply(observation, action)

    assert result["farmer"] == ["EAST"]


def test_local_work_fill_waits_until_minimum_step() -> None:
    observation = _observation()
    observation["step"] = 119
    observation["farms"][0]["farmer"] = [2, 2]
    observation["farms"][0]["tiles"][2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": False,
        "cared_today": False,
    }
    observation["private"]["inventories"][0] = {"WHEAT": 1}
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}
    rule = LocalWorkFillRule(minimum_step=120)

    assert rule.apply(observation, action)["farmer"] == ["PASS"]

    observation["step"] = 120
    assert rule.apply(observation, action)["farmer"] == ["FEED"]


def _capharv_observation(
    *,
    player: int = 0,
    command: list | None = None,
    kind: str = "PASTURE",
    animal: str | None = "COW",
    yield_units: int = 3,
    shed_total: int = 90,
    carried: int = 8,
    hand: list[int] | None = None,
) -> tuple[dict, dict]:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[2][2] = {
        "kind": kind,
        "animal": animal,
        "yield_units": yield_units,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": True,
    }
    farm = {
        "farmer": [2, 2],
        "hands": [hand or [0, 0]],
        "tiles": tiles,
    }
    farms = [farm, {"farmer": [0, 0], "hands": [], "tiles": []}]
    observation = {
        "player": player,
        "step": 220,
        "farms": farms,
        "private": {
            "inventories": [{"MILK": carried}, {}],
            "shed": {"MILK": shed_total},
        },
    }
    if player == 1:
        farms[1] = farm
        farms[0] = {"farmer": [0, 0], "hands": [], "tiles": []}
    action = {"farmer": command or ["CARE"], "hands": [["PASS"]], "market": []}
    return observation, action


def test_capharv_replaces_care_when_midnight_dump_would_overflow() -> None:
    observation, action = _capharv_observation()

    result = CapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]
    assert result["hands"] == [["PASS"]]


def test_capharv_replaces_collect_fertilizer_on_overflow() -> None:
    observation, action = _capharv_observation(command=["COLLECT_FERTILIZER"])

    result = CapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_capharv_leaves_care_when_projected_load_fits_shed() -> None:
    observation, action = _capharv_observation(shed_total=80, carried=8)

    result = CapHarvRule().apply(observation, action)

    assert result == action


def test_capharv_boundary_fits_at_capacity_overflows_above() -> None:
    observation, action = _capharv_observation(
        shed_total=89, carried=8, yield_units=3
    )
    assert CapHarvRule().apply(observation, action) == action

    observation, action = _capharv_observation(
        shed_total=90, carried=8, yield_units=3
    )
    assert CapHarvRule().apply(observation, action)["farmer"] == ["HARVEST"]


def test_capharv_applies_to_second_seat_farm() -> None:
    observation, action = _capharv_observation(player=1)

    result = CapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_capharv_harvests_goose_coop_tile() -> None:
    observation, action = _capharv_observation(kind="COOP", animal="GOOSE")

    result = CapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_capharv_preserves_care_when_tile_has_no_yield() -> None:
    observation, action = _capharv_observation(yield_units=0)

    result = CapHarvRule().apply(observation, action)

    assert result == action


def test_capharv_preserves_care_when_tile_has_no_animal() -> None:
    observation, action = _capharv_observation(animal=None)

    result = CapHarvRule().apply(observation, action)

    assert result == action


def test_capharv_yields_to_already_scheduled_harvest() -> None:
    observation, action = _capharv_observation()
    action["hands"] = [["HARVEST"]]

    result = CapHarvRule().apply(observation, action)

    assert result == action


def test_capharv_ignores_non_maintenance_commands() -> None:
    observation, action = _capharv_observation(command=["FEED"])

    result = CapHarvRule().apply(observation, action)

    assert result == action


def _hourgated_observation(*, step: int, player: int = 0) -> tuple[dict, dict]:
    observation, action = _capharv_observation(player=player)
    observation["step"] = step
    return observation, action


def test_hourgated_blocks_early_day_flip() -> None:
    observation, action = _hourgated_observation(step=220)

    result = HourGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_hourgated_allows_late_day_flip() -> None:
    observation, action = _hourgated_observation(step=23)

    result = HourGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_hourgated_boundary_hour() -> None:
    blocked, blocked_action = _hourgated_observation(step=13)
    assert HourGatedCapHarvRule().apply(blocked, blocked_action) == blocked_action

    allowed, _ = _hourgated_observation(step=14)
    assert HourGatedCapHarvRule().apply(allowed, dict(blocked_action))["farmer"] == [
        "HARVEST"
    ]


def test_hourgated_applies_to_second_seat_farm() -> None:
    observation, action = _hourgated_observation(step=23, player=1)

    result = HourGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_hourgated_fails_closed_without_step() -> None:
    observation, action = _capharv_observation()
    del observation["step"]

    result = HourGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_hourgated_preserves_care_when_late_day_load_fits_shed() -> None:
    observation, action = _capharv_observation(shed_total=80, carried=8)
    observation["step"] = 23

    result = HourGatedCapHarvRule().apply(observation, action)

    assert result == action


def _value_observation(
    *,
    player: int = 0,
    command: list | None = None,
    animal: str = "COW",
    kind: str = "PASTURE",
    yield_units: int = 3,
    shed_total: int = 99,
    carried: int = 8,
    pending: int = 0,
    placed_day: int | None = 0,
    fed_today: bool = True,
    cared_today: bool = False,
    day: int = 10,
    step: int = 23,
    price: float | None = 160.0,
) -> tuple[dict, dict]:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tile: dict = {
        "kind": kind,
        "animal": animal,
        "yield_units": yield_units,
        "fed_today": fed_today,
        "cared_today": cared_today,
        "fertilizer_available": True,
        "pending_care_bonus": pending,
    }
    if placed_day is not None:
        tile["placed_day"] = placed_day
    tiles[2][2] = tile
    farm = {
        "farmer": [2, 2],
        "hands": [[0, 0]],
        "tiles": tiles,
    }
    farms = [farm, {"farmer": [0, 0], "hands": [], "tiles": []}]
    product = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}[animal]
    prices = {} if price is None else {product: price}
    observation = {
        "player": player,
        "step": step,
        "day": day,
        "farms": farms,
        "private": {
            "inventories": [{"MILK": carried}, {}],
            "shed": {"MILK": shed_total},
        },
        "market": {"prices": prices},
    }
    if player == 1:
        farms[1] = farm
        farms[0] = {"farmer": [0, 0], "hands": [], "tiles": []}
    action = {"farmer": command or ["CARE"], "hands": [["PASS"]], "market": []}
    return observation, action


def test_value_gated_flips_when_waste_dominates() -> None:
    # COW placed day 0, obs day 10: next payout next_day 12 -> d=2.
    # bonus = (0 + 1) * 1.0 / 2 = 0.5 units < waste min(3, 10) = 3.
    observation, action = _value_observation()

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_value_gated_keeps_care_when_bonus_dominates() -> None:
    # Overcap of 1 -> waste 1 unit; banked 2 + today's CARE = 3 units,
    # payout tomorrow (day 11 -> next_day 12, d=1): 3 >= 1 keeps CARE.
    observation, action = _value_observation(
        shed_total=90, carried=8, pending=2, day=11
    )

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_value_gated_tie_fails_closed() -> None:
    # Waste exactly 1 unit; bonus exactly (0 + 1) * 1.0 / 1 = 1 -> keep CARE.
    observation, action = _value_observation(shed_total=90, carried=8, day=11)

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_value_gated_applies_to_second_seat() -> None:
    observation, action = _value_observation(player=1)

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_value_gated_blocks_early_day_flip() -> None:
    observation, action = _value_observation(step=220)

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_value_gated_flips_when_max_held_cap_binds() -> None:
    # Tile at the cow cap (6): bonus headroom is 0 -> bonus worthless -> flip.
    observation, action = _value_observation(
        yield_units=6, shed_total=95, carried=8, pending=2
    )

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_value_gated_unfed_discount_tips_marginal_flip() -> None:
    # Same trade as the tie test but unfed: 1 * 0.5 / 1 = 0.5 < 1 -> flip.
    observation, action = _value_observation(
        shed_total=90, carried=8, day=11, fed_today=False
    )

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_value_gated_flips_collect_when_no_bank() -> None:
    # COLLECT banks nothing: bonus 0 < waste 3 -> flip.
    observation, action = _value_observation(command=["COLLECT_FERTILIZER"])

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result["farmer"] == ["HARVEST"]


def test_value_gated_keeps_collect_when_bank_large() -> None:
    # COLLECT with a large pending bank near payout: 3 >= 1 -> keep.
    observation, action = _value_observation(
        command=["COLLECT_FERTILIZER"],
        shed_total=90,
        carried=8,
        pending=3,
        day=11,
    )

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_value_gated_fails_closed_without_price() -> None:
    observation, action = _value_observation(price=None)

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action


def test_value_gated_goose_pays_out_daily() -> None:
    # GOOSE interval 1 -> d=1 always; (1 + 1) * 1.0 / 1 = 2 >= waste 1.
    observation, action = _value_observation(
        animal="GOOSE",
        kind="COOP",
        yield_units=2,
        shed_total=91,
        carried=8,
        pending=1,
        placed_day=6,
        day=11,
        price=50.0,
    )

    result = ValueGatedCapHarvRule().apply(observation, action)

    assert result == action
