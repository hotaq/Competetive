"""Current Arman route with semantic recovery and projected shed sales."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99566826_s0 import (
    agent as _tape,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
    recover_rejected_unit_intents,
)
from kaggriculture_rl.policy.unit_rules import AdjacentShedReturnController

_RECOVERY = WeedRecoveryController()
_SHED_RETURN = AdjacentShedReturnController(
    minimum_inventory_value=2_500.0,
    maximum_interventions_per_game=1,
    liquidation_items=("WOOL",),
)


def agent(observation, configuration=None):
    proposed = _tape(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    legalized = legalize_action(observation, liquidated, configuration)
    semantic = recover_rejected_unit_intents(
        observation,
        liquidated,
        legalized,
        configuration,
    )
    return _SHED_RETURN.apply(observation, semantic)
