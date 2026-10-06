"""V20 repaired opening followed by the V16 route."""

from artifacts.top20_taiseiu_tape_population4_20260825.tape_99297663 import agent as _raw_v20
from kaggriculture_rl.policy.fixed_handoff import FixedHandoffPolicy
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
)
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _raw_v16

_RECOVERY = WeedRecoveryController()


def _v20(observation, configuration=None):
    proposed = _raw_v20(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    return append_terminal_liquidation(observation, repaired)


def _v16(observation, configuration=None):
    del configuration
    return _raw_v16(observation)


agent = FixedHandoffPolicy(_v20, _v16, boundary=1)
