"""V20 repaired opening followed by the V12 route."""

from artifacts.top20_taiseiu_tape_population4_20260825.tape_99297663 import agent as _raw_v20
from kaggriculture_rl.policy.fixed_handoff import FixedHandoffPolicy
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
)
from submissions.top1_v12_ryo_b96_dual_b144_retriever_exact_safe_local.main import (
    agent as _v12,
)

_RECOVERY = WeedRecoveryController()


def _v20(observation, configuration=None):
    proposed = _raw_v20(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    return append_terminal_liquidation(observation, repaired)


agent = FixedHandoffPolicy(_v20, _v12, boundary=1)
