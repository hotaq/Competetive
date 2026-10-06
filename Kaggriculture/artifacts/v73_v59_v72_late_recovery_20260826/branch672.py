"""V73 fixed splice: V59 before step 672, fresh V72 afterward."""

from artifacts.v73_v59_v72_late_recovery_20260826.overlay import make_agent

BRANCH_STEP = 672
agent = make_agent(BRANCH_STEP)

