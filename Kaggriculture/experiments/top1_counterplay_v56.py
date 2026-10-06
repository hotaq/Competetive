"""v56 champion: guarded weed recovery and gated adaptive counterplay."""

from experiments.top1_counterplay_core_v56 import make_agent


agent = make_agent(
    "value_sell_first",
    "behind_floor",
    lead_gap=0,
    floor_scale=0.6,
    front_run={"STRAWBERRY": 215},
    activation_gap=100,
    activation_day=12,
    activation_deadband=(800, 1300),
)
v56_submission_agent = agent
