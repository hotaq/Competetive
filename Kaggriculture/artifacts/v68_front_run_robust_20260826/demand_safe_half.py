from artifacts.v68_front_run_robust_20260826.overlay import make_agent

agent = make_agent(
    ("MELON", "MILK", "STRAWBERRY", "WOOL"),
    quantity_scale=0.5,
    skip_current_town_demand=True,
    sum_existing_sells=True,
)
