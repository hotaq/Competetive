from artifacts.v68_front_run_robust_20260826.overlay import make_agent

agent = make_agent(
    ("MELON", "MILK", "STRAWBERRY", "WOOL"),
    quantity_scale=0.5,
    carry_debt=True,
    reserve_future_pickups=True,
)
