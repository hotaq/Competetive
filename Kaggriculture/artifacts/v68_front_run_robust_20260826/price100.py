from artifacts.v68_front_run_robust_20260826.overlay import make_agent

agent = make_agent(
    ("MELON", "MILK", "STRAWBERRY", "WOOL"),
    price_floor_ratio=1.0,
)
