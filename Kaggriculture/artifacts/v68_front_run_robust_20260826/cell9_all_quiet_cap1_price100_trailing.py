from artifacts.v68_front_run_robust_20260826.overlay import make_agent

agent = make_agent(
    ("MELON", "MILK", "STRAWBERRY", "WOOL"),
    skip_current_town_demand=True,
    sum_existing_sells=True,
    max_quantity_per_item=1,
    price_floor_ratio=1.0,
    trailing_only=True,
    correct_future_clock=True,
)
