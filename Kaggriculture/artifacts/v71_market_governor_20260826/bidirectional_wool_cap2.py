"""V71: early WOOL exit plus a conservative town-demand reserve branch."""

from artifacts.v71_market_governor_20260826.overlay import make_agent

agent = make_agent(reserve_items=("WOOL",))

