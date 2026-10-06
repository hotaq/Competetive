#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${LAB_DIR:-$ROOT_DIR/.omx/research/upstream/orbit-wars-lab}"
GAMES="${1:-1}"
SEED="${2:-777}"
SAVE_REPLAYS="${SAVE_REPLAYS:-0}"
OPPONENTS="${OPPONENTS:-external/pilkwang-structured,external/tamrazov-starwars,external/yuriygreben-architect}"

if [[ ! -d "$LAB_DIR/.venv" ]]; then
  "$ROOT_DIR/scripts/setup_orbit_wars_lab.sh"
else
  "$ROOT_DIR/scripts/sync_lab_agent.sh"
fi

cd "$LAB_DIR"
source .venv/bin/activate

export OWL_GAMES="$GAMES"
export OWL_SEED="$SEED"
export OWL_SAVE_REPLAYS="$SAVE_REPLAYS"
export OWL_OPPONENTS="$OPPONENTS"

python - <<'PY'
import json
import os
from pathlib import Path

from orbit_wars_app.schemas import TournamentConfig
from orbit_wars_app.tournament import Tournament

games = int(os.environ["OWL_GAMES"])
seed = int(os.environ["OWL_SEED"])
save_replays = os.environ.get("OWL_SAVE_REPLAYS", "0").lower() in {"1", "true", "yes", "on"}
opponents = [item.strip() for item in os.environ["OWL_OPPONENTS"].split(",") if item.strip()]
challenger = "mine/current-w1b-counter"

cfg = TournamentConfig(
    agents=[challenger, *opponents],
    games_per_pair=games,
    mode="fast",
    format="2p",
    parallel=1,
    seed_base=seed,
    save_replays=save_replays,
    shape="gauntlet",
    challenger_id=challenger,
)

run_id = Tournament(cfg, runs_root=Path("runs"), zoo_root=Path("agents")).run()
run = Path("runs") / run_id
res = json.loads((run / "results.json").read_text())
print("RUN", run)
print(json.dumps(res["summary"], indent=2))
for m in res["matches"]:
    print(
        m["match_id"],
        m["agent_ids"],
        "winner=", m["winner"],
        "scores=", m["scores"],
        "turns=", m["turns"],
        "seed=", m["seed"],
        "status=", m["status"],
        "replay=", m.get("replay_path", ""),
    )
PY
