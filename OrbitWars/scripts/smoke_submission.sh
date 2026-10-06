#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-4242}"
STEPS="${2:-80}"
SUBMIT_FILE="${SUBMIT_FILE:-submit_current_w1b_counter.py}"

echo "Smoke-validating ${SUBMIT_FILE} with self-play"
echo "  seed:  $SEED"
echo "  steps: $STEPS"

python3 -m py_compile main.py "$SUBMIT_FILE"

uv run --no-project --python 3.11 \
  --with kaggle-environments \
  --with numpy \
  python - "$SEED" "$STEPS" <<'PY'
import importlib.util
import os
import pathlib
import sys

from kaggle_environments import make

seed = int(sys.argv[1])
steps = int(sys.argv[2])
path = pathlib.Path(os.environ.get("SUBMIT_FILE", "submit_current_w1b_counter.py")).resolve()
spec = importlib.util.spec_from_file_location("submission_smoke", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

env = make("orbit_wars", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
env.run([module.agent, module.agent])
print("VALIDATION_SMOKE", [str(state.status) for state in env.state], [state.reward for state in env.state], len(env.steps))
PY
