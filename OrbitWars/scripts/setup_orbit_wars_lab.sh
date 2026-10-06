#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${LAB_DIR:-$ROOT_DIR/.omx/research/upstream/orbit-wars-lab}"
LAB_REPO="${LAB_REPO:-https://github.com/automatylicza/orbit-wars-lab.git}"
KAGGLE_ENV_SHA="${KAGGLE_ENV_SHA:-27660ecca553b96b1e3d6b282ca855fd1f274847}"

mkdir -p "$(dirname "$LAB_DIR")"

if [[ -d "$LAB_DIR/.git" ]]; then
  echo "Updating Orbit Wars Lab: $LAB_DIR"
  git -C "$LAB_DIR" pull --ff-only
else
  echo "Cloning Orbit Wars Lab: $LAB_DIR"
  git clone "$LAB_REPO" "$LAB_DIR"
fi

cd "$LAB_DIR"
uv venv --python 3.12 .venv
source .venv/bin/activate

python - <<'PY' >/dev/null 2>&1 && NEED_INSTALL=0 || NEED_INSTALL=1
import kaggle_environments
import orbit_wars_app
import trueskill
PY

if [[ "$NEED_INSTALL" == "1" ]]; then
  echo "Installing Orbit Wars Lab deps with uv"
  uv pip install -e . --no-deps
  uv pip install numpy pyyaml pydantic trueskill fastapi "uvicorn[standard]" httpx psutil "kaggle~=2.0.2"
  uv pip install "kaggle-environments @ git+https://github.com/Kaggle/kaggle-environments.git@${KAGGLE_ENV_SHA}"
else
  echo "Orbit Wars Lab deps already installed"
fi

"$ROOT_DIR/scripts/sync_lab_agent.sh"

python -m orbit_wars_app.tournament --zoo agents --runs runs list | sed -n '1,40p'
