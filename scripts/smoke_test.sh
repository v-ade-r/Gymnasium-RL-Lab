#!/usr/bin/env bash
# Run every trainer with a short smoke budget (SMOKE=1 / --smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SMOKE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

run() {
  echo ""
  echo "==> $*"
  "$@"
}

ALGO="${ROOT}/algorithms"

run python "${ALGO}/tabular/frozen_lake/Q-learning.py" --smoke
run python "${ALGO}/tabular/frozen_lake/SARSA.py" --smoke
run python "${ALGO}/discrete_control/lunar_lander/PPO.py" --smoke
run python "${ALGO}/continuous_control/halfcheetah/sac/sac.py" --smoke
run python "${ALGO}/continuous_control/halfcheetah/ppo/PPO.py" --smoke
run python "${ALGO}/continuous_control/halfcheetah/evolution/CMA-ES.py" --smoke
run python "${ALGO}/continuous_control/halfcheetah/evolution/NES.py" --smoke
run python "${ALGO}/continuous_control/halfcheetah/evolution/MAP-Elites.py" --smoke

echo ""
echo "All smoke tests passed."
