#!/usr/bin/env bash
set -u

# Stage-3 gate: train deterministic Belief on coordinated Intent-v2 only
# after Type-Oracle has beaten both Local and Last-action baselines.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1

output_dir="${OUTPUT_DIR:-results/intent_v2_headroom}"
python_bin="${PYTHON_BIN:-python}"
t_max="${T_MAX:-500000}"
seed="${SEED:-0}"
dry_run="${DRY_RUN:-0}"
method="belief"
log_path="$output_dir/logs/${method}__seed${seed}.log"
result_path="$output_dir/artifacts/$method"
mkdir -p "$output_dir/logs" "$result_path"

if grep -q "pymarl Completed" "$log_path" 2>/dev/null; then
  echo "Skipping completed $log_path"
  exit 0
fi
if [ -s "$log_path" ]; then
  echo "Refusing to overwrite incomplete log: $log_path" >&2
  echo "Move it aside after inspection, then restart." >&2
  exit 1
fi

command=(
  "$python_bin" src/main.py
  --config=deterministic_belief_mappo
  --env-config=gymma
  with
  env_args.key=epymarl/Switching-LBF-Belief-Intent-v2
  env_args.time_limit=50
  "seed=$seed"
  "t_max=$t_max"
  test_nepisode=100
  test_interval=50000
  log_interval=10000
  save_model=True
  save_model_interval=50000
  "local_results_path=$result_path"
  use_wandb=False
)

if [ "$dry_run" = "1" ]; then
  printf 'DRY RUN -> '
  printf '%q ' "${command[@]}"
  printf '> %q 2>&1\n' "$log_path"
  exit 0
fi

echo "Starting method=$method seed=$seed env=Intent-v2"
"${command[@]}" >"$log_path" 2>&1
if ! grep -q "pymarl Completed" "$log_path"; then
  echo "Incomplete training log: $log_path" >&2
  exit 1
fi
echo "Intent-v2 deterministic Belief training completed."
