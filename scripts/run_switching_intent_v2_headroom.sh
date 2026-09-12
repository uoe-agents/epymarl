#!/usr/bin/env bash
set -u

# Stage-1 gate for coordinated Intent-v2. Train only Local and Type-Oracle;
# Last-action and Belief are forbidden until an Oracle gap is demonstrated.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1

output_dir="${OUTPUT_DIR:-results/intent_v2_headroom}"
python_bin="${PYTHON_BIN:-python}"
max_jobs="${MAX_JOBS:-1}"
t_max="${T_MAX:-500000}"
seed="${SEED:-0}"
dry_run="${DRY_RUN:-0}"
mkdir -p "$output_dir/logs" "$output_dir/artifacts"

methods=(local oracle)
configs=(mappo type_oracle_mappo)
env_keys=(
  epymarl/Switching-LBF-Intent-v2
  epymarl/Switching-LBF-TypeOracle-Intent-v2
)

running_jobs() {
  jobs -pr | wc -l
}

wait_for_slot() {
  while [ "$(running_jobs)" -ge "$max_jobs" ]; do
    wait -n || true
  done
}

for idx in "${!methods[@]}"; do
  method="${methods[$idx]}"
  config="${configs[$idx]}"
  env_key="${env_keys[$idx]}"
  log_path="$output_dir/logs/${method}__seed${seed}.log"
  result_path="$output_dir/artifacts/$method"

  if grep -q "pymarl Completed" "$log_path" 2>/dev/null; then
    echo "Skipping completed $log_path"
    continue
  fi
  if [ -s "$log_path" ]; then
    echo "Refusing to overwrite incomplete log: $log_path" >&2
    echo "Move it aside after inspection, then restart." >&2
    exit 1
  fi

  command=(
    "$python_bin" src/main.py
    "--config=$config"
    --env-config=gymma
    with
    "env_args.key=$env_key"
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
    continue
  fi

  wait_for_slot
  echo "Starting method=$method seed=$seed env=$env_key"
  "${command[@]}" >"$log_path" 2>&1 &
done

wait
if [ "$dry_run" = "1" ]; then
  echo "Dry run completed; no training was started."
  exit 0
fi

failed=0
for method in "${methods[@]}"; do
  log_path="$output_dir/logs/${method}__seed${seed}.log"
  if ! grep -q "pymarl Completed" "$log_path"; then
    echo "Incomplete training log: $log_path" >&2
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  exit 1
fi
echo "All coordinated Intent-v2 headroom trainings completed."
