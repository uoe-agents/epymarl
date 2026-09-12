#!/usr/bin/env bash
set -u

# Fair fixed-condition evaluation of the latest Intent-v1 headroom models.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1

training_root="${TRAINING_ROOT:-results/intent_v1_headroom/artifacts}"
output_dir="${OUTPUT_DIR:-results/intent_v1_checkpoint_eval}"
python_bin="${PYTHON_BIN:-python}"
max_jobs="${MAX_JOBS:-2}"
test_nepisode="${TEST_NEPISODE:-1000}"
seed="${SEED:-0}"
dry_run="${DRY_RUN:-0}"
mkdir -p "$output_dir"

methods=(local oracle last_action)
configs=(mappo type_oracle_mappo last_action_mappo)
env_keys=(
  epymarl/Switching-LBF-Intent-v1
  epymarl/Switching-LBF-TypeOracle-Intent-v1
  epymarl/Switching-LBF-LastAction-Intent-v1
)
conditions=(same right_right wait_wait right_wait)
switch_modes=("[0,0]" "[1,1]" "[2,2]" "[1,2]")

running_jobs() {
  jobs -pr | wc -l
}

wait_for_slot() {
  while [ "$(running_jobs)" -ge "$max_jobs" ]; do
    wait -n || true
  done
}

for method_idx in "${!methods[@]}"; do
  method="${methods[$method_idx]}"
  config="${configs[$method_idx]}"
  env_key="${env_keys[$method_idx]}"
  model_root="$training_root/$method/models"
  latest="$({
    find "$model_root" -regextype posix-extended -type d \
      -regex '.*/[0-9]+' -printf '%f\t%h\n' 2>/dev/null || true
  } | sort -n -k1,1 | tail -1)"
  if [ -z "$latest" ]; then
    echo "Missing numeric checkpoint below $model_root" >&2
    exit 1
  fi
  IFS=$'\t' read -r load_step checkpoint_path <<< "$latest"
  echo "Selected method=$method step=$load_step path=$checkpoint_path"

  for condition_idx in "${!conditions[@]}"; do
    condition="${conditions[$condition_idx]}"
    switch_mode="${switch_modes[$condition_idx]}"
    log_path="$output_dir/${method}__${condition}__seed${seed}.log"
    if grep -q "pymarl Completed" "$log_path" 2>/dev/null; then
      echo "Skipping completed $log_path"
      continue
    fi
    if [ -s "$log_path" ]; then
      echo "Refusing to overwrite incomplete log: $log_path" >&2
      echo "Move it aside after inspection, then restart." >&2
      exit 1
    fi

    eval_seed=$((20000 + condition_idx * 100 + seed))
    command=(
      "$python_bin" src/main.py
      "--config=$config"
      --env-config=gymma
      with
      "env_args.key=$env_key"
      env_args.time_limit=50
      "env_args.initial_mode_ids=[0,0]"
      "env_args.switch_mode_ids=$switch_mode"
      env_args.fixed_switch_step=25
      "checkpoint_path=$checkpoint_path"
      "load_step=$load_step"
      "seed=$eval_seed"
      "test_nepisode=$test_nepisode"
      evaluate=True
      use_cuda=False
      use_wandb=False
    )

    if [ "$dry_run" = "1" ]; then
      printf 'DRY RUN -> '
      printf '%q ' "${command[@]}"
      printf '> %q 2>&1\n' "$log_path"
      continue
    fi

    wait_for_slot
    echo "Starting method=$method condition=$condition seed=$seed"
    "${command[@]}" >"$log_path" 2>&1 &
  done
done

wait
if [ "$dry_run" = "1" ]; then
  echo "Dry run completed; no evaluations were started."
  exit 0
fi
echo "All Intent-v1 checkpoint evaluations finished."
"$python_bin" scripts/summarize_switching_recovery.py --log-dir "$output_dir"
