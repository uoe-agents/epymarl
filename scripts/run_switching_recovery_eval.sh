#!/usr/bin/env bash
set -u

# Run from the EPyMARL repository root. Existing completed logs are skipped so
# this launcher can be restarted safely after a disconnect.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1

output_dir="${OUTPUT_DIR:-results/checkpoint_eval_recovery}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/switching_lbf}"
mkdir -p "$output_dir"

max_jobs="${MAX_JOBS:-2}"
test_nepisode="${TEST_NEPISODE:-1000}"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"

methods=(local oracle last_action belief)
configs=(mappo type_oracle_mappo last_action_mappo deterministic_belief_mappo)
env_keys=(
  epymarl/Switching-LBF-v0
  epymarl/Switching-LBF-TypeOracle-v0
  epymarl/Switching-LBF-LastAction-v0
  epymarl/Switching-LBF-Belief-v0
)

local_steps=(1802942 1802482 1802066)
oracle_steps=(1802652 1802638 1802234)
last_action_steps=(1802435 1803347 1802510)
belief_steps=(1802586 1803343 1803146)

conditions=(same left_left wait_wait left_wait)
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
  step_array_name="${method}_steps[@]"
  steps=("${!step_array_name}")

  for seed in 0 1 2; do
    load_step="${steps[$seed]}"
    checkpoint_dir="$(
      find "$checkpoint_root/${method}_seed${seed}/models" \
        -type d -name "$load_step" -print -quit 2>/dev/null
    )"
    if [ -z "$checkpoint_dir" ]; then
      echo "Missing checkpoint: method=$method seed=$seed step=$load_step" >&2
      exit 1
    fi
    checkpoint_path="$(dirname "$checkpoint_dir")"

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
        echo "Move it aside, then restart this launcher." >&2
        exit 1
      fi

      # Keep evaluation environment randomness matched across methods for each
      # condition and policy seed.
      eval_seed=$((10000 + condition_idx * 100 + seed))
      command=(
        "$python_bin" src/main.py
        "--config=$config"
        --env-config=gymma
        with
        "env_args.key=$env_key"
        env_args.time_limit=50
        "env_args.initial_mode_ids=[0,0]"
        "env_args.switch_mode_ids=$switch_mode"
        env_args.fixed_switch_step=10
        "checkpoint_path=$checkpoint_path"
        "load_step=$load_step"
        "seed=$eval_seed"
        "test_nepisode=$test_nepisode"
        evaluate=True
        use_cuda=False
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
done

wait
if [ "$dry_run" = "1" ]; then
  echo "Dry run completed; no evaluations were started."
  exit 0
fi
echo "All recovery evaluations finished."
python scripts/summarize_switching_recovery.py --log-dir "$output_dir"
