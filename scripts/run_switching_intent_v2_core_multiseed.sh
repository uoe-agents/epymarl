#!/usr/bin/env bash
set -u

# Stage-4 core gate: train only Last-action and deterministic Belief for the
# requested additional seeds. The two methods for each seed run concurrently;
# the next seed starts only after both have completed successfully.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root" || exit 1

output_dir="${OUTPUT_DIR:-results/intent_v2_headroom}"
python_bin="${PYTHON_BIN:-python}"
t_max="${T_MAX:-500000}"
seeds="${SEEDS:-1 2}"
dry_run="${DRY_RUN:-0}"

methods=(last_action belief)
scripts=(
  scripts/run_switching_intent_v2_last_action.sh
  scripts/run_switching_intent_v2_belief.sh
)

for seed in $seeds; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed in SEEDS: $seed" >&2
    exit 1
  fi

  pids=()
  labels=()
  for method_idx in "${!methods[@]}"; do
    method="${methods[$method_idx]}"
    script="${scripts[$method_idx]}"
    echo "Starting core training method=$method seed=$seed"
    env \
      OUTPUT_DIR="$output_dir" \
      PYTHON_BIN="$python_bin" \
      T_MAX="$t_max" \
      SEED="$seed" \
      DRY_RUN="$dry_run" \
      bash "$script" &
    pids+=("$!")
    labels+=("$method seed=$seed")
  done

  failed=0
  for job_idx in "${!pids[@]}"; do
    if ! wait "${pids[$job_idx]}"; then
      echo "Core training failed: ${labels[$job_idx]}" >&2
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    exit 1
  fi
done

if [ "$dry_run" = "1" ]; then
  echo "Core multi-seed dry run completed; no training was started."
else
  echo "All core Intent-v2 multi-seed trainings completed."
fi
