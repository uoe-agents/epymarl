#!/usr/bin/env python3
"""Behavior gate for historical and candidate Switching-LBF versions.

This deliberately uses simple fixed policies rather than trained checkpoints.
It guards the environment semantics that the learning experiments depend on.
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODULE_PATH = PROJECT_ROOT / "src" / "envs" / "switching_lbf.py"
MODULE_SPEC = importlib.util.spec_from_file_location("switching_lbf_gate", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Cannot load Switching-LBF module from {MODULE_PATH}")
SWITCHING_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(SWITCHING_MODULE)
SwitchingLBFEnv = SWITCHING_MODULE.SwitchingLBFEnv


EPISODE_LIMIT = 50
SWITCH_STEP = 25
CONDITIONS = ((0, 0), (1, 1), (2, 2))


def oracle_action(env):
    """Follow the target implied by the currently revealed common mode."""
    mode = env._current_modes[0]
    return env._move_action(0, env._choose_target(0, mode))


def evaluate(label, env_kwargs, policy, episodes):
    returns = []
    positives = []
    load_counts = []
    foods_collected = []
    for mode_ids in CONDITIONS:
        env = SwitchingLBFEnv(
            max_episode_steps=EPISODE_LIMIT,
            initial_mode_ids=mode_ids,
            switch_mode_ids=mode_ids,
            fixed_switch_step=SWITCH_STEP,
            seed=0,
            **env_kwargs,
        )
        try:
            for episode in range(episodes):
                env.reset(seed=episode)
                episode_return = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = 0 if policy == "noop" else oracle_action(env)
                    _, reward, terminated, truncated, info = env.step([action])
                    episode_return += reward
                returns.append(episode_return)
                positives.append(episode_return > 0)
                load_counts.append(info["switching_lbf_teammate_load_count"])
                foods_collected.append(info["switching_lbf_foods_collected"])
        finally:
            env.close()
    return {
        "label": label,
        "episodes": len(returns),
        "return_mean": float(np.mean(returns)),
        "positive_rate": float(np.mean(positives)),
        "teammate_load_mean": float(np.mean(load_counts)),
        "foods_collected_mean": float(np.mean(foods_collected)),
    }


def shared_mode_rates(env_kwargs, episodes):
    initial_shared = []
    switched_shared = []
    env = SwitchingLBFEnv(
        max_episode_steps=EPISODE_LIMIT,
        fixed_switch_step=1,
        seed=0,
        **env_kwargs,
    )
    try:
        for episode in range(episodes):
            env.reset(seed=episode)
            initial_shared.append(env._current_modes[0] == env._current_modes[1])
            env.step([0])
            env.step([0])
            switched_shared.append(env._current_modes[0] == env._current_modes[1])
    finally:
        env.close()
    return float(np.mean(initial_shared)), float(np.mean(switched_shared))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/switching_lbf_version_gate.csv"),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    fixed_kwargs = {"use_load_positions": True}
    intent_kwargs = {
        "base_key": "lbforaging:Foraging-2s-10x10-3p-3f-coop-v3",
        "teammate_modes": ("left_priority", "right_priority", "wait"),
        "use_load_positions": True,
    }
    coordinated_kwargs = {**intent_kwargs, "shared_teammate_mode": True}
    rows = [
        evaluate("legacy_v0_noop", {}, "noop", args.episodes),
        evaluate("fixed_v1_noop", fixed_kwargs, "noop", args.episodes),
        evaluate("intent_v1_noop", intent_kwargs, "noop", args.episodes),
        evaluate("intent_v1_oracle_heuristic", intent_kwargs, "oracle", args.episodes),
        evaluate("intent_v2_noop", coordinated_kwargs, "noop", args.episodes),
        evaluate(
            "intent_v2_oracle_heuristic",
            coordinated_kwargs,
            "oracle",
            args.episodes,
        ),
    ]
    v1_shared_rates = shared_mode_rates(intent_kwargs, args.episodes)
    v2_shared_rates = shared_mode_rates(coordinated_kwargs, args.episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']:<28} n={row['episodes']:4d} "
            f"return={row['return_mean']:.4f} "
            f"positive={row['positive_rate']:.3f} "
            f"loads={row['teammate_load_mean']:.2f} "
            f"foods={row['foods_collected_mean']:.3f}"
        )
    print(
        "intent_v1_shared_mode_rate "
        f"initial={v1_shared_rates[0]:.3f} switched={v1_shared_rates[1]:.3f}"
    )
    print(
        "intent_v2_shared_mode_rate "
        f"initial={v2_shared_rates[0]:.3f} switched={v2_shared_rates[1]:.3f}"
    )

    by_label = {row["label"]: row for row in rows}
    failures = []
    if by_label["legacy_v0_noop"]["teammate_load_mean"] != 0:
        failures.append("historical v0 behavior changed")
    if by_label["fixed_v1_noop"]["return_mean"] <= 0:
        failures.append("Fixed-v1 teammates did not collect food")
    if by_label["intent_v1_noop"]["return_mean"] != 0:
        failures.append("Intent-v1 permits reward without ego participation")
    if by_label["intent_v1_oracle_heuristic"]["return_mean"] <= 0.05:
        failures.append("Intent-v1 oracle heuristic has insufficient reward")
    if by_label["intent_v2_noop"]["return_mean"] != 0:
        failures.append("Intent-v2 permits reward without ego participation")
    if by_label["intent_v2_oracle_heuristic"]["return_mean"] <= 0.05:
        failures.append("Intent-v2 oracle heuristic has insufficient reward")
    if v1_shared_rates[0] != 0:
        failures.append("Intent-v1 no longer reproduces independent initial modes")
    if v2_shared_rates != (1.0, 1.0):
        failures.append("Intent-v2 teammates do not keep a shared mode")

    print(f"saved={args.output}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: version semantics and coordinated Intent-v2 prerequisites hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
