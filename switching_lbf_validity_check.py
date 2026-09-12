import csv
from collections import deque
from pathlib import Path

import numpy as np

from envs.switching_lbf import SwitchingLBFEnv


MODES = ("greedy", "left_priority", "wait")
N_EPISODES = 1000
EPISODE_LIMIT = 50
SWITCH_STEP = 25


def ego_action(env):
    foods = env._food_positions()
    if not foods:
        return 0
    player = env.base_env.players[0]
    if player.position is None:
        return 0
    pos = tuple(int(v) for v in player.position)
    occupied = {
        tuple(int(v) for v in player.position)
        for player in env.base_env.players[1:]
        if player.position is not None
    }
    height, width = env.base_env.field.shape[:2]
    directions = ((-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4))
    best = None
    for target in sorted(
        foods,
        key=lambda food: (abs(pos[0] - food[0]) + abs(pos[1] - food[1]), food),
    ):
        queue = deque([(pos, [])])
        visited = {pos}
        path = None
        while queue:
            current, current_path = queue.popleft()
            if current == target:
                path = current_path
                break
            for dr, dc, action in directions:
                nxt = (current[0] + dr, current[1] + dc)
                if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                    continue
                if nxt in occupied and nxt != target:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, current_path + [action]))
        if path is not None:
            best = path
            break
    if best:
        return best[0]
    if pos in foods:
        return env.env.action_space[0].n - 1
    return 0


def run_condition(initial_modes, switch_modes):
    before_returns = []
    after_returns = []
    total_returns = []
    before_positive = []
    after_positive = []
    switched_flags = []

    env = SwitchingLBFEnv(
        base_key="lbforaging:Foraging-2s-10x10-3p-3f-v3",
        max_episode_steps=EPISODE_LIMIT,
        initial_mode_ids=initial_modes,
        switch_mode_ids=switch_modes,
        fixed_switch_step=SWITCH_STEP,
        seed=0,
    )
    for episode in range(N_EPISODES):
        env.reset(seed=episode)
        before_return = 0.0
        after_return = 0.0
        before_positive_steps = 0
        after_positive_steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = ego_action(env)
            _, reward, terminated, truncated, info = env.step([action])
            if info["switching_lbf_step"] <= SWITCH_STEP:
                before_return += reward
                before_positive_steps += int(reward > 0)
            else:
                after_return += reward
                after_positive_steps += int(reward > 0)
        before_returns.append(before_return)
        after_returns.append(after_return)
        total_returns.append(before_return + after_return)
        before_positive.append(before_positive_steps > 0)
        after_positive.append(after_positive_steps > 0)
        switched_flags.append(info["switching_lbf_switched"])
    env.close()

    return {
        "initial": "+".join(MODES[i] for i in initial_modes),
        "switch": "+".join(MODES[i] for i in switch_modes),
        "before_mean": float(np.mean(before_returns)),
        "after_mean": float(np.mean(after_returns)),
        "total_mean": float(np.mean(total_returns)),
        "before_std": float(np.std(before_returns)),
        "after_std": float(np.std(after_returns)),
        "before_positive_rate": float(np.mean(before_positive)),
        "after_positive_rate": float(np.mean(after_positive)),
        "switched_rate": float(np.mean(switched_flags)),
    }


def main():
    conditions = [
        ((0, 0), (0, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (2, 2)),
        ((0, 0), (1, 2)),
        ((1, 1), (1, 1)),
        ((2, 2), (2, 2)),
    ]
    rows = []
    for initial_modes, switch_modes in conditions:
        row = run_condition(initial_modes, switch_modes)
        rows.append(row)
        print(
            f"{row['initial']:>22} -> {row['switch']:<22} "
            f"before={row['before_mean']:.4f} "
            f"after={row['after_mean']:.4f} "
            f"total={row['total_mean']:.4f} "
            f"positive={row['before_positive_rate']:.3f}/"
            f"{row['after_positive_rate']:.3f}",
            flush=True,
        )

    output = Path("results/switching_lbf_validity_1000.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
