import numpy as np

from envs.switching_lbf import SwitchingLBFEnv


def run(ego_mode, initial=(0, 0), switch=(0, 0), n=100):
    returns = []
    positive = []
    action_changes = []
    for seed in range(n):
        env = SwitchingLBFEnv(
            base_key="lbforaging:Foraging-2s-10x10-3p-3f-v3",
            max_episode_steps=50,
            initial_mode_ids=initial,
            switch_mode_ids=switch,
            fixed_switch_step=25,
            seed=seed,
        )
        env.reset(seed=seed)
        total = 0.0
        positive_steps = 0
        prev = None
        changes = 0
        for t in range(50):
            if ego_mode == "noop":
                action = 0
            elif ego_mode == "random":
                action = int(env.env.action_space[0].sample())
            else:
                action = int(env.env.action_space[0].sample())
            _, reward, terminated, truncated, info = env.step([action])
            total += reward
            positive_steps += int(reward > 0)
            current = (info["switching_lbf_teammate_0_mode"], info["switching_lbf_teammate_1_mode"])
            changes += int(prev is not None and current != prev)
            prev = current
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
        positive.append(positive_steps > 0)
        action_changes.append(changes)
    return float(np.mean(returns)), float(np.mean(positive)), float(np.mean(action_changes))


for mode in ((0, 0), (1, 1), (2, 2)):
    print("fixed", mode, "noop", run("noop", mode, mode))
    print("fixed", mode, "random", run("random", mode, mode))
