import numpy as np

from envs.switching_lbf import SwitchingLBFEnv


def snapshot(env):
    players = []
    for player in env.base_env.players:
        players.append(None if player.position is None else tuple(int(v) for v in player.position))
    foods = env._food_positions()
    return players, foods


def greedy_action(env, t):
    del t
    foods = env._food_positions()
    player = env.base_env.players[0]
    if not foods or player.position is None:
        return 0
    pos = tuple(int(v) for v in player.position)
    target = min(foods, key=lambda f: (abs(pos[0] - f[0]) + abs(pos[1] - f[1]), f))
    return env._move_action(0, target)


def run(policy, seed):
    env = SwitchingLBFEnv(
        base_key="lbforaging:Foraging-2s-10x10-3p-3f-v3",
        max_episode_steps=50,
        initial_mode_ids=(0, 0),
        switch_mode_ids=(2, 2),
        fixed_switch_step=25,
        seed=seed,
    )
    env.reset(seed=seed)
    print("initial", snapshot(env), "switch", env._switch_step)
    total = 0.0
    for t in range(50):
        action = policy(env, t)
        before = snapshot(env)
        _, reward, terminated, truncated, info = env.step([action])
        after = snapshot(env)
        total += reward
        if seed == 0:
            print("t", t, "action", action, "reward", reward, "before", before, "after", after, "switched", info["switching_lbf_switched"])
        if terminated or truncated:
            break
    env.close()
    return total


def random_policy(env, t):
    del t
    return int(env.env.action_space[0].sample())


def main():
    for name, policy in (("greedy", greedy_action), ("random", random_policy)):
        values = [run(policy, seed) for seed in range(10)]
        print(name, "returns", values, "mean", float(np.mean(values)), "positive", float(np.mean(np.asarray(values) > 0)))


if __name__ == "__main__":
    main()
