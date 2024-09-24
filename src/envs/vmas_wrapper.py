from pathlib import Path

import gymnasium as gym

import vmas


class VMASWrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, env_name, **kwargs):
        self._env = vmas.make_env(
            env_name,
            num_envs=1,
            continuous_actions=False,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            **kwargs,
        )

        self.n_agents = self._env.unwrapped.n_agents

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _compress_info(self, info):
        if any(isinstance(i, dict) for i in info.values()):
            # info is nested dict --> flatten
            return {f"{key}/{k}": v for key, i in info.items() for k, v in i.items()}
        else:
            return info

    def reset(self, *args, **kwargs):
        obss, info = self._env.reset(*args, **kwargs)
        return obss, self._compress_info(info)

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def step(self, actions):
        obss, rews, done, truncated, info = self._env.step(actions)
        return obss, rews, done, truncated, self._compress_info(info)

    def close(self):
        return self._env.close()


# import all files within the pettingzoo library that match "**/*_v?.py" underneath library of pettingzoo
envs = Path(vmas.__path__[0]).glob("scenarios/**/*.py")
for env in envs:
    if "__" in env.stem:
        continue
    name = env.stem
    gym.register(
        f"vmas-{name}",
        entry_point="envs.vmas_wrapper:VMASWrapper",
        kwargs={
            "env_name": name,
        },
    )
