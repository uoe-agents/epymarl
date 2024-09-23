from pathlib import Path
import importlib

import gymnasium as gym
from gymnasium.spaces import Tuple

import pettingzoo


class PettingZooWrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, lib_name, env_name, **kwargs):
        env = importlib.import_module(f"pettingzoo.{lib_name}.{env_name}")
        self._env = env.parallel_env(**kwargs)
        self._env.reset()

        self.n_agents = self._env.num_agents
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple([obs[k] for k in self._env.agents])
        self.last_obs = obs
        return obs, info

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        done = all([dones[k] for k in self._env.agents])
        truncated = all([truncated[k] for k in self._env.agents])
        info = {
            f"{k}_{key}": value
            for k in self._env.agents
            for key, value in infos[k].items()
        }
        if done:
            # empty obs and rewards for PZ environments on terminated episode
            assert len(obs) == 0
            assert len(rewards) == 0
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()


# import all files within the pettingzoo library that match "**/*_v?.py" underneath library of pettingzoo
envs = Path(pettingzoo.__path__[0]).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    gym.register(
        gymkey,
        entry_point="envs.pz_wrapper:PettingZooWrapper",
        kwargs={
            "lib_name": lib,
            "env_name": filename,
        },
    )
