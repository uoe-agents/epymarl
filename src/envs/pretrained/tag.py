from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Tuple
import torch

from .ddpg import DDPG


class FrozenTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3
        self.unwrapped.n_agents = 3

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        return obss[:-1], info

    def step(self, action):
        random_action = 0
        action = tuple(action) + (random_action,)
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        return obs, rew, done, truncated, info


class RandomTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3
        self.unwrapped.n_agents = 3

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs[:-1], info

    def step(self, action):
        random_action = self.pt_action_space.sample()
        action = tuple(action) + (random_action,)
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        return obs, rew, done, truncated, info


class PretrainedTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3
        self.unwrapped.n_agents = 3

        self.prey = DDPG(14, 5, 50, 128, 0.01)
        # current file dir
        param_path = Path(__file__).parent / "prey_params.pt"
        save_dict = torch.load(param_path)
        self.prey.load_params(save_dict["agent_params"][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_prey_obs = obs[-1]
        return obs[:-1], info

    def step(self, action):
        prey_action = self.prey.step(self.last_prey_obs)
        action = tuple(action) + (prey_action,)
        obs, rew, done, truncated, info = super().step(action)
        self.last_prey_obs = obs[-1]
        obs = obs[:-1]
        rew = rew[:-1]
        return obs, rew, done, truncated, info
