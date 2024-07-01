from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Tuple
import torch

from .ddpg import DDPG


class PretrainedAdversary(gym.Wrapper):
    """Adversary with pretrained adversary agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[0]
        self.pt_observation_space = self.observation_space[0]
        self.action_space = Tuple(self.action_space[1:])
        self.observation_space = Tuple(self.observation_space[1:])
        self.n_agents = 2
        self.unwrapped.n_agents = 2

        self.adv = DDPG(8, 5, 50, 64, 0.01)
        param_path = Path(__file__).parent / "adv_params.pt"
        save_dict = torch.load(param_path)
        self.adv.load_params(save_dict["agent_params"][0])
        self.adv.policy.eval()
        self.last_adv_obs = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_adv_obs = obs[0]
        return obs[1:], info

    def step(self, action):
        adv_action = self.adv.step(self.last_adv_obs)
        action = (adv_action,) + tuple(action)
        obs, rew, done, truncated, info = super().step(action)
        self.last_adv_obs = obs[0]
        obs = obs[1:]
        rew = rew[1:]
        return obs, rew, done, truncated, info
