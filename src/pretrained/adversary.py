import gym
from gym.spaces import Tuple
from pretrained.ddpg import DDPG
import torch
import os

class PretrainedAdversary(gym.Wrapper):
    """ Adversary with pretrained adversary agent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[0]
        self.pt_observation_space = self.observation_space[0]
        self.action_space = Tuple(self.action_space[1:])
        self.observation_space = Tuple(self.observation_space[1:])
        self.n_agents = 2

        self.adv = DDPG(8, 5, 50, 64, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), 'adv_params.pt')
        save_dict = torch.load(param_path)
        self.adv.load_params(save_dict['agent_params'][0])
        self.adv.policy.eval()
        self.last_adv_obs = None


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_adv_obs = obs[0]
        return obs[1:]

    def step(self, action):
        adv_action = self.adv.step(self.last_adv_obs)
        action = (adv_action, ) + tuple(action)
        obs, rew, done, info = super().step(action)
        self.last_adv_obs = obs[0]
        obs = obs[1:]
        rew = rew[1:]
        done = done[1:]
        return obs, rew, done, info
