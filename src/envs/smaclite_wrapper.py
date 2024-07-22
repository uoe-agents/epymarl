import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import smaclite  # noqa

from .multiagentenv import MultiAgentEnv


class SMACliteWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, time_limit, **kwargs):
        self.env = gym.make(f"smaclite/{map_name}-v0", seed=seed, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)

        self.n_agents = self.env.unwrapped.n_agents
        self.episode_limit = time_limit

        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.unwrapped.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.unwrapped.get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.unwrapped.obs_size

    def get_state(self):
        return self.env.unwrapped.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.unwrapped.state_size

    def get_avail_actions(self):
        return self.env.unwrapped.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.unwrapped.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs = self.env.reset(seed=seed, options=options)
        return obs, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
