from collections.abc import Iterable
import warnings

import gym
from gym.spaces import flatdim
import numpy as np
from smac.env import MultiAgentEnv

from .wrappers import TimeLimit, FlattenObservation
import pretrained


class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        self._env.seed(self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def step(self, actions):
        """Returns reward, terminated, info"""
        actions = [int(a) for a in actions]
        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return reward, done, {}

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self.original_env, "state_size"):
            return self.original_env.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
