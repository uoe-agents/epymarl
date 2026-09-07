from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa

try:
    from .pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )


class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        max_episode_steps=None,
        **kwargs,
    ):
        # ``lbforaging`` registrations include their own 50-step TimeLimit and
        # an internal ``_max_episode_steps`` attribute.  Passing the requested
        # limit to gym.make controls the outer wrapper; for LBF, unwrapping and
        # patching the base environment removes the inner hard-coded limit too.
        self._env = gym.make(
            f"{key}", max_episode_steps=time_limit, **kwargs
        )
        if key.startswith("lbforaging:"):
            base_env = self._env.unwrapped
            for attr in (
                "_max_episode_steps",
                "_max_steps",
                "max_steps",
                "_step_limit",
            ):
                if hasattr(base_env, attr):
                    setattr(base_env, attr, time_limit)
                    break
            self._env = TimeLimit(base_env, max_episode_steps=time_limit)
        else:
            self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self._is_lbf = key.startswith("lbforaging:")
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

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

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)

        # EPyMARL runners use this flag to distinguish a time-limit timeout
        # from a terminal environment state and keep the value bootstrap valid.
        if truncated:
            self._info = dict(self._info or {})
            self._info["episode_limit"] = True

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

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
        # Keep the historical state contract for the centralized critic.  The
        # full LBF map is exposed separately through get_global_state().
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_global_state(self):
        """Return a fully observable entity state for Oracle-MAPPO.

        LBF's normal vector observation hides food and agents outside the
        acting player's sight.  Concatenating those observations therefore
        is not an oracle: food can remain absent from all three vectors.  The
        underlying ForagingEnv keeps the complete field and player list, so
        expose a fixed-size representation containing every food position and
        level followed by every player position and level.  Keep the legacy
        concatenated observation in ``get_state`` for the centralized critic.
        """
        if not self._is_lbf:
            return self.get_state()

        base_env = self._env.unwrapped
        max_num_food = int(base_env.max_num_food)
        global_state = np.zeros(
            (max_num_food + self.n_agents, 3), dtype=np.float32
        )
        # ``(-1, -1, 0)`` is the same empty-entity convention as LBF's
        # observation vector and keeps the representation at a fixed size.
        global_state[:, :2] = -1.0

        # np.nonzero is row-major, giving food slots a deterministic order.
        for i, (row, col) in enumerate(
            np.argwhere(base_env.field > 0)[:max_num_food]
        ):
            global_state[i] = (row, col, base_env.field[row, col])

        for i, player in enumerate(base_env.players[: self.n_agents]):
            if player.position is not None:
                global_state[max_num_food + i] = (
                    player.position[0],
                    player.position[1],
                    player.level,
                )

        # Keep entity features on comparable scales for the Oracle actor.  LBF
        # coordinates are map indices (e.g. 0..14) while levels are small
        # integers; feeding them raw makes the global columns much larger than
        # the local observation features.  Preserve ``-1`` for empty slots so
        # the fixed-size representation still distinguishes missing entities.
        field_shape = np.asarray(base_env.field.shape[:2], dtype=np.float32)
        row_scale = max(float(field_shape[0] - 1.0), 1.0)
        col_scale = max(float(field_shape[1] - 1.0), 1.0)
        player_level_limit = np.asarray(
            getattr(base_env, "max_player_level", 1), dtype=np.float32
        )
        level_scale = max(
            float(np.max(player_level_limit)) if player_level_limit.size else 1.0,
            float(np.max(base_env.field)) if base_env.field.size else 1.0,
            1.0,
        )
        valid = global_state[:, 0] >= 0.0
        global_state[valid, 0] /= row_scale
        global_state[valid, 1] /= col_scale
        global_state[valid, 2] /= level_scale

        return global_state.reshape(-1)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_global_state_size(self):
        if self._is_lbf:
            base_env = self._env.unwrapped
            return (int(base_env.max_num_food) + self.n_agents) * 3
        return self.get_state_size()

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

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
