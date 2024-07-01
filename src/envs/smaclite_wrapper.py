import gymnasium as gym
from gymnasium.spaces import flatdim
import smaclite

from .multiagentenv import MultiAgentEnv


class SMACliteWrapper(MultiAgentEnv):
    def __init__(self, map_name, **kwargs):
        self.env = gym.make(f"smaclite:{map_name}-v0", **kwargs)
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        obss, rews, terminated, info = self.env.step(actions)
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.__get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.__get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.obs_size

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.state_size

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss = self.env.reset(seed=seed, options=options, return_info=False)
        return obss, self.env.__get_info()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)


# registration
for preset in smaclite.env.maps.map.MapPreset:
    map_info = preset.value
    gym.register(
        f"smaclite:{map_info.name}-v0",
        entry_point="smaclite.env:SMACliteEnv",
        kwargs={"map_info": map_info},
    )
    gym.register("smaclite:custom-v0", entry_point="smaclite.env:SMACliteEnv")
