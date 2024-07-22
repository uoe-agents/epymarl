from pathlib import Path
import yaml

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .multiagentenv import MultiAgentEnv


SMACv2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def get_scenario_names():
    return [p.name for p in SMACv2_CONFIG_DIR.iterdir()]


def load_scenario(map_name, **kwargs):
    scenario_path = SMACv2_CONFIG_DIR / f"{map_name}.yaml"
    with open(scenario_path, "r") as f:
        scenario_args = yaml.load(f, Loader=yaml.FullLoader)
    scenario_args.update(kwargs)
    return StarCraftCapabilityEnvWrapper(**scenario_args["env_args"])


class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.env = load_scenario(map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()


if __name__ == "__main__":
    for scenario in get_scenario_names():
        env = load_scenario(scenario)
        env_info = env.get_env_info()
        # print name of config, number of agents, state shape, observation shape, action shape
        print(
            scenario,
            env_info["n_agents"],
            env_info["state_shape"],
            env_info["obs_shape"],
            env_info["n_actions"],
        )
        print()
