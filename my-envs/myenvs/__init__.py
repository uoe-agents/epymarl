from myenvs.water_bomber_gym import *
from myenvs.simultaneous_env import *

from gym.envs.registration import registry, register, make, spec
import gym

register(
    id="Water-bomber-v1",                     # Environment ID.
    entry_point="myenvs.water_bomber_gym:WaterBomberEnv",  # The entry point for the environment class
    kwargs={
                'x_max':4, 
                'y_max':2, 
                't_max':20, 
                'n_agents':2                                  # Arguments that go to ForagingEnv's __init__ function.
            },
)

register(
    id="Simultaneous-v1",                     # Environment ID.
    entry_point="myenvs.simultaneous_env:SimultaneousEnv",  # The entry point for the environment class
    kwargs={
                "n_agents":8, 
                "n_actions":2, 
                "n_agents_to_defeat_ennemy":None                                   # Arguments that go to ForagingEnv's __init__ function.
            },
)

if __name__ == "__main__":
  env = gym.make('Water-bomber-v1')
  #env = gym.make('module:Env-v0')
