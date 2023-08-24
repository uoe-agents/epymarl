import functools
import random
from copy import copy

import numpy as np
from gym.spaces import *

from gym import Env, spaces
#factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
from random import randint
import time
from enum import Enum


#gymkey = f"Simultaneous-{n_agent}ag-{n_actions}empty-{bonus_win}bonus_win-v0"


class CellEntity(Enum):
    # entity encodings for grid observations
    EMPTY = 0
    ENNEMY = 1

class SimultaneousEnv(Env):
    metadata = {
    "name": "water-bomber-env_v0",
    }

    def __init__(self, n_agents, n_actions, n_ennemies=1, n_agents_to_defeat_ennemy=None, obs_normalization=None):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_ennemy = n_ennemies
        if n_agents_to_defeat_ennemy is None:
            self.n_agents_to_defeat_ennemy = n_agents
        else:
            self.n_agents_to_defeat_ennemy = n_agents_to_defeat_ennemy
        self.reward_win = 1.0 # bonus_win*((n_actions+1)**(self.n_agents_to_defeat_ennemy))
        proba_others_attack = (1/n_actions**(n_agents-1))
        #self.reward_death = -n_actions/(1 - (1/n_actions**(n_agents-1))) if n_agents>1 else -0.1
        self.reward_death = -1.0/ (2**(n_agents-1) - 1)
        #-proba_others_attack*(1.0 - proba_others_attack)
        #self.reward_death = -0.1
        self.common_reward = True

        sa_observation_space = Discrete(1) #Space(None)
        self.observation_space = spaces.Tuple(tuple(n_agents * [sa_observation_space]))

        sa_action_space = Discrete(n_actions)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.render_mode = None
        self.reward_range = (self.reward_death, self.reward_win)

        print("reward_win:",self.reward_win)
        print("reward_death:",self.reward_death)
        

    def step(self, actions):
        infos = [{} for a in self.agents]
        actions = np.array(actions).reshape(-1)
        #for a in actions:
        attaquers = actions == 0
        nreward = np.zeros(self.n_agents)
        nreward[attaquers] = self.reward_win if np.sum(attaquers) >= self.n_agents_to_defeat_ennemy else self.reward_death

        nobs = [[0.0] for _ in range(self.n_agents)]
        ndone = [[True] for _ in range(self.n_agents)]
        ninfo = [None for _ in range(self.n_agents)]
        if self.common_reward:
            mean_reward = np.mean(nreward)
            nreward = [mean_reward for _ in range(self.n_agents)]
        return nobs, nreward, ndone, infos
    
    def reset(self):
        return [[0.0] for _ in range(self.n_agents)]

    def get_env_info(self):
        return {
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            }

    def normalize_obs(self, obs):
        return obs
    #def get_avail_agent_actions(self, agent_id=None):
    #    return np.ones(self.n_actions)
    
if __name__ == "__main__":
    n_agents=2
    n_actions=2
    env = SimultaneousEnv(n_agents=n_agents, n_actions=n_actions)
    #print("reward:", -1.0/((n_actions+1)**(n_agents-1)))
    total_rewards = np.zeros(n_agents)
    NB_STEPS = 100000
    nobs = env.reset()
    for _ in trange(NB_STEPS):
        actions = np.random.randint(n_actions, size=n_agents)
        nobs, nreward, ndone, infos = env.step(actions)
        #print("actions", actions)
        #print("nreward", nreward)
        if True:
            print("actions", actions)
            print("nreward", nreward)
        total_rewards += nreward
    print("Mean reward:", total_rewards/NB_STEPS)
