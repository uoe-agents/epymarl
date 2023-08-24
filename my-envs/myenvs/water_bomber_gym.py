import functools
import random
from copy import copy

import numpy as np
from gym.spaces import *

from gym import Env
#factorielle = lambda n: n * factorielle(n-1) if n > 0 else 1
from random import randint
import time
from enum import Enum

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

class WaterBomberEnv(Env):
  metadata = {
    "name": "water-bomber-v1",
  }

  def __init__(self, x_max=4, y_max=4, t_max=20, n_agents=2, add_id=False, obs_normalization=True, deterministic=False, gamma=0.9):

    self.X_MAX = x_max
    self.Y_MAX = y_max
    self.T_MAX = t_max
    self.n_agents = n_agents
    self.obs_normalization = obs_normalization
    self.deterministic = deterministic
    self.gamma=gamma

    #self.players = [Player() for _ in range(n_agents)]
    self.possible_agents = [i for i in range(n_agents)]
    self.name_agents = ["water_bomber_"+str(i) for i in range(n_agents)]
    self.symbols = {i:str(i) for i in range(n_agents)}
    #{"water_bomber_"+str(i):str(i) for i in range(n_agents)}
    self.verbose = False

    self.add_id = add_id
    self.length_id = self.n_agents if add_id else 0
    self.one_hot = np.eye(n_agents)

    self.max_discrete_obs = self.get_observation_space(0).nvec
    len_obs = len(self.max_discrete_obs)
    self.observation_space = Tuple([Box(low=-1.0, high=1.0, shape=(len_obs,)) for a in range(self.n_agents)])
    self.action_space = Tuple([self.get_action_space(a) for a in range(self.n_agents)])

  def reset(self, seed=None, options=None, deterministic=False):
    self.agents = copy(self.possible_agents)
    
    if self.deterministic and not (self.X_MAX==4 and self.Y_MAX==2 and self.T_MAX==20 and self.n_agents==2):
        print("env cannot be deterministic")
        self.deterministic = False

    if self.deterministic or deterministic:
      assert self.X_MAX==4 and self.Y_MAX==2 and self.T_MAX==20 and self.n_agents==2
      self.fires = [[2,1],[4,1]]
      self.water_bombers = [[0,0],[2,0]]
    else:
      points = {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      while len(points) < 2*self.n_agents:
        points |= {(randint(0, self.X_MAX), randint(0, self.Y_MAX))}
      list_pos = list(list(x) for x in points)
      
      self.fires = list_pos[:self.n_agents]
      self.water_bombers = list_pos[self.n_agents:]
      #self.water_bombers = {"water_bomber_"+str(i):coor for i, coor in enumerate(list_pos[self.n_agents:])}

    self.has_finished = [False]*self.n_agents

    self.timestep = 0

    observations = self._generate_observations()

    self.reward_opti = self.compute_optimal_reward()

    return observations 

  def step(self, actions):
    infos = {} 
    for agent, action in enumerate(actions):

      x, y = self.water_bombers[agent]

      if [x, y] not in self.fires:
        if action == 0 and not [x,y+1] in self.water_bombers and y<self.Y_MAX:
          self.water_bombers[agent][1] += 1
        elif action == 1 and not [x+1,y] in self.water_bombers and x<self.X_MAX:
          self.water_bombers[agent][0] += 1
        elif action == 2 and not [x,y-1] in self.water_bombers and y>0:
          self.water_bombers[agent][1] -= 1
        elif action == 3 and not [x-1,y] in self.water_bombers and x>0:
          self.water_bombers[agent][0] -= 1
    

    self.has_finished = {a:self.water_bombers[a] in self.fires for a in self.agents}
    rewards = [self._compute_reward() for a in self.agents]
    self.timestep += 1
    truncations = [False for a in self.agents]
    observations = self._generate_observations()
    
    if self.timestep > self.T_MAX:
      truncations = [True for a in self.agents]

    if self.verbose:
      print()
      print("observations",observations)
      print("rewards",rewards)
      #print("terminations",terminations)
      print("truncations",truncations)

    return observations, rewards, truncations, infos


  def render(self):
    grid = np.full((self.Y_MAX+1, self.X_MAX+1), '_')
    for x, y in self.fires:
      grid[y, x] = "F"

    for agent, (x, y) in enumerate(self.water_bombers): #.items():
      grid[y, x] = self.symbols[agent]

    result = "\n".join(["".join([i for i in row]) for row in grid[::-1]])
    print()
    print(result)

  @functools.lru_cache(maxsize=None)
  def get_observation_space(self, agent):
    l = sum([[self.X_MAX+1, self.Y_MAX+1] for _ in range(2*self.n_agents)]+[[self.T_MAX]], []) #
    if self.add_id:
      l += self.length_id*[2]
    return  MultiDiscrete(l)
    #return Dict({ Tuple
    #  'observation': MultiDiscrete(l), #+[13]
    #  'action_mask': MultiBinary(5)
    #})

  @functools.lru_cache(maxsize=None)
  def get_action_space(self, agent):
      return Discrete(4)


  def _is_terminated(self):
    fires = copy(self.fires).sort()
    water_bombers = copy(list(self.water_bombers.values())).sort()

    return fires == water_bombers

  def _compute_reward(self):
    if np.all([self.has_finished[a] for a in self.agents]):
      return 1.0/self.reward_opti 
    else:
      return 0.0 

 
  def normalize_obs(self, obs):
    # ASSUMES ALL AGENTS HAVE SAME OBS SPACE
    normalized_obs = copy(obs)

    agent = self.possible_agents[0]
    normalized_obs = 2*normalized_obs/(self.max_discrete_obs+1) - 1.0
    return normalized_obs

  def get_state(self):
    for agent in self.agents:
      x, y = self.water_bombers[agent]

    occupied_positions = self.water_bombers
    #list(self.water_bombers.values())

    state = sum(self.fires + occupied_positions +[[self.timestep]], [])
    return state

  def _generate_observations(self):
     #+[[self.timestep]]
    state = self.get_state()
    observations = []
    for a in self.agents:
      obs_perso = np.concatenate((state, self.one_hot[a])) if self.add_id else state
      if self.obs_normalization:
        obs_perso = np.array(obs_perso, dtype=float)
        obs_perso = self.normalize_obs(obs_perso)
      else:
        obs_perso = np.array(obs_perso, dtype=int)
      observations.append(obs_perso)
      

    return observations

  def compute_optimal_reward(self):
    norm_1 = lambda x, y: np.linalg.norm(np.array(x, dtype=float)-np.array(y, dtype=float), ord=1)
    duree_min = min([max([norm_1(self.water_bombers[a],f) for f in self.fires]) for a in self.agents])
    return self.T_MAX - duree_min + 2.0

  def get_env_info(self):
    return {
      "n_actions": 4,
      "n_agents": self.n_agents,
    }

def main_1():
  env = WaterBomberEnv(x_max=4, y_max=1, t_max=20, n_agents=2, add_id=True)
  #parallel_api_test(env, num_cycles=1_000_000)
  observations = env.reset(seed=42, deterministic=False, )
  #env.render()
  #print("observations initiale:", observations)
  total_reward = 0.0
  done = False
  while not done:
    # this is where you would insert your policy
    #print(observations)
    actions = [np.random.choice(4) for agent in env.agents]

    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
    #print("actions:",actions)
    env.render()
    print("actions", actions)
    observations, rewards, terminations, infos = env.step(actions)
    print("observations", observations)
    print("rewards", rewards)
    print("terminations", terminations)
    print("infos", infos)
    print()
    done = np.all(np.array(terminations)==True)
    total_reward += np.mean(rewards) 
    #print(actions, observations, rewards, terminations, action_masks)
    #print("rewards:",rewards, "; total reward:", total_reward)

    #env.render()
    time.sleep(0.1)
  env.close()


if __name__ == "__main__":
  main_1()
