import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector(): # Multinomial distribution of action probabilities
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        self.epsilon = self.schedule.eval(t_env)
        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()
        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector

class EpsilonGreedyActionSelector(): # epsilon greedy action selection
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)
        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class SoftPoliciesSelector(): # Categorical distribution, softmaxed action logits
    def __init__(self, args):
        self.args = args
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions

REGISTRY["soft_policies"] = SoftPoliciesSelector

class ContinuousSelector(): # Means and standard deviations of normal distributions. [:k], [k:]
    def __init__(self, args):
        self.args = args
    def select_action(self,agent_inputs,avail_actions,t_env,test_mode=False):
        with th.no_grad():
          #print("SELECT ACTION called!")
          #print(avail_actions)
          k = agent_inputs.shape[-1]//2
          #print(f"!!!! {k}")
          #print(agent_inputs)
          #print(agent_inputs.shape)
          u, var = agent_inputs[:,:,:k], agent_inputs[:,:,k:]
          u, var = F.tanh(u), th.sqrt(F.softplus(var))
          action_dist = th.distributions.Normal(u, var)
          action = action_dist.sample()
          # For now, clip actions to [0,1] manually.
          action = th.clip(action,min=0,max=1)
          #print(action)
          return action

REGISTRY["continuous"] = ContinuousSelector