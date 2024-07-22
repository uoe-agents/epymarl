import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, norm_in=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # create network layers
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(self.in_fn(x)))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out

class DDPG():

    def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim, hidden, lr):
        self.policy = MLP(actor_input_dim, actor_output_dim, hidden)

    def step(self, obs, explore=False):
        obs = torch.Tensor(obs).unsqueeze(0)
        action = self.policy(obs)
        action = action.argmax().cpu().item()
        return action

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])