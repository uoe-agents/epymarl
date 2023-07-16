import torch as th
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from einops import rearrange, reduce, repeat
from modules.critics.mlp import MLP


def generate_other_actions(n_actions, n_agents, device):


    # print(avail_actions.shape)
    # ^ batch n_steps agents actions
    other_acts = [
        th.cat(x) for x in product(*[th.eye(n_actions, device=device) for _ in range(n_agents - 1)])
    ]
    other_acts = th.stack(other_acts)

    return other_acts


class PACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(PACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_actions)

        self.device = "cuda" if args.use_cuda else "cpu"

    def forward(self, batch, t=None, compute_all=False):
        if compute_all:
            inputs, bs, max_t, other_actions = self._build_inputs_all(batch, t=t)
        else:
            inputs, bs, max_t, other_actions = self._build_inputs_cur(batch, t=t)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, other_actions

    def _gen_all_other_actions(self, batch, bs, max_t):

        other_agents_actions = generate_other_actions(self.n_actions, self.n_agents, self.device)
        n_other_actions = other_agents_actions.shape[0]
        other_agents_actions = repeat(other_agents_actions, "e f -> n s a e f", n=bs, s=max_t, a=self.n_agents)
        return other_agents_actions

    def _gen_subsample_other_actions(self, batch, bs, max_t, sample_size):

        avail_actions = batch["avail_actions"]

        # ALL AVAIL ACTIONS ARE ZERO IF EPISODE HAS TERMINATED
        probs =avail_actions/avail_actions.sum(dim=-1).unsqueeze(-1)
        probs = th.nan_to_num(probs, nan=1.0/avail_actions.size(-1))

        avail_dist = th.distributions.OneHotCategorical(probs=probs)
        sample = avail_dist.sample([sample_size])
        samples = []
        for i in range(self.n_agents):
            samples.append(th.cat([sample[:, :, :, j, :] for j in range(self.n_agents) if j != i], dim=-1))
        samples = th.stack(samples)
        samples = rearrange(samples, "i j k l m -> k l i j m")
        return samples

    def _build_inputs_all(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs = th.cat(inputs, dim=-1)

        if self.args.use_subsampling:
            other_actions = self._gen_subsample_other_actions(batch, bs, max_t, self.args.sample_size)
        else:
            other_actions = self._gen_all_other_actions(batch, bs, max_t)

        n_other_actions = other_actions.size(3)

        inputs = repeat(inputs, "n s a f -> n s a e f", e=n_other_actions)
        inputs = th.cat((inputs, other_actions), dim=-1)

        # print(inputs.shape)

        return inputs, bs, max_t, other_actions

    def _build_inputs_cur(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        actions = []
        for i in range(self.n_agents):
            actions.append(th.cat([batch["actions_onehot"][:, :, j].unsqueeze(2) for j in range(self.n_agents)
                                   if j != i], dim=-1))
        actions = th.cat(actions, dim=2)
        inputs.append(actions)
        # inputs.append()
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t, actions

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_actions * (self.n_agents - 1)
        return input_shape

