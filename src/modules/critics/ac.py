import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        if self.args.use_rnn:
            h = self.init_hidden()
        # make empty torch arrya for qs and hs
        qs = []
        for input in inputs:
            x = F.relu(self.fc1(input))
            if self.args.use_rnn:
                h = self.rnn(x, h)
            else:
                h = F.relu(self.rnn(x))
            q = self.fc3(x)
            qs.append(q)
        q = th.stack(qs, dim=1)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # agent id
        input_shape += self.n_agents
        return input_shape
