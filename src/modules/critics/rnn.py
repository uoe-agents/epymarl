# Description: RNN critic model.
import torch as th
import torch.nn as nn
import torch.nn.functional as F



class RNN(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, max_t, bs):
        # make sure inputs are of the size (max_t, batch_size, input_shape)
        inputs = inputs.view(max_t, bs, -1)
        bs = inputs.size(1)
        h = self.init_hidden().repeat(bs, 1)
        qs = []
        for input in inputs:
            x = F.relu(self.fc1(input))
            h = self.rnn(x, h)
            q = self.fc3(h)
            qs.append(q)
        q = th.stack(qs).view(bs, max_t, 1)
        return q