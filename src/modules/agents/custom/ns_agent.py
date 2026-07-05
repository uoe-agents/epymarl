import torch.nn as nn
import torch as th

from modules.agents.custom.custom_agent import CustomAgent

class NSAgent(nn.Module):
    """
    Non-shared parameter agent wrapper.
    Creates a separate instance of the base agent for each agent in the environment.
    """
    def __init__(self, input_shape, args, agent=CustomAgent):
        super(NSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList(
            [agent(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        """Initialise hidden state for all (non-shared) agents.

        Returns list[n_agents] where each element is the per-agent hidden
        (list[layer][tuple of tensors with shape (1, hidden_dim)]).
        """
        return [agent.init_hidden() for agent in self.agents]

    def forward(self, inputs, hidden_state) -> tuple[th.Tensor, list]:
        """Forward pass for non-shared agents.

        hidden_state: list[n_agents] where each element is the per-agent hidden
                      (list[layer][tuple of tensors with shape (batch, hidden_dim)])

        Returns:
            q_out: tensor of shape (batch * n_agents, n_actions), flattened across agents.
            hiddens: list[n_agents] of per-agent hidden states (same structure as hidden_state).
        """
        inputs = inputs.view(-1, self.n_agents, self.input_shape)

        hiddens = []
        qs = []
        for i in range(self.n_agents):
            q, h = self.agents[i](inputs[:, i], hidden_state[i])
            qs.append(q)
            hiddens.append(h)

        out_dim = qs[0].size(-1)
        q_out = th.stack(qs, dim=1).reshape(-1, out_dim)
        return q_out, hiddens

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
