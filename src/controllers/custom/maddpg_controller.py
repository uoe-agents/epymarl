from controllers.custom.basic_controller import CustomBasicMAC
from controllers.custom.non_shared_controller import CustomNonSharedMAC
from controllers.maddpg_controller import gumbel_softmax, onehot_from_logits
import torch as th


class CustomMADDPGMAC(CustomBasicMAC):
    """MADDPG controller for shared-parameter CustomAgent."""

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states_flatten = self.agent(agent_inputs, self._hidden_states_flatten)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs

    def init_hidden_one_agent(self, batch_size):
        self._batch_size = batch_size
        expanded = self.expand_hidden_states(self.agent.init_hidden(), batch_size, n_agents=1)
        self._hidden_states_flatten = self._flatten_hidden(expanded)


class CustomNonSharedMADDPGMAC(CustomNonSharedMAC):
    """MADDPG controller for non-shared-parameter CustomAgent."""

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states = self.agent(agent_inputs, self._hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs
