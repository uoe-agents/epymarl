from controllers.custom.basic_controller import CustomBasicMAC
import torch as th


class CustomNonSharedMAC(CustomBasicMAC):
    """MAC for non-shared CustomAgent instances (separate parameters per agent).

    Hidden states are stored as list[n_agents], each element being the
    standard per-agent hidden (list[layer][tuple of (batch, dim)]).
    """

    def init_hidden(self, batch_size):
        self._hidden_states = [
            [tuple(x.expand(batch_size, -1) for x in layer_tuple)
             for layer_tuple in agent_h]
            for agent_h in self.agent.init_hidden()
        ]

    @property
    def hidden_states(self):
        return self._hidden_states

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states = self.agent(agent_inputs, self._hidden_states)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
