from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class CustomBasicMAC:
    """Multi-agent controller for CustomAgent (shared parameters).

    Hidden states use the custom format: list[layer][tuple of tensors],
    supporting arbitrary recurrent architectures defined via agent_arch.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        action_selector = getattr(args, "action_selector", None)
        self.action_selector = None if action_selector is None else action_REGISTRY[action_selector](args)

        self._hidden_states_flatten = None
        self._batch_size = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    @property
    def hidden_states(self):
        if self._hidden_states_flatten is None:
            return None
        return self._unflatten_hidden(self._hidden_states_flatten)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states_flatten = self.agent(agent_inputs, self._hidden_states_flatten)

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self._batch_size = batch_size
        expanded = self.expand_hidden_states(self.agent.init_hidden(), batch_size)
        self._hidden_states_flatten = self._flatten_hidden(expanded)

    def expand_hidden_states(self, hidden_states, batch_size, n_agents=None):
        """Expand agent-produced hidden states to (batch_size, n_agents, dim)."""
        n_agents = n_agents if n_agents is not None else self.n_agents
        return [
            tuple(x.unsqueeze(0).expand(batch_size, n_agents, -1) for x in h)
            for h in hidden_states
        ]

    def _flatten_hidden(self, hidden_states):
        """(batch, n_agents, dim) -> (batch*n_agents, dim) per recurrent layer."""
        return [tuple(x.reshape(-1, x.shape[-1]) for x in h) for h in hidden_states]

    def _unflatten_hidden(self, hidden_states):
        """(batch*n_agents, dim) -> (batch, n_agents, dim) per recurrent layer."""
        return [tuple(x.reshape(self._batch_size, -1, x.shape[-1]) for x in h) for h in hidden_states]

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
