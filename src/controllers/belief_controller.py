import torch as th
import torch.nn as nn

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


class BeliefMAC:
    """MAPPO controller with a shared teammate GRU and per-teammate state."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        if self.n_agents != 1:
            raise ValueError("belief_mac expects one learnable ego agent")
        self.args = args
        self.obs_dim = int(scheme["obs"]["vshape"])
        self.belief_dim = int(getattr(args, "belief_hidden_dim", 64))
        self.n_teammates = int(getattr(args, "belief_n_teammates", 2))
        self.n_actions = int(args.n_actions)

        # The Switching-LBF wrapper appends two teammate action one-hots to obs.
        # The GRU receives the visible ego observation plus one teammate action.
        self.evidence_dim = self.obs_dim + self.n_actions
        self.belief_gru = nn.GRUCell(self.evidence_dim, self.belief_dim)
        actor_input_dim = self.obs_dim + self.n_teammates * self.belief_dim
        if args.obs_agent_id:
            actor_input_dim += self.n_agents
        self.agent = agent_REGISTRY[args.agent](actor_input_dim, args)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.belief_hidden = None

    def parameters(self):
        return list(self.agent.parameters()) + list(self.belief_gru.parameters())

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )
        self.belief_hidden = self.agent.fc1.weight.new_zeros(
            batch_size, self.n_teammates, self.belief_dim
        )

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False):
        del test_mode
        obs = ep_batch["obs"][:, t]
        bs = ep_batch.batch_size
        if self.belief_hidden is None or self.belief_hidden.size(0) != bs:
            self.init_hidden(bs)

        # The final 2*n_actions features are teammate previous-action one-hots.
        action_start = max(0, self.obs_dim - self.n_teammates * self.n_actions)
        action_features = obs[:, :, action_start:]
        beliefs = []
        for teammate_idx in range(self.n_teammates):
            start = teammate_idx * self.n_actions
            end = start + self.n_actions
            teammate_action = action_features[:, :, start:end]
            evidence = th.cat([obs, teammate_action], dim=-1)
            hidden = self.belief_hidden[:, teammate_idx]
            hidden = self.belief_gru(evidence.reshape(bs, -1), hidden)
            beliefs.append(hidden)
        self.belief_hidden = th.stack(beliefs, dim=1)

        actor_inputs = [obs]
        actor_inputs.append(th.cat(beliefs, dim=-1).unsqueeze(1))
        if self.args.obs_agent_id:
            actor_inputs.append(
                th.eye(self.n_agents, device=ep_batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )
        inputs = th.cat(actor_inputs, dim=-1)
        agent_outs, self.hidden_states = self.agent(
            inputs.reshape(bs * self.n_agents, -1), self.hidden_states
        )
        if self.agent_output_type == "pi_logits":
            avail_actions = ep_batch["avail_actions"][:, t]
            reshaped_avail = avail_actions.reshape(bs * self.n_agents, -1)
            agent_outs[reshaped_avail == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(bs, self.n_agents, -1)

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.belief_gru.load_state_dict(other_mac.belief_gru.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.belief_gru.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")
        th.save(self.belief_gru.state_dict(), f"{path}/belief_gru.th")

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage)
        )
        self.belief_gru.load_state_dict(
            th.load(f"{path}/belief_gru.th", map_location=lambda storage, loc: storage)
        )
