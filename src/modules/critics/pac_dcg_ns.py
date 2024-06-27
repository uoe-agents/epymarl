# code adapted from https://github.com/wendelinboehmer/dcg
import contextlib
import itertools

import numpy as np
import torch as th
import torch.nn as nn
import torch_scatter

from modules.agents import REGISTRY as agent_REGISTRY


class DCGCriticNS:
    """Implements DCG without any parameter sharing between agents (Boehmer et al., 2020)."""

    # ================================ Constructors ===================================================================

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.n_actions = args.n_actions
        self.payoff_rank = args.cg_payoff_rank
        self.payoff_decomposition = (
            isinstance(self.payoff_rank, int) and self.payoff_rank > 0
        )
        self.iterations = args.msg_iterations
        self.normalized = args.msg_normalized
        self.anytime = args.msg_anytime

        # New utilities and payoffs
        self.utility_fun = [
            self._mlp(
                self.args.hidden_dim, args.cg_utilities_hidden_dim, self.n_actions
            )
            for _ in range(self.n_agents)
        ]

        payoff_out = (
            2 * self.payoff_rank * self.n_actions
            if self.payoff_decomposition
            else self.n_actions**2
        )

        self.edges_from = None
        self.edges_to = None
        self.edges_n_in = None
        self._set_edges(self._edge_list(args.cg_edges))

        self.payoff_fun = [
            self._mlp(2 * self.args.hidden_dim, args.cg_payoffs_hidden_dim, payoff_out)
            for _ in range(len(self.edges_from))
        ]

    # ================== DCG Core Methods =============================================================================

    def annotations(self, ep_batch, t, compute_grads=False, actions=None):
        """Returns all outputs of the utility and payoff functions."""
        with th.no_grad() if not compute_grads else contextlib.suppress():
            # Compute all hidden states
            agent_inputs = self._build_inputs(ep_batch, t).view(
                ep_batch.batch_size, self.n_agents, -1
            )
            for i, ag in enumerate(self.agents):
                self.hidden_states[i] = ag(
                    agent_inputs[:, i, :], self.hidden_states[i]
                )[1].view(ep_batch.batch_size, -1)
            # Compute all utility functions
            f_i, f_ij = [], []
            for i, f in enumerate(self.utility_fun):
                f_i.append(f(self.hidden_states[i]).reshape(ep_batch.batch_size, -1))
            f_i = th.stack(f_i, dim=-2)
            # Compute all payoff functions
            if len(self.payoff_fun) > 0:
                for i, f in enumerate(self.payoff_fun):
                    f_ij.append(self.single_payoff(f, i, self.hidden_states))
                f_ij = th.stack(f_ij, dim=-3)
            else:
                f_ij = f_i.new_zeros(*f_i.shape[:-2], 0, self.n_actions, self.n_actions)
        return f_i, f_ij

    def single_payoff(self, payoff_fun, edge, hidden_states):
        """Computes one payoff at a time, as each payoff function does have different parameters."""
        # Construct the inputs for all edges' payoff functions and their flipped counterparts
        n = self.n_actions
        inputs = th.stack(
            [
                th.cat(
                    [
                        hidden_states[self.edges_from[edge]],
                        hidden_states[self.edges_to[edge]],
                    ],
                    dim=-1,
                ),
                th.cat(
                    [
                        hidden_states[self.edges_to[edge]],
                        hidden_states[self.edges_from[edge]],
                    ],
                    dim=-1,
                ),
            ],
            dim=0,
        )
        # Without action batching, all payoffs of a sample are computed at once
        output = payoff_fun(inputs)
        if self.payoff_decomposition:
            # If the payoff matrix is decomposed, we need to de-decompose it here: ...
            dim = list(output.shape[:-1])
            # ... reshape output into left and right bases of the matrix, ...
            output = output.view(*[np.prod(dim) * self.payoff_rank, 2, n])
            # ... outer product between left and right bases, ...
            output = th.bmm(
                output[:, 0, :].unsqueeze(dim=-1), output[:, 1, :].unsqueeze(dim=-2)
            )
            # ... and finally sum over the above outer products of payoff_rank base-pairs.
            output = output.view(*(dim + [self.payoff_rank, n, n])).sum(dim=-3)
        else:
            # Without decomposition, the payoff_fun output must only be reshaped
            output = output.view(*output.shape[:-1], n, n)
        # The output of the backward messages must be transposed
        output[1] = output[1].transpose(dim0=-2, dim1=-1).clone()
        # Compute the symmetric average of each edge with it's flipped counterpart
        return output.mean(dim=0)

    # ================== Override methods of DeepCoordinationGraphMAC =================================================

    def _edge_list(self, arg):
        """Specifies edges for various topologies."""
        edges = []
        wrong_arg = (
            "Parameter cg_edges must be either a string:{'vdn','line','cycle','star','full'}, "
            "an int for the number of random edges (<= n_agents!), "
            "or a list of either int-tuple or list-with-two-int-each for direct specification."
        )
        # Parameter cg_edges must be either a string:{'vdn','line','cycle','star','full'}, ...
        if isinstance(arg, str):
            if arg == "vdn":  # no edges = VDN
                pass
            elif arg == "line":  # arrange agents in a line
                edges = [(i, i + 1) for i in range(self.n_agents - 1)]
            elif arg == "cycle":  # arrange agents in a circle
                edges = [(i, i + 1) for i in range(self.n_agents - 1)] + [
                    (self.n_agents - 1, 0)
                ]
            elif arg == "star":  # arrange all agents in a star around agent 0
                edges = [(0, i + 1) for i in range(self.n_agents - 1)]
            elif arg == "full":  # fully connected CG
                edges = [
                    [(j, i + j + 1) for i in range(self.n_agents - j - 1)]
                    for j in range(self.n_agents - 1)
                ]
                edges = [e for l in edges for e in l]
            else:
                assert False, wrong_arg
        # ... an int for the number of random edges (<= (n_agents-1)!), ...
        if isinstance(arg, int):
            assert 0 <= arg <= factorial(self.n_agents - 1), wrong_arg
            for i in range(arg):
                found = False
                while not found:
                    e = (randrange(self.n_agents), randrange(self.n_agents))
                    if e[0] != e[1] and e not in edges and (e[1], e[0]) not in edges:
                        edges.append(e)
                        found = True
        # ... or a list of either int-tuple or list-with-two-int-each for direct specification.
        if isinstance(arg, list):
            assert all(
                [
                    (isinstance(l, list) or isinstance(l, tuple))
                    and (len(l) == 2 and all([isinstance(i, int) for i in l]))
                    for l in arg
                ]
            ), wrong_arg
            edges = arg
        return edges

    def q_values(self, f_i, f_ij, actions):
        """Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020)."""
        n_batches = actions.shape[0]
        # Use the utilities for the chosen actions
        values = f_i.gather(dim=-1, index=actions).squeeze(dim=-1).mean(dim=-1)
        # Use the payoffs for the chosen actions (if the CG contains edges)
        if len(self.edges_from) > 0:
            f_ij = f_ij.view(
                n_batches, len(self.edges_from), self.n_actions * self.n_actions
            )
            edge_actions = actions.gather(
                dim=-2, index=self.edges_from.view(1, -1, 1).expand(n_batches, -1, 1)
            ) * self.n_actions + actions.gather(
                dim=-2, index=self.edges_to.view(1, -1, 1).expand(n_batches, -1, 1)
            )
            values = values + f_ij.gather(dim=-1, index=edge_actions).squeeze(
                dim=-1
            ).mean(dim=-1)
        # Return the Q-values for the given actions
        return values

    def greedy(self, f_i, f_ij, available_actions=None):
        """Finds the maximum Q-values and corresponding greedy actions for given utilities and payoffs.
        (Algorithm 3 in Boehmer et al., 2020)"""
        # All relevant tensors should be double to reduce accumulating precision loss
        in_f_i, f_i = f_i, f_i.double() / self.n_agents
        in_f_ij, f_ij = f_ij, f_ij.double() / len(self.edges_from)
        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        if available_actions is not None:
            f_i = f_i.masked_fill(available_actions == 0, -float("inf"))
        # Initialize best seen value and actions for anytime-extension
        best_value = in_f_i.new_empty(f_i.shape[0]).fill_(-float("inf"))
        best_actions = f_i.new_empty(
            best_value.shape[0], self.n_agents, 1, dtype=th.int64, device=f_i.device
        )
        # Without edges (or iterations), CG would be the same as VDN: mean(f_i)
        utils = f_i
        # Perform message passing for self.iterations: [0] are messages to *edges_to*, [1] are messages to *edges_from*
        if len(self.edges_from) > 0 and self.iterations > 0:
            messages = f_i.new_zeros(
                2, f_i.shape[0], len(self.edges_from), self.n_actions
            )
            for iteration in range(self.iterations):
                # Recompute messages: joint utility for each edge: "sender Q-value"-"message from receiver"+payoffs/E
                joint0 = (utils[:, self.edges_from] - messages[1]).unsqueeze(
                    dim=-1
                ) + f_ij
                joint1 = (utils[:, self.edges_to] - messages[0]).unsqueeze(
                    dim=-1
                ) + f_ij.transpose(dim0=-2, dim1=-1)
                # Maximize the joint Q-value over the action of the sender
                messages[0] = joint0.max(dim=-2)[0]
                messages[1] = joint1.max(dim=-2)[0]
                # Normalization as in Kok and Vlassis (2006) and Wainwright et al. (2004)
                if self.normalized:
                    messages -= messages.mean(dim=-1, keepdim=True)
                # Create the current utilities of all agents, based on the messages
                msg = torch_scatter.scatter_add(
                    src=messages[0], index=self.edges_to, dim=1, dim_size=self.n_agents
                )
                msg += torch_scatter.scatter_add(
                    src=messages[1],
                    index=self.edges_from,
                    dim=1,
                    dim_size=self.n_agents,
                )
                utils = f_i + msg
                # Anytime extension (Kok and Vlassis, 2006)
                if self.anytime:
                    # Find currently best actions and the (true) value of these actions
                    actions = utils.max(dim=-1, keepdim=True)[1]
                    value = self.q_values(in_f_i, in_f_ij, actions)
                    # Update best_actions only for the batches that have a higher value than best_value
                    change = value > best_value
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]
        # Return the greedy actions and the corresponding message output averaged across agents
        if not self.anytime or len(self.edges_from) == 0 or self.iterations <= 0:
            _, best_actions = utils.max(dim=-1, keepdim=True)
        return best_actions

    def _set_edges(self, edge_list):
        """Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation."""
        self.edges_from = th.zeros(len(edge_list), dtype=th.long)
        self.edges_to = th.zeros(len(edge_list), dtype=th.long)
        for i, edge in enumerate(edge_list):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(
            src=self.edges_to.new_ones(len(self.edges_to)),
            index=self.edges_to,
            dim=0,
            dim_size=self.n_agents,
        ) + torch_scatter.scatter_add(
            src=self.edges_to.new_ones(len(self.edges_to)),
            index=self.edges_from,
            dim=0,
            dim_size=self.n_agents,
        )
        self.edges_n_in = self.edges_n_in.float()

    def _build_agents(self, input_shape):
        """Overloads method to build a list of input-encoders for the different agents."""
        self.agents = [
            agent_REGISTRY["rnn_feat"](input_shape, self.args)
            for _ in range(self.n_agents)
        ]

    def cuda(self):
        """Overloads methornn_d to make sure all encoders, utilities and payoffs are on the GPU."""
        for ag in self.agents:
            ag.cuda()
        for f in self.utility_fun:
            f.cuda()
        for f in self.payoff_fun:
            f.cuda()
        if self.edges_from is not None:
            self.edges_from = self.edges_from.cuda()
            self.edges_to = self.edges_to.cuda()
            self.edges_n_in = self.edges_n_in.cuda()

    def parameters(self):
        """Overloads method to make sure the parameters of all encoders, utilities and payoffs are returned."""
        param = itertools.chain(
            *[ag.parameters() for ag in self.agents],
            *[f.parameters() for f in self.utility_fun],
            *[f.parameters() for f in self.payoff_fun],
        )

        return param

    def state_dict(self):
        return (
            [ag.state_dict() for ag in self.agents]
            + [f.state_dict() for f in self.utility_fun]
            + [f.state_dict() for f in self.payoff_fun]
        )

    def load_state_dict(self, other_mac):
        """Overloads method to make sure the parameters of all encoders, utilities and payoffs are swapped."""
        for i in range(len(self.agents)):
            self.agents[i].load_state_dict(other_mac.agents[i].state_dict())
        for i in range(len(self.utility_fun)):
            self.utility_fun[i].load_state_dict(other_mac.utility_fun[i].state_dict())
        for i in range(len(self.payoff_fun)):
            self.payoff_fun[i].load_state_dict(other_mac.payoff_fun[i].state_dict())

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """Creates an MLP with the specified input and output dimensions and (optional) hidden layers."""
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        return input_shape

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs = batch["state"][:, ts].repeat(1, self.n_agents, 1)
        return inputs

    def init_hidden(self, batch_size):
        """Overloads method to make sure the hidden states of all agents are intialized."""
        self.hidden_states = [
            ag.init_hidden().expand(batch_size, -1) for ag in self.agents
        ]  # bv

    def forward(
        self,
        ep_batch,
        t,
        actions=None,
        policy_mode=True,
        test_mode=False,
        compute_grads=False,
    ):
        """This is the main function that is called by learner and runner.
        If policy_mode=True,    returns the greedy policy (for controller) for the given ep_batch at time t.
        If policy_mode=False,   returns either the Q-values for given 'actions'
                                        or the actions of of the greedy policy for 'actions==None'."""
        # Get the utilities and payoffs after observing time step t
        f_i, f_ij = self.annotations(ep_batch, t, compute_grads, actions)
        # We either return the values for the given batch and actions...
        if actions is not None and not policy_mode:
            values = self.q_values(f_i, f_ij, actions)
            return values
        # ... or greedy actions  ... or the computed Q-values (for the learner)
        actions = self.greedy(
            f_i, f_ij, available_actions=ep_batch["avail_actions"][:, t]
        )
        if policy_mode:  # ... either as policy tensor for the runner ...
            policy = f_i.new_zeros(ep_batch.batch_size, self.n_agents, self.n_actions)
            policy.scatter_(
                dim=-1, index=actions, src=policy.new_ones(1, 1, 1).expand_as(actions)
            )
            return policy
        else:  # ... or as action tensor for the learner
            return actions
