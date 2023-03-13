from .centralV import CentralVCritic

import torch
import torch as th
from torch import nn

import math
import numpy as np
from perceiver.model.core import InputAdapter, PerceiverEncoder, CrossAttention
from einops import rearrange


class CentralVCriticSK(CentralVCritic):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.input_shape = self._get_input_shape(scheme)
        self.state_shape = scheme["state"]["vshape"]

        # SAF related setting
        self.use_policy_pool = self.args.use_policy_pool
        self.use_SK = self.args.use_SK
        self.n_policy = self.args.n_policy
        self.N_SK_slots = self.args.N_SK_slots
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.activation = self.args.activation

        # latent kl setting
        self.latent_kl = self.args.latent_kl
        self.latent_dim = self.args.latent_dim

        # SAF module
        self.SAF = Communication_and_policy(input_dim=self.input_shape,
                                            key_dim=self.input_shape,
                                            N_SK_slots=self.N_SK_slots,
                                            n_agents=self.n_agents, n_policy=self.n_policy,
                                            hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                                            activation=self.activation, latent_kl=self.latent_kl,
                                            latent_dim=self.latent_dim)

    def _build_inputs(self, batch, t=None):
        inputs, bs, max_t = super()._build_inputs(batch, t)
        # print(f'{batch.batch_size=}, {batch.max_seq_length=}')
        if self.args.use_SK:
            # Transform batch first using SAF
            # communicate among different agents using SK
            states = inputs[0]
            # print(f'{states.shape=}')

            states_saf = self.SAF(states.clone())
            
            # print(f'{states_saf.shape=}')
            inputs[0] = states_saf
        return inputs, bs, max_t
        

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'swish': nn.SiLU()
}
class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_list, 
        out_dim, 
        std=np.sqrt(2),
        bias_const=0.0,
        activation='tanh'):
        
        super().__init__()
        assert activation in ['relu', 'tanh', 'gelu', 'swish']
        
        self.layers = nn.ModuleList()
        self.layers.append(layer_init(nn.Linear(in_dim, hidden_list[0])))
        self.layers.append(activations[activation])
        
        for i in range(len(hidden_list)-1):
            self.layers.append(layer_init(nn.Linear(hidden_list[i], hidden_list[i+1])))
            self.layers.append(activations[activation])
        self.layers.append(layer_init(nn.Linear(hidden_list[-1],out_dim), std=std, bias_const=bias_const))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

# Input adapater for perceiver
class agent_input_adapter(InputAdapter):
    def __init__(self, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)
        self.pos_encoding = nn.Parameter(
            torch.empty(max_seq_len, num_input_channels))
        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
        b, l, dim = x.shape  # noqa: E741
        p_enc = rearrange(self.pos_encoding[:l], "... -> () ...")
        return x * self.scale + p_enc
        
# ######inter-agent communication


class Communication_and_policy(nn.Module):
    def __init__(self, input_dim, key_dim, N_SK_slots, n_agents, n_policy, hidden_dim, n_layers, activation, latent_kl, latent_dim):
        super(Communication_and_policy, self).__init__()
        self.N_SK_slots = N_SK_slots

        self.n_agents = n_agents

        self.n_policy = n_policy
        self.latent_kl = latent_kl
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.n_agents = n_agents
        self.n_policy = n_policy

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.device = th.device(
            'cuda' if th.cuda.is_available() else 'cpu')

        self.policy_keys = th.nn.Parameter(
            th.randn(self.n_policy, 1, key_dim)).to(self.device)
        self.policy_attn = nn.MultiheadAttention(
            embed_dim=key_dim, num_heads=1, batch_first=False)

        self.query_projector_s1 = MLP(input_dim,
                                      [self.hidden_dim]*self.n_layers,
                                      key_dim,
                                      std=1.0,
                                      activation=self.activation)  # for sending out message to sk

        self.original_state_projector = MLP(input_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation)  # original agent's own state
        self.policy_query_projector = MLP(input_dim,
                                          [self.hidden_dim]*self.n_layers,
                                          key_dim,
                                          std=1.0,
                                          activation=self.activation)  # for query-key attention pick policy form pool

        self.combined_state_projector = MLP(2*key_dim,
                                            [self.hidden_dim]*self.n_layers,
                                            key_dim,
                                            std=1.0,
                                            activation=self.activation).to(self.device)  # responsible for independence of the agent
        # shared knowledge(workspace)

        input_adapter = agent_input_adapter(num_input_channels=key_dim, max_seq_len=n_agents).to(
            self.device)  # position encoding included as well, so we know which agent is which

        self.PerceiverEncoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=N_SK_slots,  # N
            num_latent_channels=key_dim,  # D
            num_cross_attention_qk_channels=key_dim,  # C
            num_cross_attention_heads=1,
            num_self_attention_heads=1,  # small because observational space is small
            num_self_attention_layers_per_block=self.n_layers,
            num_self_attention_blocks=self.n_layers,
            dropout=0.0,
        ).to(self.device)
        self.SK_attention_read = CrossAttention(
            num_heads=1,
            num_q_input_channels=key_dim,
            num_kv_input_channels=key_dim,
            num_qk_channels=key_dim,
            num_v_channels=key_dim,
            dropout=0.0,
        ).to(self.device)

        if self.latent_kl:
            self.encoder = nn.Sequential(
                nn.Linear(int(2*key_dim), 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

            self.encoder_prior = nn.Sequential(
                nn.Linear(key_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, int(2*self.latent_dim)),
            ).to(self.device)

        self.previous_state = th.randn(5, 1, key_dim).to(self.device)

    def forward(self, state):
        # sate has shape (bsz,N_agents,embsz)
        # communicate among agents using perceiver
        state = state.to(self.device).permute(1, 0, 2)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        # message (bsz,N_agent,dim), for communication
        message_to_send = self.query_projector_s1(state)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        # use perceiver arttecture to collect information from all agents by attention

        SK_slots = self.PerceiverEncoder(message_to_send)
        message = self.SK_attention_read(message_to_send, SK_slots)

        # message plus original state
        # shape (bsz,N_agents,2*dim)
        state_with_message = th.cat([state_encoded, message], 2)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_with_message = self.combined_state_projector(
            state_with_message)  # (N_agents,bsz,dim)

        state_with_message = state_with_message.permute(
            1, 0, 2)  # (bsz,N_agents,dim)

        # print(state_with_message.shape)
        return state_with_message

    def forward_NoCommunication(self, state):
        # jsut encoder the original state without communication
        state = state.to(self.device)
        N_agents, bsz, Embsz = state.shape
        state = state.permute(1, 0, 2)
        state_encoded = self.original_state_projector(
            state)  # state_encoded, for agent's internal uses

        state_without_message = th.cat([state_encoded, th.zeros(
            state_encoded.shape).to(self.device)], 2)  # without information from other agents

        state_without_message = state_without_message.permute(
            1, 0, 2)  # (N_agents,bsz,2*dim)

        state_without_message = self.combined_state_projector(
            state_without_message)  # (N_agents,bsz,dim)

        return state_without_message

    def Run_policy_attention(self, state):
        '''
        state hasshape (bsz,N_agents,embsz)
        '''
        state = state.permute(1, 0, 2)  # (N_agents,bsz,embsz)
        state = state.to(self.device)
        # how to pick rules and if they are shared across agents
        query = self.policy_query_projector(state)
        N_agents, bsz, Embsz = query.shape

        keys = self.policy_keys.repeat(1, bsz, 1)  # n_ploicies,bsz,Embsz,

        _, attention_score = self.policy_attn(query, keys, keys)

        attention_score = nn.functional.gumbel_softmax(
            attention_score, tau=1, hard=True, dim=2)  # (Bz, N_agents , N_behavior)

        return attention_score

    def information_bottleneck(self, state_with_message, state_without_message, s_agent_previous_t):


        z_ = self.encoder(
            th.cat((state_with_message, state_without_message), dim=2))
        mu, sigma = z_.chunk(2, dim=2)
        z = (mu + sigma * th.randn_like(sigma)).reshape(z_.shape[0], -1)
        z_prior = self.encoder_prior(s_agent_previous_t)
        mu_prior, sigma_prior = z_prior.chunk(2, dim=2)
        KL = 0.5 * th.sum(((mu - mu_prior) ** 2 + sigma ** 2)/(sigma_prior ** 2 + 1e-8) + th.log(1e-8 + (sigma_prior ** 2)/(
            sigma ** 2 + 1e-8)) - 1) / np.prod(th.cat((state_with_message, state_without_message), dim=2).shape)

        return z, KL