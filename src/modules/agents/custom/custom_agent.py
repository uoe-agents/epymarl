import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_utils import net_from_args, SequentialCustomNetwork

class CustomAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CustomAgent, self).__init__()
        self.args = args

        if not isinstance(input_shape, tuple): 
            self.in_shape = (input_shape,)
        else:
            self.in_shape = input_shape
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        target_dim = args.n_actions
        self.net, self.out_dim = net_from_args(
            args.agent_arch,
            self.in_shape,
            target_dim=target_dim,
            last_layer_bias=args.last_layer_bias,
        )
        self.net = SequentialCustomNetwork(self.net, self.in_shape)
        
    def forward(self, input, h=None):
        v, h = self.net(input, h)
        return v, h

    def init_hidden(self, batch_size=1):
        return self.net.init_hidden(batch_size)
