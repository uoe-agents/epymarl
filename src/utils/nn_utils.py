import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

# Any keys specified in the config layer dicts (other than "type") are passed
# directly as keyword arguments to the underlying PyTorch layer class.
# Some kwargs can be inferred from the current input shape using `infer_kwargs`.
# These kwargs must NOT be provided explicitly in the config.
net_config = {
    # relu: no kwargs
    "relu": {
        "class": nn.ReLU,
    },
    # linear: in_features (inferred), out_features, bias(optional)
    "linear": {
        "class": nn.Linear,
        "infer_kwargs": {
            "in_features": lambda in_shape: in_shape[0],
        },
        "required": ["out_features"],
    },
    # batchNorm1d: num_features (inferred)
    "batchNorm1d": {
        "class": nn.BatchNorm1d,
        "infer_kwargs": {
            "num_features": lambda in_shape: in_shape[0],
        },
    },
    # flatten: no kwargs
    "flatten": {
        "class": nn.Flatten,
    },
    # gru: input_size (inferred), hidden_size
    "gru": {
        "class": nn.GRUCell,
        "infer_kwargs": {
            "input_size": lambda in_shape: in_shape[0],
        },
        "required": ["hidden_size"],
    },
    # lstm: input_size (inferred), hidden_size
    "lstm": {
        "class": nn.LSTMCell,
        "infer_kwargs": {
            "input_size": lambda in_shape: in_shape[0],
        },
        "required": ["hidden_size"],
    }
}

def get_layer_output_shape(layer, layer_type, input_shape):
    """
    Compute the output shape of a layer given its type and input shape.
    
    Args:
        layer: The PyTorch layer instance
        layer_type: String identifier for the layer type
        input_shape: Shape of the input tensor (tuple, without batch dimension)
    
    Returns:
        output_shape: Shape of the output tensor (tuple, with batch dimension)
    """
    x = torch.empty(1, *input_shape)
    with torch.no_grad():
        if layer_type.startswith("batchNorm"):
            output_shape = (1, *input_shape)
        elif layer_type == "flatten":
            output_shape = (1, prod(input_shape))
        elif layer_type == "lstm":
            output_shape = layer(x)[0].shape
        else:
            output_shape = layer(x).shape
    return output_shape[1:]

def layer_from_dict(layer_dict, input_shape):
    """
    Build a PyTorch layer from a dictionary representation.
    
    Args:
        layer_dict: Dictionary with 'type' key and any additional keys to be used
                    as keyword arguments for the underlying PyTorch layer. All
                    non-'type' keys are passed directly as **kwargs.
        input_shape: Shape of the input tensor (tuple)
    
    Returns:
        layer: An instance of the PyTorch layer class specified in the config
        output_shape: Shape of the output tensor (tuple), no batch dimension
    """
    assert isinstance(layer_dict, dict), f"Expected dictionary, got {type(layer_dict)}"
    assert "type" in layer_dict, f"Layer dictionary must have 'type' key: {layer_dict}"
    
    layer_type = layer_dict["type"]
    assert layer_type in net_config.keys(), f"Unexpected layer type: {layer_type}. Available types: {', '.join(list(net_config.keys()))}"

    layer_config = net_config[layer_type]

    # Start with any kwargs inferred from the input shape
    kwargs = {}
    infer_kwargs = layer_config.get("infer_kwargs", {})
    for name, fn in infer_kwargs.items():
        kwargs[name] = fn(input_shape)

    # Add parameters from layer_dict (user-specified args)
    for key, value in layer_dict.items():
        if key == "type":
            continue
        if key in infer_kwargs:
            raise ValueError(
                f"Layer type '{layer_type}' received YAML kwarg '{key}' which is "
                f"also inferred via infer_kwargs. Remove it from YAML to avoid ambiguity."
            )
        kwargs[key] = value

    # Ensure all required kwargs are present in the YAML
    required_keys = layer_config.get("required", [])
    missing_required = [k for k in required_keys if k not in layer_dict]
    if missing_required:
        raise ValueError(
            f"Layer type '{layer_type}' is missing required arguments: "
            f"{', '.join(missing_required)}"
        )

    # Instantiate layer
    layer = layer_config["class"](**kwargs)
    
    # Compute output shape
    output_shape = get_layer_output_shape(layer, layer_type, input_shape)

    return layer, output_shape

def net_from_args(layer_list, input_shape, target_dim=None, last_layer_bias=False):
    """
    Build a PyTorch Sequential network from a list of layer dictionaries.
    
    Args:
        layer_list: List of dictionaries, each with 'type' key and other arguments
        input_shape: Shape of the input tensor (tuple), no batch dimension
        target_dim: Optional output size (int). If provided and the stack does not already
                    end at that size, a final linear layer is appended.
        last_layer_bias: If True, the auto-appended final linear layer (when target_dim is used)
                         uses bias; if False, bias is disabled on that layer.
    
    Returns:
        net: nn.Sequential network, flattened output
        output_shape: Final output shape (tuple), no batch dimension
    """
    assert isinstance(layer_list, list), f"Expected list of layer dictionaries, got {type(layer_list)}"
    
    # Automatically prepend flatten layer
    layer_list = [{"type": "flatten"}] + layer_list
    
    # Process each layer sequentially
    layers = []
    current_shape = input_shape
    
    for layer_dict in layer_list:
        layer, current_shape = layer_from_dict(layer_dict, current_shape)
        layers.append(layer)
    
    # Optionally append final linear layer if target_dim is provided
    if target_dim is not None:
        assert isinstance(target_dim, int), "Target dimension must be an integer"
        if current_shape != (target_dim,):
            final_layer_dict = {
                "type": "linear",
                "out_features": target_dim,
                "bias": last_layer_bias,
            }
            layer, current_shape = layer_from_dict(final_layer_dict, current_shape)
            layers.append(layer)
    
    return nn.Sequential(*layers), current_shape


class SequentialCustomNetwork(nn.Module):
    def __init__(self, net, input_shape):
        super(SequentialCustomNetwork, self).__init__()
        self.net = net
        self.input_shape = input_shape

    def init_hidden(self, batch_size=1):
        # Traverse the network
        device = next(self.net.parameters()).device
        v = torch.rand(batch_size,*self.input_shape).to(device)
        hidden_lst = []
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.GRUCell):
                    v = layer(v)
                    hidden_lst.append((torch.zeros_like(v),))
                elif isinstance(layer, nn.LSTMCell):
                    v, c = layer(v)
                    hidden_lst.append((torch.zeros_like(v), torch.zeros_like(c)))
                else:
                    v = layer(v)
                    hidden_lst.append(tuple())
        return hidden_lst

    def forward(self, v, hidden_lst=None):
        # Assumes v has shape (batch_size, *input_shape)
        hidden_lst = hidden_lst if hidden_lst is not None else self.init_hidden(batch_size=v.shape[0])
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.GRUCell):
                h = hidden_lst[i][0]
                v = layer(v, h)
                hidden_lst[i] = (v,)
            elif isinstance(layer, nn.LSTMCell):
                h, c = hidden_lst[i]
                v, c = layer(v, (h,c))
                hidden_lst[i] = (v, c)
            else:
                v = layer(v)
        return v, hidden_lst
    
    def get_output_shape(self):
        with torch.no_grad():
            v = torch.empty(1,*self.input_shape)
            v, hidden_lst = self.forward(v)
        return v.shape[1:]

    def __getitem__(self, idx):
        return self.net[idx]