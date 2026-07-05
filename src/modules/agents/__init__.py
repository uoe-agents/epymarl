from functools import partial

from .rnn_agent import RNNAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_ns_agent import RNNNSAgent
from .custom import CustomAgent, NSAgent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

REGISTRY["custom"] = CustomAgent
REGISTRY["custom_ns"] = partial(NSAgent, agent=CustomAgent)
