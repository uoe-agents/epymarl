import os
import sys

from smac.env import StarCraft2Env
import smaclite  # noqa

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smac_wrapper import SMACWrapper
from .smaclite_wrapper import SMACliteWrapper


def smac_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    return SMACWrapper(StarCraft2Env(**kwargs))


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAClite only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = smac_fn

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

REGISTRY["gymma"] = gymma_fn
