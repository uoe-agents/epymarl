REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC, NonSharedMADDPGMAC
from .custom import (
    CustomBasicMAC,
    CustomNonSharedMAC,
    CustomMADDPGMAC,
    CustomNonSharedMADDPGMAC,
)

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["maddpg_ns_mac"] = NonSharedMADDPGMAC

REGISTRY["custom_basic_mac"] = CustomBasicMAC
REGISTRY["custom_non_shared_mac"] = CustomNonSharedMAC
REGISTRY["custom_maddpg_mac"] = CustomMADDPGMAC
REGISTRY["custom_maddpg_ns_mac"] = CustomNonSharedMADDPGMAC
