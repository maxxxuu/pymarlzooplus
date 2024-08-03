from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .mat_controller import MATMAC


REGISTRY = {"basic_mac": BasicMAC,
            "non_shared_mac": NonSharedMAC,
            "maddpg_mac": MADDPGMAC,
            "mat_mac": MATMAC
           }
