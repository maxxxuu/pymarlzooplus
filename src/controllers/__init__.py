from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .happo_controller import happoMAC

REGISTRY = {"basic_mac": BasicMAC,
            "non_shared_mac": NonSharedMAC,
            "maddpg_mac": MADDPGMAC,
            "happo_mac": happoMAC}
