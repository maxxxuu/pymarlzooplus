from functools import partial

from smac.env import MultiAgentEnv
from envs.gym_wrapper import _GymmaWrapper
from envs.pettingzoo_wrapper import _PettingZooWrapper
from envs.overcooked_wrapper import _OvercookedWrapper
from envs.pressureplate_wrapper import _PressurePlateWrapper
from envs.capturetarget_wrapper import _CaptureTargetWrapper
import envs.lbf_registration
import envs.mpe_registration


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


# In this way, the user don't need to install requirements

try:
    REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
except:
    pass

try:
    REGISTRY["pettingzoo"] = partial(env_fn, env=_PettingZooWrapper)
except:
    pass

try:
    REGISTRY["overcooked"] = partial(env_fn, env=_OvercookedWrapper)
except:
    pass

try:
    REGISTRY["pressureplate"] = partial(env_fn, env=_PressurePlateWrapper)
except:
    pass
  
try:
    REGISTRY["capture_target"] = partial(env_fn, env=_CaptureTargetWrapper)
except:
    pass

