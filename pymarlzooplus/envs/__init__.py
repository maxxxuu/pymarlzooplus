from functools import partial

from smac.env import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY_availability = [
    "gymma",
    "pettingzoo",
    "overcooked",
    "pressureplate",
    "capturetarget",
    "boxpushing",
]

# In this way, the user doesn't need to install all requirements
REGISTRY = {}

try:
    import pymarlzooplus.envs.lbf_registration
except ImportError:
    pass

try:
    import pymarlzooplus.envs.mpe_registration
except ImportError:
    pass

try:
    import pymarlzooplus.envs.rware_v1_registration
except ImportError:
    pass

try:
    from pymarlzooplus.envs.gym_wrapper import _GymmaWrapper
    REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
except ImportError:
    pass

try:
  from pymarlzooplus.envs.pettingzoo_wrapper import _PettingZooWrapper
  REGISTRY["pettingzoo"] = partial(env_fn, env=_PettingZooWrapper)
except ImportError:
    pass

try:
    from pymarlzooplus.envs.overcooked_wrapper import _OvercookedWrapper
    REGISTRY["overcooked"] = partial(env_fn, env=_OvercookedWrapper)
except ImportError:
    pass

try:
    from pymarlzooplus.envs.pressureplate_wrapper import _PressurePlateWrapper
    REGISTRY["pressureplate"] = partial(env_fn, env=_PressurePlateWrapper)
except ImportError:
    pass
  
try:
    from pymarlzooplus.envs.capturetarget_wrapper import _CaptureTargetWrapper
    REGISTRY["capturetarget"] = partial(env_fn, env=_CaptureTargetWrapper)
except ImportError:
    pass

try:
    from pymarlzooplus.envs.boxpushing_wrapper import _BoxPushingWrapper
    REGISTRY["boxpushing"] = partial(env_fn, env=_BoxPushingWrapper)
except ImportError:
    pass

